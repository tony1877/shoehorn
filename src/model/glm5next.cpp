#include "model/glm5next.h"

#include "kernels/moe_cpu.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace e8::model {

namespace {

constexpr int kGraphSize = 32768;
constexpr int kKvPad     = 256;

// "glm5-next" (llama.cpp's converter) or "glm5next" (some quantizers), keys prefixed the same way
thread_local std::string g_arch = "glm5-next";
std::string key(const char * k) {
    return g_arch + "." + k;
}

// the MLA latent is both K and V of a 512-wide head: f16 (or f32) only, as flash attention's quantized paths at that
// head size are the least exercised; at ~1 KB per token per layer it is small anyway
ggml_type latent_type(const LoadOptions & o) {
    return o.kv_type == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
}

} // namespace

bool Glm5NextHparams::load(const GgufSet & s, std::string & err) {
    const GgufFile & f = s.meta();
    if (f.arch() != "glm5-next" && f.arch() != "glm5next") {
        err = "architecture \"" + f.arch() + "\" is not glm5-next";
        return false;
    }
    g_arch = f.arch();
    const int64_t n_all = f.i64(key("block_count"), 0);
    n_layer             = n_all - f.i64(key("nextn_predict_layers"), 0);
    n_embd              = f.i64(key("embedding_length"), 0);
    n_head              = f.i64(key("attention.head_count"), 0);
    n_ctx_train         = f.i64(key("context_length"), 0);
    n_ff                = f.i64(key("feed_forward_length"), 0);
    rms_eps             = f.f32(key("attention.layer_norm_rms_epsilon"), 1e-5f);
    norm_eps            = f.f32(key("attention.layer_norm_epsilon"), 1e-6f);
    kda_dim             = f.i64(key("kda.head_dim"), 0);
    kda_lower_bound     = f.f32(key("kda.gate_lower_bound"), -INFINITY);
    d_conv              = f.i64(key("ssm.conv_kernel"), 0);
    q_lora              = f.i64(key("attention.q_lora_rank"), 0);
    kv_lora             = f.i64(key("attention.kv_lora_rank"), 0);
    k_mla               = f.i64(key("attention.key_length_mla"), 0);
    v_mla               = f.i64(key("attention.value_length_mla"), 0);
    idx_n_head          = f.i64(key("attention.indexer.head_count"), 0);
    idx_dim             = f.i64(key("attention.indexer.key_length"), 0);
    idx_top_k           = f.i64(key("attention.indexer.top_k"), 0);
    kpool               = f.i64(key("attention.indexer.kpool"), 0);
    select_tail         = f.b(key("attention.indexer.kpool_select_tail"), true);
    hc                  = f.i64(key("hyper_connection.count"), 0);
    hc_iters            = f.i64(key("hyper_connection.sinkhorn_iterations"), 0);
    hc_eps              = f.f32(key("hyper_connection.epsilon"), 1e-6f);
    n_expert            = f.i64(key("expert_count"), 0);
    n_expert_used       = f.i64(key("expert_used_count"), 0);
    n_ff_exp            = f.i64(key("expert_feed_forward_length"), 0);
    n_dense             = f.i64(key("leading_dense_block_count"), 0);
    expert_weights_scale = f.f32(key("expert_weights_scale"), 1.0f);
    expert_weights_norm  = f.b(key("expert_weights_norm"), true);
    add_bos             = f.b("tokenizer.ggml.add_bos_token", false);
    bos                 = f.i64("tokenizer.ggml.bos_token_id", -1);
    if (f.i64(key("expert_gating_func"), 2) != 2) {
        err = "glm5-next: only the sigmoid router is supported";
        return false;
    }

    auto per_layer = [&](const char * k, float def) {
        std::vector<float> v = f.f32_arr(key(k));
        if (v.size() == 1) v.assign((size_t) n_all, v[0]);
        v.resize((size_t) n_all, def);
        return v;
    };
    clamp_exp   = per_layer("swiglu_clamp_exp", 0.0f);
    clamp_shexp = f.has(key("swiglu_clamp_shexp")) ? per_layer("swiglu_clamp_shexp", 0.0f) : clamp_exp;

    const std::vector<int64_t> kv = f.i64_arr(key("attention.head_count_kv"));
    recurrent.assign((size_t) n_layer, false);
    for (int64_t il = 0; il < n_layer; il++) recurrent[(size_t) il] = il < (int64_t) kv.size() && kv[(size_t) il] == 0;
    const std::vector<int64_t> it = f.i64_arr(key("attention.indexer.types"));
    idx_full.assign((size_t) n_layer, true);
    for (int64_t il = 0; il < n_layer && il < (int64_t) it.size(); il++) idx_full[(size_t) il] = it[(size_t) il] != 0;

    const ggml_tensor * emb = s.tensor("token_embd.weight");
    n_vocab                 = emb ? emb->ne[1] : 0;
    if (n_layer <= 0 || n_embd <= 0 || n_head <= 0 || n_vocab <= 0 || hc != 4 || hc_iters <= 0 || kda_dim <= 0 ||
        d_conv < 2 || kv_lora <= 0 || k_mla <= 0 || v_mla <= 0 || idx_dim <= 0 || idx_n_head <= 0 || kpool < 2 ||
        idx_top_k % kpool != 0 || n_expert <= 0 || n_expert_used <= 0 || !select_tail ||
        f.i64(key("rope.dimension_count"), 0) != 0) {
        err = "incomplete or unsupported glm5-next hyperparameters";
        return false;
    }
    return true;
}

Glm5Next::Glm5Next() = default;

Glm5Next::~Glm5Next() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (mtp_buf_) ggml_backend_buffer_free(mtp_buf_);
    if (mtp_ctx_) ggml_free(mtp_ctx_);
    for (int i = 0; i < 2; i++) {
        if (rec_buf_[i]) ggml_backend_buffer_free(rec_buf_[i]);
        if (rec_ctx_[i]) ggml_free(rec_ctx_[i]);
    }
    if (ec_.buf) ggml_backend_buffer_free(ec_.buf);
    if (ec_.ctx) ggml_free(ec_.ctx);
    if (wbuf_gpu_) ggml_backend_buffer_free(wbuf_gpu_);
    for (auto b : map_bufs_) {
        if (b) ggml_backend_buffer_free(b);
    }
    for (int i = 0; i < 2; i++) {
        if (sbuf_[i]) ggml_backend_buffer_free(sbuf_[i]);
        if (wctx_[i]) ggml_free(wctx_[i]);
        if (sctx_[i]) ggml_free(sctx_[i]);
    }
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
}

bool Glm5Next::load(const std::string & path, const LoadOptions & opt, std::string & err) {
    opt_ = opt;
    if (!file_.open(path, err) || !hp_.load(file_, err)) return false;
    const auto & h = hp_;

    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) {
        err = "cannot initialize the CPU backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu_, opt.n_threads > 0 ? opt.n_threads : (int) std::max(1u, std::thread::hardware_concurrency()));
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) gpu_ = ggml_backend_dev_init(dev, nullptr);
    const int  n_gpu_layers = gpu_ ? std::clamp(opt.n_gpu_layers, 0, (int) h.n_layer) : 0;
    const bool out_gpu      = gpu_ && opt.output_gpu && n_gpu_layers > 0;

    const size_t     n_tens = file_.tensors().size() + 8;
    ggml_init_params ip     = { ggml_tensor_overhead() * n_tens, nullptr, true };
    wctx_[0]                = ggml_init(ip);
    wctx_[1]                = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> to_gpu, to_map;
    bool missing = false;
    auto make = [&](const std::string & name, bool gpu, bool required = true) -> ggml_tensor * {
        const ggml_tensor * m = file_.tensor(name);
        if (!m) {
            if (required && !missing) {
                err     = "tensor missing: " + name;
                missing = true;
            }
            return nullptr;
        }
        ggml_tensor * t = ggml_dup_tensor(wctx_[gpu ? 0 : 1], m);
        ggml_set_name(t, name.c_str());
        (gpu ? to_gpu : to_map).emplace_back(t, m);
        return t;
    };

    tok_embd_ = file_.tensor("token_embd.weight");
    if (!tok_embd_) {
        err = "tensor missing: token_embd.weight";
        return false;
    }
    output_      = make("output.weight", out_gpu);
    output_norm_ = make("output_norm.weight", out_gpu);

    layers_.resize((size_t) h.n_layer);
    for (int64_t il = 0; il < h.n_layer; il++) {
        Glm5NextLayer &   L = layers_[(size_t) il];
        const bool        g = il >= h.n_layer - n_gpu_layers;
        const std::string p = "blk." + std::to_string(il) + ".";
        L.on_gpu    = g;
        L.attn_norm = make(p + "attn_norm.weight", g);
        L.ffn_norm  = make(p + "ffn_norm.weight", g);
        L.hca_fn    = make(p + "hc_attn_fn.weight", g);
        L.hca_base  = make(p + "hc_attn_base.weight", g);
        L.hca_scale = make(p + "hc_attn_scale.weight", g);
        L.hcf_fn    = make(p + "hc_ffn_fn.weight", g);
        L.hcf_base  = make(p + "hc_ffn_base.weight", g);
        L.hcf_scale = make(p + "hc_ffn_scale.weight", g);
        if (h.recurrent[(size_t) il]) {
            L.wq     = make(p + "attn_q.weight", g);
            L.wk     = make(p + "attn_k.weight", g);
            L.wv     = make(p + "attn_v.weight", g);
            L.conv_q = make(p + "ssm_conv1d_q.weight", g);
            L.conv_k = make(p + "ssm_conv1d_k.weight", g);
            L.conv_v = make(p + "ssm_conv1d_v.weight", g);
            L.f_a    = make(p + "ssm_f_a.weight", g);
            L.f_b    = make(p + "ssm_f_b.weight", g);
            L.beta   = make(p + "ssm_beta.weight", g);
            L.a      = make(p + "ssm_a", g);
            L.dt_b   = make(p + "ssm_dt.bias", g);
            L.g_a    = make(p + "ssm_g_a.weight", g);
            L.g_b    = make(p + "ssm_g_b.weight", g);
            L.o_norm = make(p + "ssm_norm.weight", g);
        } else {
            L.q_a       = make(p + "attn_q_a.weight", g);
            L.q_a_norm  = make(p + "attn_q_a_norm.weight", g);
            L.q_b       = make(p + "attn_q_b.weight", g);
            L.kv_a      = make(p + "attn_kv_a_mqa.weight", g);
            L.kv_a_norm = make(p + "attn_kv_a_norm.weight", g);
            L.k_b       = make(p + "attn_k_b.weight", g);
            L.v_b       = make(p + "attn_v_b.weight", g);
            const bool full = h.idx_full[(size_t) il];
            L.idx_k_norm   = make(p + "indexer.k_norm.weight", g, full);
            L.idx_k_norm_b = make(p + "indexer.k_norm.bias", g, full);
            L.idx_proj     = make(p + "indexer.proj.weight", g, full);
            L.idx_k        = make(p + "indexer.attn_k.weight", g, full);
            L.idx_q_b      = make(p + "indexer.attn_q_b.weight", g, full);
            L.idx_gate     = make(p + "indexer_compressor_gate.weight", g, full);
            L.idx_ape      = make(p + "indexer_compressor_ape.weight", g, full);
        }
        L.wo = make(p + "attn_output.weight", g);
        if (il < h.n_dense) {
            L.ffn_gate = make(p + "ffn_gate.weight", g);
            L.ffn_up   = make(p + "ffn_up.weight", g);
            L.ffn_down = make(p + "ffn_down.weight", g);
        } else {
            L.gate_inp   = make(p + "ffn_gate_inp.weight", g);
            L.exp_bias   = make(p + "exp_probs_b.bias", g);
            const bool xg = g && opt.experts_gpu;  // experts normally stay in the mapping
            L.gate_exps  = make(p + "ffn_gate_exps.weight", xg);
            L.up_exps    = make(p + "ffn_up_exps.weight", xg);
            L.down_exps  = make(p + "ffn_down_exps.weight", xg);
            L.gate_shexp = make(p + "ffn_gate_shexp.weight", g);
            L.up_shexp   = make(p + "ffn_up_shexp.weight", g);
            L.down_shexp = make(p + "ffn_down_shexp.weight", g);
        }
        if (missing) return false;
    }
    if (missing) return false;

    if (gpu_ && ggml_get_first_tensor(wctx_[0])) {
        wbuf_gpu_ = ggml_backend_alloc_ctx_tensors_from_buft(wctx_[0], ggml_backend_get_default_buffer_type(gpu_));
        if (!wbuf_gpu_) {
            err = "not enough VRAM for the GPU layers (lower --gpu-layers)";
            return false;
        }
        ggml_backend_buffer_set_usage(wbuf_gpu_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        gpu_bytes_ = ggml_backend_buffer_get_size(wbuf_gpu_);
        for (auto & [t, m] : to_gpu) ggml_backend_tensor_set(t, file_.data(m), 0, ggml_nbytes(t));
    }
    map_bufs_.assign(file_.n_files(), nullptr);
    for (size_t i = 0; i < file_.n_files(); i++) {
        const auto & mf = file_.map(i);
        map_bufs_[i]    = ggml_backend_cpu_buffer_from_ptr((void *) mf.data(), (size_t) mf.size());
        ggml_backend_buffer_set_usage(map_bufs_[i], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    for (auto & [t, m] : to_map) {
        if (ggml_backend_tensor_alloc(map_bufs_[file_.file_of(m)], t, (void *) file_.data(m)) != GGML_STATUS_SUCCESS) {
            err = std::string("cannot map tensor ") + ggml_get_name(t);
            return false;
        }
        cpu_bytes_ += ggml_nbytes(t);
    }

    // state
    n_ctx_              = std::max(kKvPad, (opt.n_ctx + kKvPad - 1) / kKvPad * kKvPad);
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (h.n_layer * 10 + 8), nullptr, true };
    sctx_[0]            = ggml_init(sp);
    sctx_[1]            = ggml_init(sp);
    const size_t nl     = (size_t) h.n_layer;
    for (auto * v : { &kv_cache_, &idx_raw_, &idx_pool_, &conv_state_, &ssm_state_, &conv_ck_, &ssm_ck_ }) v->assign(nl, nullptr);
    for (size_t l = 0; l < nl; l++) {
        ggml_context * c = sctx_[layers_[l].on_gpu ? 0 : 1];
        if (h.recurrent[l]) {
            conv_state_[l] = ggml_new_tensor_1d(c, GGML_TYPE_F32, 3 * (h.d_conv - 1) * h.kda_inner());
            ssm_state_[l]  = ggml_new_tensor_1d(c, GGML_TYPE_F32, h.kda_dim * h.kda_dim * h.n_head);
            conv_ck_[l]    = ggml_dup_tensor(c, conv_state_[l]);
            ssm_ck_[l]     = ggml_dup_tensor(c, ssm_state_[l]);
        } else {
            kv_cache_[l] = ggml_new_tensor_2d(c, latent_type(opt), h.kv_lora, n_ctx_);
            if (layers_[l].idx_k) {
                idx_raw_[l]  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * h.idx_dim, n_ctx_);
                idx_pool_[l] = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.idx_dim, n_ctx_ / h.kpool + 1);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        ggml_backend_t be = i == 0 ? gpu_ : cpu_;
        if (!be || !ggml_get_first_tensor(sctx_[i])) continue;
        sbuf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(sctx_[i], ggml_backend_get_default_buffer_type(be));
        if (!sbuf_[i]) {
            err = "not enough memory for the KV cache / recurrent state (lower --ctx)";
            return false;
        }
    }
    // MTP first: the expert cache (auto) takes the VRAM that is left
    if (opt.mtp && gpu_) {
        std::string e;
        if (!load_mtp(e)) {
            fprintf(stderr, "MTP: off (%s)\n", e.c_str());
        } else {
            ggml_init_params rp = { ggml_tensor_overhead() * (nl * 6 + 8), nullptr, true };
            rec_ctx_[0] = ggml_init(rp);
            rec_ctx_[1] = ggml_init(rp);
            for (auto * v : { &rec_x_, &rec_g_, &rec_beta_, &sp_conv_, &sp_ssm_ }) v->assign(nl, nullptr);
            for (size_t l = 0; l < nl; l++) {
                if (!h.recurrent[l]) continue;
                ggml_context * c = rec_ctx_[layers_[l].on_gpu ? 0 : 1];
                rec_x_[l]    = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3 * h.kda_inner(), kRecMax);
                rec_g_[l]    = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.kda_inner(), kRecMax);
                rec_beta_[l] = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_head, kRecMax);
                sp_conv_[l]  = ggml_dup_tensor(c, conv_state_[l]);
                sp_ssm_[l]   = ggml_dup_tensor(c, ssm_state_[l]);
            }
            for (int i = 0; i < 2; i++) {
                ggml_backend_t be = i == 0 ? gpu_ : cpu_;
                if (!ggml_get_first_tensor(rec_ctx_[i])) continue;
                rec_buf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(rec_ctx_[i], ggml_backend_get_default_buffer_type(be));
                if (!rec_buf_[i]) {
                    err = "not enough memory for the MTP verify buffers";
                    return false;
                }
            }
        }
    }
    if (!init_expert_cache(err)) return false;
    if (mtp_) {  // the MTP block becomes layer n_layer: dsa() / moe() take it like any other
        mtp_il_ = (int64_t) layers_.size();
        layers_.push_back(mtp_layer_);
        for (auto * v : { &conv_state_, &ssm_state_, &conv_ck_, &ssm_ck_ }) v->push_back(nullptr);
        if (ec_.C > 0)
            for (auto * v : { &ec_.gate, &ec_.up, &ec_.down, &ec_.gmap, &ec_.cmap, &ec_.gmask }) v->push_back(nullptr);
        sel_out_.push_back(nullptr);
        kernels::PrefetchHint hn;
        hn.limit = h.clamp_exp[(size_t) mtp_il_];
        hints_.push_back(hn);
    }
    reset();

    std::vector<ggml_backend_t> bes;
    if (gpu_) bes.push_back(gpu_);
    bes.push_back(cpu_);
    sched_ = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    graph_meta_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    return true;
}

bool Glm5Next::eval_last(const int32_t * tokens, int n, float * logits, std::string & err) {
    last_only_    = true;
    const bool ok = eval(tokens, n, logits, err);
    last_only_    = false;
    return ok;
}

void Glm5Next::checkpoint_save() {
    for (size_t l = 0; l < conv_state_.size(); l++) {
        if (!conv_state_[l]) continue;
        ggml_backend_tensor_copy(conv_state_[l], conv_ck_[l]);
        ggml_backend_tensor_copy(ssm_state_[l], ssm_ck_[l]);
    }
    ck_n_past_ = n_past_;
}

void Glm5Next::checkpoint_restore() {
    for (size_t l = 0; l < conv_state_.size(); l++) {
        if (!conv_state_[l]) continue;
        ggml_backend_tensor_copy(conv_ck_[l], conv_state_[l]);
        ggml_backend_tensor_copy(ssm_ck_[l], ssm_state_[l]);
    }
    n_past_ = ck_n_past_;
    history_.resize((size_t) n_past_);
    mtp_keep_h(n_past_);
    mtp_n_ = std::max(1, std::min(mtp_n_, n_past_));
}

// Sequence state blob: n_past, the checkpoint's n_past, then the latent KV and indexer rows below n_past, the pooled
// keys of the pools below it, and the whole recurrent state and its checkpoint.
template <class F> static void glm_for_state(const std::vector<ggml_tensor *> & kv, const std::vector<ggml_tensor *> & ir,
                                             const std::vector<ggml_tensor *> & ip,
                                             std::initializer_list<const std::vector<ggml_tensor *> *> whole, int n,
                                             int kpool, F f) {
    auto rows = [&](ggml_tensor * t, int64_t r) {
        if (t) f(t, (size_t) std::min<int64_t>(r, t->ne[1]) * t->nb[1]);
    };
    for (size_t l = 0; l < kv.size(); l++) {
        rows(kv[l], n);
        rows(ir[l], n);
        rows(ip[l], n / kpool + 1);
    }
    for (const auto * w : whole)
        for (ggml_tensor * t : *w)
            if (t) f(t, ggml_nbytes(t));
}

bool Glm5Next::save_seq(std::vector<uint8_t> & out) {
    if (rec_n_ > 0) return false;  // in the middle of a verify
    const int hdr[6] = { n_past_, ck_n_past_, mtp_h0_, mtp_n_, (int) history_.size(), (int) mtp_h_.size() };
    out.assign((const uint8_t *) hdr, (const uint8_t *) hdr + sizeof hdr);
    out.insert(out.end(), (const uint8_t *) history_.data(), (const uint8_t *) (history_.data() + history_.size()));
    out.insert(out.end(), (const uint8_t *) mtp_h_.data(), (const uint8_t *) (mtp_h_.data() + mtp_h_.size()));
    glm_for_state(kv_cache_, idx_raw_, idx_pool_, { &conv_state_, &ssm_state_, &conv_ck_, &ssm_ck_ }, n_past_, (int) hp_.kpool,
                  [&](ggml_tensor * t, size_t n) {
                      const size_t o = out.size();
                      out.resize(o + n);
                      if (n) ggml_backend_tensor_get(t, out.data() + o, 0, n);
                  });
    return true;
}

bool Glm5Next::load_seq(const std::vector<uint8_t> & in) {
    int hdr[6];
    if (in.size() < sizeof hdr) return false;
    std::memcpy(hdr, in.data(), sizeof hdr);
    if (hdr[0] < 0 || hdr[0] > n_ctx_ || hdr[4] < 0 || hdr[5] < 0 ||
        in.size() < sizeof hdr + (size_t) hdr[4] * 4 + (size_t) hdr[5] * 4) return false;
    size_t o  = sizeof hdr;
    std::vector<int32_t> hist((size_t) hdr[4]);
    std::vector<float>   mh((size_t) hdr[5]);
    std::memcpy(hist.data(), in.data() + o, hist.size() * 4);
    o += hist.size() * 4;
    std::memcpy(mh.data(), in.data() + o, mh.size() * 4);
    o += mh.size() * 4;
    bool   ok = true;
    glm_for_state(kv_cache_, idx_raw_, idx_pool_, { &conv_state_, &ssm_state_, &conv_ck_, &ssm_ck_ }, hdr[0], (int) hp_.kpool,
                  [&](ggml_tensor * t, size_t n) {
                      if (!ok || o + n > in.size()) {
                          ok = false;
                          return;
                      }
                      if (n) ggml_backend_tensor_set(t, in.data() + o, 0, n);
                      o += n;
                  });
    if (!ok || o != in.size()) {
        reset();
        return false;
    }
    n_past_    = hdr[0];
    ck_n_past_ = hdr[1];
    mtp_h0_    = hdr[2];
    mtp_n_     = hdr[3];
    history_   = std::move(hist);
    mtp_h_     = std::move(mh);
    rec_n_     = 0;
    return true;
}

void Glm5Next::reset() {
    for (auto b : sbuf_) {
        if (b) ggml_backend_buffer_clear(b, 0);
    }
    n_past_ = 0;
    history_.clear();
    mtp_h_.clear();
    mtp_h0_ = 0;
    mtp_n_  = 1;
}

// ---- building blocks -------------------------------------------------------------------------------------------

// mHC read: x [n_embd, hc, n] -> sum_h x[:, h] * pre[h] ([n_embd, n]); post [hc, n] and comb [hc, hc, n] for the write
ggml_tensor * Glm5Next::hc_pre(ggml_context * ctx, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
                               ggml_tensor ** post, ggml_tensor ** comb) {
    const auto &  h  = hp_;
    const int64_t hc = h.hc, n = x->ne[2];
    ggml_tensor * mixes = ggml_mul_mat(ctx, fn, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, x, hc * h.n_embd, n), h.rms_eps));
    auto part = [&](int64_t i0, int64_t ne0, int64_t si) {  // mixes[i0 : i0+ne0] * scale[si] + base[i0 : i0+ne0]
        ggml_tensor * v = ggml_view_2d(ctx, mixes, ne0, n, mixes->nb[1], ggml_row_size(mixes->type, i0));
        v               = ggml_mul(ctx, v, ggml_view_1d(ctx, scale, 1, ggml_row_size(scale->type, si)));
        return ggml_add(ctx, v, ggml_view_1d(ctx, base, ne0, ggml_row_size(base->type, i0)));
    };
    ggml_tensor * pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, part(0, hc, 0)), 1.0f, h.hc_eps);
    *post             = ggml_scale(ctx, ggml_sigmoid(ctx, part(hc, hc, 1)), 2.0f);
    *comb             = ggml_dsv4_hc_comb(ctx, mixes, scale, base, h.hc_eps, (int32_t) h.hc_iters);
    return ggml_dsv4_hc_pre(ctx, x, pre);
}

ggml_tensor * Glm5Next::kda(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n) {
    const auto &          h  = hp_;
    const Glm5NextLayer & L  = layers_[(size_t) il];
    const int64_t         S = h.kda_dim, H = h.n_head, D = h.kda_inner();

    ggml_tensor * xq = ggml_mul_mat(ctx, L.wq, cur), * xk = ggml_mul_mat(ctx, L.wk, cur), * xv = ggml_mul_mat(ctx, L.wv, cur);
    // per-channel log decay: lower_bound * sigmoid(exp(A_log) * (f_b f_a x + dt_b)); ssm_a holds -exp(A_log)
    ggml_tensor * g = ggml_add(ctx, ggml_mul_mat(ctx, L.f_b, ggml_mul_mat(ctx, L.f_a, cur)), L.dt_b);  // [D, n]
    ggml_tensor * A = ggml_reshape_3d(ctx, L.a, 1, H, 1);
    if (std::isfinite(h.kda_lower_bound)) {
        g = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, ggml_mul(ctx, ggml_reshape_3d(ctx, g, S, H, n), A), -1.0f)), h.kda_lower_bound);
    } else {
        g = ggml_mul(ctx, ggml_reshape_3d(ctx, ggml_softplus(ctx, g), S, H, n), A);
    }
    g                  = ggml_reshape_4d(ctx, g, S, H, n, 1);
    ggml_tensor * beta = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.beta, cur)), 1, H, n, 1);
    if (record_ && !rec_x_.empty() && rec_x_[(size_t) il] && n <= kRecMax) {  // verify: keep the inputs for commit()
        ggml_tensor * rx = rec_x_[(size_t) il];
        ggml_tensor * x3[3] = { xq, xk, xv };
        for (int i = 0; i < 3; i++)
            ggml_build_forward_expand(gf, ggml_cpy(ctx, x3[i], ggml_view_2d(ctx, rx, D, n, rx->nb[1], ggml_row_size(rx->type, (size_t) i * D))));
        ggml_tensor * rg = rec_g_[(size_t) il], * rb = rec_beta_[(size_t) il];
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, g, D, n), ggml_view_2d(ctx, rg, D, n, rg->nb[1], 0)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, beta, H, n), ggml_view_2d(ctx, rb, H, n, rb->nb[1], 0)));
    }
    ggml_tensor * out = kda_core(ctx, gf, il, xq, xk, xv, g, beta, n);

    // RMSNorm(o) * sigmoid(g_b g_a x)
    ggml_tensor * g2 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.g_b, ggml_mul_mat(ctx, L.g_a, cur)), S, H, n);
    ggml_tensor * o  = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, out), S, H, n), h.rms_eps), L.o_norm);
    o                = ggml_mul(ctx, o, ggml_sigmoid(ctx, g2));
    return ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, o, D, n));
}

// conv + KDA recurrence over the pre-conv projections xq/xk/xv [D, n]: updates the layer's state, returns [S, H, n, 1]
ggml_tensor * Glm5Next::kda_core(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * xq, ggml_tensor * xk,
                                 ggml_tensor * xv, ggml_tensor * g, ggml_tensor * beta, int n) {
    const auto &          h  = hp_;
    const Glm5NextLayer & L  = layers_[(size_t) il];
    const int64_t         S = h.kda_dim, H = h.n_head, D = h.kda_inner(), K = h.d_conv;
    ggml_tensor *         cs = conv_state_[(size_t) il];
    // causal conv over [state | batch] for each of Q, K, V; the last K-1 inputs become the state
    auto conv = [&](int qkv, ggml_tensor * w, ggml_tensor * cw) {
        ggml_tensor * st  = ggml_view_3d(ctx, cs, K - 1, D, 1, ggml_row_size(cs->type, K - 1), ggml_row_size(cs->type, (K - 1) * D),
                                         ggml_row_size(cs->type, (size_t) qkv * (K - 1) * D));
        ggml_tensor * x   = ggml_reshape_3d(ctx, ggml_is_contiguous(w) ? w : ggml_cont(ctx, w), D, n, 1);
        ggml_tensor * cin = ggml_concat(ctx, st, ggml_transpose(ctx, x), 0);  // [K-1+n, D, 1]
        ggml_tensor * last = ggml_view_3d(ctx, cin, K - 1, D, 1, cin->nb[1], cin->nb[2], ggml_row_size(cin->type, n));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, last, st));
        ggml_tensor * y = ggml_silu(ctx, ggml_ssm_conv(ctx, cin, ggml_reshape_2d(ctx, cw, K, D)));  // [D, n, 1]
        return ggml_reshape_4d(ctx, y, S, H, n, 1);
    };
    ggml_tensor * q = conv(0, xq, L.conv_q);
    ggml_tensor * k = conv(1, xk, L.conv_k);
    ggml_tensor * v = conv(2, xv, L.conv_v);

    // l2 norm (FLA's, eps 1e-6)
    auto l2 = [&](ggml_tensor * x) { return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / (float) S), 1.0f / sqrtf((float) S)); };
    q = l2(q);
    k = l2(k);

    ggml_tensor * ss  = ssm_state_[(size_t) il];
    ggml_tensor * res = ggml_gated_delta_net(ctx, q, k, v, g, beta, ggml_reshape_4d(ctx, ss, S, S, H, 1), 1);
    ggml_tensor * out = ggml_view_4d(ctx, res, S, H, n, 1, ggml_row_size(res->type, S), ggml_row_size(res->type, S * H),
                                     ggml_row_size(res->type, S * H * n), 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_1d(ctx, res, S * S * H, ggml_row_size(res->type, S * H * n)), ss));
    return out;
}

ggml_tensor * Glm5Next::dsa(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n, int n_kv, bool sparse,
                            Inputs & in, ggml_tensor ** prev_mask) {
    const auto &          h = hp_;
    const Glm5NextLayer & L = layers_[(size_t) il];
    const int64_t         H = h.n_head, R = h.kv_lora;
    auto rms = [&](ggml_tensor * x, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, h.rms_eps), w); };

    ggml_tensor * qr = rms(ggml_mul_mat(ctx, L.q_a, cur), L.q_a_norm);                               // [q_lora, n]
    ggml_tensor * q  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.q_b, qr), h.k_mla, H, n);           // [k_mla, H, n]
    ggml_tensor * kv = rms(ggml_mul_mat(ctx, L.kv_a, cur), L.kv_a_norm);                             // [R, n]
    // queries absorb W_kb so they score the latent directly
    ggml_tensor * qa = ggml_mul_mat(ctx, L.k_b, ggml_permute(ctx, q, 0, 2, 1, 3));                   // [R, n, H]

    ggml_tensor * kc = kv_cache_[(size_t) il];
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kc, kv, in.kvidx));

    ggml_tensor * mask = in.mask;
    if (L.idx_k) {
        // indexer rows: LayerNorm'd key | pool gate, per token
        const int64_t d = h.idx_dim, r = h.kpool;
        ggml_tensor * raw  = idx_raw_[(size_t) il];
        ggml_tensor * pool = idx_pool_[(size_t) il];
        ggml_tensor * ik   = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, ggml_mul_mat(ctx, L.idx_k, cur), h.norm_eps), L.idx_k_norm), L.idx_k_norm_b);
        ggml_tensor * ig   = ggml_mul_mat(ctx, L.idx_gate, cur);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, raw, ggml_concat(ctx, ik, ig, 0), in.kvidx));
        {
            // re-pool every pool this batch touches: per channel, softmax over the members of (gate + ape) weighs
            // their keys (an incomplete pool gets a provisional key that is masked until it completes)
            const int64_t np   = in.pool_blk->ne[0];
            ggml_tensor * rows = ggml_reshape_3d(ctx, ggml_get_rows(ctx, raw, in.pool_rows), 2 * d, r, np);
            ggml_tensor * pk   = ggml_view_3d(ctx, rows, d, r, np, rows->nb[1], rows->nb[2], 0);
            ggml_tensor * pg   = ggml_view_3d(ctx, rows, d, r, np, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, d));
            ggml_tensor * lg   = ggml_cont(ctx, ggml_permute(ctx, ggml_add(ctx, pg, L.idx_ape), 1, 0, 2, 3));  // [r, d, np]
            ggml_tensor * pr   = ggml_reshape_3d(ctx, ggml_soft_max(ctx, ggml_reshape_2d(ctx, lg, r, d * np)), r, d, np);
            ggml_tensor * pkt  = ggml_cont(ctx, ggml_permute(ctx, pk, 1, 0, 2, 3));
            ggml_tensor * pk2  = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, pr, pkt)), d, np);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, pool, pk2, in.pool_blk));
        }
        if (sparse) {
            // score = sum_h relu(q_h . pooled) * w_h over complete visible pools; keep the top K pools
            const int64_t Hi = h.idx_n_head, NB = n_kv / r, Kp = h.idx_top_k / r;
            ggml_tensor * iq = ggml_mul_mat(ctx, L.idx_q_b, qr);                                     // [d*Hi, n]
            ggml_tensor * w  = ggml_scale(ctx, ggml_mul_mat(ctx, L.idx_proj, cur), 1.0f / sqrtf((float) (d * Hi)));  // [Hi, n]
            ggml_tensor * s  = ggml_mul_mat(ctx, ggml_view_2d(ctx, pool, d, NB, pool->nb[1], 0), ggml_reshape_2d(ctx, iq, d, Hi * n));
            s = ggml_mul(ctx, ggml_reshape_3d(ctx, ggml_relu(ctx, s), NB, Hi, n), ggml_reshape_3d(ctx, w, 1, Hi, n));
            s = ggml_cont(ctx, ggml_permute(ctx, s, 1, 0, 2, 3));                                    // [Hi, NB, n]
            s = ggml_add(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, s), NB, n), in.bvis);
            ggml_tensor * top = ggml_top_k(ctx, s, (int) Kp);                                         // [Kp, n]
            ggml_tensor * blk = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, NB, n), -INFINITY);
            ggml_tensor * zer = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, Kp, n), 0.0f);
            blk = ggml_set_rows(ctx, blk, zer, ggml_reshape_3d(ctx, top, Kp, n, 1));
            blk = ggml_add(ctx, ggml_reshape_2d(ctx, blk, NB, n), in.bvis);
            ggml_tensor * tok = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, blk, 1, NB, n), r, NB, n, 1), r * NB, n);
            // union of the selected pools and the token's own incomplete pool (0/-inf masks)
            ggml_tensor * u = ggml_clamp(ctx, ggml_add(ctx, ggml_exp(ctx, tok), ggml_exp(ctx, in.tail)), 0.0f, 1.0f);
            mask            = ggml_cast(ctx, ggml_log(ctx, u), GGML_TYPE_F16);
        }
        *prev_mask = mask;
    } else if (sparse && *prev_mask) {  // a layer sharing the previous indexer's selection
        mask = *prev_mask;
    }

    // attention over the latent: K = V = the cached rows, then W_vb per head
    ggml_tensor * K = ggml_view_3d(ctx, kc, R, n_kv, 1, kc->nb[1], kc->nb[1] * n_kv, 0);
    ggml_tensor * a = ggml_flash_attn_ext(ctx, qa, K, K, mask, 1.0f / sqrtf((float) h.k_mla), 0.0f, 0.0f);  // [R, H, n]
    ggml_prec_set_acc(a, GGML_PREC_F32);
    ggml_tensor * o = ggml_mul_mat(ctx, L.v_b, ggml_permute(ctx, a, 0, 2, 1, 3));                    // [v_mla, n, H]
    o               = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                              // [v_mla, H, n]
    return ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, o, h.v_mla * H, n));
}

ggml_tensor * Glm5Next::ffn(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * cur,
                            float limit) {
    ggml_tensor * g = ggml_mul_mat(ctx, gate, cur), * u = ggml_mul_mat(ctx, up, cur);
    return ggml_mul_mat(ctx, down, limit > 1e-6f ? ggml_swiglu_clamp(ctx, g, u, limit) : ggml_swiglu_split(ctx, g, u));
}

ggml_tensor * Glm5Next::moe(ggml_context * ctx, int64_t il, ggml_tensor * cur, int n) {
    const auto &          h  = hp_;
    const Glm5NextLayer & L  = layers_[(size_t) il];
    const int64_t         ne = h.n_embd, k = h.n_expert_used;
    const size_t          l  = (size_t) il;
    const float           lim = h.clamp_exp[l];

    // sigmoid router: selection on probs + bias, weights from probs (renormalized, scaled)
    ggml_tensor * probs = ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.gate_inp, cur));                    // [n_expert, n]
    ggml_tensor * sel   = ggml_argsort_top_k(ctx, ggml_add(ctx, probs, L.exp_bias), (int) k);       // [k, n]
    ggml_tensor * w     = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, h.n_expert, n), sel), k, n);
    if (h.expert_weights_norm) w = ggml_div(ctx, w, ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY));
    if (h.expert_weights_scale != 1.0f) w = ggml_scale(ctx, w, h.expert_weights_scale);
    w = ggml_reshape_3d(ctx, w, 1, k, n);

    ggml_tensor * x3  = ggml_reshape_3d(ctx, cur, ne, 1, n);
    ggml_tensor * out = nullptr;
    ggml_tensor * sel_c = nullptr;
    if (ec_.C > 0 && ec_.gmap[l]) {
        sel_c       = ggml_cont(ctx, sel);
        sel_out_[l] = sel_c;
        ggml_set_output(sel_c);
    }
    if (sel_c && n < 32) {  // see Qwen4Exp::moe: cached experts on the GPU, the rest from the mapping on the CPU
        auto look = [&](ggml_tensor * table) {
            ggml_tensor * v = ggml_get_rows(ctx, ggml_reshape_2d(ctx, table, 1, h.n_expert), ggml_reshape_1d(ctx, sel_c, k * n));
            return ggml_reshape_3d(ctx, v, 1, k, n);
        };
        ggml_tensor * gm   = look(ec_.gmask[l]);
        ggml_tensor * jix  = ggml_reshape_3d(ctx, ec_.jidx, 1, k, 1);
        ggml_tensor * gidf = ggml_add(ctx, ggml_mul(ctx, gm, ggml_sub(ctx, look(ec_.gmap[l]), jix)), jix);
        ggml_tensor * gids = ggml_reshape_2d(ctx, ggml_cast(ctx, gidf, GGML_TYPE_I32), k, n);
        ggml_tensor * cids = ggml_reshape_2d(ctx, ggml_cast(ctx, look(ec_.cmap[l]), GGML_TYPE_I32), k, n);
        ggml_tensor * wg   = ggml_mul(ctx, w, gm);
        ggml_tensor * wc   = ggml_sub(ctx, w, wg);
        // lookahead (the next layer's router on this layer's input starts its expert reads early): on by default here,
        // 2.5 vs 1.8 tok/s from a 980 PRO (E8_LOOKAHEAD=0 turns it off)
        static const bool la = !std::getenv("E8_LOOKAHEAD") || std::atoi(std::getenv("E8_LOOKAHEAD")) != 0;
        ggml_tensor * next_ids = nullptr;
        if (la && l + 1 < layers_.size() && layers_[l + 1].gate_inp && hints_[l].gate) {
            next_ids = ggml_cont(ctx, ggml_argsort_top_k(ctx, ggml_add(ctx, ggml_sigmoid(ctx, ggml_mul_mat(ctx, layers_[l + 1].gate_inp, cur)),
                                                                           layers_[l + 1].exp_bias), (int) k));
        }
        ggml_tensor * cpu_out = kernels::moe_cpu_sparse(ctx, cur, cids, ggml_reshape_2d(ctx, wc, k, n), L.gate_exps, L.up_exps,
                                                        L.down_exps, next_ids, &hints_[l]);
        out = ggml_add(ctx, experts(ctx, ec_.gate[l], ec_.up[l], ec_.down[l], x3, gids, wg, n, lim), cpu_out);
    } else {
        out = experts(ctx, L.gate_exps, L.up_exps, L.down_exps, x3, sel, w, n, lim);
    }
    return ggml_add(ctx, out, ffn(ctx, L.gate_shexp, L.up_shexp, L.down_shexp, cur, h.clamp_shexp[l]));
}

ggml_tensor * Glm5Next::experts(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * x3,
                                ggml_tensor * ids, ggml_tensor * w, int n, float limit) {
    const int64_t ne = hp_.n_embd, k = ids->ne[0];
    ggml_tensor * gt = ggml_mul_mat_id(ctx, gate, x3, ids);
    ggml_tensor * u  = ggml_mul_mat_id(ctx, up, x3, ids);
    ggml_tensor * a  = limit > 1e-6f ? ggml_swiglu_clamp(ctx, gt, u, limit) : ggml_swiglu_split(ctx, gt, u);
    ggml_tensor * d  = ggml_mul(ctx, ggml_mul_mat_id(ctx, down, a, ids), w);  // [ne, k, n]
    ggml_tensor * out = ggml_view_2d(ctx, d, ne, n, d->nb[2], 0);
    for (int64_t i = 1; i < k; i++) out = ggml_add(ctx, out, ggml_view_2d(ctx, d, ne, n, d->nb[2], (size_t) i * d->nb[1]));
    return out;
}

bool Glm5Next::init_expert_cache(std::string & err) {
    const auto & h  = hp_;
    const size_t nl = (size_t) h.n_layer;
    sel_out_.assign(nl, nullptr);
    hints_.assign(nl, {});
    for (size_t l = 0; l < nl; l++) hints_[l].limit = h.clamp_exp[l];
    if (!gpu_ || opt_.expert_cache_gb == 0 || opt_.experts_gpu) return true;
    uint64_t per_expert = 0;
    int      n_gpu      = 0;
    for (const auto & L : layers_) {
        if (!L.on_gpu || !L.gate_exps) continue;
        n_gpu++;
        if (!per_expert)
            for (ggml_tensor * t : { L.gate_exps, L.up_exps, L.down_exps }) per_expert += t->nb[2];
    }
    if (!n_gpu || !per_expert) return true;
    double budget = opt_.expert_cache_gb * 1e9;
    if (budget < 0) {  // auto: free VRAM minus room for compute buffers
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &free, &total);
        budget = std::max(0.0, (double) free - 3.0e9);
    }
    const int k = (int) h.n_expert_used;
    const int C = (int) std::min<int64_t>(h.n_expert - 1, (int64_t) (budget / ((double) per_expert * n_gpu)) - k);
    if (C < 1) return true;
    ggml_init_params ip = { ggml_tensor_overhead() * (nl * 8 + 8), nullptr, true };
    ec_.ctx             = ggml_init(ip);
    for (auto * v : { &ec_.gate, &ec_.up, &ec_.down, &ec_.gmap, &ec_.cmap, &ec_.gmask }) v->assign(nl, nullptr);
    auto slots = [&](ggml_tensor * t) { return ggml_new_tensor_3d(ec_.ctx, t->type, t->ne[0], t->ne[1], C + k); };
    for (size_t l = 0; l < nl; l++) {
        const auto & L = layers_[l];
        if (!L.on_gpu || !L.gate_exps) continue;
        ec_.gate[l]  = slots(L.gate_exps);
        ec_.up[l]    = slots(L.up_exps);
        ec_.down[l]  = slots(L.down_exps);
        ec_.gmap[l]  = ggml_new_tensor_1d(ec_.ctx, GGML_TYPE_F32, h.n_expert);
        ec_.cmap[l]  = ggml_new_tensor_1d(ec_.ctx, GGML_TYPE_F32, h.n_expert);
        ec_.gmask[l] = ggml_new_tensor_1d(ec_.ctx, GGML_TYPE_F32, h.n_expert);
    }
    ec_.jidx = ggml_new_tensor_1d(ec_.ctx, GGML_TYPE_F32, k);
    ec_.buf  = ggml_backend_alloc_ctx_tensors_from_buft(ec_.ctx, ggml_backend_get_default_buffer_type(gpu_));
    if (!ec_.buf) {
        err = "not enough VRAM for the expert cache (set --expert-cache-gb lower)";
        return false;
    }
    ggml_backend_buffer_clear(ec_.buf, 0);
    {
        std::vector<float> j((size_t) k);
        for (int i = 0; i < k; i++) j[(size_t) i] = (float) (C + i);
        ggml_backend_tensor_set(ec_.jidx, j.data(), 0, j.size() * sizeof(float));
    }
    ec_.C     = C;
    ec_.bytes = ggml_backend_buffer_get_size(ec_.buf);
    ec_.slot_of.assign(nl, std::vector<int>((size_t) h.n_expert, -1));
    ec_.expert_in.assign(nl, std::vector<int>((size_t) C, -1));
    ec_.heat.assign(nl, std::vector<float>((size_t) h.n_expert, 0.0f));
    ec_.dummy.assign(nl, 0);
    for (size_t l = 0; l + 1 < nl; l++) {
        const auto & N = layers_[l + 1];
        if (!N.gate_exps) continue;
        hints_[l].gate    = N.gate_exps;
        hints_[l].up      = N.up_exps;
        hints_[l].down    = N.down_exps;
        hints_[l].slot_of = ec_.gmap[l + 1] ? &ec_.slot_of[l + 1] : nullptr;
    }
    for (size_t l = 0; l < nl; l++) {
        if (!ec_.gmap[l]) continue;
        std::vector<float> gm((size_t) h.n_expert, (float) C), cm((size_t) h.n_expert), mk((size_t) h.n_expert, 0.0f);
        for (int64_t e = 0; e < h.n_expert; e++) cm[(size_t) e] = (float) e;
        ggml_backend_tensor_set(ec_.gmap[l], gm.data(), 0, gm.size() * 4);
        ggml_backend_tensor_set(ec_.cmap[l], cm.data(), 0, cm.size() * 4);
        ggml_backend_tensor_set(ec_.gmask[l], mk.data(), 0, mk.size() * 4);
    }
    fprintf(stderr, "expert cache: %d of %lld experts per layer in VRAM (%.1f GB)\n", C, (long long) h.n_expert, ec_.bytes / 1e9);
    return true;
}

void Glm5Next::update_expert_cache(int n) {
    const auto & h = hp_;
    if (ec_.C <= 0) return;
    const float decay     = std::pow(0.995f, (float) n);
    const int   max_swaps = std::max(4, std::min(64, n));
    std::vector<int32_t> ids;
    for (size_t l = 0; l < sel_out_.size(); l++) {
        ggml_tensor * s = sel_out_[l];
        if (!s || !s->buffer) continue;
        ids.resize((size_t) ggml_nelements(s));
        ggml_backend_tensor_get(s, ids.data(), 0, ids.size() * sizeof(int32_t));
        auto & heat = ec_.heat[l];
        auto & slot = ec_.slot_of[l];
        auto & in   = ec_.expert_in[l];
        for (float & x : heat) x *= decay;
        std::vector<int> used;
        for (int32_t e : ids) {
            if (e < 0 || e >= h.n_expert) continue;
            heat[(size_t) e] += 1.0f;
            ec_.uses++;
            ec_.uses_recent++;
            if (slot[(size_t) e] >= 0) {
                ec_.hits++;
                ec_.hits_recent++;
            } else {
                used.push_back(e);
            }
        }
        std::sort(used.begin(), used.end());
        used.erase(std::unique(used.begin(), used.end()), used.end());
        std::sort(used.begin(), used.end(), [&](int a, int b) { return heat[(size_t) a] > heat[(size_t) b]; });
        int  swaps   = 0;
        bool changed = false;
        for (int e : used) {
            if (swaps >= max_swaps) break;
            int target = -1;
            for (int c = 0; c < ec_.C && target < 0; c++)
                if (in[(size_t) c] < 0) target = c;
            if (target < 0) {
                float coldest = 1e30f;
                for (int c = 0; c < ec_.C; c++) {
                    const float hv = heat[(size_t) in[(size_t) c]];
                    if (hv < coldest) {
                        coldest = hv;
                        target  = c;
                    }
                }
                if (heat[(size_t) e] < coldest * 1.25f + 0.5f) continue;
                slot[(size_t) in[(size_t) target]] = -1;
            }
            const Glm5NextLayer & L = layers_[l];
            auto copy = [&](ggml_tensor * dst, ggml_tensor * src) {
                ggml_backend_tensor_set(dst, (const char *) src->data + (size_t) e * src->nb[2], (size_t) target * dst->nb[2], dst->nb[2]);
            };
            copy(ec_.gate[l], L.gate_exps);
            copy(ec_.up[l], L.up_exps);
            copy(ec_.down[l], L.down_exps);
            in[(size_t) target] = e;
            slot[(size_t) e]    = target;
            ec_.swaps++;
            swaps++;
            changed = true;
        }
        if (changed) {
            int & dm = ec_.dummy[l];  // the CPU-side dummy is a cached expert (see Qwen4Exp::update_expert_cache)
            if (slot[(size_t) dm] < 0) {
                for (int64_t e = 0; e < h.n_expert; e++) {
                    if (slot[(size_t) e] >= 0) {
                        dm = (int) e;
                        break;
                    }
                }
            }
            std::vector<float> gm((size_t) h.n_expert), cm((size_t) h.n_expert), mk((size_t) h.n_expert);
            for (int64_t e = 0; e < h.n_expert; e++) {
                const bool c   = slot[(size_t) e] >= 0;
                gm[(size_t) e] = c ? (float) slot[(size_t) e] : (float) ec_.C;
                cm[(size_t) e] = c ? (float) dm : (float) e;
                mk[(size_t) e] = c ? 1.0f : 0.0f;
            }
            ggml_backend_tensor_set(ec_.gmap[l], gm.data(), 0, gm.size() * 4);
            ggml_backend_tensor_set(ec_.cmap[l], cm.data(), 0, cm.size() * 4);
            ggml_backend_tensor_set(ec_.gmask[l], mk.data(), 0, mk.size() * 4);
        }
        sel_out_[l] = nullptr;
    }
}

std::string Glm5Next::cache_report() const {
    char prof[512] = "";
    if (n_timed_) {
        const double d = 1e3 / (double) n_timed_;
        snprintf(prof, sizeof prof, "\nper decode token: build %.1f ms, alloc %.1f ms, inputs %.1f ms, compute %.1f ms, cache update %.1f ms",
                 t_build_ * d, t_alloc_ * d, t_inputs_ * d, t_compute_ * d, t_update_ * d);
    }
    {
        double gu = 0, dn = 0;
        long long calls = 0;
        kernels::moe_cpu_times(gu, dn, calls);
        if (calls && n_timed_) {
            const double per = (double) calls / (double) (hp_.n_layer - hp_.n_dense);
            char extra[200];
            snprintf(extra, sizeof extra, "\nCPU MoE ops per pass: gate/up %.1f ms, down %.1f ms (%lld calls)", gu * 1e3 / per,
                     dn * 1e3 / per, calls);
            strncat(prof, extra, sizeof prof - strlen(prof) - 1);
        }
    }
    if (ec_.C <= 0) return std::string("expert cache: off") + prof;
    char buf[256];
    snprintf(buf, sizeof buf, "expert cache: %d slots/layer (%.2f GB), hit rate %.1f%% overall, %.1f%% recent, %lld swaps",
             ec_.C, ec_.bytes / 1e9, 100.0 * ec_.hits / std::max<int64_t>(1, ec_.uses),
             100.0 * ec_.hits_recent / std::max<int64_t>(1, ec_.uses_recent), (long long) ec_.swaps);
    return std::string(buf) + prof;
}

// ---- graphs ----------------------------------------------------------------------------------------------------

ggml_tensor * Glm5Next::mixer(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * res, int n, int n_kv, bool sparse,
                              Inputs & in, ggml_tensor ** prev_mask) {
    const Glm5NextLayer & L = layers_[(size_t) il];
    ggml_tensor *         post = nullptr, * comb = nullptr;
    ggml_tensor *         cur = hc_pre(ctx, res, L.hca_fn, L.hca_scale, L.hca_base, &post, &comb);
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp_.rms_eps), L.attn_norm);
    cur = hp_.recurrent[(size_t) il] ? kda(ctx, gf, il, cur, n) : dsa(ctx, gf, il, cur, n, n_kv, sparse, in, prev_mask);
    return ggml_dsv4_hc_post(ctx, cur, res, post, comb);
}

ggml_tensor * Glm5Next::ffn_block(ggml_context * ctx, int64_t il, ggml_tensor * res, int n) {
    const Glm5NextLayer & L = layers_[(size_t) il];
    ggml_tensor *         post = nullptr, * comb = nullptr;
    ggml_tensor *         cur = hc_pre(ctx, res, L.hcf_fn, L.hcf_scale, L.hcf_base, &post, &comb);
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp_.rms_eps), L.ffn_norm);
    cur = L.ffn_gate ? ffn(ctx, L.ffn_gate, L.ffn_up, L.ffn_down, cur, hp_.clamp_shexp[(size_t) il]) : moe(ctx, il, cur, n);
    return ggml_dsv4_hc_post(ctx, cur, res, post, comb);
}

ggml_tensor * Glm5Next::head_norm(ggml_context * ctx, ggml_tensor * res) {
    const int64_t ne = hp_.n_embd, n = res->ne[2];
    ggml_tensor * m  = ggml_view_2d(ctx, res, ne, n, res->nb[2], 0);  // mean of the streams
    for (int64_t s = 1; s < hp_.hc; s++) m = ggml_add(ctx, m, ggml_view_2d(ctx, res, ne, n, res->nb[2], (size_t) s * res->nb[1]));
    return ggml_mul(ctx, ggml_rms_norm(ctx, ggml_scale(ctx, m, 1.0f / (float) hp_.hc), hp_.rms_eps), output_norm_);
}

ggml_cgraph * Glm5Next::build_graph(ggml_context * ctx, int n, int n_kv, bool sparse, Inputs & in, ggml_tensor *& out) {
    const auto &  h  = hp_;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    in.embd          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n);
    ggml_set_input(in.embd);
    new_inputs(ctx, n, n_kv, sparse, in);
    ggml_tensor * res  = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, in.embd, h.n_embd, 1, n), h.n_embd, h.hc, n, 1);
    ggml_tensor * prev = nullptr;
    for (int64_t il = 0; il < h.n_layer; il++) {
        res = mixer(ctx, gf, il, res, n, n_kv, sparse, in, &prev);
        res = ffn_block(ctx, il, res, n);
    }
    h_all_ = nullptr;
    if (mtp_) {  // every row's post-norm hidden feeds the MTP block
        ggml_tensor * hn = head_norm(ctx, res);
        h_all_           = hn;
        ggml_set_output(hn);
        if (last_only_ && n > 1) hn = ggml_view_2d(ctx, hn, hn->ne[0], 1, hn->nb[1], hn->nb[1] * (size_t) (n - 1));
        out = ggml_mul_mat(ctx, output_, hn);
    } else {
        if (last_only_ && n > 1) {
            res = ggml_view_3d(ctx, res, res->ne[0], res->ne[1], 1, res->nb[1], res->nb[2], res->nb[2] * (size_t) (n - 1));
        }
        out = ggml_mul_mat(ctx, output_, head_norm(ctx, res));
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    return gf;
}

void Glm5Next::new_inputs(ggml_context * ctx, int n, int n_kv, bool sparse, Inputs & in) {
    const int64_t r = hp_.kpool;
    in.mask         = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n);
    ggml_set_input(in.mask);
    in.kvidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
    ggml_set_input(in.kvidx);
    const int64_t np = (n_past_ + n - 1) / r - n_past_ / r + 1;
    in.pool_rows     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, np * r);
    ggml_set_input(in.pool_rows);
    in.pool_blk = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, np);
    ggml_set_input(in.pool_blk);
    if (sparse) {
        in.bvis = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv / r, n);
        ggml_set_input(in.bvis);
        in.tail = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n);
        ggml_set_input(in.tail);
    }
}

bool Glm5Next::sparse_at(int n) const {
    // dense while every token's complete pools fit the budget (the indexer would select them all)
    return (n_past_ + n) / hp_.kpool > hp_.idx_top_k / hp_.kpool;
}

bool Glm5Next::set_inputs(const Inputs & in, int n, int n_kv) {
    auto set = [](ggml_tensor * t, const void * d, size_t nb) {
        if (t && t->buffer) ggml_backend_tensor_set(t, d, 0, nb);
    };
    const int64_t            r = hp_.kpool;
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n);
    const ggml_fp16_t        zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n; i++) {
        ggml_fp16_t * mr = mask.data() + (size_t) i * n_kv;
        const int     nz = std::min(n_kv, n_past_ + i + 1);
        std::fill(mr, mr + nz, zero);
        std::fill(mr + nz, mr + n_kv, ninf);
    }
    set(in.mask, mask.data(), mask.size() * sizeof(ggml_fp16_t));
    {
        std::vector<int64_t> rows((size_t) n);
        for (int i = 0; i < n; i++) rows[(size_t) i] = n_past_ + i;
        set(in.kvidx, rows.data(), rows.size() * sizeof(int64_t));
    }
    {
        const int64_t        b0 = n_past_ / r, np = in.pool_blk->ne[0];
        std::vector<int32_t> pr((size_t) (np * r));
        std::vector<int64_t> pb((size_t) np);
        for (int64_t i = 0; i < np; i++) {
            pb[(size_t) i] = b0 + i;
            for (int64_t j = 0; j < r; j++) pr[(size_t) (i * r + j)] = (int32_t) std::min<int64_t>((b0 + i) * r + j, n_ctx_ - 1);
        }
        set(in.pool_rows, pr.data(), pr.size() * sizeof(int32_t));
        set(in.pool_blk, pb.data(), pb.size() * sizeof(int64_t));
    }
    if (in.bvis) {
        const int64_t      NB = in.bvis->ne[0];
        std::vector<float> bv((size_t) (NB * n)), tl((size_t) n_kv * n);
        for (int i = 0; i < n; i++) {
            const int64_t p = n_past_ + i, nb = (p + 1) / r;
            // ties (several pools scoring exactly 0) go to the lower index, as torch.topk does
            for (int64_t b = 0; b < NB; b++) bv[(size_t) (i * NB + b)] = b < nb ? -1e-30f * (float) b : -INFINITY;
            float * tr = tl.data() + (size_t) i * n_kv;
            std::fill(tr, tr + n_kv, -INFINITY);
            std::fill(tr + std::min<int64_t>(n_kv, nb * r), tr + std::min<int64_t>(n_kv, p + 1), 0.0f);
        }
        set(in.bvis, bv.data(), bv.size() * sizeof(float));
        set(in.tail, tl.data(), tl.size() * sizeof(float));
    }
    return true;
}

bool Glm5Next::embed_rows(const int32_t * tokens, int n, std::vector<float> & out) const {
    const auto & h  = hp_;
    const size_t rs = ggml_row_size(tok_embd_->type, h.n_embd);
    const auto * tt = ggml_get_type_traits(tok_embd_->type);
    const char * tb = (const char *) file_.data(tok_embd_);
    out.resize((size_t) (h.n_embd * n));
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= h.n_vocab) return false;
        const char * row = tb + rs * (size_t) tokens[i];
        if (tok_embd_->type == GGML_TYPE_F32) std::memcpy(out.data() + (size_t) i * h.n_embd, row, sizeof(float) * h.n_embd);
        else tt->to_float(row, out.data() + (size_t) i * h.n_embd, h.n_embd);
    }
    return true;
}

bool Glm5Next::eval(const int32_t * tokens, int n, float * logits, std::string & err) {
    if (n <= 0 || n > opt_.n_ubatch) {
        err = "eval: batch size must be 1.." + std::to_string(opt_.n_ubatch);
        return false;
    }
    if (n_past_ + n > n_ctx_) {
        err = "eval: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    const int  n_kv   = std::min(n_ctx_, (n_past_ + n + kKvPad - 1) / kKvPad * kKvPad);
    const bool sparse = sparse_at(n);
    using clk         = std::chrono::steady_clock;
    auto sec          = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0     = clk::now();
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    Inputs           in;
    ggml_tensor *    out = nullptr;
    ggml_cgraph *    gf  = build_graph(ctx, n, n_kv, sparse, in, out);
    const auto t1 = clk::now();
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval: cannot allocate the compute graph";
        return false;
    }
    const auto t2 = clk::now();
    {
        std::vector<float> e;
        if (!embed_rows(tokens, n, e)) {
            ggml_free(ctx);
            err = "eval: token id out of range";
            return false;
        }
        ggml_backend_tensor_set(in.embd, e.data(), 0, e.size() * sizeof(float));
    }
    set_inputs(in, n, n_kv);
    const auto t3 = clk::now();
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "eval: graph compute failed";
        return false;
    }
    if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
    if (h_all_ && h_all_->buffer) {
        std::vector<float> hr((size_t) ggml_nelements(h_all_));
        ggml_backend_tensor_get(h_all_, hr.data(), 0, ggml_nbytes(h_all_));
        mtp_push_h(hr.data(), n_past_, n);
    }
    h_all_        = nullptr;
    const auto t4 = clk::now();
    update_expert_cache(n);
    const auto t5 = clk::now();
    if (n == 1) {
        t_build_ += sec(t0, t1);
        t_alloc_ += sec(t1, t2);
        t_inputs_ += sec(t2, t3);
        t_compute_ += sec(t3, t4);
        t_update_ += sec(t4, t5);
        n_timed_++;
    }
    static const bool prof = std::getenv("E8_EVAL_PROF") != nullptr;
    if (prof) {
        fprintf(stderr, "eval n=%d: build %.1f alloc %.1f inputs %.1f compute %.1f update %.1f ms, %d splits\n", n,
                1e3 * sec(t0, t1), 1e3 * sec(t1, t2), 1e3 * sec(t2, t3), 1e3 * sec(t3, t4), 1e3 * sec(t4, t5),
                ggml_backend_sched_get_n_splits(sched_));
    }
    ggml_free(ctx);
    history_.insert(history_.end(), tokens, tokens + n);
    n_past_ += n;
    // plain decoding (no drafts) still keeps the MTP rows current, in batches
    if (mtp_ && !record_ && (int64_t) mtp_h_.size() / hp_.n_embd > 256 && !mtp_fill(n_past_, err)) return false;
    return true;
}

// ---- layer-major prompt processing -------------------------------------------------------------------------------

bool Glm5Next::run_part(int part, int64_t il, int n, float * r, float * logits, std::string & err) {
    const auto &  h      = hp_;
    const int64_t hcd    = h.hc * h.n_embd;
    const int     n_kv   = std::min(n_ctx_, (n_past_ + n + kKvPad - 1) / kKvPad * kKvPad);
    const bool    sparse = sparse_at(n);
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    Inputs           in;
    in.res = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h.n_embd, h.hc, part == 2 ? 1 : n);
    ggml_set_input(in.res);
    ggml_tensor * out = nullptr, * prev = nullptr;
    if (part == 0) {
        new_inputs(ctx, n, n_kv, sparse, in);
        out = mixer(ctx, gf, il, in.res, n, n_kv, sparse, in, &prev);
    } else if (part == 1) {
        out = ffn_block(ctx, il, in.res, n);
    } else if (part == 2) {
        out = ggml_mul_mat(ctx, output_, head_norm(ctx, in.res));
    } else {
        out = head_norm(ctx, in.res);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "prefill: cannot allocate the compute graph";
        return false;
    }
    ggml_backend_tensor_set(in.res, part == 2 ? r + (size_t) (n - 1) * hcd : r, 0, ggml_nbytes(in.res));
    if (part == 0) set_inputs(in, n, n_kv);
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "prefill: graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(out, part >= 2 ? logits : r, 0, ggml_nbytes(out));
    ggml_free(ctx);
    return true;
}

bool Glm5Next::prefill(const int32_t * tokens, int N, float * logits, std::string & err) {
    const auto & h = hp_;
    static const int min_layered = std::getenv("E8_LAYERED_MIN") ? std::atoi(std::getenv("E8_LAYERED_MIN")) : 256;
    if (N < min_layered) return CausalLM::prefill(tokens, N, logits, err) && (!mtp_ || mtp_fill(n_past_, err));
    if (n_past_ + N > n_ctx_) {
        err = "prefill: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    // layer by layer over chunks of S tokens, so each layer's experts are read from disk once per chunk (see Qwen4Exp)
    const int64_t hcd = h.hc * h.n_embd;
    const int     S   = std::getenv("E8_PREFILL_CHUNK") ? std::atoi(std::getenv("E8_PREFILL_CHUNK")) : 65536;
    const int     um  = std::getenv("E8_MOE_UBATCH") ? std::atoi(std::getenv("E8_MOE_UBATCH")) : 4096;
    std::vector<float> rr, e;
    auto prefetch = [&](int64_t il) {
        std::vector<std::pair<void *, size_t>> ranges;
        if (il < h.n_layer) {
            const Glm5NextLayer & L = layers_[(size_t) il];
            for (ggml_tensor * t : { L.gate_exps, L.up_exps, L.down_exps })
                if (t && t->data && t->buffer && ggml_backend_buffer_is_host(t->buffer)) ranges.emplace_back(t->data, ggml_nbytes(t));
        }
        return std::thread([ranges]() {
#if defined(_WIN32)
            std::vector<WIN32_MEMORY_RANGE_ENTRY> w;
            for (auto & r : ranges) w.push_back({ r.first, r.second });
            if (!w.empty()) PrefetchVirtualMemory(GetCurrentProcess(), w.size(), w.data(), 0);
#else
            static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
            std::vector<std::pair<char *, size_t>> pieces;
            for (auto & r : ranges) {
                char * a = (char *) ((uintptr_t) r.first & ~(pg - 1));
                char * e = (char *) r.first + r.second;
                for (; a < e; a += (16u << 20)) pieces.emplace_back(a, std::min<size_t>(16u << 20, (size_t) (e - a)));
            }
            std::atomic<size_t>      next{ 0 };
            std::vector<std::thread> th;
            for (int t = 0; t < 8; t++) {
                th.emplace_back([&]() {
                    volatile char sink = 0;
                    for (size_t i; (i = next++) < pieces.size();) {
                        madvise(pieces[i].first, pieces[i].second, MADV_WILLNEED);
                        for (size_t o = 0; o < pieces[i].second; o += pg) sink = sink + pieces[i].first[o];
                    }
                });
            }
            for (auto & x : th) x.join();
#endif
        });
    };
    for (int c0 = 0; c0 < N; c0 += S) {
        const int cn   = std::min(S, N - c0);
        const int base = n_past_;
        rr.resize((size_t) cn * hcd);
        if (!embed_rows(tokens + c0, cn, e)) {
            err = "prefill: token id out of range";
            return false;
        }
        for (int i = 0; i < cn; i++)
            for (int64_t j = 0; j < h.hc; j++)
                std::memcpy(rr.data() + (size_t) i * hcd + j * h.n_embd, e.data() + (size_t) i * h.n_embd, sizeof(float) * h.n_embd);
        history_.insert(history_.end(), tokens + c0, tokens + c0 + cn);
        std::thread pf = prefetch(0);
        for (int64_t il = 0; il < h.n_layer; il++) {
            int ua = 2048;
            if (!h.recurrent[(size_t) il]) {  // keep the mask and the sparse temporaries near 256 MB
                const int64_t kv = std::max<int64_t>(kKvPad, base + cn);
                ua = (int) std::clamp<int64_t>(48'000'000 / kv / 32 * 32, 32, 512);
            }
            for (int m0 = 0; m0 < cn; m0 += ua) {
                const int n = std::min(ua, cn - m0);
                n_past_     = base + m0;
                if (!run_part(0, il, n, rr.data() + (size_t) m0 * hcd, nullptr, err)) return false;
            }
            pf.join();
            pf = prefetch(il + 1);
            for (int m0 = 0; m0 < cn; m0 += um) {
                const int n = std::min(um, cn - m0);
                n_past_     = base + m0;
                if (!run_part(1, il, n, rr.data() + (size_t) m0 * hcd, nullptr, err)) return false;
            }
        }
        pf.join();
        n_past_ = base + cn;
        if (mtp_) {  // the chunk's post-norm hidden rows feed the MTP rows, in pieces
            std::vector<float> hn;
            for (int o = 0; o < cn; o += 4096) {
                const int b = std::min(4096, cn - o);
                hn.resize((size_t) b * h.n_embd);
                if (!run_part(3, 0, b, rr.data() + (size_t) o * hcd, hn.data(), err)) return false;
                mtp_push_h(hn.data(), base + o, b);
                if (!mtp_fill(base + o + b, err)) return false;
            }
        }
        if (c0 + cn == N && logits && !run_part(2, 0, cn, rr.data(), logits, err)) return false;
    }
    return true;
}

// ---- MTP speculation -----------------------------------------------------------------------------------------

bool Glm5Next::load_mtp(std::string & err) {
    const auto &      h  = hp_;
    const GgufFile &  f  = file_.meta();
    if (f.i64(key("nextn_predict_layers"), 0) != 1) {
        err = "the GGUF has no MTP block";
        return false;
    }
    const std::string p  = "blk." + std::to_string(h.n_layer) + ".";
    ggml_init_params  ip = { ggml_tensor_overhead() * 64, nullptr, true };
    mtp_ctx_             = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> up;
    bool missing = false;
    auto mk = [&](const std::string & name) -> ggml_tensor * {
        const ggml_tensor * m = file_.tensor(name);
        if (!m) {
            if (!missing) err = "MTP tensor missing: " + name;
            missing = true;
            return nullptr;
        }
        ggml_tensor * t = ggml_dup_tensor(mtp_ctx_, m);
        ggml_set_name(t, ("mtp." + name).c_str());
        up.emplace_back(t, m);
        return t;
    };
    Glm5NextLayer & L = mtp_layer_;
    L.on_gpu          = true;
    L.attn_norm       = mk(p + "attn_norm.weight");
    L.ffn_norm        = mk(p + "ffn_norm.weight");
    L.q_a             = mk(p + "attn_q_a.weight");
    L.q_a_norm        = mk(p + "attn_q_a_norm.weight");
    L.q_b             = mk(p + "attn_q_b.weight");
    L.kv_a            = mk(p + "attn_kv_a_mqa.weight");
    L.kv_a_norm       = mk(p + "attn_kv_a_norm.weight");
    L.k_b             = mk(p + "attn_k_b.weight");
    L.v_b             = mk(p + "attn_v_b.weight");
    L.wo              = mk(p + "attn_output.weight");
    L.idx_k_norm      = mk(p + "indexer.k_norm.weight");
    L.idx_k_norm_b    = mk(p + "indexer.k_norm.bias");
    L.idx_proj        = mk(p + "indexer.proj.weight");
    L.idx_k           = mk(p + "indexer.attn_k.weight");
    L.idx_q_b         = mk(p + "indexer.attn_q_b.weight");
    L.idx_gate        = mk(p + "indexer_compressor_gate.weight");
    L.idx_ape         = mk(p + "indexer_compressor_ape.weight");
    L.gate_inp        = mk(p + "ffn_gate_inp.weight");
    L.exp_bias        = mk(p + "exp_probs_b.bias");
    L.gate_exps       = mk(p + "ffn_gate_exps.weight");
    L.up_exps         = mk(p + "ffn_up_exps.weight");
    L.down_exps       = mk(p + "ffn_down_exps.weight");
    L.gate_shexp      = mk(p + "ffn_gate_shexp.weight");
    L.up_shexp        = mk(p + "ffn_up_shexp.weight");
    L.down_shexp      = mk(p + "ffn_down_shexp.weight");
    mtp_eh_           = mk(p + "nextn.eh_proj.weight");
    mtp_enorm_        = mk(p + "nextn.enorm.weight");
    mtp_hnorm_        = mk(p + "nextn.hnorm.weight");
    mtp_head_norm_    = mk(p + "nextn.shared_head_norm.weight");
    if (missing) return false;
    // its caches, like an attention layer's: latent KV, indexer rows, pooled keys
    ggml_tensor * kv   = ggml_new_tensor_2d(mtp_ctx_, latent_type(opt_), h.kv_lora, n_ctx_);
    ggml_tensor * raw  = ggml_new_tensor_2d(mtp_ctx_, GGML_TYPE_F32, 2 * h.idx_dim, n_ctx_);
    ggml_tensor * pool = ggml_new_tensor_2d(mtp_ctx_, GGML_TYPE_F32, h.idx_dim, n_ctx_ / h.kpool + 1);
    mtp_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(mtp_ctx_, ggml_backend_get_default_buffer_type(gpu_));
    if (!mtp_buf_) {
        err = "not enough VRAM for the MTP block";
        return false;
    }
    ggml_backend_buffer_clear(mtp_buf_, 0);
    for (auto & [t, m] : up) ggml_backend_tensor_set(t, file_.data(m), 0, ggml_nbytes(t));
    kv_cache_.push_back(kv);  // index n_layer, as mtp_il_ will be
    idx_raw_.push_back(raw);
    idx_pool_.push_back(pool);
    mtp_ = true;
    fprintf(stderr, "MTP: blk.%lld, %.2f GB in VRAM\n", (long long) h.n_layer, ggml_backend_buffer_get_size(mtp_buf_) / 1e9);
    return true;
}

void Glm5Next::mtp_keep_h(int upto) {
    const int64_t ne   = hp_.n_embd;
    const int     rows = (int) ((int64_t) mtp_h_.size() / ne);
    mtp_h_.resize((size_t) (std::clamp(upto - mtp_h0_, 0, rows) * ne));
}

void Glm5Next::mtp_push_h(const float * rows, int pos0, int n) {
    const int64_t ne = hp_.n_embd;
    mtp_keep_h(pos0);  // a rewind: forget what lay past pos0
    if (mtp_h0_ + (int) ((int64_t) mtp_h_.size() / ne) != pos0) {  // a gap (a restored checkpoint)
        mtp_h_.clear();
        mtp_h0_ = pos0;
        mtp_n_  = std::max(mtp_n_, pos0 + 1);
    }
    mtp_h_.insert(mtp_h_.end(), rows, rows + (size_t) n * ne);
}

bool Glm5Next::mtp_fill(int upto, std::string & err) {
    if (!mtp_) return true;
    const int64_t ne   = hp_.n_embd;
    const int     rows = (int) ((int64_t) mtp_h_.size() / ne);
    upto               = std::min(upto, (int) history_.size());
    mtp_n_             = std::max(mtp_n_, mtp_h0_ + 1);
    while (mtp_n_ < upto && mtp_n_ - 1 < mtp_h0_ + rows) {
        const int j0 = mtp_n_;
        const int b  = std::min({ 512, upto - j0, mtp_h0_ + rows - (j0 - 1) });
        if (!mtp_run(history_.data() + j0, mtp_h_.data() + (size_t) (j0 - 1 - mtp_h0_) * ne, b, j0, nullptr, nullptr, err)) return false;
        mtp_n_ += b;
    }
    const int drop = std::min(rows, mtp_n_ - 1 - mtp_h0_);  // keep the rows from mtp_n_-1 on: the next row's input
    if (drop > 0) {
        mtp_h_.erase(mtp_h_.begin(), mtp_h_.begin() + (ptrdiff_t) (drop * ne));
        mtp_h0_ += drop;
    }
    return true;
}

bool Glm5Next::mtp_run(const int32_t * tokens, const float * hrows, int n, int pos0, int32_t * argmax, float * h_out,
                       std::string & err) {
    const auto &  h  = hp_;
    const int64_t ne = h.n_embd;
    if (pos0 + n > n_ctx_) {
        err = "MTP: context full";
        return false;
    }
    std::vector<float> e;
    if (!embed_rows(tokens, n, e)) {
        err = "MTP: token id out of range";
        return false;
    }
    const int saved = n_past_;
    n_past_         = pos0;  // dsa() / set_inputs() place rows at n_past_
    const int  n_kv   = std::min(n_ctx_, (pos0 + n + kKvPad - 1) / kKvPad * kKvPad);
    const bool sparse = sparse_at(n);

    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    Inputs           in;
    in.embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, n);
    ggml_set_input(in.embd);
    ggml_tensor * hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, n);
    ggml_set_input(hin);
    new_inputs(ctx, n, n_kv, sparse, in);

    const Glm5NextLayer & L = layers_[(size_t) mtp_il_];
    auto rms = [&](ggml_tensor * x, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, h.rms_eps), w); };
    // x = eh_proj([enorm(embedding) | hnorm(hidden)]), then a plain pre-norm block
    ggml_tensor * x    = ggml_mul_mat(ctx, mtp_eh_, ggml_concat(ctx, rms(in.embd, mtp_enorm_), rms(hin, mtp_hnorm_), 0));
    ggml_tensor * prev = nullptr;
    ggml_tensor * a    = dsa(ctx, gf, mtp_il_, rms(x, L.attn_norm), n, n_kv, sparse, in, &prev);  // writes this block's KV rows
    ggml_tensor * hout = nullptr, * am = nullptr;
    if (argmax || h_out) {
        x = ggml_add(ctx, x, a);
        x = ggml_add(ctx, x, moe(ctx, mtp_il_, rms(x, L.ffn_norm), n));
        ggml_tensor * last = ggml_view_2d(ctx, x, ne, 1, x->nb[1], x->nb[1] * (size_t) (n - 1));
        if (h_out) {  // the chained draft reads the block's own output hidden
            hout = ggml_cont(ctx, last);  // pre shared-head-norm, as vLLM chains DeepSeek MTP
            ggml_set_output(hout);
            ggml_build_forward_expand(gf, hout);
        }
        if (argmax) {
            ggml_tensor * lg = ggml_mul_mat(ctx, output_, rms(last, mtp_head_norm_));
            am               = ggml_argmax(ctx, lg);
            ggml_set_output(am);
            ggml_build_forward_expand(gf, am);
        }
    } else {
        ggml_build_forward_expand(gf, a);
    }
    bool ok = true;
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        err = "MTP: cannot allocate the compute graph";
        ok  = false;
    }
    if (ok) {
        ggml_backend_tensor_set(in.embd, e.data(), 0, ggml_nbytes(in.embd));
        ggml_backend_tensor_set(hin, hrows, 0, ggml_nbytes(hin));
        set_inputs(in, n, n_kv);
        if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
            err = "MTP: graph compute failed";
            ok  = false;
        }
    }
    if (ok && hout) ggml_backend_tensor_get(hout, h_out, 0, ggml_nbytes(hout));
    if (ok && am) ggml_backend_tensor_get(am, argmax, 0, sizeof(int32_t));
    ggml_free(ctx);
    n_past_ = saved;
    return ok;
}

bool Glm5Next::mtp_draft(int32_t next, int k, std::vector<int32_t> & drafts, std::string & err) {
    drafts.clear();
    if (!mtp_ || k <= 0 || n_past_ < 1 || n_past_ + k + 1 >= n_ctx_) return true;
    if (!mtp_fill(n_past_, err)) return false;
    const int64_t ne   = hp_.n_embd;
    const int     rows = (int) ((int64_t) mtp_h_.size() / ne);
    if (mtp_n_ != n_past_ || n_past_ - 1 < mtp_h0_ || n_past_ - 1 >= mtp_h0_ + rows) return true;  // no hidden: no drafts
    std::vector<float> hv((size_t) ne);
    int32_t            tok = next, d = 0;
    // the row for `next` is real (it is evaluated next whatever the drafts): it stays
    if (!mtp_run(&tok, mtp_h_.data() + (size_t) (n_past_ - 1 - mtp_h0_) * ne, 1, n_past_, &d, hv.data(), err)) return false;
    mtp_n_ = n_past_ + 1;
    drafts.push_back(d);
    for (int i = 1; i < k; i++) {
        tok = d;
        if (!mtp_run(&tok, hv.data(), 1, n_past_ + i, &d, hv.data(), err)) return false;
        drafts.push_back(d);
    }
    return true;
}

bool Glm5Next::verify(const int32_t * tokens, int n, float * logits, std::string & err) {
    if (n > kRecMax || rec_x_.empty()) {
        err = "verify: batch above " + std::to_string(kRecMax) + " or no MTP";
        return false;
    }
    for (size_t l = 0; l < sp_conv_.size(); l++) {  // the recurrent state before the batch
        if (!sp_conv_[l]) continue;
        ggml_backend_tensor_copy(conv_state_[l], sp_conv_[l]);
        ggml_backend_tensor_copy(ssm_state_[l], sp_ssm_[l]);
    }
    sp_n_past_    = n_past_;
    record_       = true;
    rec_n_        = n;
    const bool ok = eval(tokens, n, logits, err);
    record_       = false;
    if (!ok) rec_n_ = 0;
    return ok;
}

bool Glm5Next::commit(int keep, std::string & err) {
    if (rec_n_ <= 0 || keep < 1 || keep > rec_n_) {
        err = "commit: no verify to keep from";
        return false;
    }
    if (keep == rec_n_) {
        rec_n_ = 0;
        return true;
    }
    const auto & h = hp_;
    for (size_t l = 0; l < sp_conv_.size(); l++) {
        if (!sp_conv_[l]) continue;
        ggml_backend_tensor_copy(sp_conv_[l], conv_state_[l]);
        ggml_backend_tensor_copy(sp_ssm_[l], ssm_state_[l]);
    }
    // replay the kept tokens through the recurrences only (no attention, no experts)
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    const int64_t    D = h.kda_inner(), H = h.n_head, S = h.kda_dim;
    for (size_t l = 0; l < rec_x_.size(); l++) {
        if (!rec_x_[l]) continue;
        ggml_tensor * rx = rec_x_[l];
        auto part = [&](int i) { return ggml_cont(ctx, ggml_view_2d(ctx, rx, D, keep, rx->nb[1], ggml_row_size(rx->type, (size_t) i * D))); };
        ggml_tensor * g    = ggml_reshape_4d(ctx, ggml_view_2d(ctx, rec_g_[l], D, keep, rec_g_[l]->nb[1], 0), S, H, keep, 1);
        ggml_tensor * beta = ggml_reshape_4d(ctx, ggml_view_2d(ctx, rec_beta_[l], H, keep, rec_beta_[l]->nb[1], 0), 1, H, keep, 1);
        ggml_build_forward_expand(gf, kda_core(ctx, gf, (int64_t) l, part(0), part(1), part(2), g, beta, keep));
    }
    bool ok = true;
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf) || ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        err = "commit: replay failed";
        ok  = false;
    }
    ggml_free(ctx);
    n_past_ = sp_n_past_ + keep;
    history_.resize((size_t) n_past_);
    mtp_keep_h(n_past_);
    rec_n_ = 0;
    return ok;
}


} // namespace e8::model
