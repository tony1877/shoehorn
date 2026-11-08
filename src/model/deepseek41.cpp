#include "model/deepseek41.h"

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

constexpr int kGraphSize = 65536;
constexpr int kPad       = 256;

std::string key(const char * k) {
    return std::string("deepseek41.") + k;
}

} // namespace

bool Deepseek41Hparams::load(const GgufSet & s, std::string & err) {
    const GgufFile & f = s.meta();
    if (f.arch() != "deepseek41") {
        err = "architecture \"" + f.arch() + "\" is not deepseek41";
        return false;
    }
    n_layer       = f.i64(key("block_count"), 0);
    n_embd        = f.i64(key("embedding_length"), 0);
    n_head        = f.i64(key("attention.head_count"), 0);
    head_dim      = f.i64(key("attention.key_length"), 0);
    n_rot         = f.i64(key("rope.dimension_count"), 0);
    q_lora        = f.i64(key("attention.q_lora_rank"), 0);
    n_swa         = f.i64(key("attention.sliding_window"), 0);
    o_groups      = f.i64(key("attention.output_group_count"), 0);
    o_lora        = f.i64(key("attention.output_lora_rank"), 0);
    idx_n_head    = f.i64(key("attention.indexer.head_count"), 0);
    idx_dim       = f.i64(key("attention.indexer.key_length"), 0);
    idx_top_k     = f.i64(key("attention.indexer.top_k"), 0);
    hc            = f.i64(key("hyper_connection.count"), 0);
    hc_iters      = f.i64(key("hyper_connection.sinkhorn_iterations"), 0);
    hc_eps        = f.f32(key("hyper_connection.epsilon"), 1e-6f);
    rms_eps       = f.f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    freq_base     = f.f32(key("rope.freq_base"), 10000.0f);
    comp_freq_base = f.f32(key("attention.compress_rope_freq_base"), 160000.0f);
    yarn_factor   = f.str(key("rope.scaling.type"), "") == "yarn" ? f.f32(key("rope.scaling.factor"), 1.0f) : 1.0f;
    n_ctx_orig    = f.i64(key("rope.scaling.original_context_length"), 0);
    beta_fast     = f.f32(key("rope.scaling.yarn_beta_fast"), 32.0f);
    beta_slow     = f.f32(key("rope.scaling.yarn_beta_slow"), 1.0f);
    n_expert      = f.i64(key("expert_count"), 0);
    n_expert_used = f.i64(key("expert_used_count"), 0);
    n_ff_exp      = f.i64(key("expert_feed_forward_length"), 0);
    expert_weights_scale = f.f32(key("expert_weights_scale"), 1.0f);
    expert_weights_norm  = f.b(key("expert_weights_norm"), true);
    add_bos       = f.b("tokenizer.ggml.add_bos_token", false);
    bos           = f.i64("tokenizer.ggml.bos_token_id", -1);
    if (f.i64(key("expert_gating_func"), 0) != 4) {
        err = "deepseek41: only the sqrt(softplus) router is supported";
        return false;
    }
    auto per_layer = [&](const char * k) {
        std::vector<float> v = f.f32_arr(key(k));
        if (v.size() == 1) v.assign((size_t) n_layer, v[0]);
        v.resize((size_t) n_layer, 0.0f);
        return v;
    };
    clamp_exp   = per_layer("swiglu_clamp_exp");
    clamp_shexp = f.has(key("swiglu_clamp_shexp")) ? per_layer("swiglu_clamp_shexp") : clamp_exp;

    ratio = f.i64_arr(key("attention.compress_ratios"));
    if ((int64_t) ratio.size() < n_layer) {
        err = "deepseek41: compress_ratios shorter than block_count";
        return false;
    }
    ratio.resize((size_t) n_layer);
    // each compressed layer reads the KV / top-k of the latest source layer at or before it
    const std::vector<int64_t> ks = f.i64_arr(key("attention.kv_source_layers"));
    const std::vector<int64_t> is = f.i64_arr(key("attention.indexer.source_layers"));
    kv_src.assign((size_t) n_layer, -1);
    topk_src.assign((size_t) n_layer, -1);
    int64_t last_kv = -1, last_idx = -1;
    for (int64_t il = 0; il < n_layer; il++) {
        if (std::find(ks.begin(), ks.end(), il) != ks.end()) last_kv = il;
        if (std::find(is.begin(), is.end(), il) != is.end()) last_idx = il;
        if (ratio[(size_t) il] == 0) continue;
        if (ratio[(size_t) il] > 2 || last_kv < 0 || last_idx < last_kv) {
            err = "deepseek41: layer " + std::to_string(il) + " has a compress ratio without a kv / index source";
            return false;
        }
        kv_src[(size_t) il]   = last_kv;
        topk_src[(size_t) il] = last_idx;
    }
    cand_src = f.i64(key("attention.indexer.candidate_source_layer"), -1);
    if (cand_src >= 0) {
        cand_blocks     = f.i64(key("attention.indexer.candidate_top_k_blocks"), 0);
        cand_block_size = f.i64(key("attention.indexer.candidate_block_size"), 0);
        if (cand_src >= n_layer || topk_src[(size_t) cand_src] != cand_src || cand_blocks <= 0 || cand_block_size <= 0) {
            err = "deepseek41: invalid candidate selection";
            return false;
        }
    }

    // Engram
    engram_layer.assign((size_t) n_layer, -1);
    const std::vector<int64_t> el = f.i64_arr(key("ple.layers"));
    for (size_t e = 0; e < el.size(); e++)
        if (el[e] >= 0 && el[e] < n_layer) engram_layer[(size_t) el[e]] = (int64_t) e;
    if (!el.empty()) {
        ple_ngram    = f.i64(key("ple.ngram_size"), 0);
        ple_per_gram = f.i64(key("ple.heads_per_ngram"), 0);
        ple_head_dim = f.i64(key("embedding_length_per_layer_input"), 0);
        ple_n_heads  = (ple_ngram - 1) * ple_per_gram;
        engram_pad   = f.i64(key("engram.pad_token_id"), 0);
        for (int64_t v : f.i64_arr(key("ple.layer_multipliers"))) ple_mult.push_back((uint64_t) v);
        for (int64_t v : f.i64_arr(key("ple.head_offsets"))) ple_offset.push_back((uint64_t) v);
        for (int64_t v : f.i64_arr(key("ple.head_vocab_sizes"))) ple_vocab.push_back((uint64_t) v);
        for (int64_t v : f.i64_arr(key("engram.token_map"))) token_map.push_back((int32_t) v);
        const size_t ne = el.size();
        if (ple_ngram < 2 || ple_head_dim <= 0 || ple_mult.size() != ne * (size_t) ple_ngram ||
            ple_offset.size() != ne * (size_t) ple_n_heads || ple_vocab.size() != ne * (size_t) ple_n_heads ||
            engram_pad < 0 || engram_pad >= (int64_t) token_map.size()) {
            err = "deepseek41: inconsistent Engram metadata";
            return false;
        }
        for (uint64_t p : ple_vocab)
            if (p < 2) {
                err = "deepseek41: Engram bucket count below 2";
                return false;
            }
    }
    const ggml_tensor * emb = s.tensor("token_embd.weight");
    n_vocab                 = emb ? emb->ne[1] : 0;
    if (n_layer <= 0 || n_embd <= 0 || n_head <= 0 || head_dim <= 0 || n_rot <= 0 || n_rot > head_dim || q_lora <= 0 ||
        n_swa <= 0 || o_groups <= 0 || o_lora <= 0 || n_head % o_groups != 0 || hc != 4 || n_vocab <= 0 ||
        n_expert <= 0 || n_expert_used <= 0 || idx_dim < n_rot || idx_top_k <= 0) {
        err = "incomplete or unsupported deepseek41 hyperparameters";
        return false;
    }
    return true;
}

Deepseek41::Deepseek41() = default;

Deepseek41::~Deepseek41() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (ec_.buf) ggml_backend_buffer_free(ec_.buf);
    if (ec_.ctx) ggml_free(ec_.ctx);
    if (wbuf_gpu_) ggml_backend_buffer_free(wbuf_gpu_);
    for (auto b : map_bufs_)
        if (b) ggml_backend_buffer_free(b);
    if (sbuf_) ggml_backend_buffer_free(sbuf_);
    if (sctx_) ggml_free(sctx_);
    for (int i = 0; i < 2; i++)
        if (wctx_[i]) ggml_free(wctx_[i]);
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
}

bool Deepseek41::load(const std::string & path, const LoadOptions & opt, std::string & err) {
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
    if (!gpu_) {
        err = "deepseek41 needs a GPU";
        return false;
    }
    const int n_gpu_layers = std::clamp(opt.n_gpu_layers, 0, (int) h.n_layer);

    const size_t     n_tens = file_.tensors().size() + 8;
    ggml_init_params ip     = { ggml_tensor_overhead() * n_tens, nullptr, true };
    wctx_[0]                = ggml_init(ip);
    wctx_[1]                = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> to_gpu, to_map;
    bool missing = false;
    // shape3: wo_a is stored 2D [a, b*groups] and used 3D [a, b, groups] (same bytes)
    auto make = [&](const std::string & name, bool gpu, bool required = true, int64_t groups = 0) -> ggml_tensor * {
        const ggml_tensor * m = file_.tensor(name);
        if (!m) {
            if (required && !missing) {
                err     = "tensor missing: " + name;
                missing = true;
            }
            return nullptr;
        }
        ggml_tensor * t = groups ? ggml_new_tensor_3d(wctx_[gpu ? 0 : 1], m->type, m->ne[0], m->ne[1] / groups, groups)
                                 : ggml_dup_tensor(wctx_[gpu ? 0 : 1], m);
        ggml_set_name(t, name.c_str());
        (gpu ? to_gpu : to_map).emplace_back(t, m);
        return t;
    };
    tok_embd_     = file_.tensor("token_embd.weight");
    engram_table_ = file_.tensor("per_layer_token_embd.weight");
    if (!tok_embd_) {
        err = "tensor missing: token_embd.weight";
        return false;
    }
    output_      = make("output.weight", true);
    output_norm_ = make("output_norm.weight", true);
    layers_.resize((size_t) h.n_layer);
    for (int64_t il = 0; il < h.n_layer; il++) {
        Deepseek41Layer & L = layers_[(size_t) il];
        const bool        g = il >= h.n_layer - n_gpu_layers;
        const std::string p = "blk." + std::to_string(il) + ".";
        const size_t      l = (size_t) il;
        L.on_gpu    = g;
        L.attn_norm = make(p + "attn_norm.weight", g);
        L.ffn_norm  = make(p + "ffn_norm.weight", g);
        L.sinks     = make(p + "attn_sinks.weight", g);
        L.hca_fn    = make(p + "hc_attn_fn.weight", g);
        L.hca_base  = make(p + "hc_attn_base.weight", g);
        L.hca_scale = make(p + "hc_attn_scale.weight", g);
        L.hcf_fn    = make(p + "hc_ffn_fn.weight", g);
        L.hcf_base  = make(p + "hc_ffn_base.weight", g);
        L.hcf_scale = make(p + "hc_ffn_scale.weight", g);
        L.q_a       = make(p + "attn_q_a.weight", g);
        L.q_a_norm  = make(p + "attn_q_a_norm.weight", g);
        L.q_b       = make(p + "attn_q_b.weight", g);
        L.wkv       = make(p + "attn_kv.weight", g);
        L.kv_norm   = make(p + "attn_kv_a_norm.weight", g);
        L.wo_a      = make(p + "attn_output_a.weight", g, true, h.o_groups);
        L.wo_b      = make(p + "attn_output_b.weight", g);
        if (h.kv_src[l] == il) {
            L.comp_kv    = make(p + "attn_compressor_kv.weight", g);
            L.comp_norm  = make(p + "attn_compressor_norm.weight", g);
            L.comp_gate  = make(p + "attn_compressor_gate.weight", g, h.ratio[l] > 1);
            L.idx_k      = make(p + "indexer.attn_k.weight", g);
            L.idx_k_norm = make(p + "indexer.k_norm.weight", g);
        }
        if (h.topk_src[l] == il) {
            L.idx_q_b  = make(p + "indexer.attn_q_b.weight", g);
            L.idx_proj = make(p + "indexer.proj.weight", g);
        }
        if (h.engram_layer[l] >= 0) {
            L.engram_kv   = make(p + "engram_kv.weight", g);
            L.engram_gate = make(p + "engram_gate.weight", g);
        }
        L.gate_inp   = make(p + "ffn_gate_inp.weight", g);
        L.exp_bias   = make(p + "exp_probs_b.bias", g);
        const bool xg = g && opt.experts_gpu;
        L.gate_exps  = make(p + "ffn_gate_exps.weight", xg);
        L.up_exps    = make(p + "ffn_up_exps.weight", xg);
        L.down_exps  = make(p + "ffn_down_exps.weight", xg);
        L.gate_shexp = make(p + "ffn_gate_shexp.weight", g);
        L.up_shexp   = make(p + "ffn_up_shexp.weight", g);
        L.down_shexp = make(p + "ffn_down_shexp.weight", g);
        if (missing) return false;
    }
    if (h.ple_n_heads > 0 && (!engram_table_ || engram_table_->ne[0] != h.ple_head_dim)) {
        err = "tensor missing or misshaped: per_layer_token_embd.weight";
        return false;
    }
    if (ggml_get_first_tensor(wctx_[0])) {
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

    // state (VRAM): window rings for every layer, compressed rows and compressor inputs for the kv source layers
    n_ctx_ = std::max(kPad, (opt.n_ctx + kPad - 1) / kPad * kPad);
    R_     = (int) ((h.n_swa + std::max(opt.n_ubatch, 512) + kPad - 1) / kPad * kPad);
    const size_t     nl = (size_t) h.n_layer;
    ggml_init_params sp = { ggml_tensor_overhead() * (nl * 8 + 8), nullptr, true };
    sctx_               = ggml_init(sp);
    for (auto * v : { &ring_, &comp_k_, &comp_ik_, &comp_in_, &ring_ck_, &comp_in_ck_ }) v->assign(nl, nullptr);
    const ggml_type kt = opt.kv_type == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    for (size_t l = 0; l < nl; l++) {
        ring_[l]    = ggml_new_tensor_2d(sctx_, kt, h.head_dim, R_);
        ring_ck_[l] = ggml_dup_tensor(sctx_, ring_[l]);
        if (h.kv_src[l] == (int64_t) l) {
            comp_k_[l]     = ggml_new_tensor_2d(sctx_, kt, h.head_dim, n_ctx_ / h.ratio[l] + 1);
            comp_ik_[l]    = ggml_new_tensor_2d(sctx_, GGML_TYPE_F32, h.idx_dim, n_ctx_ / h.ratio[l] + 1);
            comp_in_[l]    = ggml_new_tensor_2d(sctx_, GGML_TYPE_F32, 2 * h.head_dim, R_);
            comp_in_ck_[l] = ggml_dup_tensor(sctx_, comp_in_[l]);
        }
    }
    sbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(sctx_, ggml_backend_get_default_buffer_type(gpu_));
    if (!sbuf_) {
        err = "not enough VRAM for the KV cache (lower --ctx)";
        return false;
    }
    if (!init_expert_cache(err)) return false;
    reset();

    std::vector<ggml_backend_t> bes = { gpu_, cpu_ };
    sched_ = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    graph_meta_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    return true;
}

void Deepseek41::reset() {
    if (sbuf_) ggml_backend_buffer_clear(sbuf_, 0);
    ring_pos_.assign((size_t) R_, -1);
    ring_pos_ck_ = ring_pos_;
    history_.clear();
    n_past_ = ck_n_past_ = 0;
    rec_n_  = 0;
}

bool Deepseek41::eval_last(const int32_t * tokens, int n, float * logits, std::string & err) {
    last_only_    = true;
    const bool ok = eval(tokens, n, logits, err);
    last_only_    = false;
    return ok;
}

// The rings are the only state a later token overwrites (the compressed rows are positional): a checkpoint copies them
void Deepseek41::checkpoint_save() {
    for (size_t l = 0; l < ring_.size(); l++) {
        ggml_backend_tensor_copy(ring_[l], ring_ck_[l]);
        if (comp_in_[l]) ggml_backend_tensor_copy(comp_in_[l], comp_in_ck_[l]);
    }
    ring_pos_ck_ = ring_pos_;
    ck_n_past_   = n_past_;
}

void Deepseek41::checkpoint_restore() {
    for (size_t l = 0; l < ring_.size(); l++) {
        ggml_backend_tensor_copy(ring_ck_[l], ring_[l]);
        if (comp_in_[l]) ggml_backend_tensor_copy(comp_in_ck_[l], comp_in_[l]);
    }
    ring_pos_ = ring_pos_ck_;
    n_past_   = ck_n_past_;
    history_.resize((size_t) n_past_);
}

bool Deepseek41::save_seq(std::vector<uint8_t> & out) {
    if (rec_n_ > 0) return false;
    const int hdr[4] = { n_past_, ck_n_past_, (int) history_.size(), R_ };
    out.assign((const uint8_t *) hdr, (const uint8_t *) hdr + sizeof hdr);
    auto put = [&](const void * p, size_t n) { out.insert(out.end(), (const uint8_t *) p, (const uint8_t *) p + n); };
    put(history_.data(), history_.size() * 4);
    put(ring_pos_.data(), ring_pos_.size() * 4);
    put(ring_pos_ck_.data(), ring_pos_ck_.size() * 4);
    auto tens = [&](ggml_tensor * t, size_t n) {
        if (!t || !n) return;
        const size_t o = out.size();
        out.resize(o + n);
        ggml_backend_tensor_get(t, out.data() + o, 0, n);
    };
    for (size_t l = 0; l < ring_.size(); l++) {
        const int rows = comp_k_[l] ? (int) (n_past_ / hp_.ratio[l] + 1) : 0;
        tens(ring_[l], ggml_nbytes(ring_[l]));
        tens(ring_ck_[l], ggml_nbytes(ring_ck_[l]));
        if (comp_k_[l]) {
            tens(comp_k_[l], (size_t) rows * comp_k_[l]->nb[1]);
            tens(comp_ik_[l], (size_t) rows * comp_ik_[l]->nb[1]);
            tens(comp_in_[l], ggml_nbytes(comp_in_[l]));
            tens(comp_in_ck_[l], ggml_nbytes(comp_in_ck_[l]));
        }
    }
    return true;
}

bool Deepseek41::load_seq(const std::vector<uint8_t> & in) {
    int hdr[4];
    if (in.size() < sizeof hdr) return false;
    std::memcpy(hdr, in.data(), sizeof hdr);
    if (hdr[0] < 0 || hdr[0] > n_ctx_ || hdr[2] < 0 || hdr[3] != R_) return false;
    size_t o  = sizeof hdr;
    bool   ok = true;
    auto get = [&](void * p, size_t n) {
        if (o + n > in.size()) return ok = false;
        std::memcpy(p, in.data() + o, n);
        o += n;
        return true;
    };
    std::vector<int32_t> hist((size_t) hdr[2]), rp((size_t) R_), rpc((size_t) R_);
    get(hist.data(), hist.size() * 4);
    get(rp.data(), rp.size() * 4);
    get(rpc.data(), rpc.size() * 4);
    auto tens = [&](ggml_tensor * t, size_t n) {
        if (!ok || !t || !n) return;
        if (o + n > in.size()) {
            ok = false;
            return;
        }
        ggml_backend_tensor_set(t, in.data() + o, 0, n);
        o += n;
    };
    for (size_t l = 0; l < ring_.size(); l++) {
        const int rows = comp_k_[l] ? (int) (hdr[0] / hp_.ratio[l] + 1) : 0;
        tens(ring_[l], ggml_nbytes(ring_[l]));
        tens(ring_ck_[l], ggml_nbytes(ring_ck_[l]));
        if (comp_k_[l]) {
            tens(comp_k_[l], (size_t) rows * comp_k_[l]->nb[1]);
            tens(comp_ik_[l], (size_t) rows * comp_ik_[l]->nb[1]);
            tens(comp_in_[l], ggml_nbytes(comp_in_[l]));
            tens(comp_in_ck_[l], ggml_nbytes(comp_in_ck_[l]));
        }
    }
    if (!ok || o != in.size()) {
        reset();
        return false;
    }
    n_past_      = hdr[0];
    ck_n_past_   = hdr[1];
    history_     = std::move(hist);
    ring_pos_    = std::move(rp);
    ring_pos_ck_ = std::move(rpc);
    rec_n_       = 0;
    return true;
}

// ---- building blocks -------------------------------------------------------------------------------------------

void Deepseek41::dbg(ggml_context * ctx, ggml_cgraph * gf, int64_t il, const char * name, ggml_tensor * t) {
    static const int layer = std::getenv("E8_DS_DUMP") ? std::atoi(std::getenv("E8_DS_DUMP")) : -1;  // the layer to trace
    if (layer < 0 || il != layer) return;
    t = ggml_cont(ctx, t);
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
    dbg_.emplace_back(std::string(name) + "-" + std::to_string(il), t);
}

// RoPE on the last n_rot dims of each head (after n_nope); compressed layers use their base with YaRN
ggml_tensor * Deepseek41::rope(ggml_context * ctx, ggml_tensor * t, ggml_tensor * pos, int64_t n_nope, int64_t il, bool back) {
    const auto & h     = hp_;
    const bool   comp  = h.ratio[(size_t) il] != 0;
    const float  base  = comp ? h.comp_freq_base : h.freq_base;
    const float  scale = comp ? 1.0f / h.yarn_factor : 1.0f;
    const float  ext   = comp && h.yarn_factor != 1.0f ? 1.0f : 0.0f;
    const float  attn  = ext == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / scale));
    const float  bf = comp ? h.beta_fast : 0.0f, bs = comp ? h.beta_slow : 0.0f;
    const int    orig = comp ? (int) h.n_ctx_orig : 0;
    ggml_tensor * r = back ? ggml_rope_ext_back(ctx, t, pos, nullptr, (int) h.n_rot, 0, orig, base, scale, ext, attn, bf, bs)
                           : ggml_rope_ext(ctx, t, pos, nullptr, (int) h.n_rot, 0, orig, base, scale, ext, attn, bf, bs);
    return ggml_rope_set_offset(r, (int) n_nope);
}

void Deepseek41::mixes(ggml_context * ctx, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
                       ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) {
    const auto &  h  = hp_;
    const int64_t hc = h.hc, n = x->ne[2];
    ggml_tensor * mx = ggml_mul_mat(ctx, fn, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, x, hc * h.n_embd, n), h.rms_eps));
    auto part = [&](int64_t i0, int64_t ne0, int64_t si) {
        ggml_tensor * v = ggml_view_2d(ctx, mx, ne0, n, mx->nb[1], ggml_row_size(mx->type, i0));
        v               = ggml_mul(ctx, v, ggml_view_1d(ctx, scale, 1, ggml_row_size(scale->type, si)));
        return ggml_add(ctx, v, ggml_view_1d(ctx, base, ne0, ggml_row_size(base->type, i0)));
    };
    *pre  = ggml_scale_bias(ctx, ggml_sigmoid(ctx, part(0, hc, 0)), 1.0f, h.hc_eps);
    *post = ggml_scale(ctx, ggml_sigmoid(ctx, part(hc, hc, 1)), 2.0f);
    *comb = ggml_dsv4_hc_comb(ctx, mx, scale, base, h.hc_eps, (int32_t) h.hc_iters);
}

ggml_tensor * Deepseek41::engram(ggml_context * ctx, int64_t il, ggml_tensor * x, ggml_tensor * rows) {
    const auto &            h  = hp_;
    const Deepseek41Layer & L  = layers_[(size_t) il];
    const int64_t           ne = h.n_embd, hc = h.hc, n = x->ne[2];
    ggml_tensor * kv    = ggml_mul_mat(ctx, L.engram_kv, rows);  // [ne * (hc + 1), n]
    ggml_tensor * k     = ggml_cont(ctx, ggml_view_3d(ctx, kv, ne, hc, n, ggml_row_size(kv->type, ne), kv->nb[1], 0));
    ggml_tensor * v     = ggml_cont(ctx, ggml_view_3d(ctx, kv, ne, 1, n, ggml_row_size(kv->type, ne), kv->nb[1], ggml_row_size(kv->type, hc * ne)));
    ggml_tensor * eg    = ggml_is_quantized(L.engram_gate->type) ? ggml_cast(ctx, L.engram_gate, GGML_TYPE_F32) : L.engram_gate;
    ggml_tensor * dot   = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, x, h.rms_eps), ggml_rms_norm(ctx, k, h.rms_eps)), eg);
    dot                 = ggml_scale(ctx, ggml_sum_rows(ctx, dot), 1.0f / sqrtf((float) ne));  // [1, hc, n]
    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1
    ggml_tensor * mag  = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, mag, sign));
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_repeat_4d(ctx, v, ne, hc, n, 1), gate));
}

ggml_tensor * Deepseek41::attention(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n, Inputs & in,
                                    Build & b) {
    const auto &            h  = hp_;
    const Deepseek41Layer & L  = layers_[(size_t) il];
    const size_t            l  = (size_t) il;
    const int64_t           D = h.head_dim, H = h.n_head, nope = h.head_dim - h.n_rot, di = h.idx_dim;
    auto rms = [&](ggml_tensor * x, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, h.rms_eps), w); };

    dbg(ctx, gf, il, "attn_norm", cur);
    ggml_tensor * qa_out = ggml_mul_mat(ctx, L.q_a, cur);
    dbg(ctx, gf, il, "q_a_out", qa_out);
    ggml_tensor * qr = rms(qa_out, L.q_a_norm);                                                      // [q_lora, n]
    dbg(ctx, gf, il, "qr_norm", qr);
    ggml_tensor * q0 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.q_b, qr), D, H, n);
    dbg(ctx, gf, il, "q_pre_rope", q0);
    ggml_tensor * q  = rope(ctx, q0, in.pos, nope, il);
    dbg(ctx, gf, il, "q", q);
    ggml_tensor * kv0 = rms(ggml_mul_mat(ctx, L.wkv, cur), L.kv_norm);
    dbg(ctx, gf, il, "kv_pre_rope", kv0);
    ggml_tensor * kv = rope(ctx, ggml_reshape_3d(ctx, kv0, D, 1, n), in.pos, nope, il);
    dbg(ctx, gf, il, "kv", kv);
    ggml_tensor * ring = ggml_set_rows(ctx, ring_[l], ggml_reshape_2d(ctx, kv, D, n), in.ring_idx);
    ggml_build_forward_expand(gf, ring);

    const int64_t r   = h.ratio[l];
    const int64_t src = h.kv_src[l];
    Inputs::Comp & cp = in.c[r];
    // the source layer's compressed rows (the ones written by this graph when it is in it)
    if (r != 0 && src == il) {
        ggml_tensor * sk = ggml_mul_mat(ctx, L.comp_kv, cur);
        ggml_tensor * sg = L.comp_gate ? ggml_mul_mat(ctx, L.comp_gate, cur) : ggml_scale(ctx, sk, 0.0f);
        ggml_tensor * ci = ggml_set_rows(ctx, comp_in_[l], ggml_concat(ctx, sk, sg, 0), in.ring_idx);
        ggml_build_forward_expand(gf, ci);
        if (cp.n_blk > 0) {
            // pool each completed pair: per channel, softmax over the members of the gate scores weighs their kv
            const int64_t nb   = cp.n_blk;
            ggml_tensor * rows = ggml_reshape_3d(ctx, ggml_get_rows(ctx, ci, cp.blk_rows), 2 * D, r, nb);
            ggml_tensor * v    = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, rows, D, r, nb, rows->nb[1], rows->nb[2], 0), 1, 0, 2, 3));
            ggml_tensor * s    = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, rows, D, r, nb, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, D)), 1, 0, 2, 3));
            ggml_tensor * c    = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, v, ggml_soft_max(ctx, s))), D, nb);
            c                  = rms(c, L.comp_norm);
            dbg(ctx, gf, il, "comp_latent", c);
            // the indexer key comes from the RoPE-free latent; a row stands for the first token of its pair
            ggml_tensor * ik = rope(ctx, ggml_reshape_3d(ctx, rms(ggml_mul_mat(ctx, L.idx_k, c), L.idx_k_norm), di, 1, nb), cp.blk_pos, di - h.n_rot, il);
            c                = rope(ctx, ggml_reshape_3d(ctx, c, D, 1, nb), cp.blk_pos, nope, il);
            ggml_tensor * wk = ggml_set_rows(ctx, comp_k_[l], ggml_reshape_2d(ctx, c, D, nb), cp.blk_dst);
            ggml_tensor * wi = ggml_set_rows(ctx, comp_ik_[l], ggml_reshape_2d(ctx, ik, di, nb), cp.blk_dst);
            ggml_build_forward_expand(gf, wk);
            ggml_build_forward_expand(gf, wi);
            b.ck[l]  = wk;
            b.cik[l] = wi;
        }
    }

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * qp    = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, n, H]
    ggml_tensor * Kr    = ggml_view_2d(ctx, ring, D, R_, ring->nb[1], 0);
    ggml_tensor * a     = nullptr;
    if (r == 0 || cp.n_comp == 0) {  // window only
        a = ggml_flash_attn_ext(ctx, qp, Kr, Kr, in.raw_mask, scale, 0.0f, 0.0f);
    } else {
        const int64_t nc  = cp.n_comp;
        ggml_tensor * ck  = b.ck[(size_t) src] ? b.ck[(size_t) src] : comp_k_[(size_t) src];
        ggml_tensor * cik = b.cik[(size_t) src] ? b.cik[(size_t) src] : comp_ik_[(size_t) src];
        ck                = ggml_view_2d(ctx, ck, D, nc, ck->nb[1], 0);
        cik               = ggml_view_2d(ctx, cik, di, nc, cik->nb[1], 0);
        ggml_tensor * comp_mask = cp.mask;  // [nc, n] f32
        const int64_t ts        = h.topk_src[l];
        if (ts == il && nc > h.idx_top_k) {
            // lightning indexer over the visible compressed rows, in chunks of tokens (scores [Hi, chunk, nc])
            const int64_t Hi = h.idx_n_head;
            ggml_tensor * iq = rope(ctx, ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.idx_q_b, qr), di, Hi, n), in.pos, di - h.n_rot, il);
            ggml_tensor * iw = ggml_scale(ctx, ggml_mul_mat(ctx, L.idx_proj, cur), 1.0f / sqrtf((float) (di * Hi)));  // [Hi, n]
            const int64_t chunk = std::max<int64_t>(1, (INT64_C(1) << 26) / (Hi * nc));
            const int64_t bs    = h.cand_block_size;
            const bool    pick  = h.cand_src == il && cp.cand_pin;
            const bool    use   = h.cand_src >= 0 && h.cand_src < il && b.cand_sel;
            const int64_t nblk  = (nc + std::max<int64_t>(bs, 1) - 1) / std::max<int64_t>(bs, 1);
            std::vector<ggml_tensor *> tops, sels;
            for (int64_t t0 = 0; t0 < n; t0 += chunk) {
                const int64_t m  = std::min<int64_t>(chunk, n - t0);
                ggml_tensor * qc = ggml_view_2d(ctx, iq, di, Hi * m, iq->nb[1], (size_t) t0 * iq->nb[2]);
                ggml_tensor * s  = ggml_relu(ctx, ggml_mul_mat(ctx, cik, qc));                         // [nc, Hi*m]
                s = ggml_mul(ctx, ggml_reshape_3d(ctx, s, nc, Hi, m),
                             ggml_reshape_3d(ctx, ggml_view_2d(ctx, iw, Hi, m, iw->nb[1], (size_t) t0 * iw->nb[1]), 1, Hi, m));
                s = ggml_cont(ctx, ggml_permute(ctx, s, 1, 0, 2, 3));                                  // [Hi, nc, m]
                s = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, s), nc, m);
                s = ggml_add(ctx, s, ggml_view_2d(ctx, comp_mask, nc, m, comp_mask->nb[1], (size_t) t0 * comp_mask->nb[1]));
                if (pick && nblk > h.cand_blocks) {
                    // level one: best row per block of the candidate source, its newest block pinned, keep the top blocks
                    ggml_tensor * s2 = s;
                    if (nblk * bs != nc) s2 = ggml_concat(ctx, s2, ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nblk * bs - nc, m), -INFINITY), 0);
                    ggml_tensor * bm = ggml_pool_1d(ctx, s2, GGML_OP_POOL_MAX, (int) bs, (int) bs, 0);  // [nblk, m]
                    bm = ggml_add(ctx, bm, ggml_view_2d(ctx, cp.cand_pin, nblk, m, cp.cand_pin->nb[1], (size_t) t0 * cp.cand_pin->nb[1]));
                    sels.push_back(ggml_top_k(ctx, bm, (int) h.cand_blocks));
                } else if (use) {
                    // level two: own scores, only inside the source's candidate blocks
                    ggml_tensor * bsel = ggml_cont(ctx, ggml_view_2d(ctx, b.cand_sel, b.cand_sel->ne[0], m, b.cand_sel->nb[1], (size_t) t0 * b.cand_sel->nb[1]));
                    const int64_t kb   = bsel->ne[0];
                    ggml_tensor * bm   = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nblk, m), -INFINITY);
                    ggml_tensor * z    = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kb, m), 0.0f);
                    bm = ggml_set_rows(ctx, bm, z, ggml_reshape_3d(ctx, bsel, kb, m, 1));
                    bm = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, bm, bs, nblk, m, 1), bs * nblk, m);
                    bm = ggml_cont(ctx, ggml_view_2d(ctx, bm, nc, m, bm->nb[1], 0));
                    s  = ggml_add(ctx, s, bm);
                }
                tops.push_back(ggml_top_k(ctx, s, (int) h.idx_top_k));
            }
            auto join = [&](std::vector<ggml_tensor *> v) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) w.push_back(i + 1 < v.size() ? ggml_concat(ctx, v[i], v[i + 1], 1) : v[i]);
                    v.swap(w);
                }
                return v[0];
            };
            if (!sels.empty()) b.cand_sel = ggml_cont(ctx, join(sels));
            b.topk[l] = ggml_cont(ctx, join(tops));
        }
        ggml_tensor * top = ts >= 0 ? b.topk[(size_t) ts] : nullptr;
        if (top && top->ne[0] < nc) {  // -inf outside the selected rows
            ggml_tensor * all  = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nc, n), -INFINITY);
            ggml_tensor * zero = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, top->ne[0], n), 0.0f);
            all       = ggml_set_rows(ctx, all, zero, ggml_reshape_3d(ctx, top, top->ne[0], n, 1));
            comp_mask = ggml_add(ctx, ggml_reshape_2d(ctx, all, nc, n), comp_mask);
        }
        ggml_tensor * Kc   = ck->type == Kr->type ? ggml_cont(ctx, ck) : ggml_cast(ctx, ck, Kr->type);
        ggml_tensor * K    = ggml_concat(ctx, Kr, Kc, 1);                                            // [D, R + nc]
        ggml_tensor * mask = ggml_concat(ctx, in.raw_mask, ggml_cast(ctx, comp_mask, GGML_TYPE_F16), 0);
        a = ggml_flash_attn_ext(ctx, qp, K, K, mask, scale, 0.0f, 0.0f);
        if (top && top->ne[0] < nc) ggml_flash_attn_ext_set_n_kv_max(a, (int32_t) (std::min<int64_t>(R_, h.n_swa) + top->ne[0]));
    }
    ggml_flash_attn_ext_add_sinks(a, L.sinks);
    ggml_prec_set_acc(a, GGML_PREC_F32);                                                             // [D, H, n]
    dbg(ctx, gf, il, "fa", a);
    ggml_tensor * o = rope(ctx, ggml_reshape_3d(ctx, a, D, H, n), in.pos, nope, il, true);           // undo the RoPE
    dbg(ctx, gf, il, "attn_derope", o);
    // grouped low-rank output: [H/G*D, G, n] -> per group wo_a -> [o_lora*G, n] -> wo_b
    const int64_t G = h.o_groups;
    o               = ggml_permute(ctx, ggml_reshape_3d(ctx, o, H / G * D, G, n), 0, 2, 1, 3);       // [H/G*D, n, G]
    ggml_tensor * oa = ggml_mul_mat(ctx, L.wo_a, o);                                                 // [o_lora, n, G]
    oa               = ggml_cont_2d(ctx, ggml_permute(ctx, oa, 0, 2, 1, 3), h.o_lora * G, n);
    dbg(ctx, gf, il, "attn_wo_a", oa);
    ggml_tensor * ao = ggml_mul_mat(ctx, L.wo_b, oa);
    dbg(ctx, gf, il, "attn_out", ao);
    return ao;
}

ggml_tensor * Deepseek41::mixer(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * res, ggml_tensor ** pre, int n,
                                Inputs & in, Build & b) {
    const auto &            h = hp_;
    const Deepseek41Layer & L = layers_[(size_t) il];
    const int64_t           e = h.engram_layer[(size_t) il];
    if (e >= 0 && (size_t) e < in.engram.size() && in.engram[(size_t) e]) {
        dbg(ctx, gf, il, "engram_rows", in.engram[(size_t) e]);
        res = engram(ctx, il, res, in.engram[(size_t) e]);
        dbg(ctx, gf, il, "engram_out", res);
    }
    ggml_tensor * ap = nullptr, * post = nullptr, * comb = nullptr;
    mixes(ctx, res, L.hca_fn, L.hca_scale, L.hca_base, &ap, &post, &comb);
    // single pass: the attention input collapses the streams with the previous sublayer's pre-mix (stream 0 at first)
    ggml_tensor * cur = *pre ? ggml_dsv4_hc_pre(ctx, res, *pre) : ggml_cont(ctx, ggml_view_2d(ctx, res, h.n_embd, n, res->nb[2], 0));
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, h.rms_eps), L.attn_norm);
    cur = attention(ctx, gf, il, cur, n, in, b);
    *pre = ap;
    return ggml_dsv4_hc_post(ctx, cur, res, post, comb);
}

ggml_tensor * Deepseek41::ffn_block(ggml_context * ctx, int64_t il, ggml_tensor * res, ggml_tensor ** pre, int n) {
    const auto &            h = hp_;
    const Deepseek41Layer & L = layers_[(size_t) il];
    ggml_tensor *           fp = nullptr, * post = nullptr, * comb = nullptr;
    mixes(ctx, res, L.hcf_fn, L.hcf_scale, L.hcf_base, &fp, &post, &comb);
    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_dsv4_hc_pre(ctx, res, *pre), h.rms_eps), L.ffn_norm);
    cur  = ggml_add(ctx, moe(ctx, il, cur, n), ffn(ctx, L.gate_shexp, L.up_shexp, L.down_shexp, cur, h.clamp_shexp[(size_t) il]));
    *pre = fp;
    return ggml_dsv4_hc_post(ctx, cur, res, post, comb);
}

ggml_tensor * Deepseek41::head(ggml_context * ctx, ggml_tensor * res, ggml_tensor * pre) {
    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_dsv4_hc_pre(ctx, res, pre), hp_.rms_eps), output_norm_);
    return ggml_mul_mat(ctx, output_, cur);
}

ggml_tensor * Deepseek41::ffn(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * cur,
                              float limit) {
    ggml_tensor * g = ggml_mul_mat(ctx, gate, cur), * u = ggml_mul_mat(ctx, up, cur);
    return ggml_mul_mat(ctx, down, limit > 1e-6f ? ggml_swiglu_clamp(ctx, g, u, limit) : ggml_swiglu_split(ctx, g, u));
}

ggml_tensor * Deepseek41::moe(ggml_context * ctx, int64_t il, ggml_tensor * cur, int n) {
    const auto &            h  = hp_;
    const Deepseek41Layer & L  = layers_[(size_t) il];
    const int64_t           ne = h.n_embd, k = h.n_expert_used;
    const size_t            l  = (size_t) il;
    const float             lim = h.clamp_exp[l];

    ggml_tensor * lg = ggml_mul_mat(ctx, L.gate_inp, cur);
    ggml_prec_set_acc(lg, GGML_PREC_F32);  // expert selection is near-tie sensitive
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, lg));                                   // [n_expert, n]
    ggml_tensor * sel   = ggml_argsort_top_k(ctx, ggml_add(ctx, probs, L.exp_bias), (int) k);
    ggml_tensor * w     = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, h.n_expert, n), sel), k, n);
    if (h.expert_weights_norm && k > 1) w = ggml_div(ctx, w, ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY));
    if (h.expert_weights_scale != 1.0f) w = ggml_scale(ctx, w, h.expert_weights_scale);
    w = ggml_reshape_3d(ctx, w, 1, k, n);

    ggml_tensor * x3    = ggml_reshape_3d(ctx, cur, ne, 1, n);
    ggml_tensor * sel_c = nullptr;
    if (ec_.C > 0 && ec_.gmap[l]) {
        sel_c       = ggml_cont(ctx, sel);
        sel_out_[l] = sel_c;
        ggml_set_output(sel_c);
    }
    if (sel_c && n < 32) {  // cached experts on the GPU, the rest from the mapping on the CPU (see Qwen4Exp::moe)
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
        static const bool la = !std::getenv("E8_LOOKAHEAD") || std::atoi(std::getenv("E8_LOOKAHEAD")) != 0;
        ggml_tensor * next_ids = nullptr;
        if (la && l + 1 < layers_.size() && hints_[l].gate) {
            const Deepseek41Layer & N = layers_[l + 1];
            next_ids = ggml_cont(ctx, ggml_argsort_top_k(ctx, ggml_add(ctx, ggml_sqrt(ctx, ggml_softplus(ctx, ggml_mul_mat(ctx, N.gate_inp, cur))), N.exp_bias), (int) k));
        }
        ggml_tensor * cpu_out = kernels::moe_cpu_sparse(ctx, cur, cids, ggml_reshape_2d(ctx, wc, k, n), L.gate_exps, L.up_exps,
                                                        L.down_exps, next_ids, &hints_[l]);
        return ggml_add(ctx, experts(ctx, ec_.gate[l], ec_.up[l], ec_.down[l], x3, gids, wg, n, lim), cpu_out);
    }
    return experts(ctx, L.gate_exps, L.up_exps, L.down_exps, x3, sel, w, n, lim);
}

ggml_tensor * Deepseek41::experts(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * x3,
                                  ggml_tensor * ids, ggml_tensor * w, int n, float limit) {
    const int64_t ne = hp_.n_embd, k = ids->ne[0];
    ggml_tensor * gt = ggml_mul_mat_id(ctx, gate, x3, ids);
    ggml_tensor * u  = ggml_mul_mat_id(ctx, up, x3, ids);
    ggml_tensor * a  = limit > 1e-6f ? ggml_swiglu_clamp(ctx, gt, u, limit) : ggml_swiglu_split(ctx, gt, u);
    ggml_tensor * d  = ggml_mul(ctx, ggml_mul_mat_id(ctx, down, a, ids), w);
    ggml_tensor * out = ggml_view_2d(ctx, d, ne, n, d->nb[2], 0);
    for (int64_t i = 1; i < k; i++) out = ggml_add(ctx, out, ggml_view_2d(ctx, d, ne, n, d->nb[2], (size_t) i * d->nb[1]));
    return out;
}

bool Deepseek41::init_expert_cache(std::string & err) {
    const auto & h  = hp_;
    const size_t nl = (size_t) h.n_layer;
    sel_out_.assign(nl, nullptr);
    hints_.assign(nl, {});
    for (size_t l = 0; l < nl; l++) hints_[l].limit = h.clamp_exp[l];
    if (opt_.expert_cache_gb == 0 || opt_.experts_gpu) return true;
    uint64_t per_expert = 0;
    int      n_gpu      = 0;
    for (const auto & L : layers_) {
        if (!L.on_gpu) continue;
        n_gpu++;
        if (!per_expert)
            for (ggml_tensor * t : { L.gate_exps, L.up_exps, L.down_exps }) per_expert += t->nb[2];
    }
    if (!n_gpu || !per_expert) return true;
    double budget = opt_.expert_cache_gb * 1e9;
    if (budget < 0) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &free, &total);
        // compute buffers: a prefill MoE piece copies a whole layer of experts (~2.7 GB at IQ1_S) plus its temporaries
        budget = std::max(0.0, (double) free - 8.0e9);
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
        if (!L.on_gpu) continue;
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
        const auto & N    = layers_[l + 1];
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

void Deepseek41::update_expert_cache(int n) {
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
            const Deepseek41Layer & L = layers_[l];
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
            int & dm = ec_.dummy[l];
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

std::string Deepseek41::cache_report() const {
    char prof[512] = "";
    if (n_timed_) snprintf(prof, sizeof prof, "\nper decode token: compute %.1f ms", 1e3 * t_compute_ / (double) n_timed_);
    {
        double gu = 0, dn = 0;
        long long calls = 0;
        kernels::moe_cpu_times(gu, dn, calls);
        if (calls && n_timed_) {
            const double per = (double) calls / (double) hp_.n_layer;
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

// ---- inputs ----------------------------------------------------------------------------------------------------

bool Deepseek41::embed_rows(const int32_t * tokens, int n, std::vector<float> & out) const {
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

// Engram module e's table rows for n tokens at n_past_ (engram.py NgramHashState): the n-gram of each token and the
// ngram-1 before it (token ids remapped, padded once the sequence start is reached), hashed per head
void Deepseek41::engram_rows(int64_t e, const int32_t * tokens, int n, std::vector<float> & out) const {
    const auto &   h      = hp_;
    const int64_t  ng     = h.ple_ngram, nh = h.ple_n_heads, hd = h.ple_head_dim, per = h.ple_per_gram;
    const uint64_t * mult = h.ple_mult.data() + (size_t) (e * ng);
    const uint64_t * prm  = h.ple_vocab.data() + (size_t) (e * nh);
    const uint64_t * off  = h.ple_offset.data() + (size_t) (e * nh);
    const int64_t    pad  = h.token_map[(size_t) h.engram_pad];
    const size_t     rs   = ggml_row_size(engram_table_->type, hd);
    const auto *     tt   = ggml_get_type_traits(engram_table_->type);
    const char *     tab  = (const char *) file_.data(engram_table_);
    const int64_t    rows = engram_table_->ne[1];
    auto cid = [&](int64_t t) -> int64_t { return t < 0 || t >= (int64_t) h.token_map.size() ? -1 : h.token_map[(size_t) t]; };
    auto tok_at = [&](int64_t q) -> int64_t { return q < n_past_ ? history_[(size_t) q] : tokens[q - n_past_]; };
    out.assign((size_t) (n * nh * hd), 0.0f);
    std::vector<uint64_t> tk((size_t) ng);
    for (int i = 0; i < n; i++) {
        const int64_t p = n_past_ + i;
        int64_t       c = cid(tokens[i]);
        bool          blocked = c < 0;
        tk[0]         = (uint64_t) (blocked ? pad : c);
        for (int64_t s = 1; s < ng; s++) {
            c       = p - s < 0 ? -1 : cid(tok_at(p - s));
            blocked = blocked || c < 0;
            tk[(size_t) s] = (uint64_t) (blocked ? pad : c);
        }
        uint64_t rolling = tk[0] * mult[0];
        for (int64_t o = 1; o < ng; o++) {
            rolling ^= tk[(size_t) o] * mult[o];
            for (int64_t q = 0; q < per; q++) {
                const int64_t col = (o - 1) * per + q;
                const int64_t row = (int64_t) (rolling % prm[col] + off[col]);
                if (row < 0 || row >= rows) continue;
                float * dst = out.data() + ((size_t) i * nh + col) * hd;
                if (engram_table_->type == GGML_TYPE_F32) std::memcpy(dst, tab + rs * (size_t) row, sizeof(float) * hd);
                else tt->to_float(tab + rs * (size_t) row, dst, hd);
            }
        }
    }
}

void Deepseek41::new_inputs(ggml_context * ctx, int n, Inputs & in) {
    const auto & h = hp_;
    in.pos         = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(in.pos);
    in.ring_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
    ggml_set_input(in.ring_idx);
    in.raw_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, R_, n);
    ggml_set_input(in.raw_mask);
    for (int64_t rt = 1; rt <= 2; rt++) {
        if (std::find(h.ratio.begin(), h.ratio.end(), rt) == h.ratio.end()) continue;
        Inputs::Comp & cp = in.c[rt];
        cp.n_comp         = (n_past_ + n) / rt;
        cp.n_blk          = (n_past_ + n) / rt - n_past_ / rt;
        if (cp.n_comp > 0) {
            cp.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cp.n_comp, n);
            ggml_set_input(cp.mask);
            if (h.cand_src >= 0 && h.cand_block_size > 0 && h.ratio[(size_t) h.cand_src] == rt) {
                cp.cand_pin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (cp.n_comp + h.cand_block_size - 1) / h.cand_block_size, n);
                ggml_set_input(cp.cand_pin);
            }
        }
        if (cp.n_blk > 0) {
            cp.blk_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rt * cp.n_blk);
            ggml_set_input(cp.blk_rows);
            cp.blk_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, cp.n_blk);
            ggml_set_input(cp.blk_dst);
            cp.blk_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cp.n_blk);
            ggml_set_input(cp.blk_pos);
        }
    }
    if (h.ple_n_heads > 0) {
        int64_t ne = 0;
        for (int64_t e : h.engram_layer) ne = std::max(ne, e + 1);
        in.engram.assign((size_t) ne, nullptr);
        for (auto & t : in.engram) {
            t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.ple_n_heads * h.ple_head_dim, n);
            ggml_set_input(t);
        }
    }
}

// fills the inputs and records the ring slots this batch writes
void Deepseek41::set_inputs(const Inputs & in, const int32_t * tokens, int n) {
    const auto & h = hp_;
    auto set = [](ggml_tensor * t, const void * d, size_t nb) {
        if (t && t->buffer) ggml_backend_tensor_set(t, d, 0, nb);
    };
    std::vector<int32_t> pos((size_t) n);
    std::vector<int64_t> slot((size_t) n);
    for (int i = 0; i < n; i++) {
        pos[(size_t) i]             = n_past_ + i;
        slot[(size_t) i]            = (n_past_ + i) % R_;
        ring_pos_[(size_t) slot[(size_t) i]] = n_past_ + i;
    }
    set(in.pos, pos.data(), pos.size() * 4);
    set(in.ring_idx, slot.data(), slot.size() * 8);
    {  // window: slot visible when it holds a position in (t - n_swa, t]
        std::vector<ggml_fp16_t> m((size_t) R_ * n);
        const ggml_fp16_t        z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; i++) {
            const int t = n_past_ + i;
            for (int s = 0; s < R_; s++) {
                const int p = ring_pos_[(size_t) s];
                m[(size_t) i * R_ + s] = p >= 0 && p <= t && p > t - h.n_swa ? z : ni;
            }
        }
        set(in.raw_mask, m.data(), m.size() * sizeof(ggml_fp16_t));
    }
    for (int64_t rt = 1; rt <= 2; rt++) {
        const Inputs::Comp & cp = in.c[rt];
        if (cp.mask) {  // compressed row j visible to t when j < (t + 1) / ratio
            const int64_t      nc = cp.n_comp;
            std::vector<float> m((size_t) nc * n);
            for (int i = 0; i < n; i++) {
                const int64_t nv = (n_past_ + i + 1) / rt;
                for (int64_t j = 0; j < nc; j++) m[(size_t) i * nc + j] = j < nv ? 0.0f : -INFINITY;
            }
            set(cp.mask, m.data(), m.size() * 4);
        }
        if (cp.cand_pin) {  // +1e30 on the block holding each token's newest visible row
            const int64_t      nb = cp.cand_pin->ne[0], bs = h.cand_block_size;
            std::vector<float> pin((size_t) nb * n, 0.0f);
            for (int i = 0; i < n; i++) {
                const int64_t nv = (n_past_ + i + 1) / rt;
                if (nv > 0) pin[(size_t) i * nb + (nv - 1) / bs] = 1e30f;
            }
            set(cp.cand_pin, pin.data(), pin.size() * 4);
        }
        if (cp.blk_rows) {  // groups completed by this batch: their member slots, row and first position
            std::vector<int32_t> rows, bp;
            std::vector<int64_t> dst;
            for (int64_t j = n_past_ / rt; j < (n_past_ + n) / rt; j++) {
                for (int64_t k = 0; k < rt; k++) rows.push_back((int32_t) ((rt * j + k) % R_));
                dst.push_back(j);
                bp.push_back((int32_t) (rt * j));
            }
            set(cp.blk_rows, rows.data(), rows.size() * 4);
            set(cp.blk_dst, dst.data(), dst.size() * 8);
            set(cp.blk_pos, bp.data(), bp.size() * 4);
        }
    }
    if (tokens) {
        std::vector<float> er;
        for (size_t e = 0; e < in.engram.size(); e++) {
            engram_rows((int64_t) e, tokens, n, er);
            set(in.engram[e], er.data(), er.size() * 4);
        }
    }
}

bool Deepseek41::compute(ggml_context * ctx, ggml_cgraph * gf, std::string & err) {
    (void) ctx;
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        err = "graph compute failed";
        return false;
    }
    return true;
}

// ---- eval ------------------------------------------------------------------------------------------------------

bool Deepseek41::eval(const int32_t * tokens, int n, float * logits, std::string & err) {
    const auto & h = hp_;
    if (n <= 0 || n > R_ - h.n_swa) {
        err = "eval: batch size must be 1.." + std::to_string(R_ - h.n_swa);
        return false;
    }
    if (n_past_ + n > n_ctx_) {
        err = "eval: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    using clk = std::chrono::steady_clock;
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    Inputs           in;
    Build            b;
    b.topk.assign((size_t) h.n_layer, nullptr);
    b.ck.assign((size_t) h.n_layer, nullptr);
    b.cik.assign((size_t) h.n_layer, nullptr);
    in.embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n);
    ggml_set_input(in.embd);
    new_inputs(ctx, n, in);
    ggml_tensor * res = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, in.embd, h.n_embd, 1, n), h.n_embd, h.hc, n, 1);
    ggml_tensor * pre = nullptr;
    dbg(ctx, gf, 0, "embd", in.embd);  // with E8_DS_DUMP=0
    // E8_DS_DUMP: the sum of each layer's residual (attention, then FFN), against llama.cpp's eval-callback l_out sums
    static const bool dump = std::getenv("E8_DS_DUMP") != nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> dumps;
    auto mark = [&](const std::string & name, ggml_tensor * t) {
        if (!dump) return;
        t = ggml_cont(ctx, t);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        dumps.emplace_back(name, t);
    };
    for (int64_t il = 0; il < h.n_layer; il++) {
        res = mixer(ctx, gf, il, res, &pre, n, in, b);
        mark("attn_out-" + std::to_string(il), res);
        res = ffn_block(ctx, il, res, &pre, n);
        mark("l_out-" + std::to_string(il), res);
    }
    if (last_only_ && n > 1) {
        res = ggml_view_3d(ctx, res, res->ne[0], res->ne[1], 1, res->nb[1], res->nb[2], res->nb[2] * (size_t) (n - 1));
        pre = ggml_view_2d(ctx, pre, pre->ne[0], 1, pre->nb[1], pre->nb[1] * (size_t) (n - 1));
    }
    ggml_tensor * out = head(ctx, res, pre);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval: cannot allocate the compute graph";
        return false;
    }
    {
        std::vector<float> e;
        if (!embed_rows(tokens, n, e)) {
            ggml_free(ctx);
            err = "eval: token id out of range";
            return false;
        }
        ggml_backend_tensor_set(in.embd, e.data(), 0, e.size() * 4);
    }
    set_inputs(in, tokens, n);
    const auto t0 = clk::now();
    if (!compute(ctx, gf, err)) {
        ggml_free(ctx);
        return false;
    }
    if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
    for (auto & d : dbg_) dumps.push_back(d);
    dbg_.clear();
    for (auto & [name, t] : dumps) {
        std::vector<float> v((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        double sum = 0, sum0 = 0;
        for (float x : v) sum += x;
        for (int64_t i = 0; i < t->ne[0]; i++) sum0 += v[(size_t) i];
        const size_t r0 = (size_t) t->ne[0];
        fprintf(stderr, "DUMP %s sum %.6f first-token-stream0 sum %.6f  row0 %.4f %.4f %.4f .. %.4f %.4f %.4f  row1 %.4f %.4f %.4f\n",
                name.c_str(), sum, sum0, v[0], v[1], v[2], v[r0 - 3], v[r0 - 2], v[r0 - 1],
                v.size() > r0 + 2 ? v[r0] : 0.0f, v.size() > r0 + 2 ? v[r0 + 1] : 0.0f, v.size() > r0 + 2 ? v[r0 + 2] : 0.0f);
    }
    if (n == 1) {
        t_compute_ += std::chrono::duration<double>(clk::now() - t0).count();
        n_timed_++;
    }
    update_expert_cache(n);
    ggml_free(ctx);
    history_.insert(history_.end(), tokens, tokens + n);
    n_past_ += n;
    return true;
}

bool Deepseek41::verify(const int32_t * tokens, int n, float * logits, std::string & err) {
    sp_n_past_    = n_past_;
    const bool ok = eval(tokens, n, logits, err);
    rec_n_        = ok ? n : 0;
    return ok;
}

bool Deepseek41::commit(int keep, std::string & err) {
    if (rec_n_ <= 0 || keep < 1 || keep > rec_n_) {
        err = "commit: no verify to keep from";
        return false;
    }
    // the rejected tokens' ring slots hold positions past n_past (masked until overwritten); their compressed rows
    // are rebuilt when the pairs complete again
    n_past_ = sp_n_past_ + keep;
    history_.resize((size_t) n_past_);
    rec_n_ = 0;
    return true;
}

// ---- layer-major prompt processing (see Qwen4Exp::prefill) ------------------------------------------------------

// part 0 = mixer of layer il, 1 = its FFN, 2 = head (last row's logits). r: residual rows [n_embd*hc each], pre: the
// carried pre-mix [hc each] (part 0 of layer 0 has none). The top-k a source layer computes reaches later layers
// through topk_host_ (filled here from the source's own pass).
bool Deepseek41::run_part(int part, int64_t il, const int32_t * tokens, int n, float * r, float * pre, float * logits,
                          std::string & err) {
    const auto &  h   = hp_;
    const int64_t hcd = h.hc * h.n_embd;
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    Inputs           in;
    Build            b;
    b.topk.assign((size_t) h.n_layer, nullptr);
    b.ck.assign((size_t) h.n_layer, nullptr);
    b.cik.assign((size_t) h.n_layer, nullptr);
    const int nr = part == 2 ? 1 : n;
    in.res       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h.n_embd, h.hc, nr);
    ggml_set_input(in.res);
    ggml_tensor * pin = nullptr;
    if (part != 0 || il > 0) {
        pin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.hc, nr);
        ggml_set_input(pin);
    }
    ggml_tensor * p = pin, * out = nullptr;
    ggml_tensor * topk_in = nullptr, * cand_in = nullptr;
    if (part == 0) {
        new_inputs(ctx, n, in);
        const int64_t ts = h.topk_src[(size_t) il];
        if (ts >= 0 && ts != il && in.c[h.ratio[(size_t) il]].n_comp > h.idx_top_k) {  // the source layer's top-k, from its own pass
            topk_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, h.idx_top_k, n);
            ggml_set_input(topk_in);
            b.topk[(size_t) ts] = topk_in;
        }
        const Inputs::Comp & cq = in.c[h.ratio[(size_t) il]];
        if (h.cand_src >= 0 && h.cand_src < il && ts == il && cq.cand_pin &&
            cq.cand_pin->ne[0] > h.cand_blocks && !cand_host_.empty()) {
            cand_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, h.cand_blocks, n);
            ggml_set_input(cand_in);
            b.cand_sel = cand_in;
        }
        out = mixer(ctx, gf, il, in.res, &p, n, in, b);
    } else if (part == 1) {
        out = ffn_block(ctx, il, in.res, &p, n);
    } else {
        out = head(ctx, in.res, pin);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_tensor * pout = nullptr, * topk_out = nullptr, * cand_out = nullptr;
    if (part < 2) {
        pout = ggml_cont(ctx, p);
        ggml_set_output(pout);
        ggml_build_forward_expand(gf, pout);
        if (part == 0 && b.topk[(size_t) il]) {
            topk_out = b.topk[(size_t) il];
            ggml_set_output(topk_out);
            ggml_build_forward_expand(gf, topk_out);
        }
        if (part == 0 && h.cand_src == il && b.cand_sel) {
            cand_out = b.cand_sel;
            ggml_set_output(cand_out);
            ggml_build_forward_expand(gf, cand_out);
        }
    }
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "prefill: cannot allocate the compute graph";
        return false;
    }
    ggml_backend_tensor_set(in.res, part == 2 ? r + (size_t) (n - 1) * hcd : r, 0, ggml_nbytes(in.res));
    if (pin) ggml_backend_tensor_set(pin, part == 2 ? pre + (size_t) (n - 1) * h.hc : pre, 0, ggml_nbytes(pin));
    if (part == 0) {
        set_inputs(in, tokens, n);
        const size_t off = (size_t) (n_past_ - prefill_base_);
        if (topk_in) ggml_backend_tensor_set(topk_in, topk_host_[(size_t) h.topk_src[(size_t) il]].data() + off * h.idx_top_k, 0, ggml_nbytes(topk_in));
        if (cand_in) ggml_backend_tensor_set(cand_in, cand_host_.data() + off * h.cand_blocks, 0, ggml_nbytes(cand_in));
    }
    if (!compute(ctx, gf, err)) {
        ggml_free(ctx);
        return false;
    }
    if (part == 2) {
        ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
    } else {
        ggml_backend_tensor_get(out, r, 0, ggml_nbytes(out));
        ggml_backend_tensor_get(pout, pre, 0, ggml_nbytes(pout));
        const size_t off = (size_t) (n_past_ - prefill_base_);
        if (topk_out) {
            auto & v = topk_host_[(size_t) il];
            v.resize(std::max(v.size(), (off + (size_t) n) * (size_t) h.idx_top_k));
            ggml_backend_tensor_get(topk_out, v.data() + off * h.idx_top_k, 0, ggml_nbytes(topk_out));
        }
        if (cand_out) {
            cand_host_.resize(std::max(cand_host_.size(), (off + (size_t) n) * (size_t) h.cand_blocks));
            ggml_backend_tensor_get(cand_out, cand_host_.data() + off * h.cand_blocks, 0, ggml_nbytes(cand_out));
        }
    }
    ggml_free(ctx);
    return true;
}

bool Deepseek41::prefill(const int32_t * tokens, int N, float * logits, std::string & err) {
    const auto & h = hp_;
    static const int min_layered = std::getenv("E8_LAYERED_MIN") ? std::atoi(std::getenv("E8_LAYERED_MIN")) : 256;
    const int        piece       = R_ - (int) h.n_swa;  // most tokens per graph (the window ring)
    if (N < min_layered) {
        for (int i = 0; i < N; i += piece) {
            const int m = std::min(piece, N - i);
            if (!eval_last(tokens + i, m, i + m == N ? logits : nullptr, err)) return false;
        }
        return true;
    }
    if (n_past_ + N > n_ctx_) {
        err = "prefill: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    const int64_t hcd = h.hc * h.n_embd;
    const int     S   = std::getenv("E8_PREFILL_CHUNK") ? std::atoi(std::getenv("E8_PREFILL_CHUNK")) : 16384;
    const int     um  = std::getenv("E8_MOE_UBATCH") ? std::atoi(std::getenv("E8_MOE_UBATCH")) : 4096;
    std::vector<float> rr, pp, e;
    auto prefetch = [&](int64_t il) {
        std::vector<std::pair<void *, size_t>> ranges;
        if (il < h.n_layer) {
            const Deepseek41Layer & L = layers_[(size_t) il];
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
        prefill_base_  = base;
        topk_host_.assign((size_t) h.n_layer, {});
        cand_host_.clear();
        rr.resize((size_t) cn * hcd);
        pp.assign((size_t) cn * h.hc, 0.0f);
        if (!embed_rows(tokens + c0, cn, e)) {
            err = "prefill: token id out of range";
            return false;
        }
        for (int i = 0; i < cn; i++)
            for (int64_t j = 0; j < h.hc; j++)
                std::memcpy(rr.data() + (size_t) i * hcd + j * h.n_embd, e.data() + (size_t) i * h.n_embd, sizeof(float) * h.n_embd);
        // the window ring holds piece + n_swa tokens: every layer's mixer goes in pieces of at most that, and the
        // ring positions are replayed per layer
        const std::vector<int32_t> ring0 = ring_pos_;
        std::thread pf = prefetch(0);
        struct Join {  // an early error return must not leave the prefetch thread joinable
            std::thread & t;
            ~Join() {
                if (t.joinable()) t.join();
            }
        } join{ pf };
        for (int64_t il = 0; il < h.n_layer; il++) {
            ring_pos_ = ring0;
            int ua    = std::min(piece, 512);
            if (h.ratio[(size_t) il] != 0) {  // mask and indexer temporaries scale with the compressed rows
                const int64_t nc = std::max<int64_t>(1, (base + cn) / h.ratio[(size_t) il]);
                ua = (int) std::clamp<int64_t>(24'000'000 / nc / 32 * 32, 32, ua);
            }
            for (int m0 = 0; m0 < cn; m0 += ua) {
                const int m = std::min(ua, cn - m0);
                n_past_     = base + m0;
                if (!run_part(0, il, tokens + c0 + m0, m, rr.data() + (size_t) m0 * hcd, pp.data() + (size_t) m0 * h.hc, nullptr, err)) return false;
                history_.resize((size_t) base + m0);
                history_.insert(history_.end(), tokens + c0 + m0, tokens + c0 + m0 + m);
            }
            history_.resize((size_t) base);
            pf.join();
            pf = prefetch(il + 1);
            for (int m0 = 0; m0 < cn; m0 += um) {
                const int m = std::min(um, cn - m0);
                n_past_     = base + m0;
                if (!run_part(1, il, nullptr, m, rr.data() + (size_t) m0 * hcd, pp.data() + (size_t) m0 * h.hc, nullptr, err)) return false;
            }
        }
        pf.join();
        n_past_ = base + cn;
        history_.insert(history_.end(), tokens + c0, tokens + c0 + cn);
        topk_host_.clear();
        cand_host_.clear();
        if (c0 + cn == N && logits && !run_part(2, 0, nullptr, cn, rr.data(), pp.data(), logits, err)) return false;
    }
    return true;
}

} // namespace e8::model
