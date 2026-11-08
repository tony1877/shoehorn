#pragma once

// GLM-5.3-Flash ("glm5-next" in GGUF), single sequence. Mirrors llama.cpp's src/models/glm5-next.cpp at the pinned
// commit:
//   - mHC residual: hc parallel streams [n_embd, hc, T]; each block reads a sigmoid-weighted sum of them and writes
//     back through a Sinkhorn-normalized stream mix (DeepSeek V4 style); the head takes their mean
//   - KDA (Kimi delta attention, per-channel decay) in 3 of 4 layers, separate causal convs on Q/K/V, gated output
//   - nope MLA in the 4th: the KV cache holds the 512-wide latent, queries absorb W_kb, W_vb is applied after; sparse
//     attention over the top pools of 4 tokens picked by a lightning indexer on gated-softmax pooled keys (dense while
//     every complete pool fits the budget), plus the token's own incomplete pool
//   - DeepSeek MoE: sigmoid router with a selection bias, top-k renormalized and scaled, an ungated shared expert,
//     clamped SwiGLU; the first layers are dense
// Weights come from a memory-mapped (possibly split) GGUF; tensors placed on the CPU are used in place from the
// mapping, so the experts can be far larger than RAM. Hot experts are cached in VRAM (as for Flash-Next).

#include "model/causal_lm.h"
#include "model/gguf_set.h"
#include "kernels/moe_cpu.h"
#include "model/qwen35.h"  // LoadOptions

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace e8::model {

struct Glm5NextHparams {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, n_vocab = 0, n_ctx_train = 0, n_ff = 0;
    float   rms_eps = 1e-5f, norm_eps = 1e-6f;
    // KDA
    int64_t kda_dim = 0, d_conv = 0;
    float   kda_lower_bound = -INFINITY;
    // MLA
    int64_t q_lora = 0, kv_lora = 0, k_mla = 0, v_mla = 0;
    // indexer
    int64_t idx_n_head = 0, idx_dim = 0, idx_top_k = 0, kpool = 0;
    bool    select_tail = true;
    // mHC
    int64_t hc = 0, hc_iters = 0;
    float   hc_eps = 1e-6f;
    // MoE
    int64_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_dense = 0;
    float   expert_weights_scale = 1.0f;
    bool    expert_weights_norm  = true;
    std::vector<float> clamp_exp, clamp_shexp;  // per layer, 0 = none
    std::vector<bool>  recurrent, idx_full;
    bool               add_bos = false;
    int64_t            bos     = -1;

    int64_t kda_inner() const { return kda_dim * n_head; }
    bool load(const GgufSet & f, std::string & err);
};

struct Glm5NextLayer {
    ggml_tensor * attn_norm = nullptr, * ffn_norm = nullptr;
    // mHC
    ggml_tensor * hca_fn = nullptr, * hca_base = nullptr, * hca_scale = nullptr;
    ggml_tensor * hcf_fn = nullptr, * hcf_base = nullptr, * hcf_scale = nullptr;
    // KDA
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * conv_q = nullptr, * conv_k = nullptr, * conv_v = nullptr;
    ggml_tensor * f_a = nullptr, * f_b = nullptr, * beta = nullptr, * a = nullptr, * dt_b = nullptr;
    ggml_tensor * g_a = nullptr, * g_b = nullptr, * o_norm = nullptr;
    // MLA
    ggml_tensor * q_a = nullptr, * q_a_norm = nullptr, * q_b = nullptr, * kv_a = nullptr, * kv_a_norm = nullptr;
    ggml_tensor * k_b = nullptr, * v_b = nullptr;
    // indexer
    ggml_tensor * idx_k_norm = nullptr, * idx_k_norm_b = nullptr, * idx_proj = nullptr, * idx_k = nullptr;
    ggml_tensor * idx_q_b = nullptr, * idx_gate = nullptr, * idx_ape = nullptr;
    // FFN: dense, or MoE + shared expert
    ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
    ggml_tensor * gate_inp = nullptr, * exp_bias = nullptr, * gate_exps = nullptr, * up_exps = nullptr, * down_exps = nullptr;
    ggml_tensor * gate_shexp = nullptr, * up_shexp = nullptr, * down_shexp = nullptr;
    bool          on_gpu = false;
};

class Glm5Next : public CausalLM {
public:
    Glm5Next();
    ~Glm5Next() override;

    bool load(const std::string & path, const LoadOptions & opt, std::string & err);

    bool eval(const int32_t * tokens, int n, float * logits, std::string & err) override;
    void reset() override;
    bool prefill(const int32_t * tokens, int n, float * logits, std::string & err) override;
    bool eval_last(const int32_t * tokens, int n, float * logits, std::string & err) override;
    void checkpoint_save() override;
    void checkpoint_restore() override;
    bool save_seq(std::vector<uint8_t> & out) override;
    bool load_seq(const std::vector<uint8_t> & in) override;

    int64_t  n_vocab() const override { return hp_.n_vocab; }
    int      n_past() const override { return n_past_; }
    int      n_ctx() const override { return n_ctx_; }
    int64_t  n_layer() const override { return hp_.n_layer; }
    bool     add_bos() const override { return hp_.add_bos; }
    int64_t  bos_token() const override { return hp_.bos; }
    uint64_t gpu_weight_bytes() const override { return gpu_bytes_; }
    uint64_t cpu_weight_bytes() const override { return cpu_bytes_; }
    uint64_t expert_cache_bytes() const { return ec_.bytes; }
    const Glm5NextHparams & hp() const { return hp_; }
    std::string cache_report() const;

private:
    struct Inputs {
        ggml_tensor * embd = nullptr, * mask = nullptr, * res = nullptr, * kvidx = nullptr;
        ggml_tensor * bvis = nullptr, * tail = nullptr;          // sparse batches only
        ggml_tensor * pool_rows = nullptr, * pool_blk = nullptr;  // pools this batch touches
    };
    ggml_cgraph * build_graph(ggml_context * ctx, int n, int n_kv, bool sparse, Inputs & in, ggml_tensor *& out);
    void          new_inputs(ggml_context * ctx, int n, int n_kv, bool sparse, Inputs & in);
    bool          set_inputs(const Inputs & in, int n, int n_kv);
    bool          sparse_at(int n) const;
    // layer-major prompt processing (see Qwen4Exp::run_part): 0 = mixer, 1 = FFN, 2 = head
    bool          run_part(int part, int64_t il, int n, float * r, float * logits, std::string & err);
    ggml_tensor * hc_pre(ggml_context * ctx, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
                         ggml_tensor ** post, ggml_tensor ** comb);
    ggml_tensor * mixer(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * res, int n, int n_kv, bool sparse,
                        Inputs & in, ggml_tensor ** prev_mask);
    ggml_tensor * ffn_block(ggml_context * ctx, int64_t il, ggml_tensor * res, int n);
    ggml_tensor * head(ggml_context * ctx, ggml_tensor * res);
    ggml_tensor * kda(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n);
    ggml_tensor * dsa(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n, int n_kv, bool sparse,
                      Inputs & in, ggml_tensor ** prev_mask);
    ggml_tensor * ffn(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * cur,
                      float limit);
    ggml_tensor * moe(ggml_context * ctx, int64_t il, ggml_tensor * cur, int n);
    ggml_tensor * experts(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * x3,
                          ggml_tensor * ids, ggml_tensor * w, int n, float limit);
    bool          init_expert_cache(std::string & err);
    void          update_expert_cache(int n);
    bool          embed_rows(const int32_t * tokens, int n, std::vector<float> & out) const;

    Glm5NextHparams         hp_;
    GgufSet                 file_;
    ggml_backend_t          gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_sched_t    sched_ = nullptr;
    ggml_context *          wctx_[2] = { nullptr, nullptr };  // 0 = GPU copies, 1 = CPU tensors over the mapping
    ggml_backend_buffer_t   wbuf_gpu_ = nullptr;
    std::vector<ggml_backend_buffer_t> map_bufs_;
    ggml_context *          sctx_[2] = { nullptr, nullptr };
    ggml_backend_buffer_t   sbuf_[2] = { nullptr, nullptr };

    const ggml_tensor *     tok_embd_ = nullptr;
    ggml_tensor *           output_ = nullptr, * output_norm_ = nullptr;
    std::vector<Glm5NextLayer> layers_;
    // per layer: latent KV + indexer rows (key | gate) + pooled keys (MLA); conv / recurrent state (KDA)
    std::vector<ggml_tensor *> kv_cache_, idx_raw_, idx_pool_, conv_state_, ssm_state_, conv_ck_, ssm_ck_;
    int  ck_n_past_ = 0;
    bool last_only_ = false;

    LoadOptions          opt_;
    int                  n_ctx_ = 0, n_past_ = 0;
    uint64_t             gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;

    // GPU expert cache, as in Qwen4Exp
    struct ExpertCache {
        int                             C = 0;
        uint64_t                        bytes = 0;
        std::vector<ggml_tensor *>      gate, up, down, gmap, cmap, gmask;
        ggml_tensor *                   jidx = nullptr;
        std::vector<std::vector<int>>   slot_of, expert_in;
        std::vector<std::vector<float>> heat;
        std::vector<int>                dummy;
        int64_t                         uses = 0, hits = 0, swaps = 0, uses_recent = 0, hits_recent = 0;
        ggml_context *                  ctx = nullptr;
        ggml_backend_buffer_t           buf = nullptr;
    } ec_;
    std::vector<ggml_tensor *>         sel_out_;
    std::vector<kernels::PrefetchHint> hints_;  // per layer: the clamp, and the next layer's tensors for lookahead
    double  t_build_ = 0, t_alloc_ = 0, t_inputs_ = 0, t_compute_ = 0, t_update_ = 0;
    int64_t n_timed_ = 0;
};

} // namespace e8::model
