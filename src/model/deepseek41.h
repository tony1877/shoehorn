#pragma once

// DeepSeek-V4.1-Flash ("deepseek41" in GGUF), single sequence. Mirrors llama.cpp's src/models/deepseek41.cpp (on top
// of deepseek4.cpp) at ggml-org/llama.cpp a2b3018:
//   - mHC residual (4 streams, Sinkhorn-mixed), single pass: each sublayer reads the streams with the pre-mix of the
//     previous sublayer; the head collapses them with the last FFN's pre-mix
//   - attention: one 512-wide KV head (K = V), 64 query heads, RoPE on the last 64 dims (undone on the output),
//     attention sinks, grouped low-rank output. Every layer sees a 128-token window of raw keys; layers with compress
//     ratio 2 also see compressed rows (each pools two tokens, gated softmax), the top 512 of them picked by a
//     lightning indexer. The compressed KV is built in a few source layers and read by the layers after them, the
//     top-k likewise; past the candidate source layer, later indexers score only inside its top candidate blocks
//   - Engram: hashed n-gram embeddings (from a table far larger than RAM, read on the host) gated into the streams in
//     a few layers
//   - DeepSeek MoE: sqrt(softplus) router with a selection bias, renormalized and scaled, an ungated shared expert,
//     clamped SwiGLU
// Weights come from a memory-mapped split GGUF; CPU tensors are used in place from the mapping, hot experts are
// cached in VRAM (as for Flash-Next and GLM).

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

struct Deepseek41Hparams {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, n_vocab = 0, head_dim = 0, n_rot = 0, q_lora = 0, n_swa = 0;
    int64_t o_groups = 0, o_lora = 0, idx_n_head = 0, idx_dim = 0, idx_top_k = 0;
    int64_t hc = 0, hc_iters = 0;
    float   rms_eps = 1e-6f, hc_eps = 1e-6f;
    // RoPE: plain layers use freq_base; compressed layers use comp_freq_base with YaRN
    float   freq_base = 10000.0f, comp_freq_base = 160000.0f, yarn_factor = 1.0f, beta_fast = 32.0f, beta_slow = 1.0f;
    int64_t n_ctx_orig = 0;
    std::vector<int64_t> ratio;           // per layer, 0 = window only
    std::vector<int64_t> kv_src, topk_src;  // per layer: the layer whose compressed KV / top-k it reads (-1 = none)
    int64_t cand_src = -1, cand_blocks = 0, cand_block_size = 0;
    // MoE
    int64_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    float   expert_weights_scale = 1.0f;
    bool    expert_weights_norm  = true;
    std::vector<float> clamp_exp, clamp_shexp;
    // Engram
    std::vector<int64_t>  engram_layer;   // per layer: its module index, -1 = none
    int64_t               ple_ngram = 0, ple_per_gram = 0, ple_head_dim = 0, ple_n_heads = 0, engram_pad = 0;
    std::vector<uint64_t> ple_mult, ple_offset, ple_vocab;
    std::vector<int32_t>  token_map;
    bool    add_bos = false;
    int64_t bos     = -1;

    bool load(const GgufSet & f, std::string & err);
};

struct Deepseek41Layer {
    ggml_tensor * attn_norm = nullptr, * ffn_norm = nullptr, * sinks = nullptr;
    ggml_tensor * hca_fn = nullptr, * hca_base = nullptr, * hca_scale = nullptr;
    ggml_tensor * hcf_fn = nullptr, * hcf_base = nullptr, * hcf_scale = nullptr;
    ggml_tensor * q_a = nullptr, * q_a_norm = nullptr, * q_b = nullptr, * wkv = nullptr, * kv_norm = nullptr;
    ggml_tensor * wo_a = nullptr, * wo_b = nullptr;
    // compressor + indexer key (kv source layers), indexer query (top-k source layers)
    ggml_tensor * comp_kv = nullptr, * comp_gate = nullptr, * comp_norm = nullptr, * idx_k = nullptr, * idx_k_norm = nullptr;
    ggml_tensor * idx_q_b = nullptr, * idx_proj = nullptr;
    // Engram
    ggml_tensor * engram_kv = nullptr, * engram_gate = nullptr;
    // MoE
    ggml_tensor * gate_inp = nullptr, * exp_bias = nullptr, * gate_exps = nullptr, * up_exps = nullptr, * down_exps = nullptr;
    ggml_tensor * gate_shexp = nullptr, * up_shexp = nullptr, * down_shexp = nullptr;
    bool          on_gpu = false;
};

class Deepseek41 : public CausalLM, public Speculative {
public:
    Deepseek41();
    ~Deepseek41() override;

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
    std::string cache_report() const;

    // No draft block of its own (its DSpark drafter is a separate model): mtp_draft() proposes nothing, so the server
    // drafts by copying from the context (echo) and checks with verify()/commit(). Rejected tokens need no undo beyond
    // n_past: every state here is positional.
    bool has_mtp() const override { return true; }
    int  mtp_max_batch() const override { return 8; }
    bool mtp_draft(int32_t, int, std::vector<int32_t> & drafts, std::string &) override {
        drafts.clear();
        return true;
    }
    bool verify(const int32_t * tokens, int n, float * logits, std::string & err) override;
    bool commit(int keep, std::string & err) override;

private:
    struct Inputs {
        ggml_tensor * embd = nullptr, * res = nullptr, * pre = nullptr;  // pre: the carried pre-mix [hc, n] (prefill)
        ggml_tensor * pos = nullptr, * ring_idx = nullptr, * raw_mask = nullptr;  // window: ring slots, mask
        // per compress ratio (1, 2): the compressed rows visible to this batch and the groups it completes
        struct Comp {
            ggml_tensor * mask = nullptr;  // [n_comp, n] visibility (0 / -inf)
            ggml_tensor * blk_rows = nullptr, * blk_dst = nullptr, * blk_pos = nullptr;
            ggml_tensor * cand_pin = nullptr;
            int64_t       n_comp = 0, n_blk = 0;
        } c[3];
        std::vector<ggml_tensor *> engram;  // per Engram module: rows [ple_n_heads * ple_head_dim, n]
    };
    struct Build {  // per graph: what source layers hand to later ones
        std::vector<ggml_tensor *> topk;     // per top-k source: [top_k, n] I32 (null = every visible row)
        std::vector<ggml_tensor *> ck, cik;  // per kv source: its compressed rows as written by this graph
        ggml_tensor *              cand_sel = nullptr;
    };
    void          new_inputs(ggml_context * ctx, int n, Inputs & in);
    void          set_inputs(const Inputs & in, const int32_t * tokens, int n);
    ggml_tensor * rope(ggml_context * ctx, ggml_tensor * t, ggml_tensor * pos, int64_t n_nope, int64_t il, bool back = false);
    void          mixes(ggml_context * ctx, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
                        ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb);
    // the two sublayers: in/out the residual [n_embd, hc, n]; *pre is the carried pre-mix (null at layer 0), replaced
    ggml_tensor * mixer(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * res, ggml_tensor ** pre, int n,
                        Inputs & in, Build & b);
    ggml_tensor * ffn_block(ggml_context * ctx, int64_t il, ggml_tensor * res, ggml_tensor ** pre, int n);
    ggml_tensor * attention(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n, Inputs & in, Build & b);
    ggml_tensor * engram(ggml_context * ctx, int64_t il, ggml_tensor * x, ggml_tensor * rows);
    ggml_tensor * head(ggml_context * ctx, ggml_tensor * res, ggml_tensor * pre);
    ggml_tensor * ffn(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * cur, float limit);
    ggml_tensor * moe(ggml_context * ctx, int64_t il, ggml_tensor * cur, int n);
    ggml_tensor * experts(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * x3,
                          ggml_tensor * ids, ggml_tensor * w, int n, float limit);
    bool          init_expert_cache(std::string & err);
    void          update_expert_cache(int n);
    bool          embed_rows(const int32_t * tokens, int n, std::vector<float> & out) const;
    void          engram_rows(int64_t e, const int32_t * tokens, int n, std::vector<float> & out) const;
    bool          run_part(int part, int64_t il, const int32_t * tokens, int n, float * r, float * pre, float * logits,
                           std::string & err);
    bool          compute(ggml_context * ctx, ggml_cgraph * gf, std::string & err);

    Deepseek41Hparams hp_;
    GgufSet           file_;
    ggml_backend_t    gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    ggml_context *    wctx_[2] = { nullptr, nullptr };
    ggml_backend_buffer_t wbuf_gpu_ = nullptr;
    std::vector<ggml_backend_buffer_t> map_bufs_;
    ggml_context *    sctx_ = nullptr;
    ggml_backend_buffer_t sbuf_ = nullptr;

    const ggml_tensor * tok_embd_ = nullptr, * engram_table_ = nullptr;
    ggml_tensor *       output_ = nullptr, * output_norm_ = nullptr;
    std::vector<Deepseek41Layer> layers_;
    // state: per layer the window ring [head_dim, R] (raw keys); per kv source layer the compressed rows
    // [head_dim, n_ctx/2] + their indexer keys [idx_dim, n_ctx/2] and the compressor inputs ring [2*head_dim, R]
    std::vector<ggml_tensor *> ring_, comp_k_, comp_ik_, comp_in_;
    std::vector<ggml_tensor *> ring_ck_, comp_in_ck_;  // checkpoint copies of the rings
    std::vector<int32_t>       ring_pos_, ring_pos_ck_;  // position held by each ring slot (-1 = none)
    int                        R_ = 0;                   // ring slots: window + the largest batch
    std::vector<int32_t>       history_;
    int  ck_n_past_ = 0;
    bool last_only_ = false;
    int  rec_n_ = 0, sp_n_past_ = 0;
    // layer-major prefill: the top-k (and candidate blocks) source layers computed for the chunk, by token
    int                               prefill_base_ = 0;
    std::vector<std::vector<int32_t>> topk_host_;
    std::vector<int32_t>              cand_host_;

    LoadOptions          opt_;
    int                  n_ctx_ = 0, n_past_ = 0;
    uint64_t             gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;

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
    std::vector<kernels::PrefetchHint> hints_;
    // E8_DS_DUMP: named intermediates of layer 0 (and every layer's residual), their sums printed after the eval
    std::vector<std::pair<std::string, ggml_tensor *>> dbg_;
    void dbg(ggml_context * ctx, ggml_cgraph * gf, int64_t il, const char * name, ggml_tensor * t);
    double  t_compute_ = 0;
    int64_t n_timed_ = 0;
};

} // namespace e8::model
