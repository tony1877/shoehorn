// llama_logits <model.gguf> <ids.txt> <out.f32> [--last N] [--ctx N]: llama.cpp's logits for the token ids (whitespace
// separated), from position 0, as n_rows * n_vocab float32 (the last N rows, default all). The reference for checking a
// native model implementation against llama.cpp (compare with `shoehorn logits`). MoE experts stay in the file mapping
// on the CPU, the rest on the GPU.

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: llama_logits <model.gguf> <ids.txt> <out.f32> [--last N] [--ctx N]\n");
        return 1;
    }
    int last = 0, n_ctx = 8192;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--last")) last = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--ctx")) n_ctx = atoi(argv[i + 1]);
    }
    std::ifstream     f(argv[2]);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<llama_token> toks;
    if (!strcmp(argv[3], "--tokenize")) {  // llama_logits <model> <text.txt> --tokenize <ids.txt>: text to ids only
        llama_model_params vp = llama_model_default_params();
        vp.vocab_only         = true;
        llama_model *       vm = llama_model_load_from_file(argv[1], vp);
        const std::string   s  = ss.str();
        toks.resize(s.size() + 16);
        const int nt = llama_tokenize(llama_model_get_vocab(vm), s.c_str(), (int32_t) s.size(), toks.data(), (int32_t) toks.size(), true, true);
        FILE * o = fopen(argc > 4 ? argv[4] : "ids.txt", "w");
        for (int i = 0; i < nt; i++) fprintf(o, "%d ", toks[(size_t) i]);
        fclose(o);
        printf("%d tokens\n", nt);
        return 0;
    }
    for (int t; ss >> t;) toks.push_back(t);
    if (toks.empty()) {
        fprintf(stderr, "no token ids\n");
        return 1;
    }
    const int n = (int) toks.size();
    if (last <= 0 || last > n) last = n;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = 999;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    llama_model_tensor_buft_override ov[2] = { { "_exps", ggml_backend_dev_buffer_type(cpu) }, { nullptr, nullptr } };
    mp.tensor_buft_overrides = ov;
    llama_model * model      = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = (uint32_t) std::max(n_ctx, n + 16);
    cp.n_batch = cp.n_ubatch = 512;
    llama_context * ctx      = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<float> out((size_t) last * nv);
    llama_batch        b = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n; i += 512) {
        const int m = std::min(512, n - i);
        b.n_tokens  = m;
        for (int j = 0; j < m; j++) {
            b.token[j]     = toks[(size_t) (i + j)];
            b.pos[j]       = i + j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = i + j >= n - last;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "llama_decode failed\n");
            return 1;
        }
        for (int j = 0; j < m; j++)
            if (b.logits[j]) std::memcpy(out.data() + (size_t) (i + j - (n - last)) * nv, llama_get_logits_ith(ctx, j), sizeof(float) * nv);
        fprintf(stderr, "\r%d / %d", i + m, n);
    }
    fprintf(stderr, "\n");
    FILE * o = fopen(argv[3], "wb");
    fwrite(out.data(), sizeof(float), out.size(), o);
    fclose(o);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
