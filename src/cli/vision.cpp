// `shoehorn vision <base.gguf> [--res r.gguf] --mmproj enc.gguf --image a.png [--image b.png ...] [-p TEXT] [-n N]
//                  [--layout exact|index] [--ctx N]`
//
// Image input check (docs/VISION.md): the vision encoder (llama.cpp's mtmd, on the CPU) turns each image into
// embeddings; shoehorn evaluates them with M-RoPE positions (t, y, x), then answers greedily (plain decoding, no
// speculation). --layout exact is Qwen-VL's: text after an image continues max(h, w) positions later (rope shift);
// index lets text continue at the sequence position instead (no shift), for comparison. Prints the first answer
// token's top 5 and the answer.

#include "cli/commands.h"
#include "model/qwen35.h"

#include "common.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace e8::cli {

int vision(const std::vector<std::string> & args) {
    std::string              model, res, mmproj, prompt = "Describe this image.", layout = "exact";
    std::vector<std::string> images;
    int                      n_gen = 256, n_ctx = 8192, threads = 0;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--res") res = val();
        else if (a == "--mmproj") mmproj = val();
        else if (a == "--image") images.push_back(val());
        else if (a == "-p") prompt = val();
        else if (a == "-n") n_gen = std::atoi(val().c_str());
        else if (a == "--layout") layout = val();
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (model.empty() && a[0] != '-') model = a;
        else {
            fprintf(stderr, "vision: unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (model.empty() || mmproj.empty() || images.empty() || (layout != "exact" && layout != "index")) {
        fprintf(stderr, "usage: shoehorn vision <base.gguf> [--res r.gguf] --mmproj enc.gguf --image F [--image F ...] "
                        "[-p TEXT] [-n N] [--layout exact|index] [--ctx N]\n");
        return 1;
    }

    // tokenizer (vocab only) and the encoder; the vocab model also tells mtmd the decoder uses M-RoPE
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    llama_model * vm      = llama_model_load_from_file(model.c_str(), mp);
    if (!vm) {
        fprintf(stderr, "vision: cannot read the vocabulary of %s\n", model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vm);
    mtmd_context_params cp    = mtmd_context_params_default();
    cp.use_gpu                = false;  // next to a 16 GB base the encoder does not fit in VRAM
    cp.print_timings          = false;
    if (threads > 0) cp.n_threads = threads;
    mtmd_context * mc = mtmd_init_from_file(mmproj.c_str(), vm, cp);
    if (!mc || !mtmd_support_vision(mc)) {
        fprintf(stderr, "vision: %s is not a vision encoder\n", mmproj.c_str());
        return 1;
    }
    std::vector<mtmd_bitmap *> bitmaps;
    for (const auto & f : images) {
        mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(mc, f.c_str(), false, mtmd_helper_init_opt_default());
        if (!w.bitmap) {
            fprintf(stderr, "vision: cannot read image %s\n", f.c_str());
            return 1;
        }
        bitmaps.push_back(w.bitmap);
    }
    std::string text = "<|im_start|>user\n";
    for (size_t i = 0; i < images.size(); i++) text += mtmd_default_marker();
    text += prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text     in{ text.c_str(), text.size(), false, true };
    if (mtmd_tokenize(mc, chunks, &in, (const mtmd_bitmap **) bitmaps.data(), bitmaps.size()) != 0) {
        fprintf(stderr, "vision: tokenizing the prompt failed\n");
        return 1;
    }

    model::LoadOptions o;
    o.n_gpu_layers  = 999;
    o.n_ctx         = n_ctx;
    o.n_threads     = threads;
    o.residual_path = res;
    model::Qwen35 m;
    std::string   err;
    if (!m.load(model, o, err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t      nv = m.hp().n_vocab, ne = m.hp().n_embd;
    const int          ub = m.max_batch();
    std::vector<float> last((size_t) nv);
    m.reset();
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks); c++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
        const bool               fin = c + 1 == mtmd_input_chunks_size(chunks);
        model::EvalOpts          eo;
        eo.last_only = true;
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t              nt  = 0;
            const llama_token * tks = mtmd_input_chunk_get_tokens_text(ch, &nt);
            for (size_t i = 0; i < nt; i += (size_t) ub) {
                const int k = (int) std::min(nt - i, (size_t) ub);
                if (!m.eval(tks + i, k, eo, fin && i + k == nt ? last.data() : nullptr, nullptr, err)) {
                    fprintf(stderr, "eval: %s\n", err.c_str());
                    return 1;
                }
            }
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (mtmd_encode_chunk(mc, ch) != 0) {
            fprintf(stderr, "vision: encoding the image failed\n");
            return 1;
        }
        const float *               embd = mtmd_get_output_embd(mc);
        const mtmd_image_tokens *   it   = mtmd_input_chunk_get_tokens_image(ch);
        const int                   n    = (int) mtmd_input_chunk_get_n_tokens(ch);
        const int                   npos = (int) mtmd_input_chunk_get_n_pos(ch);
        const int                   p0   = m.n_past() - m.rope_shift();  // the image's first rotary position
        std::vector<mtmd_decoder_pos> dp((size_t) n);
        for (int i = 0; i < n; i++) dp[(size_t) i] = mtmd_image_tokens_get_decoder_pos(it, p0, (size_t) i);
        printf("image %zu: %d tokens, %d positions, encoded in %.2f s\n", c, n, npos,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        for (int i = 0; i < n; i += ub) {
            const int            k = std::min(n - i, ub);
            std::vector<int32_t> pos((size_t) k * 4);
            for (int j = 0; j < k; j++) {
                const mtmd_decoder_pos & d = dp[(size_t) (i + j)];
                pos[(size_t) j]         = (int32_t) d.t;
                pos[(size_t) (k + j)]   = (int32_t) d.y;
                pos[(size_t) (2 * k + j)] = (int32_t) d.x;
                pos[(size_t) (3 * k + j)] = (int32_t) d.z;
            }
            eo.embd = embd + (size_t) i * ne;
            eo.pos  = pos.data();
            if (!m.eval(nullptr, k, eo, fin && i + k == n ? last.data() : nullptr, nullptr, err)) {
                fprintf(stderr, "eval (image): %s\n", err.c_str());
                return 1;
            }
        }
        if (layout == "exact") m.set_rope_shift(m.rope_shift() + n - npos);
    }

    std::vector<int> top((size_t) nv);
    for (int64_t v = 0; v < nv; v++) top[(size_t) v] = (int) v;
    std::partial_sort(top.begin(), top.begin() + 5, top.end(), [&](int a, int b) { return last[(size_t) a] > last[(size_t) b]; });
    printf("first token top 5:");
    for (int j = 0; j < 5; j++) printf(" %s(%.2f)", common_token_to_piece(vocab, top[(size_t) j], true).c_str(), last[(size_t) top[(size_t) j]]);
    printf("\n---\n");
    const auto tg = std::chrono::steady_clock::now();
    int        produced = 0;
    for (; produced < n_gen; produced++) {
        const int32_t t = (int32_t) (std::max_element(last.begin(), last.end()) - last.begin());
        if (llama_vocab_is_eog(vocab, t)) break;
        printf("%s", common_token_to_piece(vocab, t, true).c_str());
        fflush(stdout);
        if (!m.eval(&t, 1, model::EvalOpts{}, last.data(), nullptr, err)) {
            fprintf(stderr, "\neval: %s\n", err.c_str());
            return 1;
        }
    }
    printf("\n---\n%d tokens in %.1f s\n", produced, std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count());
    mtmd_input_chunks_free(chunks);
    for (auto * b : bitmaps) mtmd_bitmap_free(b);
    mtmd_free(mc);
    llama_model_free(vm);
    return 0;
}

} // namespace e8::cli
