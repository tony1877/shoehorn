// `shoehorn encode-server --mmproj enc.gguf --model meta.gguf [--host 127.0.0.1] [--port 8091] [--api-key-file F]`
// `shoehorn encode-server --model base.gguf --save-meta meta.gguf`
//
// Remote vision encoder (docs/VISION.md): another machine's GPU encodes images for `serve --encoder-url`. Images arrive
// already decoded (the serving machine's pixels, so both ends preprocess the same input); this end runs llama.cpp's
// mtmd on them (on the GPU when there is one) and returns the embeddings. The text model is only read for its
// metadata (vocab only), so --model can be a metadata-only copy of the base GGUF: --save-meta writes one (a few MB
// instead of 16 GB).
//
// POST /encode   headers X-Width, X-Height; body: width x height x 3 bytes of RGB
//                -> body: n x n_embd float32 (little-endian); headers X-Tokens (n), X-Embd (n_embd)
// GET  /health   {"status": "ok", "gpu": true|false, "mmproj": ...}

#include "cli/commands.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"

#include <httplib.h>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace e8::cli {

namespace {

// a copy of `in` with its metadata only (no tensors): enough for a vocab-only load
bool save_meta(const std::string & in, const std::string & out) {
    gguf_init_params ip  = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context *   src = gguf_init_from_file(in.c_str(), ip);
    if (!src) {
        fprintf(stderr, "cannot read %s\n", in.c_str());
        return false;
    }
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    const bool ok = gguf_write_to_file(dst, out.c_str(), /*only_meta=*/false);
    gguf_free(dst);
    gguf_free(src);
    if (!ok) fprintf(stderr, "cannot write %s\n", out.c_str());
    return ok;
}

// the text model's width (<arch>.embedding_length), what each image embedding must have
int embedding_length(const std::string & path) {
    gguf_init_params ip = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context *   g  = gguf_init_from_file(path.c_str(), ip);
    if (!g) return -1;
    int                     n    = -1;
    const int64_t           ka   = gguf_find_key(g, "general.architecture");
    if (ka >= 0) {
        const int64_t k = gguf_find_key(g, (std::string(gguf_get_val_str(g, ka)) + ".embedding_length").c_str());
        if (k >= 0) n = (int) gguf_get_val_u32(g, k);
    }
    gguf_free(g);
    return n;
}

std::string key_from_file(const std::string & path) {
    std::ifstream      f(path, std::ios::binary);
    std::stringstream  ss;
    ss << f.rdbuf();
    std::istringstream ks(ss.str());
    std::string        key;
    for (std::string line; std::getline(ks, line);) {  // last non-comment line, as serve reads it
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty() && line[0] != '#') key = line;
    }
    return key;
}

} // namespace

int encode_server(const std::vector<std::string> & args) {
    std::string mmproj, model, host = "127.0.0.1", key_file, meta_out;
    int         port = 8091;
    bool        cpu  = false;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--mmproj") mmproj = val();
        else if (a == "--model") model = val();
        else if (a == "--host") host = val();
        else if (a == "--port") port = std::atoi(val().c_str());
        else if (a == "--api-key-file") key_file = val();
        else if (a == "--save-meta") meta_out = val();
        else if (a == "--cpu") cpu = true;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (!meta_out.empty()) {
        if (model.empty()) {
            fprintf(stderr, "usage: shoehorn encode-server --model base.gguf --save-meta meta.gguf\n");
            return 1;
        }
        if (!save_meta(model, meta_out)) return 1;
        printf("wrote %s (metadata of %s)\n", meta_out.c_str(), model.c_str());
        return 0;
    }
    if (mmproj.empty() || model.empty()) {
        fprintf(stderr,
                "usage: shoehorn encode-server --mmproj enc.gguf --model meta.gguf [--host 127.0.0.1] [--port 8091]\n"
                "                              [--api-key-file F] [--cpu]\n"
                "       shoehorn encode-server --model base.gguf --save-meta meta.gguf (metadata-only copy for --model)\n"
                "  a remote vision encoder for `serve --encoder-url http://HOST:PORT` (docs/VISION.md)\n");
        return 1;
    }
    const std::string api_key = key_file.empty() ? std::string() : key_from_file(key_file);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    llama_model * text    = llama_model_load_from_file(model.c_str(), mp);
    if (!text) {
        fprintf(stderr, "cannot load the model metadata from %s\n", model.c_str());
        return 1;
    }
    // on the GPU when the encoder fits in the free VRAM with room for a big image's activations (ggml aborts when an
    // allocation does not fit, so this is checked first); else, or with --cpu, on the CPU
    size_t fr = 0, tot = 0;
    if (ggml_backend_dev_t g = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) ggml_backend_dev_memory(g, &fr, &tot);
    std::ifstream       mf(mmproj, std::ios::binary | std::ios::ate);
    const double        need = (mf ? (double) mf.tellg() : 1e9) / 1e9 + 1.5;
    mtmd_context_params cp   = mtmd_context_params_default();
    cp.use_gpu               = !cpu && fr / 1e9 >= need;
    cp.print_timings       = false;
    mtmd_context * mc      = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!mc || !mtmd_support_vision(mc)) {
        fprintf(stderr, "cannot load the vision encoder %s\n", mmproj.c_str());
        return 1;
    }
    const int  n_embd = embedding_length(model);  // (a vocab-only load leaves the model's dimensions at 0)
    if (n_embd <= 0) {
        fprintf(stderr, "no <arch>.embedding_length in %s\n", model.c_str());
        return 1;
    }
    const bool gpu    = cp.use_gpu;
    std::mutex mu;  // one encode at a time (one encoder context)

    httplib::Server http;
    auto authorized = [&](const httplib::Request & req, httplib::Response & resp) {
        if (api_key.empty() || req.get_header_value("Authorization") == "Bearer " + api_key) return true;
        resp.status = 401;
        resp.set_content("{\"error\": \"unauthorized\"}", "application/json");
        return false;
    };
    http.Get("/health", [&](const httplib::Request &, httplib::Response & resp) {
        std::string m = mmproj;
        for (char & c : m) if (c == '\\') c = '/';
        resp.set_content("{\"status\": \"ok\", \"gpu\": " + std::string(gpu ? "true" : "false") + ", \"mmproj\": \"" + m + "\"}",
                         "application/json");
    });
    http.Post("/encode", [&](const httplib::Request & req, httplib::Response & resp) {
        if (!authorized(req, resp)) return;
        const int w = std::atoi(req.get_header_value("X-Width").c_str()), h = std::atoi(req.get_header_value("X-Height").c_str());
        if (w <= 0 || h <= 0 || req.body.size() != (size_t) w * (size_t) h * 3) {
            resp.status = 400;
            resp.set_content("{\"error\": \"need X-Width, X-Height and width x height x 3 bytes of RGB\"}", "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        const auto                  t0 = std::chrono::steady_clock::now();
        mtmd_bitmap *               bm = mtmd_bitmap_init((uint32_t) w, (uint32_t) h, (const unsigned char *) req.body.data());
        mtmd_input_chunks *         ch = mtmd_input_chunks_init();
        const char *                marker = mtmd_default_marker();
        mtmd_input_text             tx{ marker, strlen(marker), false, true };
        const mtmd_bitmap *         bms[1] = { bm };
        int32_t                     rc     = mtmd_tokenize(mc, ch, &tx, bms, 1);
        const mtmd_input_chunk *    img    = nullptr;
        for (size_t c = 0; rc == 0 && c < mtmd_input_chunks_size(ch); c++) {
            const mtmd_input_chunk * k = mtmd_input_chunks_get(ch, c);
            if (mtmd_input_chunk_get_type(k) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = k;
        }
        if (rc != 0 || !img || mtmd_encode_chunk(mc, img) != 0) {
            mtmd_input_chunks_free(ch);
            mtmd_bitmap_free(bm);
            resp.status = 500;
            resp.set_content("{\"error\": \"encoding failed\"}", "application/json");
            return;
        }
        const size_t  n = mtmd_input_chunk_get_n_tokens(img);
        const float * e = mtmd_get_output_embd(mc);
        resp.set_content(std::string((const char *) e, n * (size_t) n_embd * sizeof(float)), "application/octet-stream");
        resp.set_header("X-Tokens", std::to_string(n));
        resp.set_header("X-Embd", std::to_string(n_embd));
        mtmd_input_chunks_free(ch);
        mtmd_bitmap_free(bm);
        fprintf(stderr, "encoded %dx%d: %zu tokens in %.2f s\n", w, h, n,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    });
    http.set_payload_max_length((size_t) 64 << 20);  // a 4096 x 4096 RGB image is 48 MB
    fprintf(stderr, "encode-server: %s on the %s, listening on %s:%d%s\n", mmproj.c_str(), gpu ? "GPU" : "CPU", host.c_str(),
            port, api_key.empty() ? " (no API key)" : "");
    if (!http.listen(host, port)) {
        fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    mtmd_free(mc);
    llama_model_free(text);
    return 0;
}

} // namespace e8::cli
