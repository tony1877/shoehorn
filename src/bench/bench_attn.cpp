// `shoehorn bench-attn [--kv 32768,131072,409600] [--batch 512,1024,2048,4096] [--types q8_0/q4_0,q8_0/q8_0,f16/f16]
//                      [--sparse 0,12288] [--heads 24] [--kv-heads 4] [--head-dim 256] [--layers 16] [--reps 3]`
//
// Times prefill attention (ggml_flash_attn_ext) on the GPU at the shapes Qwen35::build_graph uses with the KV in
// VRAM: queries [hd, batch, heads], K/V viewed out of a [hd, kv_heads, n_kv] cache, an F16 mask, F32 accumulation.
// Prints ms per layer, TFLOPS* (dense-equivalent: 4 * batch * heads * hd * n_kv per layer, so sparse rows compare
// directly) and what a prompt batch at that depth costs in attention over `layers` full-attention layers.
//   --sparse 0      dense: one call over all heads, every column visible (a batch deep in a long prompt sees ~all)
//   --sparse N      approximate attention as the sparse prefill runs it: one call per KV head (its query heads), every
//                   query seeing N columns (the last 4096 plus random 64-row pages) through
//                   ggml_flash_attn_ext_set_n_kv_max, so the kernel gathers only those
// E8_FA_NCOLS2=1|2|4|8 in the environment forces the kernel's GQA tile height (see ggml-cuda/fattn.cu).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace e8::cli {

namespace {

std::vector<long long> parse_list(const std::string & s) {
    std::vector<long long> v;
    std::stringstream      ss(s);
    std::string            t;
    while (std::getline(ss, t, ',')) {
        if (!t.empty()) v.push_back(std::atoll(t.c_str()));
    }
    return v;
}

bool type_from(const std::string & s, ggml_type & t) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && s == n) {
            t = (ggml_type) i;
            return true;
        }
    }
    return false;
}

// Fills a tensor with random values: one block of up to 4096 rows, quantized to its type and repeated.
void fill(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t row   = t->ne[0];
    const int64_t chunk = std::min<int64_t>(ggml_nrows(t), 4096);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>   f((size_t) (row * chunk));
    for (float & x : f) x = nd(rng);
    std::vector<uint8_t> q(ggml_row_size(t->type, row) * (size_t) chunk);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(q.data(), f.data(), q.size());
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) q.data(), row * chunk);
    } else {
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, chunk, row, nullptr);
    }
    const size_t total = ggml_nbytes(t);
    for (size_t off = 0; off < total; off += q.size()) {
        ggml_backend_tensor_set(t, q.data(), off, std::min(q.size(), total - off));
    }
}

}  // namespace

int bench_attn(const std::vector<std::string> & args) {
    std::vector<long long>   kvs     = { 32768, 131072, 262144, 409600 };
    std::vector<long long>   batches = { 1024 };
    std::vector<long long>   sparse  = { 0 };
    std::vector<std::string> types   = { "q8_0/q4_0" };
    int heads = 24, kv_heads = 4, hd = 256, layers = 16, reps = 3;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--kv") kvs = parse_list(val());
        else if (a == "--batch") batches = parse_list(val());
        else if (a == "--sparse") sparse = parse_list(val());
        else if (a == "--types") {
            types.clear();
            std::stringstream ss(val());
            std::string       t;
            while (std::getline(ss, t, ',')) types.push_back(t);
        } else if (a == "--heads") heads = std::atoi(val().c_str());
        else if (a == "--kv-heads") kv_heads = std::atoi(val().c_str());
        else if (a == "--head-dim") hd = std::atoi(val().c_str());
        else if (a == "--layers") layers = std::atoi(val().c_str());
        else if (a == "--reps") reps = std::max(1, std::atoi(val().c_str()));
        else {
            fprintf(stderr, "bench-attn: unknown option %s\n", a.c_str());
            return 1;
        }
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "bench-attn: no GPU device\n");
        return 1;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        fprintf(stderr, "bench-attn: backend init failed\n");
        return 1;
    }
    size_t fr = 0, tot = 0;
    ggml_backend_dev_memory(dev, &fr, &tot);
    printf("bench-attn: %s, %.1f of %.1f GB free; heads %d, kv heads %d, head dim %d, %d attention layers\n",
           ggml_backend_dev_description(dev), fr / 1e9, tot / 1e9, heads, kv_heads, hd, layers);
    printf("%-11s %7s %8s %7s %10s %8s %14s\n", "K/V", "batch", "n_kv", "sparse", "ms/layer", "TFLOPS*", "attn s/batch");

    std::mt19937 rng(42);
    for (const std::string & ty : types) {
        const size_t slash = ty.find('/');
        ggml_type    kt, vt;
        if (slash == std::string::npos || !type_from(ty.substr(0, slash), kt) || !type_from(ty.substr(slash + 1), vt)) {
            fprintf(stderr, "bench-attn: bad --types entry %s (e.g. q8_0/q4_0)\n", ty.c_str());
            continue;
        }
        for (long long nkv : kvs) {
            // the cache, as Qwen35 holds it: [hd, kv_heads, n_kv]
            ggml_init_params cp = { 4 * ggml_tensor_overhead(), nullptr, true };
            ggml_context *   cc = ggml_init(cp);
            ggml_tensor *    kc = ggml_new_tensor_3d(cc, kt, hd, kv_heads, nkv);
            ggml_tensor *    vc = ggml_new_tensor_3d(cc, vt, hd, kv_heads, nkv);
            ggml_backend_buffer_t cb = ggml_backend_alloc_ctx_tensors(cc, be);
            if (!cb) {
                printf("%-11s %7s %8lld   cache does not fit\n", ty.c_str(), "-", nkv);
                ggml_free(cc);
                continue;
            }
            fill(kc, rng);
            fill(vc, rng);
            for (long long nb : batches) {
                for (long long ns : sparse) {
                    const bool       sp = ns > 0 && ns < nkv;
                    ggml_init_params gp = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                    ggml_context *   cg = ggml_init(gp);
                    ggml_tensor *    q  = ggml_new_tensor_3d(cg, GGML_TYPE_F32, hd, heads, nb);
                    ggml_tensor *    m  = ggml_new_tensor_4d(cg, GGML_TYPE_F16, nkv, nb, 1, 1);
                    ggml_tensor *    qp = ggml_permute(cg, q, 0, 2, 1, 3);  // [hd, nb, heads]
                    ggml_cgraph *    gf = ggml_new_graph(cg);
                    if (!sp) {
                        ggml_tensor * K = ggml_view_3d(cg, kc, hd, kv_heads, nkv, ggml_row_size(kt, hd), kc->nb[2], 0);
                        ggml_tensor * V = ggml_view_3d(cg, vc, hd, kv_heads, nkv, ggml_row_size(vt, hd), vc->nb[2], 0);
                        ggml_tensor * a = ggml_flash_attn_ext(cg, qp, ggml_permute(cg, K, 0, 2, 1, 3), ggml_permute(cg, V, 0, 2, 1, 3),
                                                              m, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                        ggml_prec_set_acc(a, GGML_PREC_F32);
                        ggml_build_forward_expand(gf, a);
                    } else {
                        const int g = heads / kv_heads;
                        for (int j = 0; j < kv_heads; j++) {  // one call per KV head over its g query heads
                            ggml_tensor * k  = ggml_view_3d(cg, kc, hd, nkv, 1, kc->nb[2], kc->nb[2] * (size_t) nkv, kc->nb[1] * (size_t) j);
                            ggml_tensor * v  = ggml_view_3d(cg, vc, hd, nkv, 1, vc->nb[2], vc->nb[2] * (size_t) nkv, vc->nb[1] * (size_t) j);
                            ggml_tensor * qj = ggml_view_3d(cg, qp, hd, nb, g, qp->nb[1], qp->nb[2], qp->nb[2] * (size_t) (j * g));
                            ggml_tensor * a  = ggml_flash_attn_ext(cg, qj, k, v, m, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                            ggml_prec_set_acc(a, GGML_PREC_F32);
                            ggml_flash_attn_ext_set_n_kv_max(a, (int32_t) ns);
                            ggml_build_forward_expand(gf, a);
                        }
                    }
                    ggml_backend_buffer_t gb = ggml_backend_alloc_ctx_tensors(cg, be);
                    if (!gb) {
                        printf("%-11s %7lld %8lld   inputs do not fit\n", ty.c_str(), nb, nkv);
                        ggml_free(cg);
                        continue;
                    }
                    fill(q, rng);
                    {  // dense: all visible; sparse: the last 4096 columns plus random whole 64-column pages, ns in all
                        std::vector<ggml_fp16_t> row((size_t) nkv, ggml_fp32_to_fp16(sp ? -INFINITY : 0.0f));
                        if (sp) {
                            const long long win = std::min<long long>(4096, ns);
                            for (long long c = nkv - win; c < nkv; c++) row[(size_t) c] = ggml_fp32_to_fp16(0.0f);
                            const long long npages = (nkv - win) / 64, want = (ns - win) / 64;
                            std::vector<long long> pg((size_t) npages);
                            for (long long i = 0; i < npages; i++) pg[(size_t) i] = i;
                            std::shuffle(pg.begin(), pg.end(), rng);
                            for (long long i = 0; i < std::min(want, npages); i++) {
                                for (int c = 0; c < 64; c++) row[(size_t) (pg[(size_t) i] * 64 + c)] = ggml_fp32_to_fp16(0.0f);
                            }
                        }
                        for (long long r = 0; r < nb; r++) ggml_backend_tensor_set(m, row.data(), (size_t) r * m->nb[1], row.size() * 2);
                    }
                    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                    if (!ggml_gallocr_alloc_graph(ga, gf)) {
                        printf("%-11s %7lld %8lld   graph does not fit\n", ty.c_str(), nb, nkv);
                    } else {
                        ggml_backend_graph_compute(be, gf);  // warm-up
                        ggml_backend_synchronize(be);
                        std::vector<double> t;
                        for (int r = 0; r < reps; r++) {
                            const auto t0 = std::chrono::steady_clock::now();
                            ggml_backend_graph_compute(be, gf);
                            ggml_backend_synchronize(be);
                            t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                        }
                        std::sort(t.begin(), t.end());
                        const double s     = t[t.size() / 2];
                        const double flops = 4.0 * (double) nb * heads * hd * (double) nkv;
                        printf("%-11s %7lld %8lld %7lld %10.2f %8.1f %14.2f\n", ty.c_str(), nb, nkv, sp ? ns : 0LL, s * 1e3,
                               flops / s / 1e12, s * layers);
                        fflush(stdout);
                    }
                    ggml_gallocr_free(ga);
                    ggml_backend_buffer_free(gb);
                    ggml_free(cg);
                }
            }
            ggml_backend_buffer_free(cb);
            ggml_free(cc);
        }
    }
    ggml_backend_free(be);
    return 0;
}

// `shoehorn bench-mm [--batch 1024] [--types iq4_xs,q4_0,q4_K,q8_0,nvfp4,mxfp4] [--shapes 5120x17408,17408x5120,5120x10240]
//                    [--reps 5]`
// Times a prompt-batch matmul (weights [K, M] of each type times F32 activations [K, batch]) on the GPU, with the
// activations at the default precision (Q8) and, where the type allows it, quantized to 4-bit (GGML_PREC_Q4: NVFP4 /
// MXFP4 on Blackwell's FP4 tensor cores). Prints ms and TOPS (2 * K * M * batch per call).
int bench_mm(const std::vector<std::string> & args) {
    std::vector<long long>   batches = { 1024 };
    std::vector<std::string> types   = { "iq4_xs", "q4_0", "q4_K", "q8_0", "nvfp4", "mxfp4" };
    std::vector<std::string> shapes  = { "5120x17408", "17408x5120", "5120x10240" };
    int                      reps    = 5;
    auto split = [](const std::string & s) {
        std::vector<std::string> v;
        std::stringstream        ss(s);
        std::string              t;
        while (std::getline(ss, t, ',')) v.push_back(t);
        return v;
    };
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--batch") batches = parse_list(val());
        else if (a == "--types") types = split(val());
        else if (a == "--shapes") shapes = split(val());
        else if (a == "--reps") reps = std::max(1, std::atoi(val().c_str()));
        else {
            fprintf(stderr, "bench-mm: unknown option %s\n", a.c_str());
            return 1;
        }
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t     be  = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (!be) {
        fprintf(stderr, "bench-mm: no GPU backend\n");
        return 1;
    }
    printf("bench-mm: %s\n%-8s %-12s %6s %8s %8s %8s %8s\n", ggml_backend_dev_description(dev), "type", "K x M", "batch",
           "ms(q8)", "TOPS", "ms(q4)", "TOPS");
    std::mt19937 rng(7);
    for (const std::string & ts : types) {
        ggml_type wt;
        if (!type_from(ts, wt)) {
            fprintf(stderr, "bench-mm: unknown type %s\n", ts.c_str());
            continue;
        }
        for (const std::string & sh : shapes) {
            const size_t  x = sh.find('x');
            const int64_t K = std::atoll(sh.substr(0, x).c_str()), M = std::atoll(sh.substr(x + 1).c_str());
            for (long long nb : batches) {
                double ms[2] = { -1, -1 };
                for (int p = 0; p < 2; p++) {
                    ggml_init_params gp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                    ggml_context *   cg = ggml_init(gp);
                    ggml_tensor *    w  = ggml_new_tensor_2d(cg, wt, K, M);
                    ggml_tensor *    xx = ggml_new_tensor_2d(cg, GGML_TYPE_F32, K, nb);
                    ggml_tensor *    y  = ggml_mul_mat(cg, w, xx);
                    // (NVFP4 / MXFP4 on Blackwell default to 4-bit activations: Q8 has to be asked for)
                    if (!ggml_prec_set_src(y, p == 1 ? GGML_PREC_Q4 : GGML_PREC_Q8, 1)) {
                        ggml_free(cg);
                        continue;
                    }
                    ggml_cgraph * gf = ggml_new_graph(cg);
                    ggml_build_forward_expand(gf, y);
                    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(cg, be);
                    if (!buf) {
                        ggml_free(cg);
                        continue;
                    }
                    fill(w, rng);
                    fill(xx, rng);
                    ggml_backend_graph_compute(be, gf);
                    ggml_backend_synchronize(be);
                    std::vector<double> t;
                    for (int r = 0; r < reps; r++) {
                        const auto t0 = std::chrono::steady_clock::now();
                        ggml_backend_graph_compute(be, gf);
                        ggml_backend_synchronize(be);
                        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                    }
                    std::sort(t.begin(), t.end());
                    ms[p] = t[t.size() / 2] * 1e3;
                    ggml_backend_buffer_free(buf);
                    ggml_free(cg);
                }
                const double ops = 2.0 * (double) K * (double) M * (double) nb;
                printf("%-8s %-12s %6lld %8.3f %8.1f", ts.c_str(), sh.c_str(), nb, ms[0], ms[0] > 0 ? ops / ms[0] / 1e9 : 0.0);
                if (ms[1] > 0) printf(" %8.3f %8.1f", ms[1], ops / ms[1] / 1e9);
                printf("\n");
                fflush(stdout);
            }
        }
    }
    ggml_backend_free(be);
    return 0;
}

}  // namespace e8::cli
