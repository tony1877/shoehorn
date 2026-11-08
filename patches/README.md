# ggml patches

shoehorn builds against an unmodified llama.cpp/ggml pinned as the `third_party/llama.cpp` submodule, then applies
these patches to it at build time (`scripts/build.sh`, step 3; applied once, skipped when already applied).

| Patch | Touches | What it does |
|---|---|---|
| `ggml-zero-copy.patch` | `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA buffers backed by mapped pinned host RAM (zero-copy), freed with `cudaFreeHost` |
| `ggml-weight-prefetch.patch` | `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu` | Stages the next split's host-resident weights into two VRAM slots on a second stream while the current split computes (off with `E8_NO_PREFETCH`) |
| `ggml-q8-1-quantized-sums.patch` | `ggml/src/ggml-cuda/quantize.cu` | The int8 activation blocks carry the sum of their quantized values, not the exact float sum, so K-quant minimums are applied to what the dot product saw (upstream fix; the exact sum biased every Q2_K..Q5_K GPU matmul by a constant offset) |

## The submodule shows as modified — that is expected

After any build, `git status` in the shoehorn repo reports `m third_party/llama.cpp` (`-dirty`), and inside the
submodule these three files differ from upstream:

- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/quantize.cu`

Those edits are the patches above, not stray local changes. Don't commit them inside the submodule and don't
discard them (a rebuild would just reapply them). The source of truth is this directory: to change a patch, edit the
submodule, then regenerate it, e.g. `git -C third_party/llama.cpp diff > patches/<name>.patch` (split per patch).

To confirm the dirty state is only the patches:

```bash
cd third_party/llama.cpp && for p in ../../patches/*.patch; do git apply --reverse --check "$p" && echo "applied: $p"; done
```
