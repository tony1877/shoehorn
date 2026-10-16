# Status: what is done and what is left (2026-10-15)

One page across the project; details live in the linked documents.

## Done

| Area | State | Details |
|---|---|---|
| Linux port | Every phase done: CUDA and ROCm builds, KLD parity with Windows (~4-5e-4 on sm_86/89/120 and MI300X), tarballs, `.deb` / `.rpm`, container images, systemd units, sensors, 9-distribution matrix | [LINUX-PLAN.md](LINUX-PLAN.md), [LINUX-TODO.md](LINUX-TODO.md), [linux.md](linux.md) |
| Release | v0.2.0 on GitHub (vision; Windows zip via `scripts/package-windows.ps1`, Linux CUDA / ROCm tarballs, `.deb`, `.rpm`, SHA256SUMS), MIT license | [releases/v0.2.0.md](releases/v0.2.0.md) |
| GPU fit profiles | 24 GB+ (residual partly in VRAM), 16 GB (IQ4_XS base), 12 GB (IQ3_XXS, KLD 0.0048 vs Q8), 8 GB (IQ2_XXS, KLD 0.0093, 6.9 GB VRAM) | `gpu_fit()` in `src/server/server.cpp` |
| Models page | Every model removable, order kept, `serve` without a model argument, pack form picks the profile's base | `src/server/models.html` |
| Speed research | Measured and dropped: zero-copy GEMM, CPU residual slice, draft alternatives, coupled sampling, prose imatrix, small drafters, EAGLE-3 inputs, tree drafting, verify-level leaves. Kept: Q3_K residual (+7.5%) | [SPEEDUP-PLAN.md](SPEEDUP-PLAN.md), [DECODE-LIMITS.md](DECODE-LIMITS.md) |
| Vision | Images in chat requests (base64 or http(s) URLs), encoder on the GPU on 24 GB+ cards (0.03 s per image) else the CPU (2 s), prompt reuse image-aware, Models page encoder setting and mmproj download, dsh `read_image` and dropped or pasted images working (no attach button: drop or paste), Linux build checked | [VISION.md](VISION.md) |
| Server review fixes | Answer-cap check every 256 bytes; `tool_recovery` test runs in Release (`ctest -R tool_recovery`) | `tests/tool_recovery.cpp` |

## Left

### Blocked on something outside this PC

- **AUR packages** (`packaging/aur/shoehorn-{cuda,rocm}-bin`, tested with makepkg / pacman): AUR registration is
  closed. When it reopens: create the account, add `C:\Users\aless\.ssh\id_ed25519.pub`, push both.
- **ROCm tarball and image on a GPU**: `scripts/runpod/matrix.sh` with `TARBALL=...-rocm.tar.gz` on a community
  MI300X when one is in stock.
- **Hardware not available**: consumer AMD (RX 7900 / 9070), Intel Arc, a bare-metal Linux reboot test with real
  sensors (planned on the 8 GB laptop).

### Vision follow-ups

- The GPU encoder on a real 24 GB card: checked on the 5080 only with layers moved to the CPU (`SHOEHORN_ENCODER=gpu`).
- Remote encoder (`encode-server` + `serve --encoder-url`): works from the laptop's RTX 4070 (a 1657x1085 screenshot in
  2.7 s instead of 22 s). In use since 2026-10-16: the laptop runs it as a startup task ("shoehorn encode-server",
  SYSTEM, API key required), and `C:\models\start-detached.ps1` passes `--encoder-url http://10.77.0.2:8091`; an image
  pasted into dsh is encoded on the laptop (a 1657x1085 screenshot: 3.0 s, read correctly).
- v0.2.0 Windows zip replaced on 2026-10-16: the first one ran only on RTX 50xx with an AVX-512 CPU. The new one (`scripts\build.ps1 -Release`, packaged by `package-windows.ps1`) is built for sm_86/89/120a and any AVX2 CPU, as the Linux release; SHA256SUMS and the release text are updated.
- `tests/vision_smoke.py` runs by hand against a live server; not in CI (needs a GPU and the model).
- Flash-Next (`qwen4exp`) takes no images.

### Speed (small gains only; no large software lever is left on this hardware)

- Full-layer KV staging at 200K: dropped, the engine uses sparse attention, not the streamed design it was modelled on; long context costs 73 ms per verify at 200K, mostly the selected pages ([SPEEDUP-PLAN.md](SPEEDUP-PLAN.md) section 10).
- Self-distilled MTP head: exported and A/B tested, dropped. MTP kept rises by 0.02 but tok/s stays within noise, and
  the Q8_0 head shrinks the KV window to 6.9K tokens ([SPEEDUP-PLAN.md](SPEEDUP-PLAN.md) section 2).
- No speed item is left open.

### Review items left open (decisions for the owner)

- `think_after_tool` now defaults to on: every tool-result turn reasons, so agent loops are slower than before.

### Tool calls (2026-10-16, decided by the owner)

- Tool calls are read by shoehorn's XML parser (`src/server/tool_recovery.h`) whenever the text has `<tool_call>`, in
  place of llama.cpp's grammar: a `</parameter>` ends a value only when the call continues after it, so a file that
  quotes tool-call markup is written whole (the grammar cut it short and turned the quoted examples into extra
  calls). Inside an open call the end-of-turn tokens stand for their plain text (`SpecDecoder::set_call_ban`), and a
  call cut off by the end of the turn is never reported as finished.
- `tests/tool_stream_matrix.py` (live, 23 kinds of file writes x 2 + a max_tokens cut): before, markup and
  `<|im_end|>` content lost text or broke the JSON (4 of 46 calls); after, every call is whole and streams. Left: the
  model will not write `<think>` / `</think>` as text (it leaves them out), and when a file mentions `<|im_end|>` it
  sometimes writes another marker. `ctest -R tool_recovery` covers the parser cases.

### Small

- The Models page shows the RTX 5080 as "17 GB" (VRAM rounding).
- AUR files already point at 0.2.0 (push them when AUR registration reopens).
