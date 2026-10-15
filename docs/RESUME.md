# Resume prompt

Paste everything below the line into a new Claude Code session in `C:\src\shoehorn`.

---

You are picking up work on **shoehorn** (`C:\src\shoehorn`, GitHub `tony1877/shoehorn`, branch `main`): a
llama.cpp/ggml-based inference server for the Qwen3.8 27B. It runs a 4-bit base in VRAM plus a residual in pinned RAM
(about Q8 quality), with MTP speculative decoding, on Windows and Linux. The bundled agent **dsh** (`agent/`) uses it.

## Read first

1. `docs/STATUS.md`: what is done and what is left (the work list).
2. `docs/VISION.md`: image input (the most recent feature).
3. `docs/SPEEDUP-PLAN.md`: every speed idea tried and its result. Do not retry dropped ideas.
4. `git log --oneline -15` and `git status`. A modified `third_party/llama.cpp` is normal: shoehorn's ggml patches
   are applied at build time.

## This PC

- **Server:** `C:\src\shoehorn\build\bin\shoehorn.exe serve` on port 8090, started at logon by the
  `shoehorn-watchdog` task through `C:\models\start-detached.ps1`. Log: `C:\models\bench\shoehorn.log`. API key: last
  line of `~\.dsh\llama-api-key.txt`.
- **Model list:** `~\.shoehorn\models.json`. The 27B entries have `"mmproj"` set to
  `E:/shoehorn/vision/mmproj-Qwen3.8-27B-Uncensored-f16.gguf`. Packs are in `E:\shoehorn\`.
- **dsh:** web UI on 127.0.0.1:3080. The token URL is in `~\.dsh\dsh.log` (written by C:\Tools\bin\start-dsh.ps1; `dsh-web.log` is stale). Its settings
  are in `~\.dsh\settings.yaml`; the 27B models there have `input: [text, image]`.
- **Harness:** exec bridge on port 9999 (`C:\Tools\bin\brg.ps1`), LAN proxy on 3080, SearXNG on 8888.
- **Building:**
  - Live binary: `scripts\build.ps1`. It pauses the watchdog; stop the server on port 8090 first, because the exe is
    locked.
  - Experiments: build the `build-zc` folder with the VS 2022 environment loaded, so you can work without touching
    the live server.
  - Linux: build in WSL Debian with `bash scripts/build.sh --backend cpu --release --build-dir ~/...`.
- **GPU:** the RTX 5080 (16 GB) holds one 27B at a time. Before GPU tests, unload the live model with
  `POST localhost:8090/models/unload`, or run a test server on port 8099 with `--models-file` / `--settings` pointing
  at scratch copies, so `~\.shoehorn` is never modified. Reload afterwards with
  `POST /models/load {"alias":"qwen3.8-27b-uncensored-shoehorn"}`, so dsh works again.

## How to work

- **Autonomy:** build, test, commit and push to `main` yourself. End commit messages with the Co-Authored-By line. Ask
  before publishing a release or spending money.
- **Spending:** RunPod is allowed up to **$5** in total for this session. Check the balance first, never leave a pod
  running unwatched, and delete each pod when its job ends.
- **Testing:**
  - Run a test against the real model before calling something done: `tests/vision_smoke.py`,
    `tests/server_policy.py`, and `ctest -R tool_recovery` in the build folder.
  - The user likes agent-facing checks to also go through the dsh web UI.
- **Windows:** change Windows behaviour only for the tasks below or with the user's approval.
- **Leave alone:**
  - `think_after_tool` defaulting to on.
  - The streaming lenient tool-call parser's current scope.
  - Do not script retries against the AUR registration page.
- **Record results:** update `docs/STATUS.md` (and `SPEEDUP-PLAN.md` for speed results) whenever something is
  finished or dropped, including measured numbers.

## Work, in this order

1. **Vision follow-ups** (`docs/STATUS.md` → Vision follow-ups):
   - fetch http(s) image URLs, with a size limit and timeout;
   - run the encoder on the GPU when VRAM allows (24 GB+ profiles), keeping the CPU otherwise;
   - test dsh's attach-image button in the chat box;
   - make the Models page's downloads also fetch a repository's `mmproj-*.gguf` and set it on the entry.
2. **Release v0.2.0 with vision:** bump the version, then build the Windows zip, the Linux CUDA and ROCm tarballs, and
   the `.deb` / `.rpm` (`scripts/package-*.sh`). Write release notes. **Ask before publishing** the GitHub release.
3. **Small speed gains**, each A/B measured with `gen --repeat 6` and `bench_prose.py`, then recorded:
   - export and test the self-distilled MTP head (`E:\shoehorn\mtp\mtp_chain.pt`, `experiments/mtp_distill/export.py`;
     estimated +4-7% on prose);
   - full-layer KV staging at 200K context (`docs/SPEED2X.md`, modelled +8%).

Start by reading the files above, then give a short plan for item 1 and begin.
