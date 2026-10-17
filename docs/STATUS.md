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

### Agent consistency (2026-10-16)

- Reasoning budgets per effort raised (low / medium / high: 4K / 16K / 32K tokens; medium, dsh's default, was 8K).
  Over budget the reasoning is closed after the next sentence or line end (within one decoding step; 512 tokens at
  most), and the first 16 answer tokens may not end the turn (`set_answer_floor`), which once left an empty reply.
  Measured: closed 25-135 tokens past the budget, 10 of 10 runs answered (before: up to 551 tokens later, mid-word,
  and 1 of 7 answers empty).
- Streams send an empty delta after 5 s of silence while a request waits for its turn, the model loads or a long
  prompt is read: longest silence 5.0 s over a 96 s read (before: silent the whole time).
- Repetition guard: 32 KiB of exact repetition inside a tool call (8 KiB elsewhere), so repetitive files are not cut.
- Non-string tool arguments are repaired when they are not valid JSON (trailing commas, single quotes, unquoted keys,
  Python literals, a code fence).
- Checked, no change needed: dsh sends no answer cap (`max_content_chars`), reports unknown tool names to the model
  (`unknown tool "x"`), and matches `edit` on LF-normalized text, writing back the file's own line endings.
- `tests/agent_consistency.py` (live): keep-alives, reasoning budget, repetition, JSON list arguments, unknown tool.
  Left: the model loses count on long runs of identical lines (asked for 150, writes until stopped).

### Agent edge cases (2026-10-17)

- Tool calls written as JSON inside `<tool_call>` (Hermes-style fine-tunes) keep the strict parse's calls; the XML
  parser only replaces them when the text uses `<function=`. (The 27B and Flash-Next both use `<function=`.)
- A call cut off by `max_tokens` (a file bigger than the output limit) is not run, and the reply now says so: "[The
  write call was cut off at the output limit ... Write large files in parts ...]", which the model sees next turn.
- dsh `edit` and `write` keep a UTF-8 byte-order mark (it was dropped: PowerShell 5.1 then misreads non-ASCII
  scripts). UTF-16 and legacy code-page files are named as such with how to convert them, instead of "binary file".
- Live checks (agent_consistency.py F-J), all passing: output-limit note; a client abort mid-answer, then the
  conversation goes on (answered, prompt reused); an exact signature recalled ~50K tokens back; a four-step tool
  loop with reasoning reuses every earlier step (587/659, 654/726, 721/793 tokens); two sessions at once both
  answered (the second waits its turn: no slots, the full context window stays).
- Checked, no change: dsh compaction keeps exact paths and file state and dsh refuses edits to unread or changed
  files; shell output is capped at 64 KB and `read` at 2000 lines; a session with images switched to Flash-Next (text
  only) goes on, the images sent as text placeholders.

### Small

- The Models page shows the RTX 5080 as "17 GB" (VRAM rounding).
- AUR files already point at 0.2.0 (push them when AUR registration reopens).
- **Idle unload switch** (2026-10-17): the dashboard header has a "model" selector: stay loaded (warm), or unload after 5 min / 15 min / 1 h idle. `POST /idle-unload {"seconds": N}` applies it without a restart (the unloader waits for its turn, so a running answer is never cut), saves `--idle-unload` in the settings file, and "stay loaded" reloads a freed model at once. `/stats` reports `idle_unload`. Checked live: freed 36 s after a 30 s setting, reloaded 14 s after "stay loaded".

### Long agent sessions (2026-10-17)

Found with a live 92-131-step research session in dsh (two compactions), then fixed and re-run twice:

- **Prompt reuse after a change in the middle** (server): the recurrent state could only resume where the last prompt
  ended, so any prune or compaction re-read the whole conversation from the cached system prompt (up to 133K tokens,
  ~90 s). Now a recurrent checkpoint is kept every `--ckpt-every` 8192 prompt tokens (`--ckpt-max` 8, ~157 MB of RAM
  each on the 27B) and a changed prompt rewinds to the last one before the change (`Qwen35::save_rec` / `rewind`).
  Checked: logits after the prompt bit-identical to a full read; 89 s -> 23 s for a prune 80K tokens in.
  `--fast-prefill` now counts from the whole prompt's end, so reading in pieces does not change which tokens get the
  residual.
- **Tool-result pruning** (dsh): results the model has not read yet (the running turn's, after its last assistant
  message) and the retained tail (16% of the window) are never pruned; a pass prunes every other eligible result at
  once, so later passes change the prompt near its end.
- **Compaction summaries**: requested with `tool_choice: "none"` (the server enforces it: no `<tool_call>` token), and
  stray tool-call or checkpoint tags at the ends of a summary are stripped (a summary once came back as a `write` call
  and its closing tags stayed in the checkpoint).
- **Re-injected context** (dsh): a memory-index-only change of the runtime context waits for the next turn instead of
  appending a whole snapshot mid-turn; after a compaction the skill catalog comes back with its normal wording instead
  of "the catalog changed"; an unchanged system prompt at a new request series is shown as "Context rewritten · system
  prompt unchanged".
- **Dashboard**: a "model" selector in the header: stay loaded, or unload after 5 min / 15 min / 1 h idle (live, saved
  as `--idle-unload`).
- Left: the first prune pass after a compaction still re-reads most of the prompt (it prunes results right after the
  checkpoint at its start). `E8_LOG_RAW=DIR` dumps each request body, the raw output and the top logits after the
  prompt (last 32).
