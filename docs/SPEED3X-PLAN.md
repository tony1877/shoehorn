# 2-3x decode on the 27B at today's quality

Status: plan (2026-10-19). Nothing built yet. Labels: [M] measured, [D] derived from measurements, [E] estimate.

Goal: double, then triple, decode speed of the 27B (`orca27b.base.gguf` + `orca27b.res.gguf`) on the RTX 5080 16 GB,
keeping the exact base + residual (B+R) verify, so the output distribution stays what it is today. One optional step
(Phase 3, B2) changes the target model slightly; it is a switch, and its cost is stated.

## 1. Where the time goes today

Production timing log, 1,033 long decodes [M]:

| | per cycle | share |
|---|---|---|
| Draft (MTP + base checks, VRAM) | 414 ms | 54% |
| Verify (B+R) | 320 ms | 42% |
| Tokens per cycle | 21.9 | |
| Overlap between the two (`pipe_s`) | 0 | |

- Decode: ~30 tok/s mean; 35.1 tok/s for prompts that fit the 20K VRAM window, 28.5-30.5 beyond it [M].
- 73% of requests (90% of decode time) are longer than the VRAM window; median prompt 52K tokens [M].
- The verify is weight-streaming bound: 8 to 64 rows cost 291 vs 292 ms (`gen --profile`, 256-token context) [M].
  Of the ~320 ms, ~250 ms is the 14.4 GB residual crossing PCIe at 57.5 GB/s [D].
- Residual kept in VRAM saves ~17 ms per GB per verify [D]; measured 358 -> 348 ms with 0.6 GB (`--res-gpu-gb`), no
  change end to end because drafting dominates [M].
- All 64 base layers are on the GPU; the 2.54 GB "CPU" weights are the bf16 token-embedding table (one row per token)
  [M]. There is nothing left to move onto the GPU without freeing VRAM first.
- VRAM at 262K context: 15.7 of 16.3 GB used [M]. Beyond ~15.96 GB the driver spills to system memory and decode
  collapses (docs/MULTISEQ.md). The desktop's virtual display also needs headroom on this card: keep every test under
  15.8 GB.

Targets: 2x = ~60 tok/s = a ~370 ms cycle; 3x = ~90 tok/s = a ~245 ms cycle (at 21.9 tokens per cycle).

## 2. Three levers

| Lever | Today | Target | Factor [E] |
|---|---|---|---|
| A. Overlap draft and verify | draft + verify = 734 ms | ~max(draft, verify) + overhead | up to ~1.6x |
| B. Fewer bytes per verify | 320 ms | ~180-200 ms | ~1.3x |
| C. Cheaper drafting | 414 ms | ~150-200 ms | ~1.4x, and it makes A possible |

No single lever reaches 2x. A and C together are the 2x path; B is what turns 2x into 3x.

## 3. Phase 0: measurements (1-2 days; gates every later phase)

All GPU runs only when the server is idle, with the watchdog paused through `~/.shoehorn/watchdog.pause`, and a VRAM
guard that stops a run above 15.8 GB.

1. **Cycle timeline.** CUDA events (or Nsight Systems) over one production-like verify + draft at ~50K context: GPU
   busy vs PCIe busy vs idle, per phase. Decides how much Phase 2 can win.
2. **Residual entropy (no GPU).** Read the Q4_K residual, histogram the 4-bit codes per block and per tensor type,
   compute the entropy with per-block and per-tensor contexts. Gate for B1: worth building only if the `ffn_*`
   residuals (9.8 of 14.4 GB) code at <= 3.5 bits per weight including scales.
3. **b3 pack on the 5080.** `orca27b-b3.base.gguf` (iq3_xxs, 10.1 GB) + its residual, with `--res-gpu-gb auto`:
   tokens per cycle, draft and verify ms, tok/s on the 13K and 97K test prompts, against the current pack. Its B+R
   KLD vs Q8 is known (0.0048 vs 0.0031, docs/LINUX-PLAN.md). Gate for B2: a net speed gain of >= 15%.
4. **Small draft model.** A ~0.5-2B model with the Qwen3.8 tokenizer, as a pre-drafter for the base: acceptance of
   its chains by the 4-bit base, on the same prompts at temp 1.0 and greedy. Gate for Phase 1: mean accepted chain
   length >= 4 at temp 1.0. If no off-the-shelf model passes, estimate the cost of distilling one onto base outputs
   (the MTP distillation tooling in docs/SPEEDUP-PLAN.md section 2 is a starting point).

### Phase 0 results (2026-10-19)

- **Item 2, residual entropy: B1 dropped.** 3,000 random Q4_K blocks per tensor: the 4-bit codes carry 3.95 bits of
  4 in every tensor type (order-0), 3.95 again conditioned on the sub-block scale. With scales, ~4.39 vs 4.5 bits per
  weight: entropy coding would save ~2% of the verify's bytes. Q4_K already spends its 16 levels evenly. [M]
- **Item 3, b3 pack: B2 fails its gate (>= 15% net gain).** Same build, `gen`, q8_0 KV, 2.80 GB of the b3 residual
  in VRAM with `--res-gpu-gb auto` [M]:

  | Prompt | Current pack | b3 (res auto) |
  |---|---|---|
  | 13K, greedy | 41.6 / 43.5 tok/s, 27.5 tokens/cycle, verify 261-281 ms | 34.5 tok/s, 20.7 tokens/cycle, verify 223 ms (res 0: 32.3) |
  | 97K, temp 1.0, seed 11 | 12.2 tok/s, 13.4 tokens/cycle, verify 321 ms | 12.9 tok/s, 12.4 tokens/cycle, verify 284 ms |
  | 97K, temp 1.0, seed 12 | 19.0 tok/s, 18.8 tokens/cycle, verify 323 ms | 12.6 tok/s, 11.8 tokens/cycle, verify 287 ms |
  | 97K, greedy | 15.8 tok/s (earlier run, 20K window) | 20.4 tok/s, 18.4 tokens/cycle, verify 283 ms |

  The verify is consistently ~37-40 ms faster (2.8 GB less over PCIe), but the 3-bit base's drafts are accepted
  less (short prompt: 20.7 vs 27.5 tokens/cycle), which costs more than the verify saves on short prompts (-20%)
  and leaves long prompts within run-to-run noise (12-19 tok/s for the same pack and settings, by seed). b3's only
  remaining role: freeing ~3.6 GB of VRAM if Phase 1 needs it.
- **Item 1, overlap (measured directly with `gen --pipe`, the shadow drafter from M3):** 6K-token prompt, ctx 8192,
  4 seeds at temp 1.0 each [M]:

  | | draft ms/cycle | verify ms/cycle | tokens/cycle | tok/s |
  |---|---|---|---|---|
  | sequential | 592 | 261 | 23.6 | 26.0 |
  | `--pipe` | 302 | 298 | 18.1 | 28.4 (+9%) |

  The overlap hides ~290 ms of drafting per cycle; contention adds ~37 ms (+14%) to the verify, and the pipelined
  drafts are shorter (18.1 vs 23.6 tokens/cycle), which eats most of the gain. With today's draft length kept, the
  same overlap would give ~23.6 / 0.60 s = ~39 tok/s (+50%) [D]. `--pipe` needs more VRAM than the sequential path:
  at ctx 16384 it crossed 15.8 GB and was stopped. Phase 2's work is therefore (a) the shorter pipelined drafts
  (find why: MTP off in the shadow? drafts cut at the verify boundary?) and (b) VRAM for the shadow state.
- **Overlap, follow-up (greedy A/B, ctx 8192, 500 tokens, 3 prompts) [M]: not a win; do not ship `--pipe`.**
  The 4-seed sampled runs above swing about ±15% between identical batches, so this A/B is greedy:

  | Prompt | sequential | `--pipe`, E8_PIPE_MIN=16 | `--pipe`, E8_PIPE_MIN=4 (default) |
  |---|---|---|---|
  | docs prose, 6K | 66.6 tok/s (42.3 tok/cycle) | 62.2 (29.5) | 29.7 (15.8) |
  | Webex task, 1.6K | 55.1 (32.1) | 52.5 (27.9) | 46.1 (22.3) |
  | code, 4.5K | 30.4 (18.7) | 31.7 (19.4) | 34.7 (15.8) |

  The verify runs 260 -> 300-316 ms while the shadow drafts (+15-20% contention, on every cycle), the pre-drafts
  are used only when the whole previous draft was accepted, and drafts made against the verify's deadline are
  shorter. It helps only where cycles are short anyway (code, +4-14%) and loses up to 55% where drafts are long.
  Greedy outputs with `--pipe` match sequential up to a point and then differ (rounding in different verify batch
  sizes flips a near-tie, as noted in M3), so they are not bit-identical runs. The `E8_PIPE_MIN` sweep with
  sampling (4 / 16 / 24 / 64: 23.4 / 27.2 / 22.5 / 27.1 tok/s) is inside the noise.
  What overlap would need to pay: drafting that does not slow the verify (the shadow's GPU work competes with the
  residual matmuls the weight-prefetch patch runs on the GPU), and pre-drafts that survive a partial accept (draft
  from several likely rejection points, as ShadowSpec does with trees).
- **Overlap with the cycle kept long (`E8_PIPE_MIN=64`, greedy) [M]:** docs 60.5 vs 66.6 tok/s sequential, Webex
  task 53.1 vs 55.1, code 32.3 vs 30.4. Cycles stay long (35 vs 29.5 tokens on docs) but the verify still runs at
  306-321 ms instead of ~260.
- **Overlap with CUDA stream priorities (verify streams highest, shadow lowest; experimental ggml-cuda change,
  reverted) [M]:** docs 60.3 vs 66.2, Webex task 53.1 vs 59.1; the verify went 308 -> 290 ms on docs, still well
  above sequential. **The machine hard-reset (Kernel-Power 41, no driver or WHEA event) during the next run (code
  prompt, overlap + priorities).** Cause not established; the change was reverted and not committed, and concurrent
  shadow drafting should not be run on this box again without finding out why.
  Conclusion: the overlap as designed does not pay on this hardware; contention on the verify is not removed by
  priorities.
- **Item 4, drafter: DFlash2 (stock, 2B, Q8_0) fails on this model.** No small Qwen3.8 exists (the family is 27B,
  Flash-Next, 2.4T). DFlash2 (z-lab / Inco AI, trained for Qwen3.8-27B, reads target layers 5/19/33/47/61) was run in
  llama.cpp PR #27342 (`E:\shoehorn\llama-dflash`) against `orca27b.base.gguf` (the 4-bit base, i.e. the level it
  would draft for), 4 requests x 600 tokens, temp 1.0 / top-p 0.95 / top-k 20 [M]:

  | Request | drafted | accepted | tokens per verify |
  |---|---|---|---|
  | prose (explain DESIGN.md) | 1597 | 369 | 2.60 |
  | prose (essay) | 1780 | 343 | 2.33 |
  | code (C++ from spec.h) | 1285 | 415 | 3.24 |
  | code (Python script) | 1545 | 376 | 2.68 |

  Its card reports 4.1-5.5 against stock Qwen3.8-27B Q4_K_M; against the orca fine-tune's base it reaches 2.3-3.2.
  shoehorn's own MTP chain of 6 gives ~4.3 per base check (0.556 kept x 6 + 1, 2026-10-19). Stock DFlash2 would
  draft worse than what runs today; it would need distilling onto the orca base's hidden states first. DSpark
  (1.86B) reports lower acceptance than the MTP head even on stock Qwen3.8 and was not tried.
- Run-to-run noise on the 97K prompt (same pack, different seed: 12.2 vs 19.0 tok/s) means any long-prompt A/B needs
  several seeds per side; single runs decide nothing.

### Where Phase 0 leaves the plan

| Lever | Status | Projected [D] |
|---|---|---|
| B1 residual entropy coding | dropped (3.95 of 4 bits) | - |
| B2 3-bit base | dropped as a speedup (net slower or noise) | - |
| C stock DFlash2 drafter | dropped as-is (worse than MTP on the orca base) | - |
| A overlap (`--pipe` as built) | dropped: greedy A/B -55% to +14% by prompt; the verify slows 15-20% | - |
| A overlap, redesigned | open: needs drafting that does not contend with the verify, and tree pre-drafts | unknown |
| C distilled drafter | open: DFlash2 or the MTP head distilled onto orca base outputs | unknown |

Related work: this design is SubSpec (NeurIPS 2025: a low-bit substitute of the offloaded model drafts for it);
ShadowSpec (ICML 2026) adds drafting during weight loading (Phase 2 here), refining drafts from freshly verified KV,
and deep trees, for +17-54% over SubSpec on an RTX 5090 capped at 8 GB.

## 4. Phase 1: cascade drafting (lever C; largest single gain)

Today each draft round runs MTP proposals and then a base check; a cycle needs many sequential base passes, each
reading 13.7 GB of VRAM. Add a third, cheaper level:

    tiny model (~0.5-2B, VRAM) -> base (4-bit, VRAM) -> B+R (exact verify, residual over PCIe)

- The tiny model proposes long chains (~1-3 ms per token).
- The base checks whole chains in batched passes: a pass costs about the same at 8 or 64 rows [M], so a long chain
  is checked for the price of a short one.
- B+R verifies exactly as today: the output distribution does not change.
- VRAM: ~1-1.5 GB for the tiny model and its KV. Sources: VRAM KV window 20K -> 8K frees ~0.3 GB (a smaller window
  slows prompts that fit it, 10% of decode time today [M]); the b3 base (B2) frees ~3.6 GB.
- Target: draft 414 -> ~180 ms [E]. Alone: ~44 tok/s (1.45x) [E].
- Check: `gen --compare` (greedy identity) and the temp-1.0 acceptance and chi-squared checks already used for
  speculative sampling.

## 5. Phase 2: overlap drafting with the verify (lever A)

Tried before and dropped (docs/DESIGN.md, M3): on 16 GB it fit only without MTP, the verify slowed 330 -> 420 ms from
GPU contention, and continuations were kept in 3-8 of ~19 cycles. What is different now:

- The cascade drafter is mostly tiny-model work, so it needs far less GPU time during the PCIe-bound verify. The
  Phase 0 timeline says how much idle GPU there actually is.
- VRAM for the second drafting state comes from B2 or the smaller window.
- Drafting the next cycle assumes the current drafts are all kept; when the verify rejects some, the speculative work
  is discarded and that cycle drafts again (as `pipe_s` / Phase 2 in docs/MULTISEQ.md describe). Full-accept cycle
  rate is therefore part of the gain and is logged.

Cycle: draft + verify -> about max(draft, verify) + overhead. Target with Phase 1: ~360 ms, ~61 tok/s (2x) [E].

## 6. Phase 3: fewer bytes per verify (lever B)

**B1, lossless: entropy-coded residual.** Store the residual entropy-coded in pinned RAM; send the compressed bytes
over PCIe; decode on the GPU into a VRAM staging ring (VRAM reads at ~850 GB/s vs 57.5 GB/s over PCIe), then run
the existing matmul. Start with `ffn_*`: 68% of the bytes and the least sensitive per GB (section 8). Output is
bit-identical. Gain [E]: 0-35% of the verify, depending on Phase 0 item 2. Risk: decode throughput must stay above
PCIe throughput while the GPU also runs the verify; prototype the decoder alone first.

**B2, near-lossless: the b3 pack.** A 3-bit base (10.1 GB) frees ~3.6 GB of VRAM for residual (`--res-gpu-gb`), so
~25% fewer bytes cross PCIe, and it pays for Phase 1's VRAM. Cost: B+R KLD vs Q8 0.0031 -> 0.0048 [M]. The only change
to the target model in this plan; a switch the user decides on. Risk: the 3-bit base is a weaker mid-level checker
(82% top-1 alone vs 91% for the 4-bit base [M]), so it is kept only if Phase 0 item 3 shows a net gain.

Target with Phases 1-2: verify ~200 ms, cycle ~240 ms, ~90 tok/s (3x) [E].

## 7. Summary

| Setup | Draft ms | Verify ms | Cycle ms | tok/s [E] | Output |
|---|---|---|---|---|---|
| Today [M] | 414 | 320 | 734 | ~30 | - |
| + Phase 1 cascade | ~180 | 320 | ~500 | ~44 | identical distribution |
| + Phase 2 overlap | ~180 | 320 | ~360 | ~61 (2x) | identical distribution |
| + Phase 3 (B1 and/or B2) | ~180 | ~200 | ~240 | ~90 (3x) | identical (B1); KLD 0.0048 vs Q8 (B2) |

Order: Phase 0 -> 1 -> 2 -> 3. Effort [E]: Phase 0 1-2 days; Phase 1 1-2 weeks; Phase 2 ~1 week; B1 1-2 weeks of CUDA
work; B2 a pack/config switch after measurement.

## 8. Reference: residual sensitivity (2026-10-19)

`ppl --kld-base` against full B+R, 13K-token prompt, ctx 512 x 8 chunks, with `E8_RES_SKIP` (a measurement switch in
`src/model/qwen35.cpp`: tensors whose name starts with a listed prefix keep the base alone). [M]

- Base alone vs B+R: KLD 0.0394, top-1 90.9%.
- Per layer (residual of one layer off): 0.0005-0.0031; no hot layers (top 16 hold 35% of the summed layer KLD).
  Layers 0-31 are about twice as sensitive as 32-63 (0.0017 vs 0.0009). Single-layer KLDs are not additive (sum
  0.087 vs 0.039 all together); use them as a ranking.
- Per tensor type (residual of that type off in every layer):

| Type | KLD | Residual GB | KLD per GB |
|---|---|---|---|
| attn_output | 0.0042 | 0.30 | 0.0139 |
| attn_q | 0.0061 | 0.60 | 0.0102 |
| output | 0.0051 | 0.72 | 0.0071 |
| ssm_out | 0.0053 | 0.85 | 0.0062 |
| ffn_down | 0.0134 | 3.26 | 0.0041 |
| attn_gate | 0.0034 | 0.85 | 0.0040 |
| attn_qkv | 0.0048 | 1.42 | 0.0034 |
| ffn_gate | 0.0089 | 3.26 | 0.0027 |
| ffn_up | 0.0082 | 3.26 | 0.0025 |

Consequences: an all-in-VRAM 27B near Q8 quality cannot come from mixing existing formats (the error is spread evenly
and the base already fills VRAM); it would need a better ~4-bit format everywhere (trellis / vector quantization).
A lossy fast mode is possible and measurable: skipping the `ffn_up` residual saves 3.26 GB per verify (~57 ms, ~8%)
at +0.008 KLD. Not part of this plan.

## 9. What was ruled out on 2026-10-19

- Smaller VRAM KV window for its own sake: drafts were no worse at 4-8K than at 20K on a 97K-token chat (greedy:
  16.1 / 18.0 / 20.0 / 15.2 tokens per cycle at 2K / 4K / 8K / 20K), but prompts that fit the window take the faster
  all-VRAM verify (272 vs ~325 ms), so shrinking it only pays when the VRAM buys something (Phase 1).
- Residual in VRAM on its own: ~10 ms per verify at 0.6 GB; no end-to-end change.
- Moving "CPU layers" to the GPU: there are none (see section 1).
