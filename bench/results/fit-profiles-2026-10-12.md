# GPU fit profiles, speed per card size (2026-10-12)

`scripts/runpod/fit_speed.sh` on RunPod pro cards (RunPod's GeForce hosts cannot initialize CUDA, see LINUX-PLAN.md).
Linux CUDA 12.8 build, prompt `linuxtest/prompt.ids` (200 tokens), 256 generated tokens.

| Profile | Card (VRAM, memory BW, PCIe) | Pack | Residual | Greedy decode | Sampled (3 seeds) | Verify / cycle |
|---|---|---|---|---|---|---|
| 24 GB | RTX A5000 (24 GB, 768 GB/s, 4.0) | r4 (iq4_xs + q4_K) | part in VRAM (`--res-gpu-gb auto`) | 13.5 tok/s | 11.8 tok/s | 309 ms |
| 16 GB | RTX 4000 Ada (20 GB, 360 GB/s, 4.0) | r3 (iq4_xs + q3_K) | streamed from RAM | 10.1 tok/s | 8.7 tok/s | 428 ms |
| 16 GB | RTX 5080 (16 GB, 960 GB/s, 5.0), Windows | r3 | streamed from RAM | 20.2 tok/s | 17.7 tok/s | 197 ms |
| 8 GB | RTX 5080 (16 GB, 960 GB/s, 5.0), Windows; base 6.93 GB in VRAM | b2 (iq2_xxs + q4_K) | streamed from RAM | 12.4 tok/s | - | 256 ms |
| 12 GB | RTX A4000 (16 GB, 448 GB/s, 4.0); base 10.07 GB in VRAM, so a 12 GB card runs the same | b3 (iq3_xxs + q4_K), packed on the pod | streamed from RAM | 6.1 tok/s | 6.0 tok/s | 612 ms |

The verify streams the residual over PCIe: on PCIe 4.0 it takes about twice the RTX 5080's PCIe 5.0 time, as the plan
predicted. Keeping part of the residual in VRAM (24 GB card) cuts the verify by ~30% despite the larger q4_K residual.
The 12 GB profile drafts less well (4.9 tokens per cycle against ~6, acceptance 0.65: the 3-bit base agrees less often
with base + residual) and streams the larger q4_K residual, so it runs at about 60% of the 16 GB profile on the same
class of card. No RTX A2000 (12 GB) was on RunPod; the A4000 does the same work with 4 GB of VRAM unused.
