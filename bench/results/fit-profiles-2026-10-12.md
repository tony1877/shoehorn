# GPU fit profiles, speed per card size (2026-10-12)

`scripts/runpod/fit_speed.sh` on RunPod pro cards (RunPod's GeForce hosts cannot initialize CUDA, see LINUX-PLAN.md).
Linux CUDA 12.8 build, prompt `linuxtest/prompt.ids` (200 tokens), 256 generated tokens.

| Profile | Card (VRAM, memory BW, PCIe) | Pack | Residual | Greedy decode | Sampled (3 seeds) | Verify / cycle |
|---|---|---|---|---|---|---|
| 24 GB | RTX A5000 (24 GB, 768 GB/s, 4.0) | r4 (iq4_xs + q4_K) | part in VRAM (`--res-gpu-gb auto`) | 13.5 tok/s | 11.8 tok/s | 309 ms |
| 16 GB | RTX 4000 Ada (20 GB, 360 GB/s, 4.0) | r3 (iq4_xs + q3_K) | streamed from RAM | 10.1 tok/s | 8.7 tok/s | 428 ms |
| 16 GB | RTX 5080 (16 GB, 960 GB/s, 5.0), Windows | r3 | streamed from RAM | 20.2 tok/s | 17.7 tok/s | 197 ms |
| 12 GB | (no RTX A2000 available on RunPod; quality measured: KLD 0.0048 vs Q8) | b3 (iq3_xxs + q4_K) | streamed | - | - | - |

The verify streams the residual over PCIe: on PCIe 4.0 it takes about twice the RTX 5080's PCIe 5.0 time, as the plan
predicted. Keeping part of the residual in VRAM (24 GB card) cuts the verify by ~30% despite the larger q4_K residual.
