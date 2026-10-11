# Linux vs Windows speed, same GPU (2026-10-11)

RTX 5080 16 GB, Ryzen 7 9800X3D, 32 GB RAM (WSL: 23 GB). Pack orca27b-r3 (iq4_xs base + q3_K residual), prompt
`linuxtest/prompt.ids` (200 tokens), 256 generated tokens. Windows: the MSVC build (`build/bin/shoehorn.exe`). Linux: the
CUDA 12.8 build from `shoehorn:cuda-dev` (GCC 13), run in WSL 2 Debian 13 on the same machine.

| | Windows | Linux (WSL 2) |
|---|---|---|
| `bench` IQ4_XS gemv, N=1 | 872.5 GB/s | 859.4 GB/s |
| `bench` IQ4_XS gemv, N=16 | 728.9 GB/s | 724.8 GB/s |
| `gen --temp 0` decode | 20.15 tok/s | 18.34 tok/s |
| cycles, tokens/cycle | 46, 5.63 | 46, 5.63 |
| draft / verify per cycle | 81.5 / 196.6 ms | 101.8 / 203.6 ms |
| `gen --temp 1 --seed 1 --repeat 3` decode | 17.68 tok/s | 18.52 tok/s |
| draft / verify per cycle | 125.3 / 196.3 ms | 115.1 / 195.4 ms |
| `bench` PCIe pinned 1 GiB H2D | 57.8 GB/s | (not run in WSL) |

GPU-bound parts (gemv, verify) agree within 2%; decode is at parity within run-to-run spread (greedy -9%, sampled
+5%). WSL adds a GPU virtualization layer of its own, so bare-metal Linux may do slightly better. Prefill in WSL
(38 tok/s) is first-touch page faults of weights read through WSL's /mnt/e bridge, not a Linux cost.

Other GPUs measured (Phase 1/3 runs, r3, `gen --temp 0` speculative decode): RTX A4000 9.4 tok/s, L4 8.3 tok/s,
MI300X 9.4 tok/s (13.6 with q8_0 KV), RTX PRO 6000 MIG 2g.48gb 20.5 tok/s. RunPod's GeForce hosts (RTX 4090, RTX 5080,
three machines, driver 580) failed CUDA initialization for this binary ("unknown error") in both the nvidia/cuda image
and a plain ubuntu image, while pro and data-center cards worked; no same-GPU RunPod number for the 5080.
