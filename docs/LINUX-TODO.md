# Linux port: Claude's work list

Only what Claude can do itself: on this PC (WSL Debian 13, Docker, the RTX 5080 when the Windows server is idle) or
on RunPod within the balance ($14.26 on 2026-10-11; budget below keeps ~$4 in reserve). In order. Costs are
estimates at today's prices (community cloud where it has stock). Full plan: [LINUX-PLAN.md](LINUX-PLAN.md).

| # | Task | Where | Est. cost |
|---|---|---|---|
| 1 | Phase 1, r3: pod `ppl --save-kld` + greedy `gen --compare`; KLD vs Windows here (gate ≤ 1e-5, same tokens) | PRO 6000 MIG pod (running) + PC | ~$0.40 |
| 2 | Phase 1, r4: same pod with `PACK=orca27b-r4`; Windows r4 reference on the 5080 | same pod + PC | ~$0.40 |
| 3 | Delete the pod; record results in LINUX-PLAN.md | — | — |
| 4 | Phase 1, other architectures: sm_89 (RTX 4000 Ada / 4090) and sm_86 (A5000 / 3090) with r3; gate KLD ≤ 1e-3 | 2 community pods, ~30 min each | ~$0.30 |
| 5 | Phase 2 on a pod: `serve` with base + residual, every page and `/stats` with real values, a chat request, Settings restart, a pack and a download job started and cancelled | sm_89 pod from step 4 (extra ~30 min) | ~$0.15 |
| 6 | Phase 0 rest: GPU builds with Clang as host compiler (Docker); check the CI workflow's GitHub runs and fix failures | PC + GitHub Actions (free) | $0 |
| 7 | Phase 6 rest: systemd units in WSL (systemd enabled): enable, kill → restart, Settings restart (exit 75), `wsl --shutdown` as the reboot test | PC | $0 |
| 8 | Phase 3, AMD: HIP build + `ppl` / `gen` on an MI300X (gfx942), gate KLD ≤ 1e-3 vs NVIDIA; check flash attention with q8_0/q4_0 KV, graph capture, `hipHostMalloc`, PCIe rate | MI300X community pod, ~1.5 h | ~$0.75 (secure: ~$3.60) |
| 9 | Phase 3: AMD sensors through `libamd_smi` (code), checked on the same MI300X pod | PC + step 8 pod | (in 8) |
| 10 | Phase 5: portable CUDA tarball built in an AlmaLinux 9 container; `shoehorn:cuda` runtime image | PC (Docker) | $0 |
| 11 | Phase 5: distribution matrix: the tarball on pods running ubuntu 22.04/24.04, debian 12/13, fedora, almalinux 9/10, archlinux images, `selftest` + short `gen` on a small model | 8 short A4000 community pods | ~$0.50 |
| 12 | Phase 5: `.deb` and `.rpm` from the tarball (fpm or native tools), installed in the matrix images | PC (Docker) | $0 |
| 13 | Phase 4: fit profiles: pick base/residual from free VRAM (code); pack a 3-bit base (`--base iq3_s`) and measure its KLD against q8 (`E:/shoehorn/q8.kld`) on the 5080 | PC (pack on CPU, KLD on the 5080 when idle) | $0 |
| 14 | Phase 4: Models page shows which profile fits the GPU | PC | $0 |
| 15 | Phase 7: Windows vs Linux speed on an RTX 5080 pod: `shoehorn bench`, `bench_prose.py`; GPU-bound parts compared, verify reported separately; numbers in `bench/results/` | RTX 5080 community pod, ~1 h | ~$0.40 |

Total ≈ $3.30 (community). Before each pod: check the balance and stock; each pod is deleted when its job reports
done or failed, and no pod runs unwatched.

Not on this list (needs something Claude does not have): consumer AMD RX 7900 / 9070 hardware, a bare-metal Linux
reboot test with real sensors, publishing to the AUR or a package repository, Intel Arc.
