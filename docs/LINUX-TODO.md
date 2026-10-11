# Linux port: to do

Working list for [LINUX-PLAN.md](LINUX-PLAN.md). Updated 2026-10-11.

## Now
- [ ] Phase 1: r3 on the pod (`ppl --save-kld`, greedy `gen --compare`); compare with the Windows reference
      (PPL 2.7637, KLD ≤ 1e-5, same greedy tokens per mode)
- [ ] Phase 1: r4 on the same pod (`PACK=orca27b-r4`); Windows reference for r4 once the download finishes
- [ ] Delete the pod; record the results in LINUX-PLAN.md

## Phase 0 (builds)
- [x] `scripts/build.sh`, CMake HIP option, dev images, CI workflow
- [x] CPU builds clean with GCC and Clang; CUDA and HIP release builds compile
- [ ] GPU builds with Clang as the host compiler
- [ ] First green CI run on GitHub

## Phase 1 (correct on NVIDIA)
- [x] `moe_cpu` madvise prefetch; `RLIMIT_MEMLOCK` handling
- [x] Linux `pack` = Windows `pack` byte for byte (866/866 tensors)
- [x] `pack` drops the template's old `eightfer.*` / `shoehorn.pack.*` keys
- [ ] sm_89 (RTX 4090 / 4000 Ada) and sm_86 (A4000 / 3090) runs; gate KLD ≤ 1e-3 across architectures
- [ ] `serve` with base + residual on a pod

## Phase 2 (server and dashboard)
- [x] POSIX `sysinfo` (processes, restart, CPU, drives, NVML, hwmon sensors, file pickers)
- [ ] On a pod: every page shows real values (drive temps and board sensors untested in WSL)
- [ ] Restart, pack and download jobs complete and can be cancelled

## Phase 3 (AMD)
- [x] HIP build compiles (`__ldcs` gap fixed, PIC)
- [ ] Run on a RunPod MI300X / MI350: `ppl` / `gen`, gate KLD ≤ 1e-3 against NVIDIA
- [ ] Check IQ4_XS speed, flash attention with q8_0/q4_0 KV, graph capture, `hipHostMalloc`, PCIe rate
- [ ] AMD sensors through `libamd_smi`
- [ ] Consumer RX 7900 / 9070 speed (needs hardware outside RunPod)

## Phase 4 (GPU fit profiles)
- [ ] Defaults from free VRAM (24 GB+: residual partly in VRAM; 16 GB: streamed; 12 GB: 3-bit base)
- [ ] Measure the 12 GB profile's quality before offering it
- [ ] Models page shows which profile fits

## Phase 5 (packaging)
- [ ] `shoehorn:cuda` / `shoehorn:rocm` images and portable tarballs (AlmaLinux 9 build) in CI
- [ ] Distribution matrix on one pod: Ubuntu 22.04/24.04/26.04, Debian 12/13, Fedora, Alma 9/10, Arch
- [ ] `.deb`, `.rpm`, AUR

## Phase 6 (service and docs)
- [x] systemd units, Settings restart under systemd, `scripts/agent.sh`, Linux agent prompt, `docs/linux.md`
- [ ] Reboot brings everything back; killing the server restarts it (on a real machine)
- [ ] Optional: port the pwsh auto-background patch to dsh's bash tool

## Phase 7 (speed)
- [ ] Windows vs Linux benchmarks on the same GPU class (`bench`, `bench_prose.py`, code benchmark)
- [ ] Per-GPU numbers in `bench/results/`
- [ ] Linux-only gains: `io_uring` for MoE streaming, huge pages for pinned buffers, thread pinning
