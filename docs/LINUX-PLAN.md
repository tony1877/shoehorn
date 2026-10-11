# Linux port plan (2026-10-11)

Goal: shoehorn's engine and server run on the major Linux distributions, on NVIDIA RTX 30xx/40xx/50xx and on AMD GPUs,
with the same output as on Windows and comparable speed per GPU. The bundled agent is the DeepSeek harness (dsh),
which already runs on Linux; only shoehorn's own agent preset text needs a Linux variant.

Decisions (2026-10-11): every distribution in the table below; NVIDIA sm_86/sm_89/sm_120 plus AMD; testing on cheap
RunPod Linux pods; packaging as chosen in [Packaging](#packaging).

## Survey: what is Windows-specific today

The engine is closer to Linux than the README's "untested" suggests: model loading, pinned and locked memory, disk
reads, `pack` and the benches already have POSIX branches. The work is around it.

| Area | Linux today | Work |
|---|---|---|
| mmap (`src/util/mmap.cpp`), safetensors, pack, qwen4exp | POSIX branches exist | Verify on 16-55 GB files |
| Pinned / locked memory (`src/model/qwen35.cpp`) | `mlock` branch exists; budget tuned to WDDM's pinned cap (half of RAM) | Linux budget (available RAM minus a reserve); `RLIMIT_MEMLOCK` raised with `setrlimit` or by the service unit |
| Disk reads (`src/io/async_reader.cpp`) | `pread` worker pool (IOCP on Windows) | Works; `io_uring` later for MoE expert streaming |
| MoE CPU kernel (`src/kernels/moe_cpu.cpp`) | `PrefetchVirtualMemory`, no POSIX branch | `madvise(MADV_WILLNEED)` |
| System info (`src/sys/sysinfo.cpp`) | Mostly stubs: GPU sensors, CPU load, drives and their temperatures, `self_exe`, `run_process`, `spawn_detached`, `wait_pid`, `pick_path` | Implement (Phase 2) |
| GPU sensors | NVML loaded the Windows way | NVML via `dlopen("libnvidia-ml.so.1")`; AMD through `libamd_smi` |
| Board / CPU sensors (`tools/hwmon`) | .NET helper on LibreHardwareMonitor's Windows driver | Read `/sys/class/hwmon` in shoehorn; no helper, no root |
| Build (`scripts/build.ps1`) | PowerShell + MSVC; CUDA arch `120a-real`; applies `patches/ggml-*.patch` | `scripts/build.sh` with the same steps, a backend switch (CUDA / HIP / CPU) and an arch list |
| Direct CUDA calls outside ggml | `bench_pcie.cpp` (pinned copy test), `cudaMemGetInfo` in `server.cpp` / `sysinfo.h`, one proc-address hook into the patched ggml (`ggml_backend_cuda_e8_alloc_mapped`) | Route through a small `gpu::` shim with CUDA and HIP builds |
| ggml patches (`ggml-weight-prefetch`, `ggml-zero-copy`) | CUDA only | Every CUDA call they use has a HIP mapping in ggml's `vendors/hip.h` (checked 2026-10-11), so they should compile under `GGML_HIP` unchanged |
| Service (`watchdog.ps1`, `start-detached.ps1`, scheduled task) | Windows only | systemd user units |
| shoehorn's agent preset (`agent-home/.agent-presets/shoehorn`) | System prompt says "Windows. The shell is PowerShell" | Linux variant of the prompt; dsh itself already switches its shell stack by platform |

## Targets

### Distributions

Tier 1 runs in CI on every change; tier 2 before each release; tier 3 is best-effort.

| Tier | Distribution | Default GCC | glibc | GPU driver source | Notes |
|---|---|---|---|---|---|
| 1 | Ubuntu 24.04 LTS | 13 | 2.39 | `ubuntu-drivers`, NVIDIA CUDA repo, AMD ROCm repo | Reference platform |
| 1 | Ubuntu 26.04 LTS | verify | verify | as above | Newest LTS |
| 2 | Ubuntu 22.04 LTS | 11 | 2.35 | as above | |
| 2 | Debian 12 / 13 | 12 / 14 | 2.36 / 2.41 | NVIDIA repo (Debian 12's packaged driver may predate sm_120); ROCm from AMD or Debian | |
| 2 | Fedora (current) | 15 | 2.41+ | RPM Fusion akmod; Fedora's own ROCm packages | Build with an nvcc-supported host compiler |
| 2 | RHEL / Rocky / Alma 9 and 10 | 11 / 14 | 2.34 / 2.39 | NVIDIA and AMD repos | 9 sets the glibc floor (2.34) |
| 3 | Arch | rolling | rolling | `nvidia-open`; `rocm-*` packages | AUR package later |

Secure Boot needs signed kernel modules for the NVIDIA driver; each distribution's install notes cover it.

### GPUs

| Vendor | Architecture | Example cards | Build target | VRAM tier | PCIe |
|---|---|---|---|---|---|
| NVIDIA | Ampere | RTX 3060 / 3080 Ti / 3090, A4000 | `86-real` | 12 / 16 / 24 GB | 4.0 |
| NVIDIA | Ada | RTX 4060 Ti 16 GB / 4080 / 4090, RTX 4000 Ada | `89-real` | 16 / 20 / 24 GB | 4.0 |
| NVIDIA | Blackwell | RTX 5070 Ti / 5080 / 5090 | `120a-real` | 16 / 32 GB | 5.0 |
| AMD | RDNA3 | RX 7800 XT / 7900 XT / 7900 XTX | `gfx1100;gfx1101` | 16 / 20 / 24 GB | 4.0 |
| AMD | RDNA4 | RX 9070 / 9070 XT | `gfx1201` | 16 GB | 5.0 |
| AMD | CDNA (CI only) | MI300X / MI350 | `gfx942;gfx950` | 192-294 GB | 5.0 |

Two hardware facts change speed, not correctness:
- **VRAM** decides what fits next to the 13.7 GB IQ4_XS base. A 24 GB card can also hold part or all of the residual
  (`--res-gpu-gb auto`), which removes most of the PCIe stream from the verify. A 12 GB card cannot hold the IQ4_XS
  base at all.
- **PCIe generation**: the verify streams the residual (11-15 GB) over PCIe. At PCIe 4.0 (about 25 GB/s) that takes
  roughly twice as long as on the 5080's PCIe 5.0 (measured 57 GB/s).

## Packaging

1. **Container images** (primary): `shoehorn:cuda` and `shoehorn:rocm`, plus a `-dev` image with the toolchain.
   They are what RunPod pods run, so every test uses the shipped artifact, and they are the simplest way to run a
   server on Linux (`docker run --gpus all` / `--device /dev/kfd --device /dev/dri`).
2. **Portable tarballs**: `shoehorn-<ver>-linux-x86_64-cuda.tar.gz` and `-rocm.tar.gz`, built in an AlmaLinux 9
   container (glibc 2.34, the oldest in the table), so one binary runs on every tier-1/2 distribution.
   - CUDA: the runtime linked statically, cuBLAS bundled; the host needs only the NVIDIA driver.
   - ROCm: depends on an installed ROCm of the matching major version; ROCm's libraries are several GB and are
     tied to the driver stack, so bundling them is not worth it.
3. **Distribution packages** (`.deb`, `.rpm`, AUR) once the tarballs are proven, generated from the same build.

The CUDA toolkit line for releases is the newest one whose driver requirement the tier-1/2 distributions meet,
favoring driver reach over new toolkit features; the Windows build stays on CUDA 13.

## Phases

### Phase 0: builds on Linux (3-4 days)

1. `scripts/build.sh`: checks the toolchain; applies `patches/ggml-*.patch` once; configures Release with
   `--backend cuda|hip|cpu`, `CMAKE_CUDA_ARCHITECTURES` (default `native`, release list `86-real;89-real;120a-real`)
   or `AMDGPU_TARGETS`, and `CMAKE_CUDA_HOST_COMPILER` when the default GCC is newer than nvcc accepts.
2. Fix GCC/Clang build errors and warnings; replace MSVC-only flags with portable CMake.
3. `gpu::` shim: `mem_get_info`, pinned host alloc/free, async copy; CUDA and HIP builds. Port `bench_pcie` and the
   two `cudaMemGetInfo` callers to it.
4. CI (GitHub Actions): CPU build and tests on `ubuntu-24.04` / `ubuntu-22.04`; compile-only CUDA and ROCm jobs in
   the `-dev` images.

Gate: clean CPU, CUDA and HIP builds with GCC and Clang; `selftest` passes on CPU.

### Phase 1: correct on NVIDIA (4-6 days)

1. `moe_cpu`: `madvise(MADV_WILLNEED)`.
2. Pinned memory: Linux budget, `RLIMIT_MEMLOCK` handling, one clear message when locking fails.
3. On RunPod: RTX 5080 (same GPU as the Windows machine), RTX 4000 Ada or 4090 (sm_89), A4000 or 3080 Ti (sm_86):
   `selftest`, `gen`, `ppl`, `serve`, with the base alone and with base + residual, on a small pack (Qwen3.5-2B,
   fast) and the 27B pack.
4. Cross-OS check on the same GGUF: `ppl --save-kld` on Windows, `--kld-base` on Linux.

Gate: KLD ≤ 1e-5 against Windows on the same GPU architecture; greedy `gen` output identical; ≤ 1e-3 across
architectures (different kernels).

### Phase 2: server and dashboard (4-6 days)

1. `sysinfo.cpp` POSIX implementations:
   - `self_exe` (`/proc/self/exe`), `wait_pid` (`pidfd_open`, falling back to `kill(pid, 0)`);
   - `spawn_detached` and `run_process` with `posix_spawn` (`POSIX_SPAWN_SETSID`), output to the log file,
     cancellation by `SIGTERM` then `SIGKILL`;
   - `cpu_sample` (`/proc/stat`), `fixed_drives` / `describe_drive` (`/proc/mounts`, `/sys/block`), `drive_temps`
     (NVMe hwmon);
   - `gpu_sensors`: NVML via `dlopen`; AMD via `libamd_smi` (Phase 3);
   - `pick_path`: `zenity` or `kdialog` when a display exists, else a typed path in the web UI.
2. Board and CPU sensors from `/sys/class/hwmon` (k10temp, coretemp, nct6775 and similar).
3. Every page (Dashboard, Sensors, Models, Settings), the Settings restart, and a Models-page download and pack job.

Gate: every page shows real values on a RunPod pod; restart, pack and download complete and can be cancelled.

### Phase 3: AMD (ROCm/HIP) (5-8 days)

1. Build with `GGML_HIP=ON` and the patched ggml; fix whatever the HIP build of the patches needs.
2. Correctness on a RunPod MI350 (gfx950) with the Phase 1 checks. Gate: KLD ≤ 1e-3 against NVIDIA.
3. Check the features the design depends on, under HIP: IQ4_XS matrix-vector speed, flash attention with q8_0/q4_0
   KV, graph capture, large mapped pinned allocations (`hipHostMalloc`), the residual stream's PCIe rate.
4. AMD sensors through `libamd_smi` (temperature, power, VRAM, PCIe rate).
5. Consumer RDNA3/RDNA4: RunPod offers no consumer AMD cards. Correctness carries over from CDNA where the kernels
   are shared, but speed needs a real RX 7900 / 9070, local or from another provider (open).

If a HIP gap blocks the speculative path, ggml's Vulkan backend is the fallback for AMD (and Intel): it runs without
the prefetch patches, so the verify would be slower.

### Phase 4: GPU fit profiles (3-5 days)

1. Pick defaults from free VRAM at load:

   | Free VRAM | Base | Residual |
   |---|---|---|
   | 24 GB or more | IQ4_XS | Part or all in VRAM (`--res-gpu-gb auto`) |
   | 16 GB | IQ4_XS | Streamed from RAM (today's setup) |
   | 12 GB | A 3-bit base (`pack --base iq3_s` or similar) | Streamed |
   | Less than 12 GB | Not supported for the 27B | |

2. Measure the 12 GB profile's quality (KLD of base + residual against Q8_0) before offering it; the residual has to
   cover a larger base error.
3. The Models page shows which profile fits the detected GPU.

Gate: each profile loads and runs `bench_prose.py` on a RunPod card of that size; numbers recorded per GPU.

### Phase 5: packaging and the distribution matrix (4-6 days)

1. Container images and tarballs built in CI (see [Packaging](#packaging)).
2. Matrix test on RunPod. The host driver comes from the pod, so one cheap GPU pod per vendor tests every
   distribution's userland: run the tarball inside the `ubuntu:22.04/24.04/26.04`, `debian:12/13`, `fedora`,
   `almalinux:9/10` and `archlinux` images, then repeat the Phase 1 checks on the small pack.
3. Then `.deb`, `.rpm` and AUR packages.

Gate: the tarball passes in every tier-1/2 image on NVIDIA and in the ROCm-supported images on AMD.

### Phase 6: running it as a service, and docs (2-3 days)

1. systemd user units in `packaging/systemd/`: `shoehorn.service` (`Restart=on-failure` in place of the watchdog,
   `LimitMEMLOCK=infinity`, logs to the journal) and `dsh.service`.
2. The Settings restart: re-exec through `spawn_detached`, or exit and let systemd restart; one behavior.
3. The shoehorn agent preset's Linux prompt (bash); `scripts/agent.sh`.
4. `docs/linux.md`: per-distribution driver setup (NVIDIA and ROCm), install from image or tarball, enabling the
   units, where logs and models go.

Gate: a reboot brings everything back; killing the server restarts it.

### Phase 7: speed (ongoing)

1. Same benchmarks on Windows and on a RunPod RTX 5080: `shoehorn bench`, `bench_prose.py`, the code benchmark.
   The GPU is the same, but the pod's CPU, RAM and PCIe slot are not, so compare the GPU-bound parts (draft time per
   cycle, base pass) and report the verify separately.
2. Per-GPU numbers for every card tested, published in `bench/results/`.
3. Linux-only gains after parity: `io_uring` for MoE expert streaming (Flash-Next), transparent huge pages for pinned
   buffers, thread pinning for CPU residual work.

## Test infrastructure on RunPod

- **Model cache**: a network volume in one data center that has the test GPUs, holding the 27B pack (31 GB, or 28 GB
  with the Q3_K residual) and a small Qwen3.5-2B pack, so pods start without a 30 GB download (about $4-5 a month).
- **Runner**: a script that creates a pod from the test image, runs the checks for one phase, saves results to the
  volume and deletes the pod. Every run is bounded by `--terminate-after`.
- **Cost per full run**, at today's prices: distribution matrix (8 images x 15 min on an RTX 4000 Ada, $0.20/h)
  about $0.40; NVIDIA GPU matrix (5080, 4090, 3080 Ti, 30 min each) about $0.50; AMD MI350 for 1 h $0.50. About
  $1.50 per full run.

## Open questions

1. Consumer AMD speed testing (RDNA3/RDNA4): local hardware or another cloud.
2. Intel Arc through Vulkan: out of scope unless asked.
3. Whether the 12 GB profile's quality is acceptable, measured in Phase 4.

Estimate: about 6-8 weeks for Phases 0-6; Phase 7 runs alongside.
