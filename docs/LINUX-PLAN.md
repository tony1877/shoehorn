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
| Direct CUDA calls outside ggml | None (checked 2026-10-11): `bench_pcie` and the VRAM queries go through ggml's backend API, and the `ggml_backend_cuda_e8_alloc_mapped` hook is plain C in `ggml-cuda.cu`, which the HIP build compiles too | No `gpu::` shim needed |
| ggml patches (`ggml-weight-prefetch`, `ggml-zero-copy`) | CUDA only | All but one CUDA call they use has a HIP mapping in ggml's `vendors/hip.h`; the exception, `__ldcs` in the zero-copy patch's SM copy kernel, is now a plain load under HIP (`E8_LDCS`). The HIP build also needs position-independent code (set by `SHOEHORN_HIP`) |
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
3. ~~`gpu::` shim~~: not needed, see the survey (ggml's backend API already is the shim).
4. CI (GitHub Actions): CPU build and tests on `ubuntu-24.04` / `ubuntu-22.04`; compile-only CUDA and ROCm jobs in
   the `-dev` images.

Gate: clean CPU, CUDA and HIP builds with GCC and Clang; `selftest` passes on CPU.

Progress (2026-10-11): `scripts/build.sh`, `SHOEHORN_HIP`, `packaging/docker/dev-{cuda,rocm}.Dockerfile` and
`.github/workflows/linux.yml` added. CPU builds clean with GCC 14 and Clang 19 (Debian 13 under WSL), and
`selftest --kernel` passes. Fixes: the AVX-512 Q4_K kernel's lambdas became `target`-attributed functions (GCC does
not give lambdas the enclosing function's ISA); `std::filesystem::u8path` (deprecated in C++20) replaced by
`e8::u8path` (`src/util/path.h`). Phase 1 items 1-2 done early: `moe_cpu` prefetches with `madvise(MADV_WILLNEED)`;
`os_lock` raises `RLIMIT_MEMLOCK` to the hard limit and says once how to raise it when `mlock` is refused (CUDA's
pinned allocations are not bound by it; the half-of-RAM cap is Windows-only and only `bench_pcie` applies it).
The CUDA release build (`86-real;89-real;120a-real`, CUDA 12.8.1, GCC 13) compiles in `shoehorn:cuda-dev`, and
the binary starts against WSL's NVIDIA driver and passes `selftest --kernel` (a CPU-backend test: no GPU
correctness check yet, that is item 3); `GGML_CUDA_NCCL` is off so the binary does not
depend on NCCL. The HIP release build (`gfx1100;gfx1101;gfx1201;gfx942;gfx950`, ROCm 7.0) compiles and links in
`shoehorn:rocm-dev` (not run: no AMD GPU here). Not done for the gate yet: GPU builds with Clang as the host
compiler, and the CI workflow's first run on GitHub. The MSVC build still compiles and passes `selftest --kernel` after these changes.

### Phase 1: correct on NVIDIA (4-6 days)

1. `moe_cpu`: `madvise(MADV_WILLNEED)`.
2. Pinned memory: Linux budget, `RLIMIT_MEMLOCK` handling, one clear message when locking fails.
3. On RunPod: RTX 5080 (same GPU as the Windows machine), RTX 4000 Ada or 4090 (sm_89), A4000 or 3080 Ti (sm_86):
   `selftest`, `gen`, `ppl`, `serve`, with the base alone and with base + residual, on a small pack (Qwen3.5-2B,
   fast) and the 27B pack.
4. Cross-OS check on the same GGUF: `ppl --save-kld` on Windows, `--kld-base` on Linux.

Gate: KLD ≤ 1e-3 against Windows (revised 2026-10-11 from ≤ 1e-5, see below) and across architectures; greedy
`gen` output identical up to near-ties.

Progress (2026-10-11): `pack` on Linux gives the Windows result. `scripts/runpod/pack.sh` built shoehorn on a RunPod
RTX PRO 6000 (MIG 2g.48gb, CUDA 12.8, GCC 13) and packed orcarouter/Qwen3.8-27B-Uncensored with imatrix-orca27b:
all 866 tensors of the r3 base are byte-identical to E:/shoehorn/orca27b-r3.base.gguf (`tools/gguf_diff.py`); only
`shoehorn.pack.id` (random) and `shoehorn.pack.source` (folder name) differ. The q3_K (r3) and q4_K (r4, base+residual
rel. RMSE 0.0051 vs 0.0105) packs are in the private HF repo Tony1877/Qwen3.8-27B-Uncensored-shoehorn, ready for the
`gen` / `ppl` checks. Note: `pack` copies the template's metadata, so a template that is itself an older pack brings
stale `eightfer.pack.*` keys (the Windows r3 says `residual_type: q4_K` there); stripping them in `pack` is pending.
Cross-OS check (2026-10-11, RTX PRO 6000 MIG pod vs the RTX 5080 here, both sm_120; same packs, 8 chunks of 512):

| Pack | PPL Windows | PPL Linux | Mean KLD | Same top-1 | Greedy `gen` (128 tokens) |
|---|---|---|---|---|---|
| r3 (q3_K residual) | 2.7637 | 2.7609 | 0.00048 | 99.0% | identical, including the spec/plain split and its logits |
| r4 (q4_K residual) | 2.7715 | 2.7702 | 0.00042 | 99.2% | same for 95 tokens, then a near-tie flips (logits within 0.07) |

The ≤ 1e-5 gate is not met, and is not reachable without matching compilers: part of the work stays on the CPU
(embeddings, output, residual parts), and ggml selects its CPU kernels by compile-time macros that MSVC and GCC set
differently. Thread count is not the cause (Windows at 8 vs 16 threads: KLD 0, 100% same top-1), nor FMA contraction
(`-ffp-contract=off` left Linux's PPL unchanged). The cross-OS gate becomes the cross-architecture one, KLD ≤ 1e-3.
RunPod's GeForce hosts (2026-10-11): a minimal CUDA program (`scripts/runpod/cuda_probe.sh`) gets `cuInit` = 999
(unknown error) on an RTX 5080 and an RTX 4090 host (driver 580, open kernel module), with or without the compat
libraries, while the device nodes (including nvidia-uvm) and a matching libcuda are present. The nodes are owned by
nobody:nogroup and NVIDIA_VISIBLE_DEVICES=void: a user-namespaced container with the GPU passed in by hand, which is
RunPod's setup, not shoehorn's. Use pro / data-center cards there (A4000, L4, PRO 6000, MI300X all worked).
RunPod notes: request `allowedCudaVersions` >= 12.8 for `nvidia/cuda:12.8` images: on hosts with an older driver the
image loads its forward-compatibility libraries, which GeForce cards reject ("failed to initialize CUDA: unknown
error", two RTX 4090 hosts). The plain `nvidia/cuda` image has no curl (start with `apt-get install curl`); the REST API does not
accept MIG GPU types, GraphQL `podFindAndDeployOnDemand` does.

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

Pod check (2026-10-11, L4, `scripts/runpod/phase2.sh`): `serve` with r3 answers every page and `/stats` (200);
NVML gives GPU temperature, power, VRAM and PCIe rate; NVMe temperatures with drive names and k10temp CPU sensors
(Tctl, CCDs) come from hwmon; a download job runs and is cancelled (no curl left); a pack job with a bad source is
refused with a message; the Settings restart re-execs into a new process with the new flag and reloads the model.
Not covered on a pod: a pack that runs to completion through the Models page (the RunPod pack job ran `pack` directly).

Progress (2026-10-11): every `sysinfo` function has a POSIX implementation (item 1; AMD sensors still Phase 3), plus
`board_sensors()` over `/sys/class/hwmon` under LibreHardwareMonitor's kind names, which the server uses when no
`--hw-monitor` is given (the default is off on Linux). The Models page hides Browse without a desktop session.
Checked under WSL (Debian 13): `run_process` exit codes and cancellation, `spawn_detached`, `wait_pid`, `cpu_sample`,
`fixed_drives` / `describe_drive`, NVML through `libnvidia-ml.so.1`, and `serve` answering every page and `/stats`
with live GPU values. Not checkable under WSL (no hwmon, no NVMe): `drive_temps`, `board_sensors`; they need the pod.

### Phase 3: AMD (ROCm/HIP) (5-8 days)

1. Build with `GGML_HIP=ON` and the patched ggml; fix whatever the HIP build of the patches needs.
2. Correctness on a RunPod MI350 (gfx950) with the Phase 1 checks. Gate: KLD ≤ 1e-3 against NVIDIA.
3. Check the features the design depends on, under HIP: IQ4_XS matrix-vector speed, flash attention with q8_0/q4_0
   KV, graph capture, large mapped pinned allocations (`hipHostMalloc`), the residual stream's PCIe rate.
4. AMD sensors through `libamd_smi` (temperature, power, VRAM, PCIe rate).
5. Consumer RDNA3/RDNA4: RunPod offers no consumer AMD cards. Correctness carries over from CDNA where the kernels
   are shared, but speed needs a real RX 7900 / 9070, local or from another provider (open).

Progress (2026-10-11, RunPod MI300X, ROCm 7.0, gfx942): the HIP build with the patched ggml runs the whole path.
r3 against the Windows RTX 5080 run: PPL 2.7675 vs 2.7637, mean KLD 0.00043, 98.9% same top-1 (gate met). Greedy
`gen` with speculation works (6.0 tokens per cycle), also with a q8_0 KV cache (flash attention on quantized KV).
`bench`: IQ4_XS matrix-vector 1.8 TB/s at N=1; 16 GiB pinned (`hipHostMalloc`) in 1.4 s; PCIe 5.0 at 57 GB/s H2D.
AMD sensors read the amdgpu driver's sysfs (hwmon temperature and power, `mem_info_vram_*`, `mem_busy_percent`)
instead of `libamd_smi`, whose structs change between ROCm releases: 44 C, 137 W, VRAM shown on the pod. Limitation:
the first AMD card in /sys/class/drm is used; a container on a multi-GPU server sees every card of the host, so the
reading may belong to another GPU there (one-GPU desktops are fine). Open: consumer RDNA3 / RDNA4 (item 5).

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

Progress (2026-10-11): `gpu_fit()` in the server picks the profile from total VRAM; without `--res-gpu-gb` the
residual goes to VRAM from 22 GB up (a 16 GB card keeps today's default); `/models/state` and the Models page show
the profile. The 12 GB profile measured against Q8 (`q8.kld`, first 40 chunks of 512, RTX 5080):

| Pack | Base in VRAM | Mean KLD vs Q8 | Same top-1 | PPL |
|---|---|---|---|---|
| r4: iq4_xs + q4_K | ~13.9 GB | 0.0020 | 97.7% | 4.002 |
| r3: iq4_xs + q3_K | ~13.9 GB | 0.0031 | 97.4% | 3.996 |
| b3: iq3_xxs + q4_K (12 GB profile) | ~10.1 GB | 0.0048 | 96.9% | 4.017 |
| b3 base alone | ~10.1 GB | 0.236 | 82.0% | 4.480 |

The residual carries the 3-bit base close to the 16 GB setup (1.5x r3's KLD). Whether to offer it is open
question 3. Speed per profile (`bench/results/fit-profiles-2026-10-12.md`, `gen` on RunPod pro cards): 24 GB (RTX A5000,
r4, residual partly in VRAM) 13.5 tok/s greedy; 16 GB (RTX 4000 Ada, r3 streamed) 10.1 tok/s. The 12 GB run waits for an
RTX A2000 on RunPod (none available on 2026-10-12).

### Phase 5: packaging and the distribution matrix (4-6 days)

1. Container images and tarballs built in CI (see [Packaging](#packaging)).
2. Matrix test on RunPod. The host driver comes from the pod, so one cheap GPU pod per vendor tests every
   distribution's userland: run the tarball inside the `ubuntu:22.04/24.04/26.04`, `debian:12/13`, `fedora`,
   `almalinux:9/10` and `archlinux` images, then repeat the Phase 1 checks on the small pack.
3. Then `.deb`, `.rpm` and AUR packages.

Gate: the tarball passes in every tier-1/2 image on NVIDIA and in the ROCm-supported images on AMD.

Progress (2026-10-11): `scripts/package-tarball.sh` builds `shoehorn-0.1.0-linux-x86_64-cuda.tar.gz` (655 MB) in an
EL9 container (CUDA 12.8; `libcudart`, `libcublas`, `libcublasLt`, `libgomp` bundled, RPATH `$ORIGIN/../lib`; needs
glibc 2.34 and the driver's `libcuda`). `scripts/runpod/matrix.sh` ran it on one community pod per image (RTX A4000 /
A4500 / 4000 Ada): libraries resolve, it starts, `selftest --kernel` and `bench --only gpu` pass on all nine:

| Image | glibc | Result |
|---|---|---|
| ubuntu:22.04 | 2.35 | pass |
| ubuntu:24.04 | 2.39 | pass |
| ubuntu:26.04 | 2.43 | pass |
| debian:12 | 2.36 | pass |
| debian:13 | 2.41 | pass |
| fedora:43 | 2.42 | pass |
| almalinux:9 | 2.34 | pass |
| almalinux:10 | 2.39 | pass |
| archlinux | 2.44 | pass |

`scripts/package-distro.sh` turns the tarball into `shoehorn-cuda_0.1.0_amd64.deb` and `shoehorn-cuda-0.1.0-1.x86_64.rpm`
(/opt/shoehorn, /usr/bin/shoehorn, systemd user units in /usr/lib/systemd/user); both install on ubuntu:24.04,
debian:12, fedora:43 and almalinux:9. The CUDA binary links `libcuda.so.1` (ggml's VMM pool), so it cannot
start without the NVIDIA driver; `GGML_CUDA_NO_VMM` removes that link but cost ~4% decode on the RTX 5080 (greedy
19.80 -> 19.05, sampled 18.35 -> 17.65 tok/s). Instead the tarball carries a CPU build too: `bin/shoehorn` is a
launcher that runs `libexec/shoehorn-cuda` when it loads and `libexec/shoehorn-cpu` (with a note) when the driver is
missing; checked in a plain ubuntu container (CPU) and on the RTX 5080 (CUDA, full speed). `FLAVOR=rocm scripts/package-tarball.sh` builds `shoehorn-0.1.0-linux-x86_64-rocm.tar.gz` (74 MB) in AlmaLinux 9 with
ROCm 7.0 from repo.radeon.com and gcc-toolset-13 (ROCm's compiler emits `__truncsfhf2`, which EL9's GCC 11 runtime
lacks); it needs glibc 2.34 and an installed ROCm 7 (/opt/rocm, RPATH), and its launcher falls back to the CPU build
without one (checked on ubuntu:24.04 and almalinux:9). Not yet run on an AMD GPU: no community MI300X on 2026-10-12
(the job: `scripts/runpod/matrix.sh` with `TARBALL=...-rocm.tar.gz` on `rocm/dev-ubuntu-24.04:7.0`). The `.deb` / `.rpm`
were rebuilt from the CUDA tarball with the launcher; with no driver they start the CPU build (ubuntu:24.04, fedora:43).
Open: container images, the AUR.

### Phase 6: running it as a service, and docs (2-3 days)

1. systemd user units in `packaging/systemd/`: `shoehorn.service` (`Restart=on-failure` in place of the watchdog,
   `LimitMEMLOCK=infinity`, logs to the journal) and `dsh.service`.
2. The Settings restart: re-exec through `spawn_detached`, or exit and let systemd restart; one behavior.
3. The shoehorn agent preset's Linux prompt (bash); `scripts/agent.sh`.
4. `docs/linux.md`: per-distribution driver setup (NVIDIA and ROCm), install from image or tarball, enabling the
   units, where logs and models go.

Gate: a reboot brings everything back; killing the server restarts it.

Progress (2026-10-11): `packaging/systemd/{shoehorn,dsh}.service`, `scripts/agent.sh`, the `persona-posix` row in
the shoehorn preset, and `docs/linux.md` written. The Settings restart under systemd (`INVOCATION_ID` set) exits with
status 75 and the unit restarts on it (`RestartForceExitStatus=75`); elsewhere it re-execs as on Windows. dsh's bash tool now has
auto-background too (the pwsh patch ported to `dsh-tool-bash-persistent`, 10 new tests); the Linux preset turns it on
and its prompt describes it as the Windows one does.

### Phase 7: speed (ongoing)

1. Same benchmarks on Windows and on a RunPod RTX 5080: `shoehorn bench`, `bench_prose.py`, the code benchmark.
   The GPU is the same, but the pod's CPU, RAM and PCIe slot are not, so compare the GPU-bound parts (draft time per
   cycle, base pass) and report the verify separately.
2. Per-GPU numbers for every card tested, published in `bench/results/`.

   Progress (2026-10-11): `bench/results/linux-vs-windows-2026-10-11.md`: same RTX 5080, Windows vs Linux (WSL 2): GPU-bound
   parts within 2%, decode at parity (greedy -9%, sampled +5%). RunPod GeForce hosts (4090, 5080) could not
   initialize CUDA for the binary in any image (driver 580, "unknown error"); pro and data-center cards worked.
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
3. Whether the 12 GB profile's quality is acceptable: measured in Phase 4 (KLD 0.0048 vs Q8, 96.9% same top-1; r3 0.0031). Owner's decision.

Estimate: about 6-8 weeks for Phases 0-6; Phase 7 runs alongside.
