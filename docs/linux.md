# shoehorn on Linux

Status (2026-10-12): runs on NVIDIA (RTX 30xx/40xx/50xx, pro and data-center cards) and AMD (ROCm 7, checked on an
MI300X), with the same output as Windows within KLD 5e-4 (99% the same top-1 token); a pack made on Linux is
byte-identical to the Windows one. Details and per-GPU results: [LINUX-PLAN.md](LINUX-PLAN.md),
[bench/results](../bench/results).

## 1. GPU driver

NVIDIA: driver 570 or newer (needed by CUDA 12.8, the first toolkit with RTX 50xx support).

| Distribution | Install |
|---|---|
| Ubuntu 22.04 / 24.04 / 26.04 | `sudo ubuntu-drivers install` (or `sudo apt install nvidia-driver-570-open`) |
| Debian 12 / 13 | NVIDIA's CUDA repository: `sudo apt install cuda-drivers` (Debian 12's own packaged driver predates RTX 50xx) |
| Fedora | RPM Fusion: `sudo dnf install akmod-nvidia` |
| RHEL / Rocky / Alma 9, 10 | NVIDIA's repository: `sudo dnf install nvidia-driver` (`dnf module install nvidia-driver:open-dkms` on 9) |
| Arch | `sudo pacman -S nvidia-open` |

With Secure Boot on, the kernel module must be signed: Ubuntu's and Fedora's packages handle it (you enroll a key
at the next boot); elsewhere follow the distribution's notes.

AMD: ROCm 6.2 or newer (7.0 for MI350), from AMD's repository or the distribution's own packages (Fedora, Arch,
Debian 13). Add yourself to the `render` and `video` groups.

Check: `nvidia-smi`, or `rocminfo | grep gfx`.

## 2. Install

Pick one; all of them carry a CPU build too, which `shoehorn` falls back to (with a note) when the GPU stack is
missing, so it always starts.

| Package | Needs | Notes |
|---|---|---|
| `shoehorn-<ver>-linux-x86_64-cuda.tar.gz` | glibc 2.34+, NVIDIA driver 570+ | CUDA runtime bundled; unpack anywhere, run `bin/shoehorn` |
| `shoehorn-<ver>-linux-x86_64-rocm.tar.gz` | glibc 2.34+, ROCm 7 in `/opt/rocm` | RDNA3 (gfx1100/1101), RDNA4 (gfx1201), MI300/MI350 |
| `shoehorn-cuda_<ver>_amd64.deb`, `shoehorn-rocm_<ver>_amd64.deb` | Debian 12+, Ubuntu 22.04+ | `sudo apt install ./shoehorn-cuda_*.deb` |
| `shoehorn-cuda-<ver>-1.x86_64.rpm`, `shoehorn-rocm-<ver>-1.x86_64.rpm` | Fedora, RHEL / Alma / Rocky 9+ | `sudo dnf install ./shoehorn-cuda-*.rpm` |
| container `shoehorn:cuda`, `shoehorn:rocm` | Docker + NVIDIA container toolkit / AMD devices | see below |
| Arch: `shoehorn-cuda-bin`, `shoehorn-rocm-bin` | Arch Linux | not on the AUR yet (its registration is paused); build the PKGBUILD from the repo: `cd packaging/aur/shoehorn-cuda-bin && makepkg -si` |

The tarball was checked on Ubuntu 22.04 / 24.04 / 26.04, Debian 12 / 13, Fedora 43, AlmaLinux 9 / 10 and Arch. The
packages install to `/opt/shoehorn`, link `/usr/bin/shoehorn` and put the systemd user units in `/usr/lib/systemd/user`.
They are built from the source tree with `scripts/package-tarball.sh` (`FLAVOR=cuda|rocm`),
`scripts/package-distro.sh <tarball>` and `scripts/package-images.sh`.

Containers:

```bash
docker run --gpus all -p 8090:8090 -v ~/models:/models -v ~/.shoehorn-data:/data shoehorn:cuda     serve /models/x.base.gguf --res /models/x.res.gguf --host 0.0.0.0
docker run --device /dev/kfd --device /dev/dri -p 8090:8090 -v ~/models:/models shoehorn:rocm serve ...
```

## 3. Build from source

```bash
sudo apt install build-essential cmake ninja-build git        # Fedora / RHEL: gcc-c++ cmake ninja-build git
git clone --recursive https://github.com/tony1877/shoehorn && cd shoehorn
scripts/build.sh                    # CUDA, for the GPU in this machine
scripts/build.sh --backend hip      # AMD
scripts/build.sh --backend cpu      # no GPU
```

The CUDA build needs the CUDA Toolkit 12.8 or newer (`nvcc`). When the system GCC is newer than the toolkit accepts
(CUDA 12.8 takes up to GCC 14), `build.sh` picks an installed older `g++-N` as nvcc's host compiler, or tells you
which to install. `--release` builds for every supported GPU (sm_86, sm_89, sm_120) and any x86-64-v3 CPU.

Without installing a toolkit: build in the development image.

```bash
docker build -t shoehorn:cuda-dev -f packaging/docker/dev-cuda.Dockerfile packaging/docker
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/src" shoehorn:cuda-dev bash scripts/build.sh --release
```

The binary is `build/bin/shoehorn`.

## 4. Locked memory

The server keeps the RAM part of a long-context KV locked in memory (`mlock`). Most distributions cap that at 8 MB
per process; shoehorn raises its limit to the hard limit at start, and prints once how to raise it further when that
is not enough. For an interactive shell: `ulimit -l unlimited` (needs a hard limit that allows it, e.g.
`* - memlock unlimited` in `/etc/security/limits.conf`). The systemd unit below sets `LimitMEMLOCK=infinity`, but a
user unit cannot go above its user manager's own limit (64 MB on Debian 13), so raise that once, as root:

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/memlock.conf
sudo systemctl daemon-reload    # then log out and in (or reboot)
```

## 5. Run as a service

```bash
install -Dm755 build/bin/shoehorn ~/.local/bin/shoehorn   # (the packages put it in /usr/bin)
mkdir -p ~/.config/systemd/user && cp packaging/systemd/*.service ~/.config/systemd/user/
# edit ExecStart in ~/.config/systemd/user/shoehorn.service (model list, flags), then:
systemctl --user daemon-reload
systemctl --user enable --now shoehorn
loginctl enable-linger "$USER"      # start at boot without logging in
```

- Logs: `journalctl --user -u shoehorn -f`.
- A crash or kill restarts the server (`Restart=on-failure`). The Settings page's restart exits with status 75, and
  systemd starts the server again with the saved settings.
- Settings and the model list live in `~/.shoehorn/` (`settings.json`, `models.json`), as on Windows.

The agent (`dsh.service`) runs `scripts/agent.sh` from the checkout (set `WorkingDirectory`); it needs Node 24 and
pnpm (`corepack enable`). Enable it like the server: `systemctl --user enable --now dsh`. `scripts/agent.sh` rebuilds
the agent when its sources changed since the last build (after a `git pull`), so updates reach a running install.

## 6. Sensors

The Sensors page reads an NVIDIA GPU through NVML (`libnvidia-ml.so.1`, part of the driver), an AMD GPU from the
amdgpu driver's sysfs (temperature, power, VRAM, memory load), drive temperatures from the
NVMe and `drivetemp` hwmon drivers, and CPU, memory and board sensors from `/sys/class/hwmon` (k10temp, coretemp,
nct6775 / it87, spd5118 / jc42). No helper program and no root. Board voltages appear only when the driver labels
them. For SATA drive temperatures load the `drivetemp` module (`echo drivetemp | sudo tee /etc/modules-load.d/drivetemp.conf`).

The Models page's folder and file pickers open `kdialog` or `zenity` when a desktop session is running; on a headless
machine type the path instead.

## 7. Which GPU, how fast

The server picks a profile from the GPU's VRAM and the Models page shows it: from 24 GB the residual goes partly to
VRAM by default, a 16 GB card streams it from RAM, a 12 GB card needs a 3-bit base (`pack --base iq3_xxs`; KLD 0.0048
against Q8, against 0.0031 for the 16 GB setup), an 8 GB card a 2-bit base (`pack --base iq2_xxs` with an imatrix;
6.9 GB of VRAM, KLD 0.0093, 96.4% the same top-1 token as Q8; needs 32 GB of RAM and a small context). The verify streams the residual over PCIe, so a PCIe 5.0 slot is
worth about twice a PCIe 4.0 one. Measured (27B, 256 tokens, greedy decode):

| GPU | Profile | Decode |
|---|---|---|
| RTX 5080 16 GB (PCIe 5.0), Windows / Linux (WSL) | 16 GB | 20.2 / 18.3 tok/s |
| RTX A5000 24 GB (PCIe 4.0) | 24 GB | 13.5 tok/s |
| RTX 4000 Ada 20 GB (PCIe 4.0) | 16 GB | 10.1 tok/s |
| RTX A4000 16 GB (PCIe 4.0), 3-bit base | 12 GB | 6.1 tok/s |
| RTX 5080 16 GB (PCIe 5.0), 2-bit base | 8 GB | 12.4 tok/s (a laptop GPU on fewer PCIe lanes: expect 3-5) |
| AMD MI300X (ROCm 7) | 24 GB+ (residual on host here) | 9.4 tok/s, 13.6 with a q8_0 KV |

On RunPod, use pro or data-center cards: its GeForce hosts (RTX 4090 / 5080) fail CUDA initialization for any program
(`cuInit` 999), which is the host's container setup, not shoehorn.
