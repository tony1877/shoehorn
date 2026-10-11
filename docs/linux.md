# shoehorn on Linux

Status (2026-10-11): builds with GCC and Clang (CPU) and with the CUDA and HIP (ROCm) backends; the HIP build is
not yet tested on an AMD GPU. GPU correctness against Windows is the next step (see [LINUX-PLAN.md](LINUX-PLAN.md), Phase 1).

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

## 2. Build

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

## 3. Locked memory

The server keeps the RAM part of a long-context KV locked in memory (`mlock`). Most distributions cap that at 8 MB
per process; shoehorn raises its limit to the hard limit at start, and prints once how to raise it further when that
is not enough. For an interactive shell: `ulimit -l unlimited` (needs a hard limit that allows it, e.g.
`* - memlock unlimited` in `/etc/security/limits.conf`). The systemd unit below sets `LimitMEMLOCK=infinity`.

## 4. Run as a service

```bash
install -Dm755 build/bin/shoehorn ~/.local/bin/shoehorn
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
pnpm (`corepack enable`). Enable it like the server: `systemctl --user enable --now dsh`.

## 5. Sensors

The Sensors page reads the GPU through NVML (`libnvidia-ml.so.1`, part of the driver), drive temperatures from the
NVMe and `drivetemp` hwmon drivers, and CPU, memory and board sensors from `/sys/class/hwmon` (k10temp, coretemp,
nct6775 / it87, spd5118 / jc42). No helper program and no root. Board voltages appear only when the driver labels
them. For SATA drive temperatures load the `drivetemp` module (`echo drivetemp | sudo tee /etc/modules-load.d/drivetemp.conf`).

The Models page's folder and file pickers open `kdialog` or `zenity` when a desktop session is running; on a headless
machine type the path instead.
