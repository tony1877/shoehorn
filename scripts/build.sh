#!/usr/bin/env bash
# Builds shoehorn on Linux: GCC or Clang + CMake/Ninja, Release, with the CUDA, HIP (ROCm) or CPU backend.
#
#   scripts/build.sh                          # CUDA for the GPU in this machine (native)
#   scripts/build.sh --backend hip            # ROCm, for the GPU in this machine
#   scripts/build.sh --backend cpu            # CPU-only
#   scripts/build.sh --release                # CUDA 86-real;89-real;120a-real, no -march=native
#   scripts/build.sh --arch "89-real" --clean # one architecture, fresh build directory
#   scripts/build.sh --cc clang               # Clang instead of GCC for the C/C++ sources
#   scripts/build.sh --build-dir ~/sh-build   # build directory (default: ./build)
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build="$root/build"
backend=cuda
arch=""
clean=0
release=0
cc=gcc

fail() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)   backend=$2; shift 2 ;;
        --arch)      arch=$2; shift 2 ;;
        --build-dir) build=$2; shift 2 ;;
        --cc)        cc=$2; shift 2 ;;
        --clean)     clean=1; shift ;;
        --release)   release=1; shift ;;
        -h|--help)   sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           fail "unknown option $1 (see --help)" ;;
    esac
done
case "$backend" in cuda|hip|cpu) ;; *) fail "--backend must be cuda, hip or cpu" ;; esac
case "$cc" in
    gcc)   cc_bin=gcc;   cxx_bin=g++ ;;
    clang) cc_bin=clang; cxx_bin=clang++ ;;
    *)     fail "--cc must be gcc or clang" ;;
esac

# 1. Tools.
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 not found. $2"; }
need cmake    "Install CMake 3.24 or newer (apt install cmake / dnf install cmake)."
need ninja    "Install Ninja (apt install ninja-build / dnf install ninja-build)."
need git      "Install git."
need "$cc_bin"  "Install $cc (apt install build-essential, or clang)."
need "$cxx_bin" "Install $cxx_bin."

cmake_args=(-G Ninja -DCMAKE_BUILD_TYPE=Release
            "-DCMAKE_C_COMPILER=$cc_bin" "-DCMAKE_CXX_COMPILER=$cxx_bin")

if [ "$backend" = cuda ]; then
    if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi
    need nvcc "Install the CUDA Toolkit 12.8 or newer (sm_120 support), or build with --backend cpu."
    [ -n "$arch" ] || { [ $release = 1 ] && arch='86-real;89-real;120a-real' || arch=native; }
    cmake_args+=(-DSHOEHORN_CUDA=ON -DSHOEHORN_HIP=OFF "-DCMAKE_CUDA_ARCHITECTURES=$arch")
    # nvcc accepts host GCC only up to a version that depends on the toolkit (CUDA 12.8: GCC 14). When the default
    # GCC is newer, use the newest accepted one that is installed.
    cuda_ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
    max_gcc=13; [ "${cuda_ver:-0}" -ge 128 ] && max_gcc=14; [ "${cuda_ver:-0}" -ge 130 ] && max_gcc=15
    gcc_major=$(gcc -dumpversion | cut -d. -f1)
    if [ "$gcc_major" -gt "$max_gcc" ]; then
        host=""
        for v in $(seq "$max_gcc" -1 11); do
            command -v "g++-$v" >/dev/null 2>&1 && { host="g++-$v"; break; }
        done
        [ -n "$host" ] || fail "GCC $gcc_major is newer than nvcc accepts (<= $max_gcc). Install g++-$max_gcc."
        cmake_args+=("-DCMAKE_CUDA_HOST_COMPILER=$host")
        echo "nvcc host compiler: $host"
    fi
elif [ "$backend" = hip ]; then
    rocm=${ROCM_PATH:-/opt/rocm}
    [ -x "$rocm/bin/hipconfig" ] || fail "ROCm not found at $rocm (set ROCM_PATH). Install ROCm 6.2 or newer."
    if [ -z "$arch" ]; then
        if [ $release = 1 ]; then
            arch='gfx1100;gfx1101;gfx1201;gfx942;gfx950'
        else
            arch=$("$rocm/bin/rocminfo" 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
            [ -n "$arch" ] || fail "no AMD GPU found by rocminfo; pass --arch gfxNNNN"
        fi
    fi
    export HIPCXX="$rocm/llvm/bin/clang"
    export HIP_PATH="$rocm"
    cmake_args+=(-DSHOEHORN_CUDA=OFF -DSHOEHORN_HIP=ON "-DGPU_TARGETS=$arch" "-DAMDGPU_TARGETS=$arch")
else
    cmake_args+=(-DSHOEHORN_CUDA=OFF -DSHOEHORN_HIP=OFF)
fi
# Release binaries run on any x86-64-v3 CPU (AVX2/FMA, Haswell / Zen 1 on); local builds use this CPU's full ISA.
if [ $release = 1 ]; then
    cmake_args+=(-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON)
else
    cmake_args+=(-DGGML_NATIVE=ON)
fi

# 2. ggml submodule.
if [ ! -f "$root/third_party/llama.cpp/ggml/CMakeLists.txt" ]; then
    git -C "$root" submodule update --init --depth 1 || fail "git submodule update failed"
fi

# 3. shoehorn's ggml patches (patches/ggml-*.patch), applied once: `git apply --check` fails when already applied.
# --ignore-whitespace: a Windows checkout used from WSL has CRLF files.
for p in "$root"/patches/ggml-*.patch; do
    [ -e "$p" ] || continue
    if git -C "$root/third_party/llama.cpp" apply --ignore-whitespace --check "$p" 2>/dev/null; then
        git -C "$root/third_party/llama.cpp" apply --ignore-whitespace "$p" || fail "could not apply $(basename "$p")"
        echo "applied $(basename "$p")"
    elif ! git -C "$root/third_party/llama.cpp" apply --ignore-whitespace --check -R "$p" 2>/dev/null; then
        fail "$(basename "$p") neither applies nor is applied (submodule at an unexpected commit?)"
    fi
done

# 4. Configure and build. The first GPU build compiles ggml's kernels and takes several minutes.
[ $clean = 1 ] && rm -rf "$build"
cmake -S "$root" -B "$build" "${cmake_args[@]}" || fail "cmake configure failed (see above)"
cmake --build "$build" --config Release || fail "build failed (see above)"

echo "Built: $build/bin/shoehorn"
