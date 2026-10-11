#!/usr/bin/env bash
# Portable Linux tarball (docs/LINUX-PLAN.md, Packaging 2): built in an EL9 container (glibc 2.34, the oldest of the
# supported distributions), with the CUDA runtime, cuBLAS and OpenMP bundled in lib/ (RPATH $ORIGIN/../lib), so the
# host needs only the NVIDIA driver (570+). (ggml's GGML_STATIC would link everything statically, which breaks
# loading the driver's libcuda / libnvidia-ml, so the runtime is bundled instead.)
#
#   scripts/package-tarball.sh            # -> dist/shoehorn-<ver>-linux-x86_64-cuda.tar.gz
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
img=shoehorn:tarball-cuda
docker=${DOCKER:-docker}  # e.g. DOCKER="sudo docker"
$docker build -q -t $img -f "$root/packaging/docker/tarball-cuda.Dockerfile" "$root/packaging/docker" >/dev/null
mkdir -p "$root/dist" "${BUILD_DIR:=$HOME/sh-build-tarball}"
$docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$root:/src" -v "$BUILD_DIR:/build" -v "$root/dist:/dist" $img bash -c '
set -euo pipefail
git config --global --add safe.directory "*"
bash scripts/build.sh --backend cuda --release --build-dir /build
ver=$(sed -n "s/^project(shoehorn VERSION \([0-9.]*\).*/\1/p" CMakeLists.txt)
name=shoehorn-$ver-linux-x86_64-cuda
st=/tmp/$name; rm -rf $st; mkdir -p $st/bin $st/lib
cp /build/bin/shoehorn $st/bin/
for l in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do cp -L /usr/local/cuda/lib64/$l $st/lib/; done
cp -L "$(g++ -print-file-name=libgomp.so.1)" $st/lib/
patchelf --set-rpath "\$ORIGIN/../lib" $st/bin/shoehorn
cp -r packaging/systemd $st/; cp docs/linux.md $st/README-linux.md; cp LICENSE* $st/ 2>/dev/null || true
echo "glibc needed: $(objdump -T $st/bin/shoehorn $st/lib/*.so* | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
ldd $st/bin/shoehorn | grep -v "linux-vdso\|ld-linux" | sed "s/^/  /"
tar -C /tmp -czf /dist/$name.tar.gz $name
ls -la /dist/$name.tar.gz
'
