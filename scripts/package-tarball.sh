#!/usr/bin/env bash
# Portable Linux tarball (docs/LINUX-PLAN.md, Packaging 2): built in an EL9 container (glibc 2.34, the oldest of the
# supported distributions), with the CUDA runtime, cuBLAS and OpenMP bundled in lib/ (RPATH $ORIGIN/../lib), so the
# host needs only the NVIDIA driver (570+). (ggml's GGML_STATIC would link everything statically, which breaks
# loading the driver's libcuda / libnvidia-ml, so the runtime is bundled instead.) A CPU build rides along: bin/shoehorn
# starts it when the NVIDIA driver is missing, so the tarball also runs (slowly) on machines without one.
#
#   scripts/package-tarball.sh            # -> dist/shoehorn-<ver>-linux-x86_64-cuda.tar.gz
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
img=shoehorn:tarball-cuda
docker=${DOCKER:-docker}  # e.g. DOCKER="sudo docker"
$docker build -q -t $img -f "$root/packaging/docker/tarball-cuda.Dockerfile" "$root/packaging/docker" >/dev/null
mkdir -p "$root/dist" "${BUILD_DIR:=$HOME/sh-build-tarball}" "$BUILD_DIR-cpu"
$docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$root:/src" -v "$BUILD_DIR:/build" -v "$BUILD_DIR-cpu:/build-cpu" \
    -v "$root/dist:/dist" $img bash -c '
set -euo pipefail
git config --global --add safe.directory "*"
bash scripts/build.sh --backend cuda --release --build-dir /build
bash scripts/build.sh --backend cpu --release --build-dir /build-cpu > /dev/null
ver=$(sed -n "s/^project(shoehorn VERSION \([0-9.]*\).*/\1/p" CMakeLists.txt)
name=shoehorn-$ver-linux-x86_64-cuda
st=/tmp/$name; rm -rf $st; mkdir -p $st/bin $st/lib $st/libexec
cp /build/bin/shoehorn $st/libexec/shoehorn-cuda
cp /build-cpu/bin/shoehorn $st/libexec/shoehorn-cpu
# bin/shoehorn: the CUDA build when the NVIDIA driver (libcuda.so.1) is there, else the CPU build
cat > $st/bin/shoehorn <<"SH"
#!/bin/sh
d=$(dirname "$(readlink -f "$0")")/../libexec
if "$d/shoehorn-cuda" --version >/dev/null 2>&1; then exec "$d/shoehorn-cuda" "$@"; fi
echo "shoehorn: no NVIDIA driver (libcuda.so.1) found, running the CPU build" >&2
exec "$d/shoehorn-cpu" "$@"
SH
chmod +x $st/bin/shoehorn
for l in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do cp -L /usr/local/cuda/lib64/$l $st/lib/; done
cp -L "$(g++ -print-file-name=libgomp.so.1)" $st/lib/
patchelf --set-rpath "\$ORIGIN/../lib" $st/libexec/shoehorn-cuda $st/libexec/shoehorn-cpu
cp -r packaging/systemd $st/; cp docs/linux.md $st/README-linux.md; cp LICENSE* $st/ 2>/dev/null || true
echo "glibc needed: $(objdump -T $st/libexec/shoehorn-* $st/lib/*.so* | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
ldd $st/libexec/shoehorn-cuda | grep -v "linux-vdso\|ld-linux" | sed "s/^/  /"
tar -C /tmp -czf /dist/$name.tar.gz $name
ls -la /dist/$name.tar.gz
'
