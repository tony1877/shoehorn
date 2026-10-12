#!/usr/bin/env bash
# Portable Linux tarballs (docs/LINUX-PLAN.md, Packaging 2), built in EL9 containers (glibc 2.34, the oldest of the
# supported distributions). A CPU build rides along in both: bin/shoehorn is a launcher that runs the GPU build when
# it can load and the CPU build otherwise, so the tarball also starts (slowly) on machines without the GPU stack.
#
#   FLAVOR=cuda: the CUDA runtime, cuBLAS and OpenMP bundled in lib/ (RPATH $ORIGIN/../lib); the host needs only the
#     NVIDIA driver (570+). (ggml's GGML_STATIC links everything statically, which breaks loading the driver's libcuda
#     and libnvidia-ml, so the runtime is bundled instead.)
#   FLAVOR=rocm: needs an installed ROCm 7.x (/opt/rocm); its libraries are several GB and tied to the kernel driver,
#     so they are not bundled. Targets gfx1100/1101 (RDNA3), gfx1201 (RDNA4), gfx942/950 (MI300/MI350).
#
#   FLAVOR=cuda scripts/package-tarball.sh   # -> dist/shoehorn-<ver>-linux-x86_64-cuda.tar.gz (default)
#   FLAVOR=rocm scripts/package-tarball.sh   # -> dist/shoehorn-<ver>-linux-x86_64-rocm.tar.gz
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
flavor=${FLAVOR:-cuda}
case "$flavor" in cuda) backend=cuda ;; rocm) backend=hip ;; *) echo "FLAVOR must be cuda or rocm" >&2; exit 1 ;; esac
img=shoehorn:tarball-$flavor
docker=${DOCKER:-docker}  # e.g. DOCKER="sudo docker"
$docker build -q -t $img -f "$root/packaging/docker/tarball-$flavor.Dockerfile" "$root/packaging/docker" >/dev/null
mkdir -p "$root/dist" "${BUILD_DIR:=$HOME/sh-build-tarball-$flavor}" "$BUILD_DIR-cpu"
$docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e FLAVOR=$flavor -e BACKEND=$backend \
    -v "$root:/src" -v "$BUILD_DIR:/build" -v "$BUILD_DIR-cpu:/build-cpu" -v "$root/dist:/dist" $img bash -c '
set -euo pipefail
git config --global --add safe.directory "*"
bash scripts/build.sh --backend $BACKEND --release --build-dir /build
bash scripts/build.sh --backend cpu --release --build-dir /build-cpu > /dev/null
ver=$(sed -n "s/^project(shoehorn VERSION \([0-9.]*\).*/\1/p" CMakeLists.txt)
name=shoehorn-$ver-linux-x86_64-$FLAVOR
st=/tmp/$name; rm -rf $st; mkdir -p $st/bin $st/lib $st/libexec
cp /build/bin/shoehorn $st/libexec/shoehorn-$FLAVOR
cp /build-cpu/bin/shoehorn $st/libexec/shoehorn-cpu
if [ $FLAVOR = cuda ]; then need="the NVIDIA driver (libcuda.so.1)"; else need="ROCm 7 (/opt/rocm)"; fi
# bin/shoehorn: the GPU build when it loads, else the CPU build
cat > $st/bin/shoehorn <<SH
#!/bin/sh
d=\$(dirname "\$(readlink -f "\$0")")/../libexec
if "\$d/shoehorn-$FLAVOR" --version >/dev/null 2>&1; then exec "\$d/shoehorn-$FLAVOR" "\$@"; fi
echo "shoehorn: $need not found, running the CPU build" >&2
exec "\$d/shoehorn-cpu" "\$@"
SH
chmod +x $st/bin/shoehorn
cp -L "$(g++ -print-file-name=libgomp.so.1)" $st/lib/
if [ $FLAVOR = cuda ]; then
    for l in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do cp -L /usr/local/cuda/lib64/$l $st/lib/; done
    patchelf --set-rpath "\$ORIGIN/../lib" $st/libexec/shoehorn-cuda $st/libexec/shoehorn-cpu
else
    patchelf --set-rpath "\$ORIGIN/../lib:/opt/rocm/lib" $st/libexec/shoehorn-rocm
    patchelf --set-rpath "\$ORIGIN/../lib" $st/libexec/shoehorn-cpu
fi
cp -r packaging/systemd $st/; cp docs/linux.md $st/README-linux.md; cp LICENSE* $st/ 2>/dev/null || true
echo "glibc needed: $(objdump -T $st/libexec/shoehorn-* $st/lib/*.so* | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
ldd $st/libexec/shoehorn-$FLAVOR | grep -v "linux-vdso\|ld-linux" | sed "s/^/  /"
tar -C /tmp -czf /dist/$name.tar.gz $name
ls -la /dist/$name.tar.gz
'
