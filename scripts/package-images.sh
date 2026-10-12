#!/usr/bin/env bash
# shoehorn:cuda and shoehorn:rocm runtime images from the tarballs in dist/ (scripts/package-tarball.sh), see
# packaging/docker/runtime.Dockerfile.
#
#   scripts/package-images.sh            # both flavors whose tarball exists
#   DOCKER="sudo docker" scripts/package-images.sh cuda
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
docker=${DOCKER:-docker}
ver=$(sed -n 's/^project(shoehorn VERSION \([0-9.]*\).*/\1/p' "$root/CMakeLists.txt")
for flavor in ${@:-cuda rocm}; do
    tgz=shoehorn-$ver-linux-x86_64-$flavor.tar.gz
    if [ ! -f "$root/dist/$tgz" ]; then echo "skip $flavor: dist/$tgz missing (scripts/package-tarball.sh)"; continue; fi
    $docker build -t "shoehorn:$flavor" -t "shoehorn:$ver-$flavor" --build-arg FLAVOR=$flavor --build-arg "TARBALL=$tgz" \
        -f "$root/packaging/docker/runtime.Dockerfile" "$root/dist"
    echo "built shoehorn:$flavor ($($docker image inspect -f '{{.Size}}' shoehorn:$flavor | awk '{printf "%.1f GB", $1/1e9}'))"
done
