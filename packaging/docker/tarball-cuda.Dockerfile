# Build environment for the portable CUDA tarball: EL9 (glibc 2.34) + CUDA 12.8 (scripts/package-tarball.sh).
FROM nvidia/cuda:12.8.1-devel-rockylinux9
RUN dnf install -y -q git gcc-c++ python3-pip patchelf binutils which findutils \
    && pip3 install -q "cmake>=3.28" ninja \
    && dnf clean all
WORKDIR /src
