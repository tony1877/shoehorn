# Build environment for the portable ROCm tarball: EL9 (glibc 2.34) + ROCm 7.0 from AMD's repository
# (scripts/package-tarball.sh FLAVOR=rocm). ROCm 7.0: gfx950 (MI350) and RDNA4 (gfx1201) support.
FROM almalinux:9
RUN printf '[ROCm-7.0]\nname=ROCm 7.0\nbaseurl=https://repo.radeon.com/rocm/el9/7.0/main\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key\n' \
        > /etc/yum.repos.d/rocm.repo \
    && dnf install -y -q epel-release dnf-plugins-core && dnf config-manager --set-enabled crb \
    && dnf install -y -q git gcc-c++ gcc-toolset-13 python3-pip patchelf binutils which findutils \
        rocm-hip-sdk hipblas-devel rocblas-devel \
    && pip3 install -q "cmake>=3.28" ninja \
    && dnf clean all
# gcc-toolset-13: ROCm's compiler emits __truncsfhf2 (float -> half), which EL9's GCC 11 runtime lacks (GCC 12+);
# the toolset links it statically, so the binary still runs on the base EL9 runtime
ENV ROCM_PATH=/opt/rocm PATH=/opt/rh/gcc-toolset-13/root/usr/bin:/opt/rocm/bin:$PATH
WORKDIR /src
