# shoehorn:rocm-dev: toolchain for building shoehorn with the HIP (ROCm) backend.
# ROCm 7.0 is the first with gfx950 (MI350); it also covers gfx1100/1101 (RDNA3), gfx1201 (RDNA4) and gfx942.
FROM rocm/dev-ubuntu-24.04:7.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build git g++ ca-certificates hipblas-dev rocblas-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
