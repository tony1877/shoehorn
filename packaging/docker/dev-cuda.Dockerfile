# shoehorn:cuda-dev: toolchain for building shoehorn with the CUDA backend (CI compile jobs, release builds).
# CUDA 12.8 is the oldest toolkit with sm_120 (RTX 50xx); its driver floor (570) gives the widest reach.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build git g++ clang ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
