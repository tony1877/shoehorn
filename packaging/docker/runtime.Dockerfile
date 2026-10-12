# shoehorn:cuda and shoehorn:rocm, the runtime images (docs/LINUX-PLAN.md, Packaging 1), built from the portable tarball
# by scripts/package-images.sh:
#
#   docker run --gpus all -p 8090:8090 -v /path/to/models:/models shoehorn:cuda serve /models/x.base.gguf --res /models/x.res.gguf
#   docker run --device /dev/kfd --device /dev/dri -p 8090:8090 -v /path/to/models:/models shoehorn:rocm serve ...
#
# The CUDA tarball bundles its CUDA runtime (the driver comes from the NVIDIA container toolkit); the ROCm image adds
# the ROCm 7.0 runtime libraries the binary links. Without a usable GPU both run the bundled CPU build.
FROM ubuntu:24.04
ARG FLAVOR=cuda
ARG TARBALL
ARG ROCM_VERSION=7.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && if [ "$FLAVOR" = rocm ]; then \
         mkdir -p /etc/apt/keyrings \
         && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
         && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION noble main" \
              > /etc/apt/sources.list.d/rocm.list \
         && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends hipblas rocblas hipblaslt rocsolver; \
       fi \
    && rm -rf /var/lib/apt/lists/*
COPY ${TARBALL} /tmp/shoehorn.tar.gz
RUN mkdir -p /opt && tar -xzf /tmp/shoehorn.tar.gz -C /opt && rm /tmp/shoehorn.tar.gz \
    && mv /opt/shoehorn-* /opt/shoehorn && ln -s /opt/shoehorn/bin/shoehorn /usr/local/bin/shoehorn \
    && if ldd /opt/shoehorn/libexec/shoehorn-$FLAVOR | grep -v libcuda | grep "not found"; then echo "missing libraries" >&2; exit 1; fi
ENV HOME=/data
VOLUME ["/data", "/models"]
EXPOSE 8090
ENTRYPOINT ["shoehorn"]
CMD ["--help"]
