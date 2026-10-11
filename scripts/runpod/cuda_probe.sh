#!/usr/bin/env bash
# RunPod diagnostic: why CUDA fails to initialize on some hosts ("unknown error"). Prints device nodes, the libcuda in
# use, the container's NVIDIA_* settings and the raw results of cuInit / cudaGetDeviceCount from a tiny program built
# here. Image: nvidia/cuda:12.8.1-devel-ubuntu24.04. Log: linuxtest/probe/$TAG.log in the HF repo.
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
LOG=/tmp/probe.log
{
echo "== $TAG $(date -u)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "== /dev"; ls -la /dev/nvidia* /dev/dri 2>&1
echo "== libcuda"; ldconfig -p | grep -E "libcuda\.so|libnvidia-ml"; ls -la /usr/lib/x86_64-linux-gnu/libcuda.so* 2>&1
echo "== env"; env | grep -E "^NVIDIA|^CUDA|^LD_LIBRARY" | sort
echo "== /proc/driver/nvidia"; cat /proc/driver/nvidia/version 2>&1 | head -2
cat > /tmp/p.cu <<'EOF'
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
int main() {
    CUresult r = cuInit(0);
    const char * s = nullptr; cuGetErrorString(r, &s);
    int dv = 0; cuDriverGetVersion(&dv);
    printf("cuInit: %d (%s), driver API version %d\n", (int) r, s ? s : "?", dv);
    int n = -1; cudaError_t e = cudaGetDeviceCount(&n);
    printf("cudaGetDeviceCount: %d (%s), count %d\n", (int) e, cudaGetErrorString(e), n);
    int rv = 0; cudaRuntimeGetVersion(&rv); printf("runtime %d\n", rv);
    if (e == cudaSuccess && n > 0) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); printf("device 0: %s sm_%d%d\n", p.name, p.major, p.minor);
        void * d = nullptr; e = cudaMalloc(&d, 1 << 20); printf("cudaMalloc: %s\n", cudaGetErrorString(e));
    }
    return 0;
}
EOF
nvcc -o /tmp/p /tmp/p.cu -lcuda 2>&1 | grep -v warning
echo "== probe"; /tmp/p
echo "== probe without compat libs"; LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v compat | paste -sd:) /tmp/p
echo "== end"
} > $LOG 2>&1
b64=$(base64 -w0 $LOG)
printf '{"key":"header","value":{"summary":"probe %s"}}\n{"key":"file","value":{"path":"linuxtest/probe/%s.log","encoding":"base64","content":"%s"}}\n' "$TAG" "$TAG" "$b64" > /tmp/c.ndjson
curl -fsS -X POST -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/x-ndjson" --data-binary @/tmp/c.ndjson \
    "https://huggingface.co/api/models/$REPO/commit/main" >/dev/null && echo uploaded
sleep infinity
