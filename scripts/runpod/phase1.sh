#!/usr/bin/env bash
# RunPod job (docs/LINUX-PLAN.md, Phase 1): build shoehorn on Linux and run the correctness checks on a pack from the
# Hugging Face repo, with the token files the Windows reference used (linuxtest/ in the repo): `ppl --save-kld` and a
# greedy `gen --compare`. Results go back to linuxtest/ in the repo; the KLD comparison runs on the Windows side
# (`shoehorn ppl --kld-base r3.linux.kld`).
#
# Pod: image nvidia/cuda:12.8.1-devel-ubuntu24.04, >= 100 GB container disk, env HF_TOKEN.
# Start command (the image has no curl): bash -c "apt-get update && apt-get install -y curl && curl -fsSL
#   https://raw.githubusercontent.com/tony1877/shoehorn/<branch>/scripts/runpod/phase1.sh | bash"
set -uo pipefail

BRANCH=${BRANCH:-main}
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
PACK=${PACK:-orca27b-r3}
TAG=${TAG:-linux}
BACKEND=${BACKEND:-cuda}  # cuda, or hip (image rocm/dev-ubuntu-24.04:7.0)
W=/work
mkdir -p $W/out
LOG=$W/out/phase1.$TAG.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

status() {
    echo "$*" > $W/out/STATUS
    hf upload "$REPO" $W/out/STATUS linuxtest/STATUS.$TAG --quiet >/dev/null 2>&1 || true
    hf upload "$REPO" "$LOG" "linuxtest/$(basename "$LOG")" --quiet >/dev/null 2>&1 || true
}
fail() {  # upload whatever results exist, so the failure can be read without SSH
    echo "FAILED: $*"
    hf upload "$REPO" $W/out linuxtest --include "*.$TAG.*" --quiet >/dev/null 2>&1 || true
    status "failed: $*"
    sleep infinity
}

echo "== $(date -u) $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB RAM"; nvidia-smi -L 2>/dev/null || rocm-smi --showproductname 2>/dev/null | head -8 || true
export DEBIAN_FRONTEND=noninteractive
# nvidia/cuda images switch on CUDA's forward-compatibility libraries when the host driver is older than the image's
# CUDA; GeForce cards reject them ("failed to initialize CUDA: unknown error"). CUDA 12.x runs on older 12.x drivers
# without them for prebuilt kernels (minor version compatibility), so take them out.
if [ -d /usr/local/cuda/compat ]; then
    rm -f /etc/ld.so.conf.d/*cuda-compat* 2>/dev/null; ldconfig 2>/dev/null
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '
' | grep -v compat | paste -sd:)
fi
apt-get update -qq && apt-get install -y -qq git cmake ninja-build g++ python3-venv curl >/dev/null || fail apt
[ "$BACKEND" = hip ] && { apt-get install -y -qq hipblas-dev rocblas-dev >/dev/null || fail "apt rocm libs"; }
python3 -m venv /opt/hf && /opt/hf/bin/pip install -q "huggingface_hub[hf_xet]" || fail pip
export PATH=/opt/hf/bin:$PATH
hf auth whoami || fail "HF_TOKEN not accepted"
status "building"
if [ -d $W/shoehorn/.git ]; then  # a restarted container keeps its disk: update instead of cloning
    git -C $W/shoehorn pull -q --ff-only || fail pull
else
    git clone -q --depth 1 -b "$BRANCH" --recurse-submodules --shallow-submodules https://github.com/tony1877/shoehorn $W/shoehorn || fail clone
fi
cd $W/shoehorn && bash scripts/build.sh --backend $BACKEND || fail build
BIN=$W/shoehorn/build/bin/shoehorn
# a community host with a broken driver makes ggml fall back to the CPU silently: stop early instead
"$BIN" bench --only gpu > $W/out/gpucheck.$TAG.txt 2>&1
if [ $? != 0 ] || grep -qE "failed to initialize|no GPU device" $W/out/gpucheck.$TAG.txt; then fail "no usable GPU on this host"; fi

status "downloading"
hf download "$REPO" "$PACK.base.gguf" "$PACK.res.gguf" linuxtest/ppl.ids linuxtest/prompt.ids --local-dir $W/m --quiet || fail download
cd $W/out

# ONLY_EXTRA=amd: skip ppl / gen (already done on this GPU) and run the AMD follow-up checks
if [ -n "${ONLY_EXTRA:-}" ]; then bash $W/shoehorn/scripts/runpod/${ONLY_EXTRA}_extra.sh; sleep infinity; fi

ulimit -l; nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv 2>/dev/null || rocminfo 2>/dev/null | grep -m3 -E "gfx|Marketing" || true
status "ppl"
"$BIN" ppl $W/m/$PACK.base.gguf --res $W/m/$PACK.res.gguf --tokens $W/m/linuxtest/ppl.ids --ctx 512 --chunks 8 \
    --save-kld "$PACK.$TAG.kld" > "ppl.$TAG.txt" 2>&1; rc=$?
cat "ppl.$TAG.txt"; [ $rc = 0 ] || fail "ppl (exit $rc)"

status "gen"
"$BIN" gen $W/m/$PACK.base.gguf --res $W/m/$PACK.res.gguf --tokens $W/m/linuxtest/prompt.ids -n 128 --temp 0 --compare \
    > "gen.$TAG.txt" 2>&1; rc=$?
cat "gen.$TAG.txt"; [ $rc = 0 ] || [ $rc = 3 ] || fail "gen (exit $rc)"  # 3: --compare found spec != plain (as on Windows)

status "uploading"
hf upload "$REPO" $W/out linuxtest --include "*.$TAG.*" || fail upload
status "done"
[ "${PHASE2:-0}" = 1 ] && bash $W/shoehorn/scripts/runpod/phase2.sh
sleep infinity
