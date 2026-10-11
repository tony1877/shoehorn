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

BRANCH=${BRANCH:-claude/qwen-custom-inference-engine-jhydpg}
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
PACK=${PACK:-orca27b-r3}
TAG=${TAG:-linux}
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

echo "== $(date -u) $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB RAM"; nvidia-smi -L || true
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake ninja-build g++ python3-venv curl >/dev/null || fail apt
python3 -m venv /opt/hf && /opt/hf/bin/pip install -q "huggingface_hub[hf_xet]" || fail pip
export PATH=/opt/hf/bin:$PATH
hf auth whoami || fail "HF_TOKEN not accepted"
status "building"
if [ -d $W/shoehorn/.git ]; then  # a restarted container keeps its disk: update instead of cloning
    git -C $W/shoehorn pull -q --ff-only || fail pull
else
    git clone -q --depth 1 -b "$BRANCH" --recurse-submodules --shallow-submodules https://github.com/tony1877/shoehorn $W/shoehorn || fail clone
fi
cd $W/shoehorn && bash scripts/build.sh --backend cuda --arch native || fail build
BIN=$W/shoehorn/build/bin/shoehorn

status "downloading"
hf download "$REPO" "$PACK.base.gguf" "$PACK.res.gguf" linuxtest/ppl.ids linuxtest/prompt.ids --local-dir $W/m --quiet || fail download
cd $W/out

ulimit -l; nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv || true
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
