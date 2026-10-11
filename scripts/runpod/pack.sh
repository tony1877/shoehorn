#!/usr/bin/env bash
# RunPod job: build shoehorn on Linux, pack Qwen3.8-27B-Uncensored with an iq4_xs base and each residual in $RESIDUALS
# (default q3_K as orca27b-r3, like E:/shoehorn/orca27b-r3, and q4_K as orca27b-r4; imatrix-orca27b), and upload the
# packs, the log and checksums to a Hugging Face repo. Each base carries its pack id (the residual must match it), so
# every pack uploads its own base.
#
# Pod: image nvidia/cuda:12.8.1-devel-ubuntu24.04, >= 250 GB container disk, env HF_TOKEN (write access to $OUT_REPO).
# Start command (the image has no curl): bash -c "apt-get update && apt-get install -y curl && curl -fsSL https://raw.githubusercontent.com/tony1877/shoehorn/<branch>/scripts/runpod/pack.sh | bash"
# The container stays up when done (RunPod restarts a container that exits); the caller deletes the pod.
set -uo pipefail

BRANCH=${BRANCH:-claude/qwen-custom-inference-engine-jhydpg}
SRC_REPO=${SRC_REPO:-orcarouter/Qwen3.8-27B-Uncensored}
TEMPLATE_REPO=${TEMPLATE_REPO:-orcarouter/Qwen3.8-27B-Uncensored-GGUF}
TEMPLATE_FILE=${TEMPLATE_FILE:-Qwen3.8-27B-Uncensored-Q4_K_M.gguf}
OUT_REPO=${OUT_REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
RESIDUALS=${RESIDUALS:-q3_K q4_K}
W=/work
mkdir -p $W/out
LOG=$W/out/linux-pack.log
exec > >(tee -a "$LOG") 2>&1

status() {  # progress marker in the HF repo, so the caller can follow without SSH
    echo "$*" > $W/out/STATUS
    hf upload "$OUT_REPO" $W/out/STATUS STATUS --quiet >/dev/null 2>&1 || true
    hf upload "$OUT_REPO" "$LOG" "$(basename "$LOG")" --quiet >/dev/null 2>&1 || true
}
fail() { echo "FAILED: $*"; status "failed: $*"; sleep infinity; }

echo "== $(date -u) $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB RAM"; nvidia-smi -L || true
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake ninja-build g++ python3-pip python3-venv curl >/dev/null || fail apt
python3 -m venv /opt/hf && /opt/hf/bin/pip install -q "huggingface_hub[hf_xet]" || fail pip
export PATH=/opt/hf/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0
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
hf download "$SRC_REPO" --local-dir $W/src --quiet || fail "download $SRC_REPO"
hf download "$TEMPLATE_REPO" "$TEMPLATE_FILE" --local-dir $W/tmpl --quiet || fail "download template"
hf download "$OUT_REPO" imatrix-orca27b.gguf --local-dir $W/imat --quiet || fail "download imatrix"

for res in $RESIDUALS; do
    NAME=orca27b-r${res:1:1}
    status "packing $NAME (residual $res)"
    "$BIN" pack --src $W/src --template "$W/tmpl/$TEMPLATE_FILE" --out "$W/out/$NAME" --base iq4_xs --res "$res"         --imatrix $W/imat/imatrix-orca27b.gguf --threads "$(nproc)" || fail "pack $NAME"
    (cd $W/out && sha256sum "$NAME".base.gguf "$NAME".res.gguf | tee "$NAME".linux.sha256)
    status "uploading $NAME"
    hf upload "$OUT_REPO" $W/out . --include "$NAME.*" || fail "upload $NAME"
    rm -f "$W/out/$NAME".base.gguf "$W/out/$NAME".res.gguf  # disk
done
status "done"
echo "== done $(date -u)"
sleep infinity
