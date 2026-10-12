#!/usr/bin/env bash
# RunPod job (docs/LINUX-PLAN.md, Phase 4 gate): one GPU fit profile on a card of its size. Builds shoehorn, gets the
# pack (from the HF repo, or packs it here from the BF16 source when PACK_BASE is set), then runs `gen` greedy and
# sampled with GEN_ARGS and logs the speed. Log: linuxtest/fit/$TAG.log in the HF repo.
#
# Env: HF_TOKEN, TAG, PACK (e.g. orca27b-r3), GEN_ARGS (e.g. "--res-gpu-gb auto"), optional PACK_BASE / PACK_RES
# (e.g. iq3_xxs / q4_K: pack $PACK here). Image: nvidia/cuda:12.8.1-devel-ubuntu24.04, a pro / data-center card.
set -uo pipefail
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
BRANCH=${BRANCH:-claude/qwen-custom-inference-engine-jhydpg}
W=/work
mkdir -p $W/out $W/m
LOG=$W/out/fit.$TAG.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
push() { hf upload "$REPO" "$LOG" "linuxtest/fit/$TAG.log" --quiet >/dev/null 2>&1 || true; }
fail() { echo "FAILED: $*"; push; sleep infinity; }

echo "== $TAG $(date -u) $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB RAM"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake ninja-build g++ python3-venv curl >/dev/null || fail apt
python3 -m venv /opt/hf && /opt/hf/bin/pip install -q "huggingface_hub[hf_xet]" || fail pip
export PATH=/opt/hf/bin:$PATH
push
git clone -q --depth 1 -b "$BRANCH" --recurse-submodules --shallow-submodules https://github.com/tony1877/shoehorn $W/shoehorn || fail clone
cd $W/shoehorn && bash scripts/build.sh --backend cuda > $W/out/build.txt 2>&1 || { tail -20 $W/out/build.txt; fail build; }
BIN=$W/shoehorn/build/bin/shoehorn
"$BIN" bench --only gpu 2>&1 | grep -E "device 0|N=1 |failed to initialize"

hf download "$REPO" linuxtest/prompt.ids --local-dir $W/m --quiet || fail "download prompt"
if [ -n "${PACK_BASE:-}" ]; then
    echo "== packing $PACK (base $PACK_BASE, residual ${PACK_RES:-q4_K})"
    hf download orcarouter/Qwen3.8-27B-Uncensored --local-dir $W/src --quiet || fail "download source"
    hf download orcarouter/Qwen3.8-27B-Uncensored-GGUF Qwen3.8-27B-Uncensored-Q4_K_M.gguf --local-dir $W/tmpl --quiet || fail "download template"
    hf download "$REPO" imatrix-orca27b.gguf --local-dir $W/imat --quiet || fail "download imatrix"
    "$BIN" pack --src $W/src --template $W/tmpl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf --out $W/m/$PACK --base "$PACK_BASE" \
        --res "${PACK_RES:-q4_K}" --imatrix $W/imat/imatrix-orca27b.gguf --threads "$(nproc)" > $W/out/pack.txt 2>&1 || { tail -5 $W/out/pack.txt; fail pack; }
    grep -E "rel. RMSE|wrote" $W/out/pack.txt
    rm -rf $W/src
else
    hf download "$REPO" "$PACK.base.gguf" "$PACK.res.gguf" --local-dir $W/m --quiet || fail "download pack"
fi
push

for a in "--temp 0" "--temp 1 --seed 1 --repeat 3"; do
    echo "== gen $a ${GEN_ARGS:-}"
    "$BIN" gen $W/m/$PACK.base.gguf --res $W/m/$PACK.res.gguf --tokens $W/m/linuxtest/prompt.ids -n 256 $a ${GEN_ARGS:-} 2>&1 \
        | grep -E "loaded|tok/s|cycles|^run |repeat 3|error|out of memory|not enough" | cut -c1-220
done
echo "== end"
push
sleep infinity
