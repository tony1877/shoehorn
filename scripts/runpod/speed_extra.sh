#!/usr/bin/env bash
# RunPod follow-up (docs/LINUX-PLAN.md, Phase 7): the speed numbers to compare with the same commands on Windows on the
# same GPU model: `bench --only gpu,pcie`, and `gen` on the pack with the Phase 1 prompt, greedy and sampled (3 seeds).
set -uo pipefail
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
TAG=${TAG:-speed}
PACK=${PACK:-orca27b-r3}
W=/work
LOG=$W/out/speed.$TAG.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
export PATH=/opt/hf/bin:$PATH
up() { echo "$*" > $W/out/STATUS; hf upload "$REPO" $W/out/STATUS linuxtest/STATUS.$TAG --quiet >/dev/null 2>&1 || true
       hf upload "$REPO" "$LOG" "linuxtest/$(basename "$LOG")" --quiet >/dev/null 2>&1 || true; }
BIN=$W/shoehorn/build/bin/shoehorn
M=$W/m
up "extra: bench"
echo "== $(nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader 2>/dev/null)"
"$BIN" bench --only gpu,pcie 2>&1 | grep -vE "^\s*$"
up "extra: gen"
for args in "--temp 0" "--temp 1 --seed 1 --repeat 3"; do
    echo "== gen $args"
    "$BIN" gen $M/$PACK.base.gguf --res $M/$PACK.res.gguf --tokens $M/linuxtest/prompt.ids -n 256 $args 2>&1 \
        | grep -E "loaded|tok/s|cycles|draft rounds|^run " | cut -c1-240
done
up "extra: done"
sleep infinity
