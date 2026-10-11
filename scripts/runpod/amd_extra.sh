#!/usr/bin/env bash
# RunPod follow-up on the MI300X pod after phase1.sh (docs/LINUX-PLAN.md, Phase 3 item 3-4): GPU sensors through
# amdgpu sysfs, pinned memory and PCIe rate (`bench --only pcie`), greedy `gen` with a q8_0 KV cache (flash attention
# on quantized KV) next to the default run. Reuses the pod's checkout, build and pack.
set -uo pipefail
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
TAG=${TAG:-mi300x}
W=/work
LOG=$W/out/amd_extra.$TAG.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
export PATH=/opt/hf/bin:$PATH
up() { echo "$*" > $W/out/STATUS; hf upload "$REPO" $W/out/STATUS linuxtest/STATUS.$TAG --quiet >/dev/null 2>&1 || true
       hf upload "$REPO" "$LOG" "linuxtest/$(basename "$LOG")" --quiet >/dev/null 2>&1 || true; }
up "extra: building"
cd $W/shoehorn && git pull -q --ff-only && bash scripts/build.sh --backend hip > /dev/null 2>&1 || { up "failed: build"; sleep infinity; }
BIN=$W/shoehorn/build/bin/shoehorn
echo "== render nodes this container can open"; ls -la /dev/dri/ 2>&1
for r in /dev/dri/renderD*; do n=$(basename $r); echo "$n -> $(readlink -f /sys/class/drm/$n/device)"; done
echo "== AMD cards in sysfs"; for c in /sys/class/drm/card*/device; do [ "$(cat $c/vendor 2>/dev/null)" = 0x1002 ] && echo "$(readlink -f $c) used $(cat $c/mem_info_vram_used)"; done
up "extra: bench"
echo "== bench gpu (sensors line = the card shoehorn picked)"; "$BIN" bench --only gpu 2>&1 | grep -E "device 0|gpu-sensors"
exit_after_sensors=${SENSORS_ONLY:-0}; [ "$exit_after_sensors" = 1 ] && { up "extra: done"; sleep infinity; }
echo "== bench gpu + pcie"; "$BIN" bench --only gpu,pcie 2>&1 | grep -vE "^\s*$" | head -40
up "extra: gen q8 KV"
echo "== gen, q8_0 KV"
"$BIN" gen $W/m/orca27b-r3.base.gguf --res $W/m/orca27b-r3.res.gguf --tokens $W/m/linuxtest/prompt.ids -n 128 --temp 0 --kv q8_0 2>&1 \
    | grep -E "loaded|^tokens|tok/s|cycles|error|warning" | cut -c1-300
up "extra: done"
sleep infinity
