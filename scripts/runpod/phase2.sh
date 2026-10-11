#!/usr/bin/env bash
# RunPod job (docs/LINUX-PLAN.md, Phase 2), run after phase1.sh on the same pod (NOSLEEP=1): `serve` with the pack,
# every page and /stats, a chat request, a download job started and cancelled, a pack job (fails fast: no source on
# the pod), and the Settings restart (re-exec, no systemd here). Results: linuxtest/phase2.$TAG.log in the repo.
set -uo pipefail
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
PACK=${PACK:-orca27b-r3}
TAG=${TAG:-linux}
W=/work
BIN=$W/shoehorn/build/bin/shoehorn
M=$W/m
P=8090
LOG=$W/out/phase2.$TAG.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
export PATH=/opt/hf/bin:$PATH
status() { echo "$*" > $W/out/STATUS; hf upload "$REPO" $W/out/STATUS linuxtest/STATUS --quiet >/dev/null 2>&1 || true
           hf upload "$REPO" "$LOG" "linuxtest/$(basename "$LOG")" --quiet >/dev/null 2>&1 || true; }
up() { for i in $(seq 600); do curl -s -o /dev/null "localhost:$P/health" && return 0; sleep 1; done; return 1; }
api() { curl -s -m 30 -X POST "localhost:$P$1" -H 'Content-Type: application/json' -d "$2"; echo; }
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fi; }

status "phase2: serve"
cd $W/out
"$BIN" serve $M/$PACK.base.gguf --res $M/$PACK.res.gguf --port $P --alias test \
    --settings $W/out/settings.json --models-file $W/out/models.json > serve.$TAG.txt 2>&1 &
up; check "serve answers" "curl -s localhost:$P/health >/dev/null"

echo "== pages"
for u in /dashboard /sensors /models /settings /settings/state /models/state /stats /v1/models; do
    printf '%-16s %s\n' "$u" "$(curl -s -o /dev/null -w '%{http_code} %{size_download}B' "localhost:$P$u")"
done
echo "== /stats hw"
curl -s localhost:$P/stats | python3 -c '
import json,sys; d=json.load(sys.stdin); h=d.get("hw",{}); m=d.get("mem",{})
print("gpu:", h.get("gpu")); print("drives:", h.get("drives")); print("sensors:", len(h.get("sensors",[])), [s["kind"]+":"+s["name"] for s in h.get("sensors",[])][:12])
print("mem: vram_total", m.get("vram_total"), "weights_gpu", m.get("weights_gpu"), "residual", m.get("residual"))'

status "phase2: chat"
echo "== chat"
api /v1/chat/completions '{"model":"test","messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":48,"temperature":0}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("reply:", repr(d["choices"][0]["message"].get("content"))[:300]); print("usage:", d.get("usage"))'

status "phase2: jobs"
echo "== download job, cancelled after 15 s"
id=$(api /models/pull '{"repo":"orcarouter/Qwen3.8-27B-Uncensored-GGUF","dir":"/work/dl","files":[{"path":"Qwen3.8-27B-Uncensored-IQ2_XXS.gguf","size":8900000000}]}' | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",0))')
sleep 15; ls -la /work/dl/* 2>/dev/null | tail -2
api /models/cancel "{\"id\":$id}" >/dev/null; sleep 5
curl -s localhost:$P/models/state | python3 -c 'import json,sys; [print("job", j.get("id"), j.get("kind"), j.get("status"), j.get("error","")) for j in json.load(sys.stdin).get("jobs",[])]'
check "no curl left running" "! pgrep -x curl >/dev/null"
echo "== pack job without a source (should fail cleanly)"
api /models/pack '{"src":"/work/nosuch","template":"/work/nosuch.gguf","out":"/work/out/x","unload":false}'
sleep 5
curl -s localhost:$P/models/state | python3 -c 'import json,sys; [print("job", j.get("id"), j.get("kind"), j.get("status"), str(j.get("error",""))[:120]) for j in json.load(sys.stdin).get("jobs",[])]'

status "phase2: restart"
echo "== Settings restart (re-exec)"
old=$(pgrep -f "shoehorn serve" | head -1)
api /settings/apply '{"flags":{"--ctx":"8192"}}'
sleep 3; up; new=$(pgrep -f "shoehorn serve" | head -1)
echo "pid $old -> $new"; check "restarted into a new process" "[ -n \"$new\" ] && [ \"$old\" != \"$new\" ]"
grep -o '"--ctx"[^,}]*' $W/out/settings.json
tail -3 serve.$TAG.txt

status "phase2: uploading"
hf upload "$REPO" $W/out linuxtest --include "phase2.$TAG.log" "serve.$TAG.txt" --quiet || true
status "phase2: done"
