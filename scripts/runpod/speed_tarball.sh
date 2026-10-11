#!/bin/sh
# RunPod job (docs/LINUX-PLAN.md, Phase 7) on a plain distribution image: the portable tarball and the pack from the HF
# repo (curl only), then `bench --only gpu,pcie` and `gen` greedy / sampled, as speed_extra.sh. Log: linuxtest/speed/$TAG.log
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
TARBALL=${TARBALL:-shoehorn-0.1.0-linux-x86_64-cuda.tar.gz}
PACK=${PACK:-orca27b-r3}
LOG=/tmp/speed.log
get() { curl -fsSL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "https://huggingface.co/$REPO/resolve/main/$1"; }
push() {  # the log through the HF commit API
    b64=$(base64 -w0 $LOG)
    printf '{"key":"header","value":{"summary":"speed %s"}}\n{"key":"file","value":{"path":"linuxtest/speed/%s.log","encoding":"base64","content":"%s"}}\n' \
        "$TAG" "$TAG" "$b64" > /tmp/c.ndjson
    curl -fsS -X POST -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/x-ndjson" --data-binary @/tmp/c.ndjson \
        "https://huggingface.co/api/models/$REPO/commit/main" >/dev/null
}
{
echo "== $TAG $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
cd /tmp && get dist/$TARBALL t.tgz && tar -xzf t.tgz && B=$(ls -d /tmp/shoehorn-*/)bin/shoehorn
"$B" bench --only gpu,pcie 2>&1 | grep -v "^ *$"
} > $LOG 2>&1
push
{
echo "== downloading $PACK"
get $PACK.base.gguf /tmp/b.gguf && get $PACK.res.gguf /tmp/r.gguf && get linuxtest/prompt.ids /tmp/p.ids
for a in "--temp 0" "--temp 1 --seed 1 --repeat 3"; do
    echo "== gen $a"
    "$B" gen /tmp/b.gguf --res /tmp/r.gguf --tokens /tmp/p.ids -n 256 $a 2>&1 | grep -E "loaded|tok/s|cycles|draft rounds|^run |error|failed" | cut -c1-240
done
echo "== end"
} >> $LOG 2>&1
push
sleep infinity
