#!/bin/sh
# RunPod job (docs/LINUX-PLAN.md, Phase 5): run the portable tarball on one distribution's userland. The pod's image
# is the distribution (ubuntu:24.04, debian:12, fedora:41, almalinux:9, archlinux, ...); the NVIDIA runtime mounts the
# host driver into it. Installs only curl, then: every library resolves, `selftest --kernel`, `bench --only gpu`.
# The result goes to linuxtest/matrix/<TAG>.log in the HF repo (HF commit API: no Python needed).
#
# Env: HF_TOKEN, TAG (e.g. ubuntu-24.04), TARBALL (file name under dist/ in the repo).
# POSIX sh: some images have no bash.
REPO=${REPO:-Tony1877/Qwen3.8-27B-Uncensored-shoehorn}
TARBALL=${TARBALL:-shoehorn-0.1.0-linux-x86_64-cuda.tar.gz}
LOG=/tmp/matrix.log

{
echo "== $TAG: $(. /etc/os-release && echo "$PRETTY_NAME"), glibc $(ldd --version 2>&1 | head -1 | grep -o '[0-9.]*$')"
cd /tmp
echo "== tarball"
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" -o t.tgz "https://huggingface.co/$REPO/resolve/main/dist/$TARBALL" && tar -xzf t.tgz || echo "FAIL download"
B=$(ls -d /tmp/shoehorn-*/ | head -1)bin/shoehorn
echo "== libraries"
ldd "$B" | grep -v "linux-vdso\|ld-linux"
if ldd "$B" | grep -q "not found"; then echo "FAIL missing libraries"; else echo "PASS libraries"; fi
echo "== run"
"$B" --version && echo "PASS starts" || echo "FAIL starts"
"$B" selftest --kernel 2>&1 | tail -2
"$B" selftest --kernel >/dev/null 2>&1 && echo "PASS selftest" || echo "FAIL selftest"
"$B" bench --only gpu 2>&1 | tail -12
"$B" bench --only gpu >/dev/null 2>&1 && echo "PASS bench gpu" || echo "FAIL bench gpu"
echo "== end"
} > $LOG 2>&1

# upload the log through the HF commit API (ndjson: header + one base64 file)
b64=$(base64 -w0 $LOG 2>/dev/null || base64 $LOG | tr -d '\n')
printf '{"key":"header","value":{"summary":"matrix %s"}}\n{"key":"file","value":{"path":"linuxtest/matrix/%s.log","encoding":"base64","content":"%s"}}\n' \
    "$TAG" "$TAG" "$b64" > /tmp/commit.ndjson
curl -fsS -X POST -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/x-ndjson" \
    --data-binary @/tmp/commit.ndjson "https://huggingface.co/api/models/$REPO/commit/main" >/dev/null && echo uploaded
sleep infinity
