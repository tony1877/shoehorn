#!/usr/bin/env bash
# Starts the bundled agent (agent/, a fork of dsh) with its web UI, set up for shoehorn serve.
#
#   scripts/agent.sh [--agent-home DIR] [--api-key KEY] [-- extra dsh web args, e.g. --no-open --trusted-host H:3080]
#
# The agent home holds the agent's settings, sessions and memory: --agent-home, else $DSH_HOME, else ~/.shoehorn/agent.
# A new one is filled from agent-home/ (settings, the `shoehorn` preset, skills); existing files are never overwritten.
# The API key is the one shoehorn serve was started with: --api-key, else $SHOEHORN_API_KEY, else "none".
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
home=""
key=""
while [ $# -gt 0 ]; do
    case "$1" in
        --agent-home) home=$2; shift 2 ;;
        --api-key)    key=$2; shift 2 ;;
        --)           shift; break ;;
        *)            break ;;
    esac
done

home=${home:-${DSH_HOME:-$HOME/.shoehorn/agent}}
mkdir -p "$home"
home=$(cd "$home" && pwd)
seed="$root/agent-home"
while IFS= read -r -d '' f; do
    dst="$home/${f#"$seed"/}"
    if [ ! -e "$dst" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$f" "$dst"
        echo "seeded $dst"
    fi
done < <(find "$seed" -type f -print0)
export DSH_HOME="$home"

export SHOEHORN_API_KEY=${key:-${SHOEHORN_API_KEY:-none}}
# everything stays on this machine: no feedback telemetry to the upstream project
export DSH_TELEMETRY_MODE=${DSH_TELEMETRY_MODE:-DISABLED}

command -v pnpm >/dev/null 2>&1 || { echo "pnpm not found: install Node 24 and run \`corepack enable\`" >&2; exit 1; }
cd "$root/agent"
stamp=.shoehorn-built
# a fresh checkout, or a pull that changed the lockfile: install the dependencies again
if [ ! -d node_modules ] || [ pnpm-lock.yaml -nt node_modules ]; then
    pnpm install --frozen-lockfile && touch node_modules
fi
# build when nothing is built yet or a source file is newer than the last build (an update to the agent)
if [ ! -f "$stamp" ] || [ -n "$(find packages vendor -path '*/node_modules' -prune -o -path '*/lib' -prune \
        -o -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.json' -o -name '*.css' \) -newer "$stamp" -print -quit)" ]; then
    echo "building the agent (sources changed since the last build)"
    pnpm run build && touch "$stamp"
fi
exec pnpm dsh web "$@"
