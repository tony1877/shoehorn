# shoehorn on a dedicated machine, dsh on this PC

dsh only talks to shoehorn over HTTP (the OpenAI-compatible API on port 8090, plus `/stats` and the dashboard for the
shoehorn panel), so the engine runs on another machine on the LAN. Everything the agent *does* (shell, files, REA,
the admin bridge, SearXNG, the Playwright browser) keeps running where dsh runs, on this PC.

## Current setup (2026-11-20)

- **The rig** (`aless@192.168.68.54`): Ubuntu 26.10, RTX PRO 5000 Blackwell 48 GB (capped at 300 W by
  `/etc/systemd/system/gpu-powercap.service`), 30 GB RAM. shoehorn runs as the user unit
  `~/.config/systemd/user/shoehorn.service` from `~/shoehorn/build/bin` on `0.0.0.0:8090`.
- **Models**: the 27B (`~/models/orca27b.base.gguf` IQ4_XS + `orca27b.res.gguf`, residual all in VRAM via
  `--res-gpu-gb 15`) is the default; `--also` adds Flash-Next (`~/models/flash-next`), GLM-5.3-Flash (`~/models/glm`)
  and DeepSeek-V4.1-Flash (`/data/deepseek`), swapped in on request.
- **27B serving**: `--ctx 262144 --dense-slots 2`: two VRAM slots, each with its own 256K KV (q8_0 K / q4_0 V, 7 GB
  each), so two conversations keep their prompts and each runs at the one-sequence speed; two at once take turns a
  decode cycle at a time, and a long prompt read yields to the other slot every 2048 tokens
  ([MULTISEQ.md](MULTISEQ.md), "VRAM slots"). The expert models use `--slots 2` (parked in RAM).
- **This PC**: `C:\models\shoehorn-url.txt` holds `http://192.168.68.54:8090`, so `C:\models\start-detached.ps1`
  starts no local shoehorn; dsh's model routes and the shoehorn panel point at the rig
  (`scripts/use-remote-shoehorn.ps1 -Local -Apply` goes back).
- **Images**: encoded on the rig itself (no `--encoder-url`); the laptop's encoder on the direct cable link is not
  reachable from the rig and is no longer used for this.

## Updating the rig

Engine changes are edited on this PC, pushed, then built and tested on the rig:

```
ssh aless@192.168.68.54
cd ~/shoehorn && git pull --ff-only && cmake --build build -j 12
systemctl --user restart shoehorn
```

`build-dev` is a second build tree for experiments (run it by hand on port 8090 after `systemctl --user stop
shoehorn`, then start the service again). The service's request timings (numbers only) go to
`~/models/bench/shoehorn-timing.jsonl`.

## Known gaps

- **The watchdog task's ports** apply at its next start (logon), as the running loop keeps its arguments.
- **Speed**: streaming over a LAN adds ~1 ms per chunk; nothing to tune.
- **Firewall**: the API is plain HTTP with a key; keep TCP 8090 reachable from this PC only.
