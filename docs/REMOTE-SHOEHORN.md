# shoehorn on a dedicated machine, dsh on this PC

dsh only talks to shoehorn over HTTP (the OpenAI-compatible API on port 8090, plus `/stats` and the dashboard for the
shoehorn panel), so the engine can run on another machine on the LAN. Everything the agent *does* (shell, files, REA,
the admin bridge, SearXNG, the Playwright browser) keeps running where dsh runs, on this PC.

## What is ready (2026-10-23)

- **shoehorn panel address** (`agent/packages/client/ui-shoehorn`): the status chip, dashboard and sensors read
  shoehorn from the `url` setting of the `ui-shoehorn` row (empty = port 8090 on the PC that served the dsh page, as
  before). The host writes it into the page as `__DSH_SHOEHORN_CONFIG__`.
- **Start script remote mode** (`C:\models\start-detached.ps1`, outside this repo; backup
  `start-detached.ps1.bak-20261023-remote`): while `C:\models\shoehorn-url.txt` holds a URL, it starts no local
  shoehorn and warms up / checks that URL instead.
- **Switch script** `scripts/use-remote-shoehorn.ps1`: flips this PC between local and remote shoehorn in one step
  (dry run by default, `-Apply` to write, `-Local` to go back). It changes the four dsh model routes (`baseURL`), the
  panel `url`, the start-script URL file, the watchdog's ports (3080 only when remote: it must not restart a local
  shoehorn that is gone), and the "services on this machine" text in both agent presets.
- **Sensors** need nothing: shoehorn serves them itself (Linux: the hwmon drivers; Windows: `shoehorn-hwmon` on the
  same machine), so they move with shoehorn.

## Moving day

1. **Server**: install shoehorn and the models (the 27B pair, ~30 GB; Flash-Next optional, ~80 GB). Start it with the
   same flags as `C:\models\start-detached.ps1` uses today, adjusted to the new paths, and `--host 0.0.0.0 --port 8090`.
   On a 48 GB GPU the whole residual fits in VRAM: `--res-gpu-gb auto`.
2. **API key**: copy `DEEPSEEK_API_KEY` from this PC's `~\.dsh\.env` into the server's `--api-key-file`. Both sides must
   use the same key; dsh sends it on every request.
3. **Firewall**: allow TCP 8090 inbound on the server, from this PC only if possible. The API is plain HTTP with a key.
4. **This PC**: `.\scripts\use-remote-shoehorn.ps1 -Url http://<server>:8090` (check the dry run), then add `-Apply`,
   stop dsh and run `C:\models\start-detached.ps1`. Confirm: the shoehorn chip shows the model, and a chat answers.
5. **Free the GPU here**: nothing else to do; shoehorn is no longer started on this PC.

## Known gaps

- **Image encoder**: shoehorn sends images to the laptop's encoder at `http://10.77.0.2:8091`, over the direct cable
  link between this PC and the laptop. A server on the normal LAN cannot reach that address: give the laptop's
  encode-server a LAN address the server can reach (and change `--encoder-url`), or let images encode on the
  server's CPU/GPU (with 48 GB the vision encoder can run on the server's own GPU).
- **The watchdog task's new ports** apply at its next start (logon), as the running loop keeps its arguments.
- **Speed**: streaming over a LAN adds ~1 ms per chunk; nothing to tune.
