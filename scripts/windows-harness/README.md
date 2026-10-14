# Windows agent support

This directory owns the machine-local launchers, authenticated exec bridge,
SearXNG provider and Qwen template used with the bundled agent. The template
honors an explicit thinking-off request even when history contains control
markers. Conversation thinking remains enabled after tool results. The
server supports `max_content_chars` as a per-request UTF-8 answer-byte cap;
tool arguments and reasoning remain separately budgeted. A capped answer
reports the OpenAI `length` finish reason.

`dsh-bridge.js` and `brg.ps1` use protocol v2 together. A 32-byte random token
in `%USERPROFILE%\.dsh\bridge-token.txt` is readable by its owner, SYSTEM and
Administrators. The owner is trusted to request elevated execution. The
bridge listens on loopback only, forwards stdout/stderr frames immediately,
preserves UTF-8 boundaries, propagates exit codes, and terminates the command
tree on disconnect, timeout or output overflow. Default bounds are two
minutes, 1 MiB request and 8 MiB combined output. JSON frames replace the
ambiguous stdout/stderr marker protocol.

```powershell
C:\Tools\bin\brg.ps1 'Get-Location'
C:\Tools\bin\brg.ps1 'EXE|["program.exe","","literal argument"]'
```

The client supplies its current directory; `-WorkingDirectory` overrides it.
`DSH_BRIDGE_TOKEN_FILE` can select an existing token file for both processes.
Simple quoted EXE strings preserve empty arguments; use JSON arrays when an
argument contains literal quote characters.

The watchdog and starter distinguish dsh's loopback 3080 listener from its
independent LAN proxy. The search provider reports discarded result rows.
`configure-local.mjs` preserves unrelated settings and configures 16,384-byte
conversation answers, 8,192-token standard thinking-off checkpoints, and
advisory investigation thresholds of 12, 24 and 48 calls. File mutations
reset the advisory count; a long investigation is not automatically blocked.

`install.ps1` backs up existing support files, secures the bridge token ACL,
and applies configuration. It refuses to install a v2 client while a bridge
is running. Activate the paired bridge changes together after inspecting
active commands. Source/UI deployment and the rebuilt inference executable
are separate operations; template/server policy requires reloading the
inference server. The installer does not restart any service.

Regression checks are `node --test tests/windows-harness.test.cjs` and
`python tests/template_policy.py` from the repository root; the Python check
requires Jinja2. The native `tests/tool_recovery.cpp` checks monotonic
argument recovery without a model. Agent checks include the file-preview
browser scenario and keyless `investigation-checkpoint` session replay.
On an idle server, `python tests/server_policy.py --api-key-file PATH` also
checks final and streamed answer bounds and both thinking-policy regressions
against the loaded model. It reports no credentials or generated text.

The configured fast-prefill path is approximate. Its fidelity measurements
are recorded in [decode limits](../../docs/DECODE-LIMITS.md); speculative
decoding's exact acceptance rules do not make base-only prefill lossless.
