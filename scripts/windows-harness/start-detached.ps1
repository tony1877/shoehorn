# Start shoehorn, then dsh (with SearXNG and the exec bridge), all detached and hidden.
#
# Win32_Process.Create makes WMI the parent, so nothing belongs to the calling console/job and
# everything keeps running after this script or its host exits. ShowWindow 0 hides the console
# windows (closing one would kill what runs inside it).
#
# Models: shoehorn serves both (it swaps them in VRAM on demand, so switching in dsh needs no restart):
#   qwen3.8-27b-uncensored-shoehorn         27B orca, ~Q8 (default)  -> dsh route local-27b-shoehorn
#   qwen3.8-flash-next-uncensored-shoehorn  Flash-Next orca IQ3_XXS  -> dsh route local-flash-shoehorn
# The llama-server router (port 8080, C:\models\models.ini: heretic and plain orca models) was
# dropped 2026-10-09; the previous version of this script is start-detached.ps1.bak-20261009-norouter.
#
# dsh is started through C:\Tools\bin\start-dsh.ps1, never directly: that script rebuilds the
# User environment (MSVC INCLUDE/LIB, C:\Tools\bin, JAVA_HOME) that a WMI-created process would
# not get, and starts SearXNG (127.0.0.1:8888) and the exec bridge (127.0.0.1:9999) first.
#
# Stop everything:
#   Stop-Process -Name shoehorn
#   foreach ($p in 3080, 8888, 9999) { Stop-Process -Id (Get-NetTCPConnection -LocalPort $p -State Listen).OwningProcess }
$dshLog   = 'C:\Users\aless\.dsh\dsh.log'

function Start-Detached([string] $commandLine, [string] $cwd) {
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $commandLine; CurrentDirectory = $cwd; ProcessStartupInformation = $startup }
    if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed ($($r.ReturnValue)): $commandLine" }
    $r.ProcessId
}
function Listening([int] $port) { [bool](Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) }
function Dsh-Listening { [bool](Get-NetTCPConnection -LocalAddress 127.0.0.1 -State Listen -LocalPort 3080 -ErrorAction SilentlyContinue) }

# ── 1. API key ────────────────────────────────────────────────────────────────
# The source of truth is DEEPSEEK_API_KEY in ~\.dsh\.env, which every dsh route sends (apiKeyEnv).
# A missing key or the old 'local' placeholder is replaced with a random one here. shoehorn gets it
# through --api-key-file, so it never appears on a process command line.
$envFile = "$env:USERPROFILE\.dsh\.env"
$keyFile = "$env:USERPROFILE\.dsh\llama-api-key.txt"
$envLines = @(Get-Content $envFile -ErrorAction SilentlyContinue)
$key = ($envLines | Where-Object { $_ -match '^DEEPSEEK_API_KEY=' } | Select-Object -First 1) -replace '^DEEPSEEK_API_KEY=', ''
if (-not $key -or $key -in 'local', 'REPLACE_ME_WITH_THE_POD_KEY') {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = 'sk-local-' + (-join ($bytes | ForEach-Object { $_.ToString('x2') }))
    $envLines = @($envLines | Where-Object { $_ -notmatch '^DEEPSEEK_API_KEY=' }) + "DEEPSEEK_API_KEY=$key"
    Set-Content $envFile $envLines
    "generated a new API key in $envFile (restart dsh so it picks it up)"
}
Set-Content $keyFile @('# shoehorn API key (from ~\.dsh\.env DEEPSEEK_API_KEY; rewritten on every start)', $key)
$auth = @{ Authorization = "Bearer $key" }
# ── 2. shoehorn (127.0.0.1:8090) ──────────────────────────────────────────────
# Every dsh route goes through it. It frees the GPU after 30 min idle, or on
# POST http://127.0.0.1:8090/unload. Must be running for dsh to work.
if (Listening 8090) { "shoehorn already running" } else {
    $e8Args = 'serve E:\shoehorn\orca27b.base.gguf --res E:\shoehorn\orca27b.res.gguf --host 0.0.0.0 --port 8090 ' +
              '--alias qwen3.8-27b-uncensored-shoehorn --ctx 262144 --kv q8_0 --kv-v q4_0 --spec auto --fast-prefill 4096 --think-after-tool 1 ' +
              '--also qwen3.8-flash-next-uncensored-shoehorn=C:\models\Qwen3.8-Flash-Next-Uncensored\Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf ' +
              "--api-key-file `"$keyFile`" --chat-template-file C:\models\templates\qwen-sharp-v22.5.0.jinja " +
              '--idle-unload 1800 --timing-log C:\models\bench\shoehorn-timing.jsonl'
    $pid3 = Start-Detached "cmd.exe /c `"C:\src\shoehorn\build\bin\shoehorn.exe $e8Args > C:\models\bench\shoehorn.log 2>&1`"" 'C:\models'
    "shoehorn started (cmd PID $pid3), log: C:\models\bench\shoehorn.log"
}
# Default model: load the shoehorn 27B now, so the first dsh message does not wait ~40 s for it.
$deadline = (Get-Date).AddSeconds(30)
while (-not (Listening 8090) -and (Get-Date) -lt $deadline) { Start-Sleep 1 }
try {
    $warm = @{ model = 'qwen3.8-27b-uncensored-shoehorn'; messages = @(@{ role = 'user'; content = 'hi' }); max_tokens = 1 } | ConvertTo-Json -Depth 5
    Invoke-RestMethod http://127.0.0.1:8090/v1/chat/completions -Method Post -Headers $auth -Body $warm -ContentType 'application/json' -TimeoutSec 300 | Out-Null
    "shoehorn 27B loaded (default model)"
} catch { "shoehorn warm-up failed: $($_.Exception.Message) (it will load on the first request)" }

# ── 3. dsh (+ SearXNG + bridge, via start-dsh.ps1) ────────────────────────────
if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue) {
    "dsh already running (PID $((Get-NetTCPConnection -State Listen -LocalPort 3080).OwningProcess)), not starting another"
} else {
    $startedAt = Get-Date
    # -WindowStyle Hidden as well: with Windows Terminal as the default console, ShowWindow 0 alone still shows a window.
    $pid2 = Start-Detached 'pwsh.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\Tools\bin\start-dsh.ps1' 'C:\Tools\bin'
    "start-dsh.ps1 started (PID $pid2)"
    $deadline = (Get-Date).AddMinutes(4)
    while (-not ((Dsh-Listening) -and (Test-Path $dshLog) -and (Get-Item $dshLog).LastWriteTime -ge $startedAt -and
                 (Select-String -Path $dshLog -Pattern 'dsh web:' -Quiet))) {
        if ((Get-Date) -gt $deadline) { throw "dsh not up after 4 min; see $dshLog" }
        Start-Sleep 2
    }
    "dsh web up (PID $((Get-NetTCPConnection -State Listen -LocalPort 3080).OwningProcess)); URL with token: the 'dsh web:' line of $dshLog"
}
foreach ($svc in @(@(8888, 'SearXNG'), @(9999, 'exec bridge'))) { "{0,-12} {1}" -f $svc[1], $(if (Listening $svc[0]) { "up on $($svc[0])" } else { 'NOT running' }) }

# dsh prints its URL (with the session token) to its log; show the newest one.
$url = Select-String -Path $dshLog -Pattern 'dsh web:\s*(\S+)' -ErrorAction SilentlyContinue | Select-Object -Last 1
''
if ($url) { "dsh URL: $($url.Matches[0].Groups[1].Value)" } else { "dsh URL: not found in $dshLog" }
