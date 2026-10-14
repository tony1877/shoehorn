# Start dsh with the FULL environment (Machine + User).
#
# dsh used to be launched with a bare Win32_Process.Create, which hands the new
# process only the Machine environment. Everything installed on the User PATH
# (MSVC cl/link/dumpbin, C:\Tools\bin with the Ghidra wrappers) plus the
# INCLUDE/LIB/LIBPATH that cl needs were invisible to the agent, so it reported
# "no VS cl, no Ghidra" while both were installed. This script rebuilds the
# environment from the registry before starting dsh, so its shells inherit it.

$machine = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
$user    = [Environment]::GetEnvironmentVariable('PATH', 'User')
$env:PATH = "$machine;$user"

foreach ($name in 'INCLUDE', 'LIB', 'LIBPATH', 'JAVA_HOME') {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if (-not $value) { $value = [Environment]::GetEnvironmentVariable($name, 'Machine') }
    if ($value) { Set-Item -Path "env:$name" -Value $value }
}

# Record what dsh is about to inherit, so the environment can be checked
# without guessing.
@(
    "started   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "cl        : $((Get-Command cl.exe -ErrorAction SilentlyContinue).Source)"
    "gh-query  : $((Get-Command gh-query.ps1 -ErrorAction SilentlyContinue).Source)"
    "java      : $((Get-Command java.exe -ErrorAction SilentlyContinue).Source)"
    "JAVA_HOME : $env:JAVA_HOME"
    "INCLUDE   : $(if ($env:INCLUDE) { 'set' } else { 'MISSING' })"
    "LIB       : $(if ($env:LIB) { 'set' } else { 'MISSING' })"
) | Set-Content "$env:USERPROFILE\.dsh\dsh-env.txt"

# Local SearXNG backs the agent's web_search (see ~\.dsh\cordis.patch.yml).
# No-op when it is already running.
& 'C:\Tools\bin\start-searxng.ps1'

# Exec bridge (brg.ps1 -> 127.0.0.1:9999) comes up with the agent. If the
# elevated scheduled task 'dsh-bridge' exists (one-time admin setup, see the
# kit README), run that so the bridge is elevated; otherwise start it with
# this script's own (non-elevated) rights. No-op when it is already running.
if (-not (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 9999 -State Listen -ErrorAction SilentlyContinue)) {
    & schtasks.exe /Query /TN 'dsh-bridge' *> $null
    if ($LASTEXITCODE -eq 0) {
        & schtasks.exe /Run /TN 'dsh-bridge' | Out-Null
    } else {
        Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList 'C:\Tools\bin\dsh-bridge.js' `
            -WindowStyle Hidden -RedirectStandardOutput "$env:USERPROFILE\.dsh\bridge.log" `
            -RedirectStandardError "$env:USERPROFILE\.dsh\bridge.err.log"
    }
    $deadline = (Get-Date).AddSeconds(15)
    while (-not (Get-NetTCPConnection -LocalPort 9999 -State Listen -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 9999 -State Listen -ErrorAction SilentlyContinue)) {
        throw 'Exec bridge did not start; inspect the dsh-bridge task and bridge.err.log.'
    }
}

# The agent is shoehorn's bundled copy (C:\src\shoehorn\agent, 2026-10-10); this machine keeps its own
# settings, sessions and memory in ~\.dsh. The old checkout C:\Users\aless\deepseek-harness is no longer used.
$env:DSH_HOME = "$env:USERPROFILE\.dsh"
Set-Location 'C:\src\shoehorn\agent'
# Reachable over the LAN (by choice, 2026-10-09). dsh only listens on 127.0.0.1 (it refuses other hosts), so a
# small TCP forwarder (dsh-lan-proxy.js) passes this PC's LAN address :3080 through to it, and dsh's browser-trust
# fence is told to accept that address. Whoever reaches it with the token link can run commands on this PC: keep the
# link private. The address is read at each start (DHCP-safe).
$lanIp = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | ForEach-Object { $_.IPv4Address.IPAddress } | Select-Object -First 1
$trustArgs = @()
if ($lanIp) {
    $trustArgs = @('--trusted-host', "$($lanIp):3080")
    if (-not (Get-NetTCPConnection -LocalAddress $lanIp -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue)) {
      Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList "C:\Tools\bin\dsh-lan-proxy.js", $lanIp -WindowStyle Hidden `
        -RedirectStandardOutput "$env:USERPROFILE\.dsh\lan-proxy.log" -RedirectStandardError "$env:USERPROFILE\.dsh\lan-proxy.err.log"
    }
}
pnpm dsh web --no-open @trustArgs *> "$env:USERPROFILE\.dsh\dsh.log"
