# Start dsh and the tools attached to it (SearXNG, exec bridge, LAN proxy), detached and hidden.
# No shoehorn: the model runs on another machine (the routes in ~\.dsh point at it).
#
#   pwsh -File start-dsh-only.ps1          # start whatever is not running
#   pwsh -File start-dsh-only.ps1 -Stop    # stop dsh, SearXNG, the bridge and the LAN proxy
#
# dsh is started through C:\Tools\bin\start-dsh.ps1, never directly: that script rebuilds the User environment
# (MSVC INCLUDE/LIB, C:\Tools\bin, JAVA_HOME) that a WMI-created process would not get, and starts SearXNG
# (127.0.0.1:8888), the exec bridge (127.0.0.1:9999) and the LAN proxy (<LAN IP>:3080) first.
# Win32_Process.Create makes WMI the parent, so everything keeps running after this script exits.
param([switch] $Stop)
$dshLog = "$env:USERPROFILE\.dsh\dsh.log"

function Listener([int] $port) { Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue }

if ($Stop) {
    # 3080 matches both dsh (127.0.0.1) and the LAN proxy (<LAN IP>).
    foreach ($port in 3080, 8888, 9999) {
        foreach ($owner in @(Listener $port | Select-Object -ExpandProperty OwningProcess -Unique)) {
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            "stopped PID $owner (port $port)"
        }
    }
    return
}

if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue) {
    "dsh already running (PID $((Listener 3080 | Select-Object -First 1).OwningProcess))"
} else {
    $startedAt = Get-Date
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    # -WindowStyle Hidden as well: with Windows Terminal as the default console, ShowWindow 0 alone still shows a window.
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = 'pwsh.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\Tools\bin\start-dsh.ps1'
        CurrentDirectory = 'C:\Tools\bin'; ProcessStartupInformation = $startup }
    if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed ($($r.ReturnValue))" }
    "start-dsh.ps1 started (PID $($r.ProcessId))"
    $deadline = (Get-Date).AddMinutes(4)
    while (-not ((Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue) -and
                 (Test-Path $dshLog) -and (Get-Item $dshLog).LastWriteTime -ge $startedAt -and
                 (Select-String -LiteralPath $dshLog -Pattern 'dsh web:' -Quiet))) {
        if ((Get-Date) -gt $deadline) { throw "dsh not up after 4 min; see $dshLog" }
        Start-Sleep 2
    }
}
foreach ($svc in @(@(3080, 'dsh'), @(8888, 'SearXNG'), @(9999, 'exec bridge'))) {
    "{0,-12} {1}" -f $svc[1], $(if (Listener $svc[0]) { "up on $($svc[0])" } else { 'NOT running' })
}
$url = Select-String -LiteralPath $dshLog -Pattern 'dsh web:\s*(\S+)' -ErrorAction SilentlyContinue | Select-Object -Last 1
''
if ($url) { "dsh URL: $($url.Matches[0].Groups[1].Value)" } else { "dsh URL: not found in $dshLog" }
