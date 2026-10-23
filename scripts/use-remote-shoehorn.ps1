<#
.SYNOPSIS
  Point this PC's dsh at a shoehorn running on a dedicated machine, or back at a local shoehorn.

.DESCRIPTION
  Switches every place that assumes shoehorn runs on the same PC as dsh:
    - dsh model routes:     every baseURL http://<host>:8090/v1 in ~\.dsh\profiles\web\cordis.patch.yml
    - shoehorn panel:       the ui-shoehorn row's url (status chip, dashboard, sensors)
    - start script:         C:\models\shoehorn-url.txt (start-detached.ps1 starts no local shoehorn while it exists)
    - watchdog:             the shoehorn-watchdog task watches 3080 only (remote) or 8090,3080 (local)
    - agent instructions:   the "services on this machine" text in both agent presets
  Dry run by default: prints what would change. -Apply writes, after backing up cordis.patch.yml.
  Restart dsh afterwards (start-detached.ps1, or stop the node process on 127.0.0.1:3080 and let the watchdog
  restart it).

  The server side is separate: run shoehorn there with --host 0.0.0.0 --port 8090 and the SAME API key as this PC
  (DEEPSEEK_API_KEY in ~\.dsh\.env, written by start-detached.ps1 to ~\.dsh\llama-api-key.txt).

.EXAMPLE
  .\scripts\use-remote-shoehorn.ps1 -Url http://192.168.68.60:8090          # dry run
  .\scripts\use-remote-shoehorn.ps1 -Url http://192.168.68.60:8090 -Apply
  .\scripts\use-remote-shoehorn.ps1 -Local -Apply                            # back to shoehorn on this PC
#>
[CmdletBinding(DefaultParameterSetName = 'Remote')]
param(
    [Parameter(ParameterSetName = 'Remote', Mandatory)][string] $Url,
    [Parameter(ParameterSetName = 'Local', Mandatory)][switch] $Local,
    [switch] $Apply,
    [string] $Profile = (Join-Path $env:USERPROFILE '.dsh\profiles\web\cordis.patch.yml'),
    [string] $UrlFile = 'C:\models\shoehorn-url.txt',
    [string] $WatchdogTask = 'shoehorn-watchdog'
)
$ErrorActionPreference = 'Stop'

if ($Local) {
    $base = 'http://127.0.0.1:8090'
} else {
    $u = $null
    if (-not [Uri]::TryCreate($Url.Trim(), 'Absolute', [ref]$u) -or $u.Scheme -notin 'http', 'https') {
        throw "Url must be an absolute http(s) URL such as http://192.168.68.60:8090 (got '$Url')"
    }
    $base = $u.GetLeftPart('Authority')
}
$mode = if ($Local) { 'local' } else { "remote $base" }
"Mode: $mode$(if (-not $Apply) { '  (dry run: add -Apply to write)' })"

# --- agent instructions: the two "services" texts, local and remote wording -------------------------------------
$i = ' ' * 16
$localA = "${i}- 127.0.0.1:8090 shoehorn: the local inference server that runs YOU.`n" +
          "${i}  The GPU's VRAM is fully used by it, so do not start GPU-heavy work,`n" +
          "${i}  and never kill ``shoehorn``."
$remoteA = "${i}- shoehorn, the inference server that runs YOU, is on a separate machine`n" +
           "${i}  ($base). It is not managed from here; this PC's GPU is free for other work."
$localB = "${i}Services - do not stop, restart or reconfigure them: 127.0.0.1:8090`n" +
          "${i}shoehorn (the local model that runs YOU; VRAM is full, so no`n" +
          "${i}GPU-heavy work and never kill shoehorn), 127.0.0.1:3080 dsh itself"
$remoteB = "${i}Services - do not stop, restart or reconfigure them: shoehorn (the model`n" +
           "${i}that runs YOU) is on a separate machine ($base), not managed from here;`n" +
           "${i}this PC's GPU is free. 127.0.0.1:3080 dsh itself"

$text = [IO.File]::ReadAllText($Profile)
$nl = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
$work = $text.Replace("`r`n", "`n")
$changes = [Collections.Generic.List[string]]::new()

# --- model routes -------------------------------------------------------------------------------------------------
$routeRx = [regex]'(?m)^(\s+baseURL:\s*)https?://[^/\s]+:8090/v1\s*$'
$routes = $routeRx.Matches($work).Count
$work = $routeRx.Replace($work, { param($m) $m.Groups[1].Value + "$base/v1" })
$changes.Add("dsh model routes: $routes baseURL -> $base/v1")

# --- shoehorn panel row -------------------------------------------------------------------------------------------
$rowRx = [regex]'(?ms)^- id: ui-shoehorn\n(?:  .*\n?)*'
$row = "- id: ui-shoehorn`n  name: `"@deepseek-ai/dsh-client-ui-shoehorn`"`n  config:`n" +
       "    # where dsh reaches shoehorn (status chip, dashboard, sensors); written by scripts/use-remote-shoehorn.ps1`n" +
       "    url: $base`n"
if ($rowRx.IsMatch($work)) { $work = $rowRx.Replace($work, $row, 1) } else { $work = $work.TrimEnd("`n") + "`n" + $row }
$changes.Add("shoehorn panel url -> $base")

# --- agent instructions -------------------------------------------------------------------------------------------
foreach ($pair in @(@($localA, $remoteA), @($localB, $remoteB))) {
    $from = if ($Local) { $pair[1] } else { $pair[0] }
    $to = if ($Local) { $pair[0] } else { $pair[1] }
    # the remote text embeds the URL: on a switch between two remote URLs, match it with any URL
    $fromRx = [regex]([regex]::Escape($from) -replace [regex]::Escape([regex]::Escape($base)), 'https?://[^)\s]+')
    $otherRemoteRx = [regex]([regex]::Escape($pair[1]) -replace [regex]::Escape([regex]::Escape($base)), 'https?://[^)\s]+')
    if ($work.Contains($to)) { $changes.Add('agent instructions: already ' + $mode); continue }
    $rx = if ($Local) { $otherRemoteRx } else { [regex]::new($fromRx.ToString() + '|' + $otherRemoteRx.ToString()) }
    if ($rx.IsMatch($work)) { $work = $rx.Replace($work, $to.Replace('$', '$$'), 1); $changes.Add('agent instructions: updated') }
    else { $changes.Add('agent instructions: text not found (edited by hand?) - left as is') }
}

$changes | ForEach-Object { "  - $_" }
$fileLine = if ($Local) { "remove $UrlFile" } else { "write $UrlFile = $base" }
"  - start script: $fileLine"
$ports = if ($Local) { '8090,3080' } else { '3080' }
"  - watchdog task '$WatchdogTask': -Ports $ports"
if (-not $Apply) { return }

$backup = "$Profile.bak-$(Get-Date -Format yyyyMMdd-HHmmss)-shoehorn-$(if ($Local) { 'local' } else { 'remote' })"
Copy-Item $Profile $backup
[IO.File]::WriteAllText($Profile, $work.Replace("`n", $nl))
if ($Local) { Remove-Item $UrlFile -ErrorAction SilentlyContinue }
else { Set-Content $UrlFile @('# shoehorn on a dedicated machine (scripts/use-remote-shoehorn.ps1); delete for a local shoehorn', $base) }
$task = Get-ScheduledTask -TaskName $WatchdogTask -ErrorAction SilentlyContinue
if ($task) {
    $action = $task.Actions[0]
    $action.Arguments = $action.Arguments -replace '-Ports\s+[\d,]+', "-Ports $ports"
    Set-ScheduledTask -TaskName $WatchdogTask -Action $action | Out-Null
} else { "  (no scheduled task '$WatchdogTask' - skipped)" }
"Written. Backup: $backup"
"Restart dsh to apply (C:\models\start-detached.ps1 after stopping dsh; the watchdog picks up its new ports at next logon)."
