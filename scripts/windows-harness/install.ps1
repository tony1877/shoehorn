# Install the tested support scripts and template. Running services are not restarted.
param([string]$DshHome=(Join-Path $env:USERPROFILE '.dsh'))
$ErrorActionPreference='Stop'
$destinations=@{
    'dsh-bridge.js'='C:\Tools\bin\dsh-bridge.js'
    'brg.ps1'='C:\Tools\bin\brg.ps1'
    'start-dsh.ps1'='C:\Tools\bin\start-dsh.ps1'
    'start-detached.ps1'='C:\models\start-detached.ps1'
    'web-search-searxng.mjs'='C:\Tools\dsh-plugins\web-search-searxng.mjs'
    'qwen-sharp-v22.5.0.jinja'='C:\models\templates\qwen-sharp-v22.5.0.jinja'
}
# Avoid installing a v2 client against a running v1 bridge. Stop the bridge as an
# explicit activation operation before calling this installer.
if(Get-NetTCPConnection -LocalPort 9999 -State Listen -ErrorAction SilentlyContinue){throw 'Stop the exec bridge before installing its coordinated v2 server and client.'}
$backup=Join-Path $DshHome ('backups\harness-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $backup | Out-Null
$tokenPath=Join-Path $DshHome 'bridge-token.txt'
if(-not(Test-Path -LiteralPath $tokenPath)){
    [IO.File]::WriteAllText($tokenPath,'')
    $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    & icacls.exe $tokenPath /inheritance:r /grant:r "*$($sid):(F)" '*S-1-5-18:(F)' '*S-1-5-32-544:(F)' | Out-Null
    if($LASTEXITCODE -ne 0){throw 'Failed to secure the bridge token ACL'}
    [IO.File]::WriteAllText($tokenPath,[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)))
}
if(([IO.File]::ReadAllText($tokenPath).Trim()).Length -lt 32){throw 'Bridge token is too short; repair it before installing.'}
foreach($name in $destinations.Keys){
    $source=Join-Path $PSScriptRoot $name
    if(Test-Path -LiteralPath $destinations[$name]){Copy-Item -LiteralPath $destinations[$name] -Destination (Join-Path $backup $name)}
    Copy-Item -LiteralPath $source -Destination $destinations[$name] -Force
}
$watchdog=Join-Path (Split-Path $PSScriptRoot -Parent) 'watchdog.ps1'
Copy-Item -LiteralPath 'C:\src\shoehorn\scripts\watchdog.ps1' -Destination (Join-Path $backup 'watchdog.ps1')
Copy-Item -LiteralPath $watchdog -Destination 'C:\src\shoehorn\scripts\watchdog.ps1' -Force
$env:DSH_HOME=$DshHome
node (Join-Path $PSScriptRoot 'configure-local.mjs')
if($LASTEXITCODE -ne 0){throw 'Local configuration failed; backups are available'}
Write-Output "Support files installed. Backup: $backup. Start the bridge and dsh to activate; inference template/server changes require a separate server restart."
