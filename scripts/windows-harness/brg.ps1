# Authenticated protocol-v2 client; output is forwarded as each frame arrives.
param(
    [Parameter(Position=0, Mandatory)][string]$Command,
    [Parameter(Position=1)][int]$Port = 9999,
    [Parameter(Position=2)][string]$RHost = '127.0.0.1',
    [string]$WorkingDirectory = $PWD.Path,
    [string]$TokenFile = $(if ($env:DSH_BRIDGE_TOKEN_FILE) { $env:DSH_BRIDGE_TOKEN_FILE } else { Join-Path $env:USERPROFILE '.dsh\bridge-token.txt' })
)
$ErrorActionPreference = 'Stop'
$client = [Net.Sockets.TcpClient]::new()
$code = 1
try {
    $token = [IO.File]::ReadAllText($TokenFile).Trim()
    if ($token.Length -lt 32) { throw 'Bridge token is missing or too short' }
    $client.Connect($RHost, $Port)
    $stream = $client.GetStream()
    $stream.ReadTimeout = 130000
    $encoding = [Text.UTF8Encoding]::new($false)
    $writer = [IO.StreamWriter]::new($stream, $encoding, 1024, $true)
    $reader = [IO.StreamReader]::new($stream, $encoding, $false, 1024, $true)
    $request = @{version=2;token=$token;command=$Command;cwd=[IO.Path]::GetFullPath($WorkingDirectory)}
    $writer.WriteLine(($request | ConvertTo-Json -Compress -EscapeHandling EscapeNonAscii))
    $writer.Flush()
    $receivedExit = $false
    while ($null -ne ($line = $reader.ReadLine())) {
        if ($line.Length -gt 16777216) { throw 'Bridge frame exceeds size limit' }
        $frame = $line | ConvertFrom-Json
        switch ($frame.kind) {
            'output' {
                if ($frame.channel -eq 'stdout') { [Console]::Out.Write([string]$frame.data); [Console]::Out.Flush() }
                elseif ($frame.channel -eq 'stderr') { [Console]::Error.Write([string]$frame.data); [Console]::Error.Flush() }
                else { throw 'Unknown bridge output channel' }
            }
            'exit' { $code=[long]$frame.code; $receivedExit=$true }
            default { throw 'Unknown bridge frame' }
        }
        if ($receivedExit) { break }
    }
    if (-not $receivedExit) { throw 'Bridge connection closed without an exit status' }
} catch {
    [Console]::Error.WriteLine('bridge failed: ' + $_.Exception.Message)
    $code = 99
} finally { $client.Dispose() }
if ($code -gt 2147483647) { exit ([int]($code - 4294967296)) }
exit ([int]$code)
