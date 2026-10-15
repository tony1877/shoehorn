<#
.SYNOPSIS
  Windows release zip: dist\shoehorn-<ver>-windows-x86_64-cuda.zip from build\bin\shoehorn.exe (scripts\build.ps1 first).
  Bundles what the exe loads besides Windows itself: cuBLAS 13 (cublas64_13, cublasLt64_13; the CUDA runtime is linked
  statically) and the VC++ runtime with OpenMP (msvcp140, vcruntime140(_1), vcomp140). The host needs only the NVIDIA
  driver (580+ for CUDA 13).
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe  = Join-Path $root 'build\bin\shoehorn.exe'
if (-not (Test-Path $exe)) { throw "build first: scripts\build.ps1 ($exe is missing)" }
$ver = (Select-String -Path (Join-Path $root 'CMakeLists.txt') -Pattern 'project\(shoehorn VERSION ([0-9.]+)').Matches[0].Groups[1].Value
$got = (& $exe --version) -replace '^shoehorn ', ''
if ($got -ne $ver) { throw "build\bin\shoehorn.exe is $got, CMakeLists.txt says ${ver}: rebuild" }

$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' }
$vs   = & (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe') -products * -version '[17.0,18.0)' -latest -property installationPath
$crt  = Get-ChildItem (Join-Path $vs 'VC\Redist\MSVC') -Directory | Where-Object Name -match '^\d' | Sort-Object Name | Select-Object -Last 1
$dlls = @(
    (Join-Path $cuda 'bin\x64\cublas64_13.dll'), (Join-Path $cuda 'bin\x64\cublasLt64_13.dll'),
    (Join-Path $crt.FullName 'x64\Microsoft.VC143.CRT\msvcp140.dll'), (Join-Path $crt.FullName 'x64\Microsoft.VC143.CRT\vcruntime140.dll'),
    (Join-Path $crt.FullName 'x64\Microsoft.VC143.CRT\vcruntime140_1.dll'), (Join-Path $crt.FullName 'x64\Microsoft.VC143.OpenMP\vcomp140.dll'))

$name  = "shoehorn-$ver-windows-x86_64-cuda"
$dist  = Join-Path $root 'dist'
$stage = Join-Path $env:TEMP $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage, $dist | Out-Null
Copy-Item (@($exe) + $dlls) $stage
Copy-Item (Join-Path $root 'README.md'), (Join-Path $root 'LICENSE') $stage
Copy-Item (Join-Path $cuda 'EULA.txt') (Join-Path $stage 'CUDA-EULA.txt')
@"
shoehorn $ver for Windows 10/11 x64 with an NVIDIA GPU (driver 580 or newer).

  shoehorn.exe serve                 start the server on port 8090; open http://127.0.0.1:8090/models to add a model
  shoehorn.exe serve model.gguf      serve one GGUF (or a packed pair: --res model.res.gguf)
  shoehorn.exe --help                every command

Images: give a model its vision encoder (mmproj GGUF) on the Models page, see docs/VISION.md in the repository.
cuBLAS (NVIDIA, see CUDA-EULA.txt) and the Microsoft VC++ runtime are included; nothing else to install.
https://github.com/tony1877/shoehorn
"@ | Set-Content -Encoding ASCII (Join-Path $stage 'README-windows.txt')

$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -Recurse -Force $stage
Write-Host ("{0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
