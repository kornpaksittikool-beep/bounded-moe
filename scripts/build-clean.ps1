<#
.SYNOPSIS
  Configure and build the external-expert-cache llama.cpp runtime from a clean tree.

.DESCRIPTION
  Reproduces the authoritative build configuration recorded in
  PRODUCTIZATION_BASELINE.md section 7. Writes only into -BuildDir; never touches
  the research build directory or the validated binaries.

.EXAMPLE
  .\build-clean.ps1 -SourceDir D:\llm-v0\productization-v1\clean-room\llama.cpp `
                    -BuildDir  D:\llm-v0\productization-v1\clean-room\build
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $SourceDir,
    [Parameter(Mandatory)] [string] $BuildDir,
    [string] $VcVars   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    [string] $CMake    = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
    [string] $Ninja    = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe',
    [string] $CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
    [switch] $Fresh,
    [string] $LogFile,
    [string[]] $ExtraCMakeArgs = @()
)

$ErrorActionPreference = 'Stop'

foreach ($p in @($VcVars, $CMake, $Ninja, $CudaRoot, $SourceDir)) {
    if (-not (Test-Path $p)) { throw "required path not found: $p" }
}

if ($Fresh -and (Test-Path $BuildDir)) {
    Write-Host "removing existing build directory $BuildDir"
    Remove-Item $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

if (-not $LogFile) { $LogFile = Join-Path $BuildDir 'build.log' }

# --- authoritative flag set -------------------------------------------------
# These defines are part of the measured configuration. Changing them produces a
# binary that is NOT comparable with the published Pareto frontier.
$projectDefines = '/DB1R4_COMPILE_OUT_DIAGNOSTICS=1 /DB1S_PROFILE=1 /DB1T_PROFILE=1 /DB1U_PROFILE=0 /DB1W_PROFILE=1'
$commonFlags    = "/Zi /O2 /Ob1 /DNDEBUG $projectDefines"

$cmakeArgs = @(
    '-G', 'Ninja'
    "-DCMAKE_MAKE_PROGRAM=$Ninja"
    '-DCMAKE_BUILD_TYPE=RelWithDebInfo'
    '-DBUILD_SHARED_LIBS=ON'
    '-DGGML_CUDA=ON'
    '-DGGML_CUDA_FA=ON'
    '-DGGML_CUDA_GRAPHS=ON'
    '-DGGML_NATIVE=ON'
    '-DGGML_BLAS=OFF'
    '-DLLAMA_BUILD_TESTS=OFF'
    '-DLLAMA_BUILD_EXAMPLES=OFF'
    "-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe"
    "-DCMAKE_C_FLAGS=$commonFlags"
    "-DCMAKE_CXX_FLAGS=$commonFlags"
    '-S', $SourceDir
    '-B', $BuildDir
)
# Caller overrides, appended last so they win. The common case is
# -DLLAMA_BUILD_UI=OFF, which makes the build fully offline.
$cmakeArgs += $ExtraCMakeArgs

$quoted = ($cmakeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '

# vcvars64.bat shells out to vswhere.exe by bare name, so the VS Installer
# directory must be on PATH before it is called.
$vsInstaller = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer'

# All output is redirected inside the batch file. Piping a native command's stderr
# through PowerShell 5.1 turns ordinary cmake status messages into NativeCommandError
# records and corrupts the exit code, so the batch owns the redirection.
$script = @"
set "PATH=$vsInstaller;%PATH%"
call "$VcVars" >nul 2>&1
if errorlevel 1 exit /b 1
"$CMake" $quoted >>"$LogFile" 2>&1
if errorlevel 1 exit /b 2
"$CMake" --build "$BuildDir" --target llama-cli llama-server llama-perplexity >>"$LogFile" 2>&1
if errorlevel 1 exit /b 3
exit /b 0
"@

$batch = Join-Path $BuildDir '_build.cmd'
Set-Content -Path $batch -Value $script -Encoding ascii

Remove-Item $LogFile -Force -ErrorAction SilentlyContinue

Write-Host "building: source=$SourceDir build=$BuildDir"
Write-Host "log: $LogFile"
$start = Get-Date

& cmd.exe /c "`"$batch`""
$code = $LASTEXITCODE

$elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 2)

switch ($code) {
    0 { Write-Host "BUILD_OK in $elapsed min" }
    1 { Write-Host "BUILD_FAILED: could not initialise MSVC environment" }
    2 { Write-Host "BUILD_FAILED: cmake configure" }
    3 { Write-Host "BUILD_FAILED: compile/link" }
    default { Write-Host "BUILD_FAILED: exit $code" }
}

if ($code -eq 0) {
    $bin = Join-Path $BuildDir 'bin'
    Get-ChildItem $bin -File |
        Where-Object { $_.Extension -in '.dll', '.exe' } |
        ForEach-Object {
            [pscustomobject]@{
                Name   = $_.Name
                Bytes  = $_.Length
                SHA256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
            }
        } | Format-Table -AutoSize
}

exit $code
