<#
.SYNOPSIS
    Start the SSD-backed external-MoE runtime with a validated profile.

.DESCRIPTION
    Resolves a named profile from profiles/profiles.json into the exact environment
    variables and command-line flags it needs, checks that this machine can actually
    host it, prints the resolved configuration, and starts llama-cli or llama-server.

    The launcher never invents or silently adjusts a semantic setting. If your machine
    cannot comfortably run the profile you asked for, it warns and asks; it does not
    quietly substitute a smaller one.

.PARAMETER Model
    Path to the GGUF. The runtime opens it read-only.

.PARAMETER Profile
    One of the profiles in profiles/profiles.json. Run with -List to see them.

.PARAMETER Server
    Start llama-server instead of the interactive CLI.

.PARAMETER Port
    Server port. Default 8080.

.PARAMETER Force
    Proceed even if a resource check fails.

.PARAMETER DryRun
    Print the resolved configuration and exit without starting anything.

.EXAMPLE
    .\run-local-moe.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf -Profile LOW_MEMORY_EXACT

.EXAMPLE
    .\run-local-moe.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf -Profile BALANCED_EXACT -Server -Port 8080
#>
[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(ParameterSetName = 'Run', Mandatory)]
    [string] $Model,

    [Parameter(ParameterSetName = 'Run')]
    [string] $Profile = 'LOW_MEMORY_EXACT',

    [Parameter(ParameterSetName = 'List')]
    [switch] $List,

    [Parameter(ParameterSetName = 'Run')] [switch] $Server,
    [Parameter(ParameterSetName = 'Run')] [int]    $Port = 8080,
    [Parameter(ParameterSetName = 'Run')] [string] $BindAddress = '127.0.0.1',
    [Parameter(ParameterSetName = 'Run')] [int]    $Context,
    [Parameter(ParameterSetName = 'Run')] [int]    $Threads,
    [Parameter(ParameterSetName = 'Run')] [string] $Prompt,
    [Parameter(ParameterSetName = 'Run')] [switch] $Force,
    [Parameter(ParameterSetName = 'Run')] [switch] $DryRun,
    [Parameter(ParameterSetName = 'Run')] [switch] $NoLog,

    [string] $BinDir,
    [string] $ProfilesJson,
    [string] $LogDir,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here

if (-not $ProfilesJson) { $ProfilesJson = Join-Path $repo 'profiles\profiles.json' }
if (-not $BinDir)       { $BinDir       = Join-Path $repo 'clean-room\build\bin' }
if (-not $LogDir)       { $LogDir       = Join-Path $repo 'launcher\sessions' }

if (-not (Test-Path $ProfilesJson)) { throw "profiles.json not found at $ProfilesJson" }
$doc = Get-Content $ProfilesJson -Raw | ConvertFrom-Json

# --------------------------------------------------------------------------
# -List
# --------------------------------------------------------------------------
if ($List) {
    Write-Host ''
    Write-Host 'Available profiles' -ForegroundColor Cyan
    Write-Host ''
    foreach ($p in $doc.profiles) {
        if ($p.PSObject.Properties.Name -contains 'hidden' -and $p.hidden) { continue }
        $mem = '{0:N2} GiB' -f $p.expected_peak_working_set_gib
        Write-Host ("  {0,-20} {1,-7} ~{2,-10} ~{3,6:N1} tok/s" -f $p.name, $p.quality_class, $mem, $p.tg_reference_toks) -ForegroundColor White
        Write-Host ("  {0,-20} {1}" -f '', $p.intended_user) -ForegroundColor DarkGray
        Write-Host ''
    }
    Write-Host 'Throughput figures are reference measurements on the machine described in profiles.json.'
    Write-Host 'They are not guarantees for your hardware.'
    Write-Host ''
    exit 0
}

# --------------------------------------------------------------------------
# resolve profile
# --------------------------------------------------------------------------
$spec = $doc.profiles | Where-Object { $_.name -eq $Profile }
if (-not $spec) {
    $known = ($doc.profiles | Where-Object { -not ($_.PSObject.Properties.Name -contains 'hidden' -and $_.hidden) } | ForEach-Object { $_.name }) -join ', '
    throw "unknown profile '$Profile'. Available: $known. Run with -List for descriptions."
}
$common = $doc.common

$exeName = if ($Server) { 'llama-server.exe' } else { 'llama-cli.exe' }
$exe     = Join-Path $BinDir $exeName

# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
$problems = New-Object System.Collections.ArrayList
$warnings = New-Object System.Collections.ArrayList

if (-not (Test-Path $Model))  { [void]$problems.Add("model file not found: $Model") }
if (-not (Test-Path $exe))    { [void]$problems.Add("$exeName not found in $BinDir") }

if (Test-Path $Model) {
    $mi = Get-Item $Model
    if ($mi.Extension -ne '.gguf') { [void]$warnings.Add("model does not have a .gguf extension: $($mi.Name)") }
    if ($mi.Length -lt 1GB)        { [void]$warnings.Add("model file is only $([math]::Round($mi.Length/1MB)) MiB, which is far smaller than the profiled model") }
}

# stale processes
$stale = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -in @('llama-cli.exe', 'llama-server.exe') })
if ($stale.Count -gt 0) {
    [void]$problems.Add("a model process is already running: " + (($stale | ForEach-Object { "$($_.Name) pid $($_.ProcessId)" }) -join ', '))
}

# memory headroom
$os        = Get-CimInstance Win32_OperatingSystem
$cs        = Get-CimInstance Win32_ComputerSystem
$availGiB  = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 2)
$totalGiB  = [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
# A server session touches the whole expert cache slab and a lot more KV cache than a
# short benchmark run does, so its steady-state working set is materially higher than
# the profile's headline figure. Use the measured server value when one exists.
$specProps = @($spec.PSObject.Properties | ForEach-Object { $_.Name })
$needGiB   = [double]$spec.expected_peak_working_set_gib
$needBasis = 'CLI benchmark'
$memProps = @($doc.memory_model.PSObject.Properties | ForEach-Object { $_.Name })
if ($Server -and ($specProps -contains 'expected_peak_working_set_gib_server') -and $spec.expected_peak_working_set_gib_server) {
    $needGiB   = [double]$spec.expected_peak_working_set_gib_server
    $needBasis = 'measured server session'
} elseif ($Server -and ($memProps -contains 'server_overhead_gib')) {
    # Not measured for this profile, but the overhead was constant across the four that
    # were, and independent of cache size. Estimating is better than quoting a CLI figure
    # that a server will exceed by two gigabytes.
    $needGiB   = $needGiB + [double]$doc.memory_model.server_overhead_gib
    $needBasis = 'estimated: CLI figure + measured server overhead'
} elseif ($Server) {
    [void]$warnings.Add('no measured server working set for this profile; the figure below is from a short CLI run and a server session will use more.')
}

if ($availGiB -lt $needGiB) {
    [void]$problems.Add("profile $Profile expects about $needGiB GiB of working set but only $availGiB GiB of RAM is available right now")
} elseif ($availGiB -lt ($needGiB * 1.25)) {
    [void]$warnings.Add("only $availGiB GiB available against an expected $needGiB GiB working set; headroom is thin")
}

# GPU
$gpuName = $null; $gpuTotalMiB = 0
try {
    $line = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($line) {
        $parts = ($line | Select-Object -First 1) -split ',\s*'
        $gpuName = $parts[0]; $gpuTotalMiB = [int]$parts[1]
    }
} catch { }

if (-not $gpuName) {
    [void]$warnings.Add('no NVIDIA GPU detected. Every profile in this file was measured with the non-MoE layers offloaded to a CUDA device; CPU-only behaviour is untested here.')
} elseif ($gpuTotalMiB -lt [int]$spec.expected_peak_vram_mib) {
    [void]$problems.Add("profile $Profile peaked at $($spec.expected_peak_vram_mib) MiB of VRAM on the reference machine, but this GPU reports only $gpuTotalMiB MiB")
}

# hardware match against the validated machine
$cpu = @(Get-CimInstance Win32_Processor)[0]
$refCpu = $doc.reference_machine.cpu
$hardwareMatches = ($cpu.Name.Trim() -like '*i5-14400F*') -and ($gpuName -like '*4060*')
if (-not $hardwareMatches) {
    [void]$warnings.Add("this machine is not the validated reference machine. Reference: $refCpu + $($doc.reference_machine.gpu). Yours: $($cpu.Name.Trim()) + $(if ($gpuName) { $gpuName } else { 'no NVIDIA GPU' }). The throughput figures in this profile will not transfer; treat them as a reference point only.")
}

if ($problems.Count -gt 0) {
    Write-Host ''
    Write-Host 'CANNOT START' -ForegroundColor Red
    foreach ($p in $problems) { Write-Host "  - $p" -ForegroundColor Red }
    if (-not $Force) {
        Write-Host ''
        Write-Host 'Fix the above, or pass -Force to start anyway. Run with -List to see smaller profiles.' -ForegroundColor Yellow
        exit 2
    }
    Write-Host '  -Force given: continuing despite the above.' -ForegroundColor Yellow
}

# --------------------------------------------------------------------------
# build the command
# --------------------------------------------------------------------------
# A profile may carry its own context/n_cpu_moe (long-context profiles need both to
# differ from the 16384/30 that every other profile shares); fall back to the common
# block when it does not, which is every profile that existed before this.
$defaultContext = if ($specProps -contains 'context')   { [int]$spec.context }   else { [int]$common.context }
$defaultNCpuMoe = if ($specProps -contains 'n_cpu_moe')  { [int]$spec.n_cpu_moe } else { [int]$common.n_cpu_moe }

$useThreads = if ($PSBoundParameters.ContainsKey('Threads')) { $Threads } else { [int]$spec.threads }
$useContext = if ($PSBoundParameters.ContainsKey('Context')) { $Context } else { $defaultContext }

if ($PSBoundParameters.ContainsKey('Threads') -and $Threads -ne [int]$spec.threads) {
    [void]$warnings.Add("thread count overridden to $Threads (profile specifies $($spec.threads)). Thread count is one of the strongest effects measured in this project; the profile's reference throughput no longer applies.")
}
if ($PSBoundParameters.ContainsKey('Context') -and $Context -ne $defaultContext) {
    [void]$warnings.Add("context overridden to $Context (this profile was measured at $defaultContext). Memory figures will change.")
}

$argList = @(
    '-m', $Model
    '-ngl', '999'
    '-ncmoe', "$defaultNCpuMoe"
    '-t', "$useThreads"
    '-fa', $common.flash_attention
    '-ctk', $common.kv_cache_type_k
    '-ctv', $common.kv_cache_type_v
    '-b', "$($common.batch)"
    '-ub', "$($common.ubatch)"
    '-c', "$useContext"
    '--load-mode', $common.load_mode
    '--jinja'
)

if ($Server) {
    $argList += @('--host', $BindAddress, '--port', "$Port")
} else {
    $argList += @('-np', "$($common.parallel)")
    if ($Prompt) { $argList += @('-p', $Prompt) }
}
$argList += $ExtraArgs

$envVars = @{}
foreach ($prop in $spec.env.PSObject.Properties) { $envVars[$prop.Name] = [string]$prop.Value }

# --------------------------------------------------------------------------
# show resolved configuration
# --------------------------------------------------------------------------
Write-Host ''
Write-Host "=== resolved configuration ===" -ForegroundColor Cyan
Write-Host ("  profile        : {0}  ({1})" -f $spec.name, $spec.quality_class)
Write-Host ("  binary         : {0}" -f $exe)
Write-Host ("  model          : {0}" -f $Model)
Write-Host ("  threads        : {0}" -f $useThreads)
Write-Host ("  context        : {0}" -f $useContext)
Write-Host ("  CPU MoE layers : {0}" -f $defaultNCpuMoe)
if ($envVars.Count -gt 0) {
    Write-Host '  environment    :'
    foreach ($k in ($envVars.Keys | Sort-Object)) { Write-Host ("{0,-34} = {1}" -f "    $k", $envVars[$k]) }
} else {
    Write-Host '  environment    : (none - resident path)'
}
Write-Host ("  expected mem   : ~{0:N2} GiB working set [{1}]   (this machine has {2:N2} GiB free of {3:N2} GiB)" -f $needGiB, $needBasis, $availGiB, $totalGiB)
Write-Host ("  reference TG   : ~{0:N1} tok/s on {1}" -f $spec.tg_reference_toks, $doc.reference_machine.cpu)
if ($Server) { Write-Host ("  server         : http://{0}:{1}" -f $BindAddress, $Port) }
Write-Host ''
Write-Host '  command:' -ForegroundColor DarkGray
Write-Host ("    {0} {1}" -f (Split-Path $exe -Leaf), (($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')) -ForegroundColor DarkGray
Write-Host ''

if ($spec.quality_class -eq 'REPACK') {
    Write-Host 'NOTE: this is a REPACK profile.' -ForegroundColor Yellow
    Write-Host '  Its token stream is deterministic but NOT bit-identical to the EXACT reference.' -ForegroundColor Yellow
    Write-Host "  $($doc.correctness_classes.REPACK.quality_statement)" -ForegroundColor Yellow
    Write-Host ''
}
if ($spec.quality_class -eq 'LONG_CONTEXT_LOW_RAM') {
    Write-Host 'NOTE: this is a LONG_CONTEXT_LOW_RAM profile.' -ForegroundColor Yellow
    Write-Host '  Its token stream is deterministic but NOT bit-identical to the EXACT reference' -ForegroundColor Yellow
    Write-Host '  past a few hundred tokens (coherent, not corrupted). See docs/LONG_CONTEXT.md.' -ForegroundColor Yellow
    Write-Host "  $($doc.correctness_classes.LONG_CONTEXT_LOW_RAM.quality_statement)" -ForegroundColor Yellow
    Write-Host ''
}

foreach ($w in $warnings) { Write-Host "WARNING: $w" -ForegroundColor Yellow }
if ($warnings.Count -gt 0) { Write-Host '' }

if ($DryRun) { Write-Host '-DryRun: not starting.' ; exit 0 }

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
$sessionLog = $null
if (-not $NoLog) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $sessionLog = Join-Path $LogDir ("{0}-{1}.log" -f $Profile, (Get-Date -Format 'yyyyMMdd-HHmmss'))
    $header = @(
        "session   : $(Get-Date -Format o)"
        "profile   : $($spec.name) ($($spec.quality_class))"
        "binary    : $exe"
        "model     : $Model"
        "env       : " + (($envVars.Keys | Sort-Object | ForEach-Object { "$_=$($envVars[$_])" }) -join ' ')
        "args      : " + ($argList -join ' ')
        "machine   : $($cpu.Name.Trim()) / $totalGiB GiB / $(if ($gpuName) { $gpuName } else { 'no NVIDIA GPU' })"
        ''
    )
    Set-Content -Path $sessionLog -Value $header -Encoding UTF8
    Write-Host "session log: $sessionLog" -ForegroundColor DarkGray
    Write-Host ''
}

foreach ($k in $envVars.Keys) { Set-Item "Env:$k" $envVars[$k] }

$proc = $null
try {
    # Run in the foreground so Ctrl+C reaches the child directly and the model can
    # shut down cleanly. Start-Process would detach it from the console.
    $proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru
    $proc.WaitForExit()
    $code = $proc.ExitCode
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Write-Host ''
        Write-Host 'stopping model...' -ForegroundColor DarkGray
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        try { [void]$proc.WaitForExit(15000) } catch { }
    }
    foreach ($k in $envVars.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }

    # never leave a detached model process behind
    $leftover = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -in @('llama-cli.exe', 'llama-server.exe') })
    if ($leftover.Count -gt 0) {
        Write-Host "cleaning up $($leftover.Count) leftover process(es)" -ForegroundColor Yellow
        foreach ($l in $leftover) { Stop-Process -Id ([int]$l.ProcessId) -Force -ErrorAction SilentlyContinue }
    }
    if ($sessionLog) { Add-Content -Path $sessionLog -Value "exited  : $(Get-Date -Format o)" -Encoding UTF8 }
}

exit $code
