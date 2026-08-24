<#
    BenchLib.psm1 - shared measurement primitives for the benchmark suite.

    Everything a third party needs to reproduce a measurement lives here:
    environment capture, stale-process detection, a single instrumented run,
    and the safety-invariant check.

    No function in this module writes outside the directory it is given.
#>

Set-StrictMode -Version Latest

$script:LlamaProcessNames = @('llama-cli.exe', 'llama-server.exe', 'llama-perplexity.exe')

# --------------------------------------------------------------------------
# process hygiene
# --------------------------------------------------------------------------

function Get-ActiveLlamaProcess {
    <#
      .SYNOPSIS
        Every llama-* process currently running, from any session.
    #>
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in $script:LlamaProcessNames })
}

function Assert-NoStaleLlama {
    param([string] $When = 'pre-flight')

    $active = @(Get-ActiveLlamaProcess)
    if ($active.Count -gt 0) {
        $desc = ($active | ForEach-Object { "$($_.Name)(pid $($_.ProcessId))" }) -join ', '
        throw "STALE_LLAMA at ${When}: $desc. Refusing to measure - a second model instance invalidates every memory figure."
    }
}

function Stop-StaleLlama {
    <#
      .SYNOPSIS
        Terminate leftover llama-* processes. Only ever called on the suite's own
        cleanup path, never on a machine the caller has not opted in for.
    #>
    foreach ($p in @(Get-ActiveLlamaProcess)) {
        Stop-Process -Id ([int]$p.ProcessId) -Force -ErrorAction SilentlyContinue
    }
}

# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------

function Get-FileIdentity {
    param([string] $Path)

    if (-not (Test-Path $Path)) { return $null }
    $i = Get-Item $Path
    [ordered]@{
        path          = $i.FullName
        bytes         = $i.Length
        last_write_utc = $i.LastWriteTimeUtc.ToString('o')
        sha256        = (Get-FileHash $i.FullName -Algorithm SHA256).Hash
    }
}

function Get-ModelIdentity {
    <#
      .SYNOPSIS
        Identify the GGUF without hashing all of it.
      .DESCRIPTION
        A full hash of a 19 GiB file evicts the Windows page cache and perturbs
        the very thing being measured. The head hash pins the header, metadata
        and tensor index, which is what identifies the model.
    #>
    param([string] $Path, [int] $HeadMiB = 16)

    if (-not (Test-Path $Path)) { throw "model not found: $Path" }
    $i = Get-Item $Path
    $fs = [System.IO.File]::OpenRead($Path)
    try {
        $n   = $HeadMiB * 1MB
        $buf = New-Object byte[] $n
        $read = $fs.Read($buf, 0, $n)
    } finally { $fs.Dispose() }

    $sha  = [System.Security.Cryptography.SHA256]::Create()
    $hash = ($sha.ComputeHash($buf, 0, $read) | ForEach-Object { $_.ToString('X2') }) -join ''

    [ordered]@{
        path            = $i.FullName
        bytes           = $i.Length
        last_write_utc  = $i.LastWriteTimeUtc.ToString('o')
        head_mib        = $HeadMiB
        head_sha256     = $hash
    }
}

function Get-BenchmarkEnvironment {
    param(
        [string] $BinDir,
        [string] $ModelPath
    )

    $os   = Get-CimInstance Win32_OperatingSystem
    $cs   = Get-CimInstance Win32_ComputerSystem
    $cpu  = @(Get-CimInstance Win32_Processor)[0]
    $cv   = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction SilentlyContinue

    $gpu = $null
    try {
        $line = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
        if ($line) {
            $parts = ($line | Select-Object -First 1) -split ',\s*'
            $gpu = [ordered]@{ name = $parts[0]; driver = $parts[1]; memory_total = $parts[2] }
        }
    } catch { $gpu = $null }

    $binaries = [ordered]@{}
    if ($BinDir -and (Test-Path $BinDir)) {
        Get-ChildItem $BinDir -File |
            Where-Object { $_.Extension -in '.exe', '.dll' } |
            Sort-Object Name |
            ForEach-Object { $binaries[$_.Name] = (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
    }

    # Storage class of the volume the model lives on - NVMe vs SATA changes the
    # external-cache numbers materially, so it belongs in the record.
    $storage = $null
    try {
        $drive = (Split-Path $ModelPath -Qualifier)
        $part  = Get-Partition -ErrorAction SilentlyContinue |
                 Where-Object { $_.DriveLetter -eq $drive.TrimEnd(':') }
        if ($part) {
            $disk = Get-PhysicalDisk -ErrorAction SilentlyContinue |
                    Where-Object { $_.DeviceId -eq $part.DiskNumber }
            if ($disk) {
                $storage = [ordered]@{
                    volume     = $drive
                    media_type = [string]$disk.MediaType
                    bus_type   = [string]$disk.BusType
                    model      = [string]$disk.FriendlyName
                }
            }
        }
    } catch { $storage = $null }

    [ordered]@{
        captured_utc      = (Get-Date).ToUniversalTime().ToString('o')
        cpu               = [ordered]@{
            name           = $cpu.Name.Trim()
            cores          = $cpu.NumberOfCores
            logical        = $cpu.NumberOfLogicalProcessors
            max_clock_mhz  = $cpu.MaxClockSpeed
        }
        ram               = [ordered]@{
            total_mib      = [math]::Round($cs.TotalPhysicalMemory / 1MB, 0)
            free_mib_at_capture = [math]::Round($os.FreePhysicalMemory / 1024, 0)
        }
        gpu               = $gpu
        storage           = $storage
        os                = [ordered]@{
            caption        = $os.Caption
            version        = $os.Version
            display_version = if ($cv) { $cv.DisplayVersion } else { $null }
            ubr            = if ($cv) { $cv.UBR } else { $null }
        }
        binaries          = $binaries
        model             = Get-ModelIdentity -Path $ModelPath
        other_ai_runtimes = @(Get-Process -ErrorAction SilentlyContinue |
                              Where-Object { $_.ProcessName -match 'ollama|lmstudio|koboldcpp|text-generation' } |
                              ForEach-Object { $_.ProcessName } | Sort-Object -Unique)
    }
}

# --------------------------------------------------------------------------
# profile resolution
# --------------------------------------------------------------------------

function Import-BenchmarkProfile {
    param(
        [Parameter(Mandatory)] [string] $ProfilesJson,
        [Parameter(Mandatory)] [string] $Name
    )

    $doc = Get-Content $ProfilesJson -Raw | ConvertFrom-Json
    $p   = $doc.profiles | Where-Object { $_.name -eq $Name }
    if (-not $p) {
        $known = ($doc.profiles | ForEach-Object { $_.name }) -join ', '
        throw "unknown profile '$Name'. Known profiles: $known"
    }
    [pscustomobject]@{
        Profile = $p
        Common  = $doc.common
        Doc     = $doc
    }
}

function ConvertTo-EnvHashtable {
    param($EnvObject)

    $h = @{}
    if ($null -ne $EnvObject) {
        foreach ($prop in $EnvObject.PSObject.Properties) { $h[$prop.Name] = [string]$prop.Value }
    }
    $h
}

# --------------------------------------------------------------------------
# a single measured run
# --------------------------------------------------------------------------

# Every environment variable the runtime reads that could change a measurement.
# All of them are cleared before each run so an inherited value from the shell
# can never silently alter a result.
$script:RuntimeEnvVars = @(
    'B1B_EXTERNAL_EXPERT_STORAGE', 'B1B_EXTERNAL_MAX_LAYERS', 'B1B_MOE_REPACK',
    'B1B_DETACH_CPU_EXPERTS', 'B1B_LOOKUP_LOGGING', 'B1B_PLACEMENT_DIAGNOSTICS',
    'B1B_SCHEDULER_PROBE', 'B1B_TRACE_BUFFER_BOUNDARY', 'B1B_TRACE_BUFFER_WRITES',
    'B1B_TRACE_POINTER_FLOW', 'B1B_TRACE_TENSOR_LIFETIME',
    'B2D_TIMING_PATH', 'B2E_PREFETCH_QD',
    'CACHE_LOAD_VERIFY_ONLY', 'CACHE_USE_CACHED_POINTER', 'CACHE_VALIDATE_CACHED_BOUNDS_ONLY',
    'LLAMA_EXPERT_B1N', 'LLAMA_EXPERT_B1P', 'LLAMA_EXPERT_B1S_OUT', 'LLAMA_EXPERT_B1T_OUT',
    'LLAMA_EXPERT_B1W', 'LLAMA_EXPERT_CACHE', 'LLAMA_EXPERT_CACHE_MB', 'LLAMA_EXPERT_DECOMP',
    'LLAMA_EXPERT_GEOMETRY_OUT', 'LLAMA_EXPERT_INDIRECTION', 'LLAMA_EXPERT_INDIRECTION_STATS',
    'LLAMA_EXPERT_LOCK_PROFILE', 'LLAMA_EXPERT_LOCK_THRESHOLD_NS', 'LLAMA_EXPERT_PERF_OUT',
    'LLAMA_EXPERT_RANGE_OUT', 'LLAMA_EXPERT_TIMING', 'LLAMA_PHASE_TRACE_OUT',
    'LLAMA_ROUTING_TRACE_OUT', 'LLAMA_STORAGE_VALIDATOR_OUT', 'LLAMA_TOKEN_TRACE_OUT',
    'LLAMA_TOP5_TRACE_INDEX', 'LLAMA_TOP5_TRACE_OUT', 'LLAMA_TRACE'
)

function Clear-RuntimeEnvironment {
    foreach ($v in $script:RuntimeEnvVars) {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
}

function Read-CounterFile {
    param([string] $Path)

    $h = [ordered]@{}
    if (Test-Path $Path) {
        foreach ($line in Get-Content $Path) {
            if ($line -match '^([^=]+)=(.*)$') { $h[$Matches[1]] = $Matches[2] }
        }
    }
    $h
}

# Invariants that must hold on every external-cache run. A non-zero value here
# is a correctness failure, not a performance observation.
$script:SafetyCounters = @(
    'resolver_failures', 'direct_read_failures', 'short_read_count', 'current_pins',
    'row_bounds_failures', 'invalid_unpins', 'evicted_while_pinned',
    'unexpected_mmap_fallbacks', 'geometry_invariant_failures',
    'resolved_pointer_range_failures', 'invalid_ids', 'external_buffer_failures',
    'unexpected_mmap_expert_bindings', 'vec_dot_reserved_pointer_violations',
    'row_offset_overflows', 'row_end_overflows'
)

function Test-SafetyInvariant {
    param([hashtable] $Counters, [System.Collections.Specialized.OrderedDictionary] $Ordered)

    $src = if ($Ordered) { $Ordered } else { $Counters }
    $violations = @()

    if ($src.Count -eq 0) { return [pscustomobject]@{ Pass = $null; Violations = @(); Reason = 'no counter file (non-external run)' } }

    foreach ($c in $script:SafetyCounters) {
        if ($src.Contains($c)) {
            $v = 0L
            if ([int64]::TryParse([string]$src[$c], [ref]$v) -and $v -ne 0) {
                $violations += "$c=$v"
            }
        }
    }
    # pins must balance unpins exactly
    if ($src.Contains('pins') -and $src.Contains('unpins')) {
        if ([int64]$src['pins'] -ne [int64]$src['unpins']) {
            $violations += "pins=$($src['pins']) != unpins=$($src['unpins'])"
        }
    }

    [pscustomobject]@{
        Pass       = ($violations.Count -eq 0)
        Violations = $violations
        Reason     = if ($violations.Count -eq 0) { 'all invariants hold' } else { 'invariant violation' }
    }
}

function Invoke-LlamaBenchRun {
    <#
      .SYNOPSIS
        Run llama-cli once under full instrumentation and return one result record.

      .PARAMETER Tag
        Short identifier; becomes the raw-log filename stem.

      .PARAMETER StopOnPressure
        Abort the run if the machine runs out of headroom. Prevents a benchmark
        from pushing the host into swap and reporting the resulting numbers as if
        they were valid.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string]   $Tag,
        [Parameter(Mandatory)] [string]   $Exe,
        [Parameter(Mandatory)] [string]   $ModelPath,
        [Parameter(Mandatory)] [string]   $OutDir,
        [Parameter(Mandatory)] [int]      $Tokens,
        [Parameter(Mandatory)] [hashtable] $EnvVars,
        [int]      $Threads       = 4,
        [int]      $NCpuMoe       = 30,
        [int]      $Context       = 16384,
        [int]      $Batch         = 256,
        [int]      $UBatch        = 256,
        [string]   $Prompt        = 'Explain_MoE_routing',
        [string]   $PromptFile,
        [int]      $Seed          = 42,
        [int]      $TimeoutSec    = 1800,
        [switch]   $StopOnPressure,
        [double]   $MinAvailGiB   = 1.5,
        [double]   $MaxWorkingSetGiB = 24
    )

    Assert-NoStaleLlama -When "before $Tag"

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $so = Join-Path $OutDir "$Tag.stdout.log"
    $se = Join-Path $OutDir "$Tag.stderr.log"
    $pf = Join-Path $OutDir "$Tag.counters.txt"
    $tf = Join-Path $OutDir "$Tag.tokens.txt"
    Remove-Item $so, $se, $pf, $tf -Force -ErrorAction SilentlyContinue

    Clear-RuntimeEnvironment
    foreach ($k in $EnvVars.Keys) { Set-Item "Env:$k" $EnvVars[$k] }
    $env:LLAMA_TOKEN_TRACE_OUT = $tf
    $isExternal = $EnvVars.ContainsKey('B1B_EXTERNAL_EXPERT_STORAGE') -and $EnvVars['B1B_EXTERNAL_EXPERT_STORAGE'] -eq '1'
    if ($isExternal) { $env:LLAMA_EXPERT_PERF_OUT = $pf }

    $arguments = @(
        '-m', $ModelPath, '-ngl', '999', '-ncmoe', "$NCpuMoe", '-t', "$Threads",
        '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
        '-b', "$Batch", '-ub', "$UBatch", '-c', "$Context", '-np', '1',
        '-n', "$Tokens", '-no-cnv', '-st', '--seed', "$Seed", '--temp', '0',
        '--load-mode', 'none', '--jinja'
    )
    if ($PromptFile) { $arguments += @('-f', $PromptFile) } else { $arguments += @('-p', $Prompt) }

    $p = $null
    $peakWs = 0L; $peakPriv = 0L; $minAvail = [double]::MaxValue
    $peakVram = 0.0; $maxActive = 0; $cpuSec = 0.0
    $stop = 'NOT_STARTED'; $exit = $null
    $start = Get-Date

    try {
        $p = Start-Process -FilePath $Exe -ArgumentList $arguments `
                           -RedirectStandardOutput $so -RedirectStandardError $se `
                           -PassThru -WindowStyle Hidden

        while (-not $p.HasExited) {
            $active = @(Get-ActiveLlamaProcess)
            $maxActive = [math]::Max($maxActive, $active.Count)

            $q = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
            if ($q) {
                $q.Refresh()
                $peakWs   = [math]::Max($peakWs,   [int64]$q.WorkingSet64)
                $peakPriv = [math]::Max($peakPriv, [int64]$q.PrivateMemorySize64)
                $cpuSec   = [double]$q.CPU
            }

            $os = Get-CimInstance Win32_OperatingSystem
            $minAvail = [math]::Min($minAvail, [double]$os.FreePhysicalMemory * 1KB)

            $g = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
            if ($g -match '^\s*([0-9.]+)') { $peakVram = [math]::Max($peakVram, [double]$Matches[1]) }

            if ($active.Count -gt 1)                                   { $stop = 'STOP_PROCESS_COUNT'; break }
            if (((Get-Date) - $start).TotalSeconds -gt $TimeoutSec)    { $stop = 'STOP_TIMEOUT';       break }
            if ($StopOnPressure) {
                if ($peakWs -ge ($MaxWorkingSetGiB * 1GB))             { $stop = 'STOP_WORKING_SET';   break }
                if ($minAvail -lt ($MinAvailGiB * 1GB))                { $stop = 'STOP_AVAILABLE_RAM'; break }
            }

            Start-Sleep -Milliseconds 500
            $p.Refresh()
        }

        if ($stop -eq 'NOT_STARTED') { $p.WaitForExit(); $p.Refresh(); $exit = $p.ExitCode }
    }
    finally {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            try { [void]$p.WaitForExit(10000) } catch { }
        }
        Clear-RuntimeEnvironment
    }

    Assert-NoStaleLlama -When "after $Tag"

    $stderrText = if (Test-Path $se) { Get-Content $se -Raw } else { '' }
    if ($null -eq $exit -and $stderrText -match 'PROCESS_EXIT\s+code=(\d+)') { $exit = [int]$Matches[1] }
    if ($stop -eq 'NOT_STARTED') { $stop = if ($exit -eq 0) { 'COMPLETED' } else { 'PROCESS_ERROR' } }

    $stdoutText = if (Test-Path $so) { Get-Content $so -Raw } else { '' }
    $raw = "$stdoutText`n$stderrText"
    $m = @([regex]::Matches($raw, 'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s'))

    $counters = Read-CounterFile $pf
    $safety   = Test-SafetyInvariant -Ordered $counters

    $tokenLines = @(Get-Content $tf -ErrorAction SilentlyContinue | Where-Object { $_ -match 'TOKEN_TRACE' })

    [pscustomobject]@{
        tag                 = $Tag
        utc                 = $start.ToUniversalTime().ToString('o')
        tokens_requested    = $Tokens
        tokens_emitted      = $tokenLines.Count
        threads             = $Threads
        n_cpu_moe           = $NCpuMoe
        context             = $Context
        batch               = $Batch
        ubatch              = $UBatch
        env                 = $EnvVars
        exit_code           = $exit
        stop_reason         = $stop
        pp_toks             = if ($m.Count) { [double]$m[-1].Groups[1].Value } else { $null }
        tg_toks             = if ($m.Count) { [double]$m[-1].Groups[2].Value } else { $null }
        wall_sec            = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)
        process_cpu_sec     = [math]::Round($cpuSec, 2)
        peak_working_set_gib = [math]::Round($peakWs   / 1GB, 3)
        peak_private_gib     = [math]::Round($peakPriv / 1GB, 3)
        min_available_ram_gib = if ($minAvail -eq [double]::MaxValue) { $null } else { [math]::Round($minAvail / 1GB, 3) }
        peak_vram_mib        = $peakVram
        max_concurrent_llama = $maxActive
        token_trace_sha256   = if (Test-Path $tf) { (Get-FileHash $tf -Algorithm SHA256).Hash } else { $null }
        safety_pass          = $safety.Pass
        safety_violations    = $safety.Violations
        counters             = $counters
        raw_logs             = [ordered]@{ stdout = $so; stderr = $se; counters = $pf; tokens = $tf }
    }
}

function Test-RunValid {
    <#
      .SYNOPSIS
        A run is valid only if it finished cleanly, emitted every requested token,
        never had a second model instance alive, and broke no safety invariant.
    #>
    param([Parameter(Mandatory)] $Run)

    $reasons = @()
    if ($Run.exit_code -ne 0)                      { $reasons += "exit_code=$($Run.exit_code)" }
    if ($Run.stop_reason -ne 'COMPLETED')          { $reasons += "stop=$($Run.stop_reason)" }
    if ($Run.tokens_emitted -ne $Run.tokens_requested) { $reasons += "tokens=$($Run.tokens_emitted)/$($Run.tokens_requested)" }
    if ($Run.max_concurrent_llama -gt 1)           { $reasons += "concurrent=$($Run.max_concurrent_llama)" }
    if ($Run.safety_pass -eq $false)               { $reasons += "safety: $($Run.safety_violations -join '; ')" }
    if ($null -eq $Run.tg_toks)                    { $reasons += 'no throughput line parsed' }

    [pscustomobject]@{ Valid = ($reasons.Count -eq 0); Reasons = $reasons }
}

Export-ModuleMember -Function `
    Get-ActiveLlamaProcess, Assert-NoStaleLlama, Stop-StaleLlama,
    Get-FileIdentity, Get-ModelIdentity, Get-BenchmarkEnvironment,
    Import-BenchmarkProfile, ConvertTo-EnvHashtable,
    Clear-RuntimeEnvironment, Read-CounterFile, Test-SafetyInvariant,
    Invoke-LlamaBenchRun, Test-RunValid
