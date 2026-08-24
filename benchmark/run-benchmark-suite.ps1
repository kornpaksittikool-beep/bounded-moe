<#
.SYNOPSIS
    Reproducible benchmark suite for the SSD-backed external MoE expert cache.

.DESCRIPTION
    Measures one or more named profiles and writes machine-readable (JSON, CSV) and
    human-readable (Markdown) results plus every raw log.

    You do not need to know anything about this project's history to run it. Point it
    at a model file and a directory of binaries and it records the rest.

    QUICK mode  - sanity, correctness and a short throughput reading. Minutes.
    FULL  mode  - publication-grade: warm-up, repeats, and run-by-run interleaving
                  between profiles so that machine drift cannot masquerade as a
                  difference between configurations.

.PARAMETER Mode
    Quick or Full.

.PARAMETER Profiles
    Profile names from profiles/profiles.json. Defaults to the five headline profiles.

.EXAMPLE
    .\run-benchmark-suite.ps1 -Mode Quick -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf

.EXAMPLE
    .\run-benchmark-suite.ps1 -Mode Full -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf `
        -Profiles LOW_MEMORY_EXACT,MAX_SPEED_REPACK -OutDir .\results\ab
#>
[CmdletBinding()]
param(
    [ValidateSet('Quick', 'Full')]
    [string]   $Mode = 'Quick',

    [Parameter(Mandatory)]
    [string]   $Model,

    [string]   $BinDir,
    [string]   $ProfilesJson,
    [string]   $OutDir,
    [string[]] $Profiles = @('TINY_MEMORY_EXACT', 'LOW_MEMORY_EXACT', 'BALANCED_EXACT', 'MAX_SPEED_EXACT', 'MAX_SPEED_REPACK'),

    [int]      $Tokens,
    [int]      $Repeats,
    [switch]   $NoWarmup,
    [switch]   $SkipStaleCheck,
    [switch]   $StopOnPressure,
    [switch]   $Interleave
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here

Import-Module (Join-Path $here 'lib\BenchLib.psm1') -Force

if (-not $ProfilesJson) { $ProfilesJson = Join-Path $repo 'profiles\profiles.json' }
if (-not $BinDir)       { $BinDir       = Join-Path $repo 'clean-room\build\bin' }
if (-not $OutDir)       { $OutDir       = Join-Path $repo ('benchmark\results\{0}-{1}' -f $Mode.ToLower(), (Get-Date -Format 'yyyyMMdd-HHmmss')) }

# ---- mode defaults --------------------------------------------------------
if (-not $PSBoundParameters.ContainsKey('Tokens'))  { $Tokens  = if ($Mode -eq 'Full') { 512 } else { 128 } }
if (-not $PSBoundParameters.ContainsKey('Repeats')) { $Repeats = if ($Mode -eq 'Full') { 3 }   else { 1 } }

foreach ($p in @($Model, $ProfilesJson)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
$exe = Join-Path $BinDir 'llama-cli.exe'
if (-not (Test-Path $exe)) { throw "llama-cli.exe not found in $BinDir" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$rawDir = Join-Path $OutDir 'raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

Write-Host ''
Write-Host "=== benchmark suite: $Mode ===" -ForegroundColor Cyan
Write-Host "  model    : $Model"
Write-Host "  binaries : $BinDir"
Write-Host "  profiles : $($Profiles -join ', ')"
Write-Host "  tokens   : $Tokens   repeats: $Repeats"
Write-Host "  output   : $OutDir"
Write-Host ''

# ---- pre-flight -----------------------------------------------------------
if (-not $SkipStaleCheck) {
    $stale = @(Get-ActiveLlamaProcess)
    if ($stale.Count -gt 0) {
        Write-Host 'STALE PROCESSES DETECTED:' -ForegroundColor Red
        $stale | ForEach-Object { Write-Host "  $($_.Name) pid $($_.ProcessId)" }
        throw 'Refusing to benchmark with a model process already running. Stop them and retry, or pass -SkipStaleCheck if you know what you are doing.'
    }
    Write-Host 'pre-flight: no stale llama processes' -ForegroundColor Green
}

Write-Host 'capturing environment...'
$environment = Get-BenchmarkEnvironment -BinDir $BinDir -ModelPath $Model
Write-Host ("  {0} / {1} MiB RAM / {2}" -f $environment.cpu.name, $environment.ram.total_mib, $(if ($environment.gpu) { $environment.gpu.name } else { 'no NVIDIA GPU detected' }))
if (@($environment.other_ai_runtimes).Count -gt 0) {
    Write-Host ("  note: other AI runtimes resident: {0}" -f (@($environment.other_ai_runtimes) -join ', ')) -ForegroundColor Yellow
}

# ---- resolve profiles -----------------------------------------------------
$resolved = foreach ($name in $Profiles) {
    $r = Import-BenchmarkProfile -ProfilesJson $ProfilesJson -Name $name
    [pscustomobject]@{
        Name    = $name
        Threads = [int]$r.Profile.threads
        Env     = ConvertTo-EnvHashtable $r.Profile.env
        Spec    = $r.Profile
        Common  = $r.Common
    }
}
$common = $resolved[0].Common
$refDoc = Get-Content $ProfilesJson -Raw | ConvertFrom-Json

# ---- build the run plan ---------------------------------------------------
# Two orderings, and choosing the wrong one produces wrong numbers.
#
# GROUPED (default): all repeats of a profile run consecutively. This is the correct
#   ordering for measuring the absolute throughput of profiles with different memory
#   footprints. Each profile's own working set stays warm in the Windows page cache
#   across its repeats.
#
# INTERLEAVED (-Interleave): A B C A B C. The machine drifts about 5% over tens of
#   minutes, which is larger than several effects worth resolving, so a small A/B
#   between two *comparable* configurations must alternate run by run.
#
#   Do NOT interleave profiles with very different footprints. Alternating a 1.4 GiB
#   profile with a 13.8 GiB one makes each run evict the other's page-cache state, and
#   every profile then measures its own cold-start cost. This was observed directly:
#   interleaved, MAX_SPEED_EXACT read 22.9 tok/s against a grouped 32.3.
$plan = New-Object System.Collections.ArrayList
$warmTokens = [math]::Min(16, $Tokens)

if ($Interleave) {
    # All warm-ups first, then alternate. The arms are comparable by assumption here,
    # so a shared warm-up phase is the right shape.
    if (-not $NoWarmup) {
        foreach ($p in $resolved) {
            [void]$plan.Add([pscustomobject]@{ Profile = $p; Repeat = 0; Warmup = $true; Tokens = $warmTokens })
        }
    }
    for ($r = 1; $r -le $Repeats; $r++) {
        foreach ($p in $resolved) {
            [void]$plan.Add([pscustomobject]@{ Profile = $p; Repeat = $r; Warmup = $false; Tokens = $Tokens })
        }
    }
} else {
    # Each profile's warm-up runs immediately before its own repeats. Putting every
    # warm-up first would leave the first measured run of each profile cold, because
    # the profiles in between evict its page-cache state.
    foreach ($p in $resolved) {
        if (-not $NoWarmup) {
            [void]$plan.Add([pscustomobject]@{ Profile = $p; Repeat = 0; Warmup = $true; Tokens = $warmTokens })
        }
        for ($r = 1; $r -le $Repeats; $r++) {
            [void]$plan.Add([pscustomobject]@{ Profile = $p; Repeat = $r; Warmup = $false; Tokens = $Tokens })
        }
    }
}

if ($Interleave -and $resolved.Count -gt 1) {
    $ws = @($resolved | ForEach-Object { [double]$_.Spec.expected_peak_working_set_gib })
    $spread = ($ws | Measure-Object -Maximum).Maximum / ($ws | Measure-Object -Minimum).Minimum
    if ($spread -gt 2) {
        Write-Host ("WARNING: -Interleave with profiles whose working sets differ by {0:N1}x. Each run will evict the others' page-cache state and every profile will measure its own cold-start cost. Interleave only comparable configurations." -f $spread) -ForegroundColor Yellow
    }
}

Write-Host ("run plan: {0} runs ({1} warm-up), ordering: {2}" -f $plan.Count, @($plan | Where-Object { $_.Warmup }).Count, $(if ($Interleave) { 'interleaved' } else { 'grouped' }))
Write-Host ''

# ---- execute --------------------------------------------------------------
$results  = New-Object System.Collections.ArrayList
$failures = New-Object System.Collections.ArrayList
$suiteStart = Get-Date

try {
    foreach ($step in $plan) {
        $p   = $step.Profile
        $tag = if ($step.Warmup) { "{0}-warmup" -f $p.Name } else { "{0}-r{1}" -f $p.Name, $step.Repeat }

        Write-Host ("-> {0} ({1} tokens)" -f $tag, $step.Tokens) -NoNewline

        $run = Invoke-LlamaBenchRun -Tag $tag -Exe $exe -ModelPath $Model -OutDir $rawDir `
                                    -Tokens $step.Tokens -EnvVars $p.Env -Threads $p.Threads `
                                    -NCpuMoe ([int]$common.n_cpu_moe) -Context ([int]$common.context) `
                                    -Batch ([int]$common.batch) -UBatch ([int]$common.ubatch) `
                                    -StopOnPressure:$StopOnPressure

        $check = Test-RunValid -Run $run
        $run | Add-Member -NotePropertyName profile        -NotePropertyValue $p.Name
        $run | Add-Member -NotePropertyName repeat         -NotePropertyValue $step.Repeat
        $run | Add-Member -NotePropertyName warmup         -NotePropertyValue $step.Warmup
        $run | Add-Member -NotePropertyName quality_class  -NotePropertyValue $p.Spec.quality_class
        $run | Add-Member -NotePropertyName valid          -NotePropertyValue $check.Valid
        $run | Add-Member -NotePropertyName invalid_reason -NotePropertyValue ($check.Reasons -join '; ')

        [void]$results.Add($run)

        if ($check.Valid) {
            Write-Host ("  TG {0,7:N3} tok/s   WS {1,6:N3} GiB   hash {2}" -f $run.tg_toks, $run.peak_working_set_gib, $run.token_trace_sha256.Substring(0, 16)) -ForegroundColor Green
        } else {
            Write-Host ("  INVALID: {0}" -f ($check.Reasons -join '; ')) -ForegroundColor Red
            [void]$failures.Add($tag)
        }
    }
}
finally {
    Stop-StaleLlama
    Clear-RuntimeEnvironment
}

$suiteMin = [math]::Round(((Get-Date) - $suiteStart).TotalMinutes, 1)

# ---- aggregate ------------------------------------------------------------
$measured = @($results | Where-Object { -not $_.warmup -and $_.valid })

$summary = foreach ($name in $Profiles) {
    $rows = @($measured | Where-Object { $_.profile -eq $name })
    if ($rows.Count -eq 0) {
        [pscustomobject]@{
            profile = $name; runs = 0; tg_mean = $null; tg_min = $null; tg_max = $null
            tg_stdev = $null; peak_ws_gib = $null; peak_private_gib = $null
            min_available_ram_gib = $null; peak_vram_mib = $null
            token_trace_sha256 = $null; hash_stable = $null; hash_vs_reference = $null; safety = 'NO VALID RUN'
            quality_class = ($resolved | Where-Object { $_.Name -eq $name }).Spec.quality_class
            tg_reference = ($resolved | Where-Object { $_.Name -eq $name }).Spec.tg_reference_toks
            delta_vs_reference_pct = $null
        }
        continue
    }
    $tg      = @($rows | ForEach-Object { $_.tg_toks })
    $mean    = ($tg | Measure-Object -Average).Average
    $stdev   = if ($tg.Count -gt 1) { [math]::Sqrt((($tg | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum) / ($tg.Count - 1)) } else { 0 }
    $hashes  = @($rows | ForEach-Object { $_.token_trace_sha256 } | Sort-Object -Unique)
    $spec    = ($resolved | Where-Object { $_.Name -eq $name }).Spec
    $ref     = [double]$spec.tg_reference_toks

    # The reference token-trace hashes are defined for a 512-token generation only.
    # At any other length the trace is a different document, so comparing it to the
    # reference would be meaningless - report $null rather than a false failure.
    $hashVerdict = $null
    if ($Tokens -eq 512) {
        if ($spec.quality_class -eq 'EXACT') {
            $hashVerdict = if ($hashes[0] -eq $refDoc.correctness_classes.EXACT.reference_512_token_trace_sha256) { 'MATCHES REFERENCE' } else { 'DIFFERS FROM REFERENCE' }
        } else {
            $prefix = $refDoc.correctness_classes.REPACK.reference_512_token_trace_sha256_prefix
            $hashVerdict = if ($hashes[0].StartsWith($prefix)) { 'MATCHES REFERENCE' } else { 'DIFFERS FROM REFERENCE' }
        }
    }

    [pscustomobject]@{
        profile               = $name
        quality_class         = $spec.quality_class
        runs                  = $rows.Count
        tg_mean               = [math]::Round($mean, 3)
        tg_min                = [math]::Round(($tg | Measure-Object -Minimum).Minimum, 3)
        tg_max                = [math]::Round(($tg | Measure-Object -Maximum).Maximum, 3)
        tg_stdev              = [math]::Round($stdev, 3)
        peak_ws_gib           = [math]::Round((($rows | ForEach-Object { $_.peak_working_set_gib }) | Measure-Object -Maximum).Maximum, 3)
        peak_private_gib      = [math]::Round((($rows | ForEach-Object { $_.peak_private_gib })    | Measure-Object -Maximum).Maximum, 3)
        min_available_ram_gib = [math]::Round((($rows | ForEach-Object { $_.min_available_ram_gib })| Measure-Object -Minimum).Minimum, 3)
        peak_vram_mib         = (($rows | ForEach-Object { $_.peak_vram_mib }) | Measure-Object -Maximum).Maximum
        pp_mean               = [math]::Round((($rows | ForEach-Object { $_.pp_toks }) | Measure-Object -Average).Average, 3)
        token_trace_sha256    = $hashes[0]
        hash_stable           = ($hashes.Count -eq 1)
        hash_vs_reference     = $hashVerdict
        safety                = if (@($rows | Where-Object { $_.safety_pass -eq $false }).Count -eq 0) { 'PASS' } else { 'FAIL' }
        tg_reference          = $ref
        delta_vs_reference_pct = if ($ref -gt 0) { [math]::Round((($mean - $ref) / $ref) * 100, 2) } else { $null }
        within_tolerance      = if ($ref -gt 0) { [math]::Abs((($mean - $ref) / $ref) * 100) -le [double]$spec.tg_reference_tolerance_pct } else { $null }
    }
}

# ---- write outputs --------------------------------------------------------
$doc = [ordered]@{
    schema_version = '1.0.0'
    suite          = [ordered]@{
        mode = $Mode; tokens = $Tokens; repeats = $Repeats
        interleaved = [bool]$Interleave; warmup = (-not $NoWarmup)
        started_utc = $suiteStart.ToUniversalTime().ToString('o')
        elapsed_min = $suiteMin
        profiles_requested = $Profiles
        failures = @($failures)
    }
    environment    = $environment
    common         = $common
    summary        = @($summary)
    runs           = @($results | ForEach-Object {
        $_ | Select-Object * -ExcludeProperty counters, raw_logs
    })
    counters       = @($results | ForEach-Object { [ordered]@{ tag = $_.tag; counters = $_.counters } })
}

$jsonPath = Join-Path $OutDir 'results.json'
$doc | ConvertTo-Json -Depth 12 | Set-Content $jsonPath -Encoding UTF8

$csvPath = Join-Path $OutDir 'results.csv'
$summary | Export-Csv $csvPath -NoTypeInformation -Encoding UTF8

$runsCsvPath = Join-Path $OutDir 'runs.csv'
$results | Select-Object profile, repeat, warmup, valid, invalid_reason, tokens_requested, tokens_emitted,
    threads, tg_toks, pp_toks, wall_sec, peak_working_set_gib, peak_private_gib,
    min_available_ram_gib, peak_vram_mib, token_trace_sha256, safety_pass, stop_reason |
    Export-Csv $runsCsvPath -NoTypeInformation -Encoding UTF8

# ---- markdown -------------------------------------------------------------
$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("# Benchmark results - $Mode mode")
[void]$md.AppendLine()
[void]$md.AppendLine("Generated $((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss')) UTC. Elapsed $suiteMin min.")
[void]$md.AppendLine()
[void]$md.AppendLine('## Machine')
[void]$md.AppendLine()
[void]$md.AppendLine('| item | value |')
[void]$md.AppendLine('|---|---|')
[void]$md.AppendLine("| CPU | $($environment.cpu.name), $($environment.cpu.cores)C/$($environment.cpu.logical)T |")
[void]$md.AppendLine("| RAM | $($environment.ram.total_mib) MiB |")
[void]$md.AppendLine("| GPU | $(if ($environment.gpu) { "$($environment.gpu.name), $($environment.gpu.memory_total), driver $($environment.gpu.driver)" } else { 'none detected' }) |")
[void]$md.AppendLine("| Storage (model volume) | $(if ($environment.storage) { "$($environment.storage.media_type) / $($environment.storage.bus_type)" } else { 'unknown' }) |")
[void]$md.AppendLine("| OS | $($environment.os.caption) $($environment.os.version) $($environment.os.display_version) |")
[void]$md.AppendLine("| Model | $(Split-Path $environment.model.path -Leaf), $($environment.model.bytes) bytes |")
[void]$md.AppendLine("| Model head SHA-256 | ``$($environment.model.head_sha256.Substring(0,32))...`` |")
if (@($environment.other_ai_runtimes).Count -gt 0) {
    [void]$md.AppendLine("| Other AI runtimes resident | $(@($environment.other_ai_runtimes) -join ', ') |")
}
[void]$md.AppendLine()
[void]$md.AppendLine('### Binaries under test')
[void]$md.AppendLine()
[void]$md.AppendLine('| file | SHA-256 |')
[void]$md.AppendLine('|---|---|')
foreach ($k in $environment.binaries.Keys) {
    [void]$md.AppendLine("| ``$k`` | ``$($environment.binaries[$k])`` |")
}
[void]$md.AppendLine()
[void]$md.AppendLine('## Results')
[void]$md.AppendLine()
[void]$md.AppendLine("$Tokens-token generation, $Repeats repeat(s) per profile, $(if ($Interleave) { 'interleaved run by run' } else { 'grouped by profile' }).")
[void]$md.AppendLine()
[void]$md.AppendLine('| profile | class | runs | TG mean | TG min-max | sd | peak WS | peak private | min avail | VRAM | hash | safety | ref TG | delta |')
[void]$md.AppendLine('|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|')
foreach ($s in $summary) {
    $hash = if ($s.token_trace_sha256) { $s.token_trace_sha256.Substring(0, 16) } else { '-' }
    if ($s.hash_stable -eq $false) { $hash += ' UNSTABLE' }
    if ($s.hash_vs_reference -eq 'DIFFERS FROM REFERENCE') { $hash += ' != REF' }
    elseif ($s.hash_vs_reference -eq 'MATCHES REFERENCE')  { $hash += ' = REF' }
    $rng  = if ($s.runs -gt 0) { "{0:N3}-{1:N3}" -f $s.tg_min, $s.tg_max } else { '-' }
    $dlt  = if ($null -ne $s.delta_vs_reference_pct) { "{0:+0.00;-0.00;0.00} %" -f $s.delta_vs_reference_pct } else { '-' }
    [void]$md.AppendLine("| ``$($s.profile)`` | $($s.quality_class) | $($s.runs) | $(if($s.runs){'{0:N3}' -f $s.tg_mean}else{'-'}) | $rng | $(if($s.runs){'{0:N3}' -f $s.tg_stdev}else{'-'}) | $(if($s.runs){'{0:N3}' -f $s.peak_ws_gib}else{'-'}) | $(if($s.runs){'{0:N3}' -f $s.peak_private_gib}else{'-'}) | $(if($s.runs){'{0:N3}' -f $s.min_available_ram_gib}else{'-'}) | $($s.peak_vram_mib) | ``$hash`` | $($s.safety) | $($s.tg_reference) | $dlt |")
}
[void]$md.AppendLine()
[void]$md.AppendLine('`ref TG` is the reference measurement recorded in `profiles/profiles.json`, taken on the machine described in that file. It is a reference point, not a guarantee for this or any other machine.')
[void]$md.AppendLine()

if ($failures.Count -gt 0) {
    [void]$md.AppendLine('## Invalid runs')
    [void]$md.AppendLine()
    foreach ($r in @($results | Where-Object { -not $_.valid })) {
        [void]$md.AppendLine("- ``$($r.tag)``: $($r.invalid_reason)")
    }
    [void]$md.AppendLine()
}

[void]$md.AppendLine('## Raw data')
[void]$md.AppendLine()
[void]$md.AppendLine('- `results.json` - full record including per-run safety counters')
[void]$md.AppendLine('- `results.csv` - per-profile summary')
[void]$md.AppendLine('- `runs.csv` - one row per run')
[void]$md.AppendLine('- `raw/` - stdout, stderr, counter dump and token trace for every run')

$mdPath = Join-Path $OutDir 'SUMMARY.md'
Set-Content -Path $mdPath -Value $md.ToString() -Encoding UTF8

# ---- report ---------------------------------------------------------------
Write-Host ''
$summary | Format-Table profile, quality_class, runs, tg_mean, tg_stdev, peak_ws_gib, safety, hash_vs_reference, delta_vs_reference_pct, within_tolerance -AutoSize
Write-Host "results : $jsonPath"
Write-Host "summary : $mdPath"
Write-Host ''

$hashFailures = @($summary | Where-Object { $_.hash_vs_reference -eq 'DIFFERS FROM REFERENCE' } | ForEach-Object { $_.profile })
if ($hashFailures.Count -gt 0) {
    Write-Host "CORRECTNESS FAILURE - token trace differs from the reference for: $($hashFailures -join ', ')" -ForegroundColor Red
    exit 1
}
if ($failures.Count -gt 0) {
    Write-Host "SUITE COMPLETED WITH $($failures.Count) INVALID RUN(S): $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'SUITE PASS' -ForegroundColor Green
exit 0
