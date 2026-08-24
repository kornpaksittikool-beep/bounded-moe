<#
.SYNOPSIS
    Compare two builds of the same code on one profile, alternating run by run.

.DESCRIPTION
    Answers exactly one question: does build B behave and perform the same as build A?

    Both arms use the identical profile, model, flags and token count. Only the binary
    directory changes. Runs alternate A B A B so that machine drift - about 5% over tens
    of minutes on the reference machine - cannot be attributed to the build.

    This is the correct use of interleaving: two configurations with identical memory
    footprints, where the effect under test may be smaller than the drift. Use
    run-benchmark-suite.ps1 (grouped) for absolute per-profile throughput instead.

.EXAMPLE
    .\run-build-ab.ps1 -Model <gguf> `
        -BinDirA D:\reference\bin -BinDirB D:\rebuilt\bin `
        -Profile LOW_MEMORY_EXACT -Repeats 3
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $BinDirA,
    [Parameter(Mandatory)] [string] $BinDirB,
    [string] $LabelA = 'A',
    [string] $LabelB = 'B',
    [string] $Profile = 'LOW_MEMORY_EXACT',
    [string] $ProfilesJson,
    [string] $OutDir,
    [int]    $Tokens  = 512,
    [int]    $Repeats = 3,
    [switch] $NoWarmup
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
Import-Module (Join-Path $here 'lib\BenchLib.psm1') -Force

if (-not $ProfilesJson) { $ProfilesJson = Join-Path $repo 'profiles\profiles.json' }
if (-not $OutDir)       { $OutDir       = Join-Path $repo ('benchmark\results\build-ab-{0}-{1}' -f $Profile, (Get-Date -Format 'yyyyMMdd-HHmmss')) }

foreach ($p in @($Model, $ProfilesJson, $BinDirA, $BinDirB)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}

$arms = @(
    [pscustomobject]@{ Label = $LabelA; BinDir = $BinDirA; Exe = (Join-Path $BinDirA 'llama-cli.exe') }
    [pscustomobject]@{ Label = $LabelB; BinDir = $BinDirB; Exe = (Join-Path $BinDirB 'llama-cli.exe') }
)
foreach ($a in $arms) { if (-not (Test-Path $a.Exe)) { throw "llama-cli.exe not found in $($a.BinDir)" } }

$r = Import-BenchmarkProfile -ProfilesJson $ProfilesJson -Name $Profile
$envVars = ConvertTo-EnvHashtable $r.Profile.env
$common  = $r.Common

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$rawDir = Join-Path $OutDir 'raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

Assert-NoStaleLlama -When 'pre-flight'

Write-Host ''
Write-Host "=== build A/B: $Profile ===" -ForegroundColor Cyan
Write-Host ("  {0,-6} {1}" -f $LabelA, $BinDirA)
Write-Host ("  {0,-6} {1}" -f $LabelB, $BinDirB)
Write-Host "  tokens: $Tokens   repeats: $Repeats   ordering: interleaved"
Write-Host "  output: $OutDir"
Write-Host ''

# record what is actually being compared
$hashes = [ordered]@{}
foreach ($a in $arms) {
    $h = [ordered]@{}
    Get-ChildItem $a.BinDir -File | Where-Object { $_.Extension -in '.dll', '.exe' } | Sort-Object Name |
        ForEach-Object { $h[$_.Name] = (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
    $hashes[$a.Label] = $h
}

$differing = @()
foreach ($name in (@($hashes[$LabelA].Keys) + @($hashes[$LabelB].Keys) | Sort-Object -Unique)) {
    $ha = if ($hashes[$LabelA].Contains($name)) { $hashes[$LabelA][$name] } else { $null }
    $hb = if ($hashes[$LabelB].Contains($name)) { $hashes[$LabelB][$name] } else { $null }
    if ($ha -ne $hb) { $differing += $name }
}
Write-Host ("binaries differing between arms: {0}" -f $(if ($differing.Count) { $differing -join ', ' } else { 'none - the arms are the same build' }))
Write-Host ''

$plan = New-Object System.Collections.ArrayList
if (-not $NoWarmup) {
    foreach ($a in $arms) { [void]$plan.Add([pscustomobject]@{ Arm = $a; Repeat = 0; Warmup = $true }) }
}
for ($i = 1; $i -le $Repeats; $i++) {
    foreach ($a in $arms) { [void]$plan.Add([pscustomobject]@{ Arm = $a; Repeat = $i; Warmup = $false }) }
}

$results = New-Object System.Collections.ArrayList
try {
    foreach ($step in $plan) {
        $tag = if ($step.Warmup) { "$($step.Arm.Label)-warmup" } else { "$($step.Arm.Label)-r$($step.Repeat)" }
        Write-Host ("-> {0,-14}" -f $tag) -NoNewline

        $run = Invoke-LlamaBenchRun -Tag $tag -Exe $step.Arm.Exe -ModelPath $Model -OutDir $rawDir `
                                    -Tokens $(if ($step.Warmup) { [math]::Min(16, $Tokens) } else { $Tokens }) `
                                    -EnvVars $envVars -Threads ([int]$r.Profile.threads) `
                                    -NCpuMoe ([int]$common.n_cpu_moe) -Context ([int]$common.context) `
                                    -Batch ([int]$common.batch) -UBatch ([int]$common.ubatch)

        $check = Test-RunValid -Run $run
        $run | Add-Member -NotePropertyName arm    -NotePropertyValue $step.Arm.Label
        $run | Add-Member -NotePropertyName repeat -NotePropertyValue $step.Repeat
        $run | Add-Member -NotePropertyName warmup -NotePropertyValue $step.Warmup
        $run | Add-Member -NotePropertyName valid  -NotePropertyValue $check.Valid
        $run | Add-Member -NotePropertyName invalid_reason -NotePropertyValue ($check.Reasons -join '; ')
        [void]$results.Add($run)

        if ($check.Valid) {
            Write-Host ("  TG {0,7:N3}  WS {1,6:N3} GiB  hash {2}" -f $run.tg_toks, $run.peak_working_set_gib, $run.token_trace_sha256.Substring(0,16)) -ForegroundColor Green
        } else {
            Write-Host ("  INVALID: {0}" -f ($check.Reasons -join '; ')) -ForegroundColor Red
        }
    }
}
finally { Stop-StaleLlama; Clear-RuntimeEnvironment }

$summary = foreach ($a in $arms) {
    $rows = @($results | Where-Object { $_.arm -eq $a.Label -and -not $_.warmup -and $_.valid })
    if ($rows.Count -eq 0) {
        [pscustomobject]@{ arm = $a.Label; runs = 0; tg_mean = $null; tg_stdev = $null; peak_ws_gib = $null; hash = $null; safety = 'NO VALID RUN' }
        continue
    }
    $tg = @($rows | ForEach-Object { $_.tg_toks })
    $mean = ($tg | Measure-Object -Average).Average
    [pscustomobject]@{
        arm = $a.Label
        runs = $rows.Count
        tg_mean = [math]::Round($mean, 3)
        tg_min  = [math]::Round(($tg | Measure-Object -Minimum).Minimum, 3)
        tg_max  = [math]::Round(($tg | Measure-Object -Maximum).Maximum, 3)
        tg_stdev = [math]::Round($(if ($tg.Count -gt 1) { [math]::Sqrt((($tg | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum) / ($tg.Count - 1)) } else { 0 }), 3)
        peak_ws_gib = [math]::Round((($rows | ForEach-Object { $_.peak_working_set_gib }) | Measure-Object -Maximum).Maximum, 3)
        hash = (@($rows | ForEach-Object { $_.token_trace_sha256 } | Sort-Object -Unique))[0]
        hash_stable = ((@($rows | ForEach-Object { $_.token_trace_sha256 } | Sort-Object -Unique)).Count -eq 1)
        safety = if (@($rows | Where-Object { $_.safety_pass -eq $false }).Count -eq 0) { 'PASS' } else { 'FAIL' }
    }
}

$sa = $summary | Where-Object { $_.arm -eq $LabelA }
$sb = $summary | Where-Object { $_.arm -eq $LabelB }
$delta = if ($sa.tg_mean -and $sb.tg_mean) { [math]::Round((($sb.tg_mean - $sa.tg_mean) / $sa.tg_mean) * 100, 2) } else { $null }
$hashMatch = ($sa.hash -eq $sb.hash)

$doc = [ordered]@{
    schema_version = '1.0.0'
    profile = $Profile; tokens = $Tokens; repeats = $Repeats; interleaved = $true
    arm_a = [ordered]@{ label = $LabelA; bin_dir = $BinDirA }
    arm_b = [ordered]@{ label = $LabelB; bin_dir = $BinDirB }
    binaries_differing = $differing
    binary_hashes = $hashes
    environment = Get-BenchmarkEnvironment -BinDir $BinDirA -ModelPath $Model
    env = $envVars
    summary = @($summary)
    delta_b_vs_a_pct = $delta
    token_hash_match = $hashMatch
    runs = @($results | ForEach-Object { $_ | Select-Object * -ExcludeProperty counters, raw_logs })
    counters = @($results | ForEach-Object { [ordered]@{ tag = $_.tag; counters = $_.counters } })
}
$doc | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $OutDir 'build-ab.json') -Encoding UTF8
$results | Select-Object arm, repeat, warmup, valid, tg_toks, peak_working_set_gib, min_available_ram_gib, token_trace_sha256, safety_pass |
    Export-Csv (Join-Path $OutDir 'runs.csv') -NoTypeInformation -Encoding UTF8

Write-Host ''
$summary | Format-Table arm, runs, tg_mean, tg_min, tg_max, tg_stdev, peak_ws_gib, safety, hash_stable -AutoSize
Write-Host ("token hash identical between arms : {0}" -f $(if ($hashMatch) { 'YES' } else { "NO  ($($sa.hash) vs $($sb.hash))" })) -ForegroundColor $(if ($hashMatch) { 'Green' } else { 'Red' })
Write-Host ("throughput {0} vs {1}              : {2:+0.00;-0.00;0.00} %" -f $LabelB, $LabelA, $delta)
Write-Host ''
Write-Host "output: $OutDir"

if (-not $hashMatch) { exit 1 }
exit 0
