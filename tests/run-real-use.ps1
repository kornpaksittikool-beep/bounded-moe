<#
.SYNOPSIS
    Real-use and stability validation against llama-server.

.DESCRIPTION
    Drives a profile through task types a 512-token benchmark never reaches: chat,
    reasoning, code generation, code explanation, long-form writing, Thai, and long
    prompts that force prompt batches far above 32 tokens.

    Records per-request time to first token, throughput and output, and samples
    process working set, private bytes and VRAM throughout, so growth across a
    sustained session is visible.

    This exists because a five-token benchmark prompt once hid a blocking crash at
    batch >= 32 for the entire research phase.

.PARAMETER Sustained
    Repeat the whole task set several times to expose growth and degradation.

.EXAMPLE
    .\run-real-use.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf -Profile LOW_MEMORY_EXACT

.EXAMPLE
    .\run-real-use.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf -Profile BALANCED_EXACT -Sustained -SustainedRounds 4
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Model,
    [string] $Profile   = 'LOW_MEMORY_EXACT',
    [string] $BinDir,
    [string] $ProfilesJson,
    [string] $OutDir,
    [int]    $Port      = 8137,
    [int]    $StartupTimeoutSec = 300,
    [int]    $RequestTimeoutSec = 600,
    [ValidateSet('chat','completion')]
    [string] $Endpoint = 'chat',
    # Reasoning is charged against the same token budget as the answer, so it is off
    # by default. Turn it on to exercise the model the way an interactive user would.
    [switch] $Thinking,
    [switch] $Sustained,
    [int]    $SustainedRounds = 3
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here

Import-Module (Join-Path $repo 'benchmark\lib\BenchLib.psm1') -Force
Import-Module (Join-Path $here 'lib\ServerLib.psm1')          -Force

if (-not $ProfilesJson) { $ProfilesJson = Join-Path $repo 'profiles\profiles.json' }
if (-not $BinDir)       { $BinDir       = Join-Path $repo 'clean-room\build\bin' }
if (-not $OutDir)       { $OutDir       = Join-Path $repo ('tests\results\realuse-{0}-{1}' -f $Profile, (Get-Date -Format 'yyyyMMdd-HHmmss')) }

$exe = Join-Path $BinDir 'llama-server.exe'
foreach ($p in @($Model, $ProfilesJson, $exe)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$doc  = Get-Content $ProfilesJson -Raw | ConvertFrom-Json
$spec = $doc.profiles | Where-Object { $_.name -eq $Profile }
if (-not $spec) { throw "unknown profile '$Profile'" }

Assert-NoStaleLlama -When 'pre-flight'

$tasks = & (Join-Path $here 'tasks\realuse-tasks.ps1')

$envVars = @{}
foreach ($prop in $spec.env.PSObject.Properties) { $envVars[$prop.Name] = [string]$prop.Value }

Write-Host ''
Write-Host "=== real-use validation: $Profile ($($spec.quality_class)) ===" -ForegroundColor Cyan
Write-Host "  tasks : $($tasks.Count)   rounds: $(if ($Sustained) { $SustainedRounds } else { 1 })   endpoint: $Endpoint   thinking: $($Thinking.IsPresent)"
Write-Host "  output: $OutDir"
Write-Host ''
Write-Host 'starting server...' -NoNewline

$samples   = New-Object System.Collections.ArrayList
$results   = New-Object System.Collections.ArrayList
$incidents = New-Object System.Collections.ArrayList
$session   = $null

try {
    $session = Start-LlamaServer -Exe $exe -ModelPath $Model -OutDir $OutDir -Tag $Profile `
                                 -EnvVars $envVars -Common $doc.common -Threads ([int]$spec.threads) `
                                 -Port $Port -StartupTimeoutSec $StartupTimeoutSec
    Write-Host " ready in $($session.LoadSeconds) s" -ForegroundColor Green
    [void]$samples.Add((Get-ServerSample -Session $session -Phase 'after-load'))

    $rounds = if ($Sustained) { $SustainedRounds } else { 1 }

    for ($round = 1; $round -le $rounds; $round++) {
        if ($rounds -gt 1) { Write-Host ''; Write-Host "--- round $round of $rounds ---" -ForegroundColor Cyan }

        foreach ($t in $tasks) {
            $tag = if ($rounds -gt 1) { "$($t.id)-round$round" } else { $t.id }
            Write-Host ("-> {0,-26}" -f $tag) -NoNewline

            $r = Invoke-LlamaCompletion -Session $session -Prompt $t.prompt -Endpoint $Endpoint `
                                        -Thinking:$Thinking -MaxTokens $t.n -TimeoutSec $RequestTimeoutSec
            [void]$samples.Add((Get-ServerSample -Session $session -Phase $tag))

            if (-not $r.ok) { [void]$incidents.Add("request $tag failed: $($r.error)") }

            if ($session.Process.HasExited) {
                [void]$incidents.Add("SERVER EXITED during $tag with code $($session.Process.ExitCode)")
                Write-Host '  SERVER DIED' -ForegroundColor Red
                [void]$results.Add([pscustomobject]@{ tag = $tag; task = $t.id; kind = $t.kind; language = $t.lang; round = $round; requested_tokens = $t.n; ok = $false; error = 'server exited' })
                break
            }

            [void]$results.Add([pscustomobject]([ordered]@{
                tag = $tag; task = $t.id; kind = $t.kind; language = $t.lang; round = $round
                requested_tokens = $t.n
                ok = $r.ok; error = $r.error
                prompt_tokens = $r.prompt_tokens; predicted_tokens = $r.predicted_tokens
                ttft_sec = $r.ttft_sec; prompt_toks = $r.prompt_toks; tg_toks = $r.tg_toks
                wall_sec = $r.wall_sec; stop_reason = $r.stop_reason
                content_chars = $r.content.Length
                reasoning_chars = $r.reasoning.Length
                truncated_in_reasoning = $r.truncated_in_reasoning
                content = $r.content
                reasoning = $r.reasoning
            }))

            if ($r.ok) {
                $note = if ($r.truncated_in_reasoning) { '  [answer truncated in reasoning]' } else { '' }
                Write-Host ("  TTFT {0,6:N2}s  TG {1,6:N2} tok/s  prompt {2,5} tok  out {3,5} tok{4}" -f `
                            $r.ttft_sec, $r.tg_toks, $r.prompt_tokens, $r.predicted_tokens, $note) `
                            -ForegroundColor $(if ($r.truncated_in_reasoning) { 'Yellow' } else { 'Green' })
            } else {
                Write-Host "  FAILED: $($r.error)" -ForegroundColor Red
            }
        }
        if ($session.Process.HasExited) { break }
    }
}
finally {
    if ($session) {
        [void]$samples.Add((Get-ServerSample -Session $session -Phase 'before-shutdown'))
        Write-Host ''
        Write-Host 'stopping server...' -ForegroundColor DarkGray
        foreach ($i in (Stop-LlamaServer -Session $session)) { [void]$incidents.Add($i) }
    } else {
        Clear-RuntimeEnvironment
        Stop-StaleLlama
    }
}

# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
if (-not $session) { throw 'server never started; nothing to report' }

# Counters are written at process exit. A force-killed server may not produce them,
# in which case the safety result is reported as unavailable rather than as a pass.
$counters = Read-CounterFile $session.CountersPath
$safety   = Test-SafetyInvariant -Ordered $counters

$measured = @($results | Where-Object { $_.ok })
$duringRequests = @($samples | Where-Object { $_.phase -ne 'after-load' })

function Stat($items, $prop, $op) {
    if (-not $items -or @($items).Count -eq 0) { return $null }
    $v = @($items | ForEach-Object { $_.$prop })
    switch ($op) {
        'min' { [math]::Round(($v | Measure-Object -Minimum).Minimum, 3) }
        'max' { [math]::Round(($v | Measure-Object -Maximum).Maximum, 3) }
        'avg' { [math]::Round(($v | Measure-Object -Average).Average, 3) }
    }
}

$wsFirst   = if ($duringRequests.Count) { ($duringRequests | Select-Object -First 1).working_set_gib } else { $null }
$wsLast    = if ($duringRequests.Count) { ($duringRequests | Select-Object -Last  1).working_set_gib } else { $null }
$vramFirst = if ($duringRequests.Count) { ($duringRequests | Select-Object -First 1).vram_mib } else { $null }
$vramLast  = if ($duringRequests.Count) { ($duringRequests | Select-Object -Last  1).vram_mib } else { $null }

# Throughput degradation across rounds is the sustained-run question that matters.
$byRound = @()
if ($measured.Count) {
    $byRound = @($measured | Group-Object round | ForEach-Object {
        [pscustomobject]@{
            round    = [int]$_.Name
            requests = $_.Count
            tg_mean  = Stat $_.Group 'tg_toks' 'avg'
            ttft_mean = Stat $_.Group 'ttft_sec' 'avg'
        }
    } | Sort-Object round)
}

$report = [ordered]@{
    schema_version = '1.0.0'
    profile        = $Profile
    quality_class  = $spec.quality_class
    started_utc    = (Get-Date).ToUniversalTime().ToString('o')
    environment    = Get-BenchmarkEnvironment -BinDir $BinDir -ModelPath $Model
    server_args    = $session.Args
    env            = $envVars
    endpoint       = $Endpoint
    thinking       = [bool]$Thinking
    rounds         = $(if ($Sustained) { $SustainedRounds } else { 1 })
    load_seconds   = $session.LoadSeconds
    requests_total = $results.Count
    requests_ok    = $measured.Count
    requests_truncated_in_reasoning = @($results | Where-Object { $_.truncated_in_reasoning }).Count
    incidents      = @($incidents)
    memory         = [ordered]@{
        working_set_first_gib  = $wsFirst
        working_set_last_gib   = $wsLast
        working_set_growth_gib = if ($null -ne $wsFirst) { [math]::Round($wsLast - $wsFirst, 3) } else { $null }
        working_set_peak_gib   = Stat $samples 'working_set_gib' 'max'
        private_peak_gib       = Stat $samples 'private_gib' 'max'
        vram_first_mib         = $vramFirst
        vram_last_mib          = $vramLast
        vram_growth_mib        = if ($null -ne $vramFirst) { $vramLast - $vramFirst } else { $null }
        vram_peak_mib          = Stat $samples 'vram_mib' 'max'
        min_available_ram_gib  = Stat $samples 'avail_ram_gib' 'min'
    }
    throughput     = [ordered]@{
        tg_min   = Stat $measured 'tg_toks' 'min'
        tg_max   = Stat $measured 'tg_toks' 'max'
        tg_mean  = Stat $measured 'tg_toks' 'avg'
        ttft_min = Stat $measured 'ttft_sec' 'min'
        ttft_max = Stat $measured 'ttft_sec' 'max'
        max_prompt_tokens = Stat $measured 'prompt_tokens' 'max'
        by_round = $byRound
    }
    safety         = [ordered]@{
        pass = $safety.Pass; reason = $safety.Reason
        violations = @($safety.Violations); counters = $counters
    }
    requests       = @($results)
    samples        = @($samples)
}

$report  | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir 'realuse.json') -Encoding UTF8
$results | Select-Object tag, kind, language, round, ok, prompt_tokens, predicted_tokens, ttft_sec, tg_toks, wall_sec, content_chars, reasoning_chars, truncated_in_reasoning, stop_reason |
    Export-Csv (Join-Path $OutDir 'requests.csv') -NoTypeInformation -Encoding UTF8
$samples | Export-Csv (Join-Path $OutDir 'samples.csv') -NoTypeInformation -Encoding UTF8

$tx = New-Object System.Text.StringBuilder
foreach ($r in $results) {
    [void]$tx.AppendLine("################ $($r.tag)  [$($r.kind) / $($r.language)] ################")
    [void]$tx.AppendLine('PROMPT:')
    # Where-Object returns a bare hashtable when exactly one item matches, and
    # indexing a hashtable with [0] looks up the key 0 rather than the first element.
    $match = @($tasks | Where-Object { $_.id -eq $r.task })
    [void]$tx.AppendLine($(if ($match.Count) { $match[0].prompt } else { '<prompt not found>' }))
    [void]$tx.AppendLine('')
    if ($r.PSObject.Properties.Name -contains 'reasoning' -and $r.reasoning) {
        [void]$tx.AppendLine('REASONING:')
        [void]$tx.AppendLine($r.reasoning)
        [void]$tx.AppendLine('')
    }
    [void]$tx.AppendLine('ANSWER:')
    [void]$tx.AppendLine($(if ($r.PSObject.Properties.Name -contains 'content' -and $r.content) { $r.content } else { '<truncated - budget consumed by reasoning>' }))
    [void]$tx.AppendLine('')
}
Set-Content (Join-Path $OutDir 'transcripts.txt') -Value $tx.ToString() -Encoding UTF8

Write-Host ''
Write-Host ("requests   : {0}/{1} ok, {2} answered but truncated in reasoning" -f $measured.Count, $results.Count, $report.requests_truncated_in_reasoning)
Write-Host ("TG range   : {0} to {1} tok/s (mean {2})" -f $report.throughput.tg_min, $report.throughput.tg_max, $report.throughput.tg_mean)
Write-Host ("TTFT range : {0} to {1} s" -f $report.throughput.ttft_min, $report.throughput.ttft_max)
Write-Host ("max prompt : {0} tokens" -f $report.throughput.max_prompt_tokens)
Write-Host ("WS growth  : {0} GiB    VRAM growth: {1} MiB" -f $report.memory.working_set_growth_gib, $report.memory.vram_growth_mib)
Write-Host ("safety     : {0}" -f $(if ($null -eq $safety.Pass) { "unavailable ($($safety.Reason))" } elseif ($safety.Pass) { 'PASS' } else { "FAIL: $($safety.Violations -join '; ')" }))
if ($byRound.Count -gt 1) { Write-Host ''; $byRound | Format-Table -AutoSize }
Write-Host ("incidents  : {0}" -f $incidents.Count)
foreach ($i in $incidents) { Write-Host "  - $i" -ForegroundColor Red }
Write-Host ''
Write-Host "output: $OutDir"

if ($incidents.Count -gt 0 -or $measured.Count -ne $results.Count -or $safety.Pass -eq $false) {
    Write-Host 'REAL-USE FAIL' -ForegroundColor Red
    exit 1
}
Write-Host 'REAL-USE PASS' -ForegroundColor Green
exit 0
