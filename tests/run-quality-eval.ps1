<#
.SYNOPSIS
    Compare two profiles on generation quality, not just throughput.

.DESCRIPTION
    Runs an identical task battery through two profiles with identical prompts,
    seeds and sampling settings, then classifies every task as one of:

      BIT_IDENTICAL   - the two arms produced exactly the same text
      TASK_EQUIVALENT - the text differs but every objective check agrees
      DIFFERENT       - at least one objective check disagrees between arms

    Objective checks are mechanical predicates over the output: a known numeric
    answer, a required word, a stated formatting constraint, a verification code
    hidden in a long prompt. They test whether a verifiable property survived, not
    whether the prose is good.

    Optionally also runs llama-perplexity over local corpora for both arms.

    Written for the EXACT-versus-REPACK question, but it will compare any two
    profiles.

.EXAMPLE
    .\run-quality-eval.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf

.EXAMPLE
    .\run-quality-eval.ps1 -Model <gguf> -ArmA LOW_MEMORY_EXACT -ArmB BALANCED_EXACT -SkipPerplexity
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Model,
    [string]   $ArmA = 'MAX_SPEED_EXACT',
    [string]   $ArmB = 'MAX_SPEED_REPACK',
    [string]   $BinDir,
    [string]   $ProfilesJson,
    [string]   $OutDir,
    [string]   $CorpusDir,
    [int]      $Port = 8139,
    [int]      $StartupTimeoutSec = 300,
    [int]      $RequestTimeoutSec = 900,
    [ValidateSet('chat','completion')]
    [string]   $Endpoint = 'chat',
    # Reasoning is charged against the same token budget as the answer. Off by default
    # so that the objective checks see an answer rather than a truncated thought.
    [switch]   $Thinking,
    [switch]   $SkipPerplexity,
    [int]      $PerplexityChunks = 24
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here

Import-Module (Join-Path $repo 'benchmark\lib\BenchLib.psm1') -Force
Import-Module (Join-Path $here 'lib\ServerLib.psm1')          -Force

if (-not $ProfilesJson) { $ProfilesJson = Join-Path $repo 'profiles\profiles.json' }
if (-not $BinDir)       { $BinDir       = Join-Path $repo 'clean-room\build\bin' }
if (-not $CorpusDir)    { $CorpusDir    = Join-Path $here 'corpora' }
if (-not $OutDir)       { $OutDir       = Join-Path $repo ('tests\results\quality-{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss')) }

$serverExe = Join-Path $BinDir 'llama-server.exe'
$pplExe    = Join-Path $BinDir 'llama-perplexity.exe'
foreach ($p in @($Model, $ProfilesJson, $serverExe)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$doc    = Get-Content $ProfilesJson -Raw | ConvertFrom-Json
$common = $doc.common
$tasks  = & (Join-Path $here 'tasks\realuse-tasks.ps1')
$checks = & (Join-Path $here 'tasks\quality-checks.ps1')

Assert-NoStaleLlama -When 'pre-flight'

function Get-ArmSpec([string] $Name) {
    $s = $doc.profiles | Where-Object { $_.name -eq $Name }
    if (-not $s) { throw "unknown profile '$Name'" }
    $e = @{}
    foreach ($prop in $s.env.PSObject.Properties) { $e[$prop.Name] = [string]$prop.Value }
    [pscustomobject]@{ Name = $Name; Spec = $s; Env = $e; Threads = [int]$s.threads }
}

$armSpecs = @((Get-ArmSpec $ArmA), (Get-ArmSpec $ArmB))

Write-Host ''
Write-Host '=== quality evaluation ===' -ForegroundColor Cyan
Write-Host ("  arm A : {0} ({1})" -f $armSpecs[0].Name, $armSpecs[0].Spec.quality_class)
Write-Host ("  arm B : {0} ({1})" -f $armSpecs[1].Name, $armSpecs[1].Spec.quality_class)
Write-Host ("  tasks : {0}   checks: {1}   thinking: {2}" -f $tasks.Count, $checks.Count, $Thinking.IsPresent)
Write-Host "  output: $OutDir"
Write-Host ''

# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
$generations = @{}
$incidents   = New-Object System.Collections.ArrayList

foreach ($arm in $armSpecs) {
    Write-Host "--- $($arm.Name) ---" -ForegroundColor Cyan
    Write-Host 'starting server...' -NoNewline
    $session = $null
    $armResults = [ordered]@{}
    try {
        $session = Start-LlamaServer -Exe $serverExe -ModelPath $Model -OutDir $OutDir -Tag $arm.Name `
                                     -EnvVars $arm.Env -Common $common -Threads $arm.Threads `
                                     -Port $Port -StartupTimeoutSec $StartupTimeoutSec
        Write-Host " ready in $($session.LoadSeconds) s" -ForegroundColor Green

        foreach ($t in $tasks) {
            Write-Host ("-> {0,-26}" -f $t.id) -NoNewline
            $r = Invoke-LlamaCompletion -Session $session -Prompt $t.prompt -Endpoint $Endpoint `
                                        -Thinking:$Thinking -MaxTokens $t.n -Seed 42 -Temperature 0 -TimeoutSec $RequestTimeoutSec
            if (-not $r.ok) { [void]$incidents.Add("$($arm.Name)/$($t.id): $($r.error)") }
            $armResults[$t.id] = $r
            $note = if (-not $r.ok) { "  FAILED: $($r.error)" } elseif ($r.truncated_in_reasoning) { '  [truncated in reasoning]' } else { '' }
            Write-Host ("  {0,5} tok  {1,6:N2} tok/s{2}" -f $r.predicted_tokens, $r.tg_toks, $note) `
                        -ForegroundColor $(if (-not $r.ok) { 'Red' } elseif ($r.truncated_in_reasoning) { 'Yellow' } else { 'Green' })
        }
    }
    finally {
        if ($session) {
            foreach ($i in (Stop-LlamaServer -Session $session)) { [void]$incidents.Add("$($arm.Name): $i") }
        } else {
            Clear-RuntimeEnvironment; Stop-StaleLlama
        }
    }
    $generations[$arm.Name] = $armResults
    Write-Host ''
}

# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
function Get-DivergenceIndex([string] $a, [string] $b) {
    $n = [math]::Min($a.Length, $b.Length)
    for ($i = 0; $i -lt $n; $i++) { if ($a[$i] -ne $b[$i]) { return $i } }
    if ($a.Length -eq $b.Length) { return -1 }
    return $n
}

$comparisons = New-Object System.Collections.ArrayList
$checkRows   = New-Object System.Collections.ArrayList

foreach ($t in $tasks) {
    $ra = $generations[$ArmA][$t.id]
    $rb = $generations[$ArmB][$t.id]

    $ta = if ($ra.ok) { $ra.content } else { $null }
    $tb = if ($rb.ok) { $rb.content } else { $null }

    $taskChecks = @($checks | Where-Object { $_.task -eq $t.id })
    $agree = 0; $disagree = 0; $bothFail = 0; $detail = New-Object System.Collections.ArrayList

    foreach ($c in $taskChecks) {
        $pa = if (-not $truncated -and $ta) { [bool](& $c.check $ta) } else { $false }
        $pb = if (-not $truncated -and $tb) { [bool](& $c.check $tb) } else { $false }
        if ($pa -eq $pb) { if ($pa) { $agree++ } else { $bothFail++ } } else { $disagree++ }
        [void]$detail.Add([pscustomobject]@{ task = $t.id; check = $c.name; arm_a = $pa; arm_b = $pb; agrees = ($pa -eq $pb) })
        [void]$checkRows.Add([pscustomobject]@{ task = $t.id; check = $c.name; arm_a = $pa; arm_b = $pb; agrees = ($pa -eq $pb) })
    }

    # A run that spent its whole token budget reasoning has no answer to score. That
    # is a budget artefact, not a quality difference, and scoring it would manufacture
    # a disagreement out of nothing.
    $truncated = $ra.truncated_in_reasoning -or $rb.truncated_in_reasoning

    $identical = ($null -ne $ta) -and ($ta -eq $tb)
    $verdict =
        if ($null -eq $ta -or $null -eq $tb) { 'INCOMPLETE' }
        elseif ($truncated)                   { 'INCOMPLETE' }
        elseif ($identical)                   { 'BIT_IDENTICAL' }
        elseif ($disagree -gt 0)              { 'DIFFERENT' }
        elseif ($taskChecks.Count -eq 0)      { 'DIVERGENT_UNSCORED' }
        else                                  { 'TASK_EQUIVALENT' }

    [void]$comparisons.Add([pscustomobject]([ordered]@{
        task = $t.id; kind = $t.kind; language = $t.lang
        verdict = $verdict
        identical = $identical
        arm_a_truncated_in_reasoning = $ra.truncated_in_reasoning
        arm_b_truncated_in_reasoning = $rb.truncated_in_reasoning
        arm_a_stop = $ra.stop_reason
        arm_b_stop = $rb.stop_reason
        divergence_char_index = if ($null -ne $ta -and $null -ne $tb) { Get-DivergenceIndex $ta $tb } else { $null }
        checks_total = $taskChecks.Count
        checks_agree_pass = $agree
        checks_agree_fail = $bothFail
        checks_disagree = $disagree
        arm_a_tokens = $ra.predicted_tokens; arm_b_tokens = $rb.predicted_tokens
        arm_a_tg = $ra.tg_toks;              arm_b_tg = $rb.tg_toks
        arm_a_chars = if ($ta) { $ta.Length } else { 0 }
        arm_b_chars = if ($tb) { $tb.Length } else { 0 }
        check_detail = @($detail)
    }))
}

# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------
$pplResults = New-Object System.Collections.ArrayList

if (-not $SkipPerplexity) {
    if (-not (Test-Path $pplExe)) {
        [void]$incidents.Add('llama-perplexity.exe not found; perplexity skipped')
    } elseif (-not (Test-Path $CorpusDir)) {
        [void]$incidents.Add("corpus directory not found: $CorpusDir; perplexity skipped")
    } else {
        $corpora = @(Get-ChildItem $CorpusDir -Filter '*.txt' -File | Sort-Object Name)
        if ($corpora.Count -eq 0) { [void]$incidents.Add("no .txt corpora in $CorpusDir; perplexity skipped") }

        Write-Host '--- perplexity ---' -ForegroundColor Cyan
        foreach ($corpus in $corpora) {
            foreach ($arm in $armSpecs) {
                Write-Host ("-> {0,-18} {1,-20}" -f $corpus.BaseName, $arm.Name) -NoNewline

                Clear-RuntimeEnvironment
                foreach ($k in $arm.Env.Keys) { Set-Item "Env:$k" $arm.Env[$k] }

                $tag = "ppl-$($corpus.BaseName)-$($arm.Name)"
                $so  = Join-Path $OutDir "$tag.stdout.log"
                $se  = Join-Path $OutDir "$tag.stderr.log"

                $pplArgs = @(
                    '-m', $Model, '-f', $corpus.FullName,
                    '-c', '512', '-b', '512', '-ub', '512', '--chunks', "$PerplexityChunks",
                    '-ngl', '999', '-ncmoe', "$($common.n_cpu_moe)", '-t', "$($arm.Threads)",
                    '-fa', $common.flash_attention
                )

                $p = Start-Process -FilePath $pplExe -ArgumentList $pplArgs `
                                   -RedirectStandardOutput $so -RedirectStandardError $se `
                                   -PassThru -WindowStyle Hidden
                $p.WaitForExit()
                Clear-RuntimeEnvironment

                $text = ((Get-Content $so -Raw -ErrorAction SilentlyContinue), (Get-Content $se -Raw -ErrorAction SilentlyContinue)) -join "`n"
                $m = [regex]::Match($text, 'Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)')

                $row = [pscustomobject]@{
                    corpus = $corpus.BaseName; corpus_bytes = $corpus.Length
                    arm = $arm.Name; quality_class = $arm.Spec.quality_class
                    chunks = $PerplexityChunks
                    ppl = if ($m.Success) { [double]$m.Groups[1].Value } else { $null }
                    stderr_estimate = if ($m.Success) { [double]$m.Groups[2].Value } else { $null }
                    exit_code = $p.ExitCode
                }
                [void]$pplResults.Add($row)

                if ($m.Success) {
                    Write-Host ("  PPL {0} +/- {1}" -f $row.ppl, $row.stderr_estimate) -ForegroundColor Green
                } else {
                    Write-Host "  no PPL parsed (exit $($p.ExitCode))" -ForegroundColor Yellow
                    [void]$incidents.Add("perplexity did not report a final estimate for $tag")
                }
            }
        }
        Write-Host ''
    }
}

$pplPairs = @()
foreach ($corpus in (@($pplResults | ForEach-Object { $_.corpus }) | Sort-Object -Unique)) {
    $a = $pplResults | Where-Object { $_.corpus -eq $corpus -and $_.arm -eq $ArmA }
    $b = $pplResults | Where-Object { $_.corpus -eq $corpus -and $_.arm -eq $ArmB }
    if ($a -and $b -and $null -ne $a.ppl -and $null -ne $b.ppl) {
        $pplPairs += [pscustomobject]@{
            corpus = $corpus
            corpus_bytes = $a.corpus_bytes
            ppl_a = $a.ppl; ppl_a_stderr = $a.stderr_estimate
            ppl_b = $b.ppl; ppl_b_stderr = $b.stderr_estimate
            delta_pct = [math]::Round((($b.ppl - $a.ppl) / $a.ppl) * 100, 3)
            # A difference smaller than arm A's own standard error is not resolvable
            # by this measurement, whatever its sign.
            within_stderr = ([math]::Abs($b.ppl - $a.ppl) -lt $a.stderr_estimate)
        }
    }
}

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
$verdictCounts = @{}
foreach ($v in @('BIT_IDENTICAL','TASK_EQUIVALENT','DIVERGENT_UNSCORED','DIFFERENT','INCOMPLETE')) {
    $verdictCounts[$v] = @($comparisons | Where-Object { $_.verdict -eq $v }).Count
}

$report = [ordered]@{
    schema_version = '1.0.0'
    generated_utc  = (Get-Date).ToUniversalTime().ToString('o')
    arm_a = [ordered]@{ name = $ArmA; quality_class = $armSpecs[0].Spec.quality_class; env = $armSpecs[0].Env; threads = $armSpecs[0].Threads }
    arm_b = [ordered]@{ name = $ArmB; quality_class = $armSpecs[1].Spec.quality_class; env = $armSpecs[1].Env; threads = $armSpecs[1].Threads }
    environment = Get-BenchmarkEnvironment -BinDir $BinDir -ModelPath $Model
    sampling = [ordered]@{ seed = 42; temperature = 0; cache_prompt = $false; endpoint = $Endpoint; thinking = [bool]$Thinking; note = 'identical prompts, seed and settings in both arms' }
    verdict_counts = $verdictCounts
    tasks_length_capped = @($comparisons | Where-Object { $_.arm_a_stop -eq 'length' -or $_.arm_b_stop -eq 'length' } | ForEach-Object { $_.task })
    checks = [ordered]@{
        total    = $checkRows.Count
        agree    = @($checkRows | Where-Object { $_.agrees }).Count
        disagree = @($checkRows | Where-Object { -not $_.agrees }).Count
        arm_a_pass = @($checkRows | Where-Object { $_.arm_a }).Count
        arm_b_pass = @($checkRows | Where-Object { $_.arm_b }).Count
    }
    perplexity = @($pplPairs)
    perplexity_raw = @($pplResults)
    comparisons = @($comparisons)
    check_detail = @($checkRows)
    incidents = @($incidents)
}

$report | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $OutDir 'quality_results.json') -Encoding UTF8
$comparisons | Select-Object task, kind, language, verdict, divergence_char_index, checks_total, checks_agree_pass, checks_agree_fail, checks_disagree, arm_a_tokens, arm_b_tokens, arm_a_stop, arm_b_stop, arm_a_truncated_in_reasoning, arm_b_truncated_in_reasoning |
    Export-Csv (Join-Path $OutDir 'quality_results.csv') -NoTypeInformation -Encoding UTF8
$checkRows | Export-Csv (Join-Path $OutDir 'quality_checks.csv') -NoTypeInformation -Encoding UTF8
if ($pplPairs.Count) { $pplPairs | Export-Csv (Join-Path $OutDir 'perplexity.csv') -NoTypeInformation -Encoding UTF8 }

# side-by-side transcripts, so a human can read what actually differed
$tx = New-Object System.Text.StringBuilder
foreach ($t in $tasks) {
    $c = $comparisons | Where-Object { $_.task -eq $t.id }
    [void]$tx.AppendLine("################ $($t.id)  [$($t.kind) / $($t.lang)]  ->  $($c.verdict) ################")
    [void]$tx.AppendLine('PROMPT:'); [void]$tx.AppendLine($t.prompt); [void]$tx.AppendLine('')
    foreach ($armName in @($ArmA, $ArmB)) {
        $g = $generations[$armName][$t.id]
        [void]$tx.AppendLine("---------------- $armName ----------------")
        if ($g.reasoning) {
            [void]$tx.AppendLine('[reasoning]')
            [void]$tx.AppendLine($g.reasoning)
            [void]$tx.AppendLine('[answer]')
        }
        [void]$tx.AppendLine($(if (-not $g.ok) { '<failed>' } elseif ($g.content) { $g.content } else { '<truncated - budget consumed by reasoning>' }))
        [void]$tx.AppendLine('')
    }
}
Set-Content (Join-Path $OutDir 'transcripts-side-by-side.txt') -Value $tx.ToString() -Encoding UTF8

Write-Host '=== verdicts ===' -ForegroundColor Cyan
foreach ($k in @('BIT_IDENTICAL','TASK_EQUIVALENT','DIVERGENT_UNSCORED','DIFFERENT','INCOMPLETE')) {
    Write-Host ("  {0,-20} {1}" -f $k, $verdictCounts[$k])
}
Write-Host ''
Write-Host ("checks: {0} total, {1} agree, {2} disagree  |  {3} passes {4}, {5} passes {6}" -f `
    $report.checks.total, $report.checks.agree, $report.checks.disagree, $ArmA, $report.checks.arm_a_pass, $ArmB, $report.checks.arm_b_pass)

# A task whose answer was cut off at the token budget is weaker evidence: a check can
# flip purely because of where the cut fell. Surface it rather than silently scoring it.
$lengthCapped = @($comparisons | Where-Object { $_.arm_a_stop -eq 'length' -or $_.arm_b_stop -eq 'length' })
if ($lengthCapped.Count -gt 0) {
    Write-Host ''
    Write-Host ("tasks whose answer hit the token budget in at least one arm ({0}): {1}" -f `
        $lengthCapped.Count, (($lengthCapped | ForEach-Object { $_.task }) -join ', ')) -ForegroundColor Yellow
    Write-Host '  a check disagreement on these is weaker evidence than on a task that finished naturally.'
}

if (@($comparisons | Where-Object { $_.verdict -eq 'DIFFERENT' }).Count -gt 0) {
    Write-Host ''
    Write-Host 'tasks where an objective check disagreed:' -ForegroundColor Yellow
    foreach ($c in @($comparisons | Where-Object { $_.verdict -eq 'DIFFERENT' })) {
        foreach ($d in @($c.check_detail | Where-Object { -not $_.agrees })) {
            Write-Host ("  {0,-22} {1,-52} {2}={3}  {4}={5}" -f $d.task, $d.check, $ArmA, $d.arm_a, $ArmB, $d.arm_b)
        }
    }
}

if ($pplPairs.Count) {
    Write-Host ''
    Write-Host '=== perplexity ===' -ForegroundColor Cyan
    $pplPairs | Format-Table corpus, ppl_a, ppl_b, delta_pct, within_stderr -AutoSize
}

Write-Host ''
foreach ($i in $incidents) { Write-Host "incident: $i" -ForegroundColor Yellow }
Write-Host "output: $OutDir"
exit 0
