<#
.SYNOPSIS
    Run llama-perplexity over a matrix of corpora and runtime configurations.

.DESCRIPTION
    The quality question "does repack cost anything" only has a clean answer when the
    two arms differ in *one* thing. Comparing a repack run at one thread count against
    an external-cache run at another confounds three variables at once.

    This script runs an explicit matrix so each variable can be isolated: same corpus,
    same chunk count, one configuration knob at a time.

    Note on what perplexity measures. It reads the model's logits directly, while greedy
    decoding only reads their argmax. Two configurations can therefore produce a
    bit-identical token stream and still report different perplexity, if their floating
    point results differ in bits that never change which token wins. Perplexity is the
    more sensitive instrument of the two.

.EXAMPLE
    .\run-perplexity-matrix.ps1 -Model <gguf> -Corpora english-356k,english-tail
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]   $Model,
    [string]   $BinDir,
    [string]   $CorpusDir,
    [string]   $OutDir,
    [string[]] $Corpora,
    [string[]] $Configs = @('plain-t12', 'plain-t4', 'repack-t12', 'external-t4'),
    [int]      $Chunks  = 24,
    # Perplexity defaults to a 512-token ubatch. The profiles run at 256, and whether a
    # numerical difference survives that change is a question worth being able to ask.
    [int]      $Batch   = 512,
    [string]   $ProfilesJson
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
Import-Module (Join-Path $repo 'benchmark\lib\BenchLib.psm1') -Force

if (-not $BinDir)       { $BinDir       = Join-Path $repo 'clean-room\build\bin' }
if (-not $CorpusDir)    { $CorpusDir    = Join-Path $here 'corpora' }
if (-not $ProfilesJson) { $ProfilesJson = Join-Path $repo 'profiles\profiles.json' }
if (-not $OutDir)       { $OutDir       = Join-Path $repo ('tests\results\ppl-matrix-{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss')) }

$exe = Join-Path $BinDir 'llama-perplexity.exe'
foreach ($p in @($Model, $exe, $CorpusDir, $ProfilesJson)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$common = (Get-Content $ProfilesJson -Raw | ConvertFrom-Json).common

# Each configuration changes exactly one thing from its neighbour, so a difference
# between two rows can be attributed.
$catalog = @{
    'plain-t12'   = @{ threads = 12; env = @{};                                                                                          note = 'plain vec_dot, resident weights - the research baseline' }
    'plain-t4'    = @{ threads = 4;  env = @{};                                                                                          note = 'plain vec_dot, resident weights, fewer threads' }
    'repack-t12'  = @{ threads = 12; env = @{ 'B1B_MOE_REPACK' = '1' };                                                                   note = 'packed Q4_K GEMM, resident weights' }
    'repack-t4'   = @{ threads = 4;  env = @{ 'B1B_MOE_REPACK' = '1' };                                                                   note = 'packed Q4_K GEMM, fewer threads' }
    'external-t4' = @{ threads = 4;  env = @{ 'B1B_EXTERNAL_EXPERT_STORAGE' = '1'; 'B1B_EXTERNAL_MAX_LAYERS' = '2'; 'LLAMA_EXPERT_CACHE_MB' = '1024' }; note = 'external expert cache, MAX_SPEED_EXACT settings' }
    'external-t4-low' = @{ threads = 4; env = @{ 'B1B_EXTERNAL_EXPERT_STORAGE' = '1'; 'B1B_EXTERNAL_MAX_LAYERS' = '30'; 'LLAMA_EXPERT_CACHE_MB' = '2560' }; note = 'external expert cache, LOW_MEMORY_EXACT settings' }
}

foreach ($c in $Configs) { if (-not $catalog.ContainsKey($c)) { throw "unknown config '$c'. Known: $($catalog.Keys -join ', ')" } }

if (-not $Corpora) {
    $Corpora = @(Get-ChildItem $CorpusDir -Filter '*.txt' -File | Sort-Object Name | ForEach-Object { $_.BaseName })
}

Assert-NoStaleLlama -When 'pre-flight'

Write-Host ''
Write-Host '=== perplexity matrix ===' -ForegroundColor Cyan
Write-Host "  corpora : $($Corpora -join ', ')"
Write-Host "  configs : $($Configs -join ', ')"
Write-Host "  chunks  : $Chunks   batch/ubatch: $Batch"
Write-Host "  output  : $OutDir"
Write-Host ''

$rows = New-Object System.Collections.ArrayList

try {
    foreach ($corpusName in $Corpora) {
        $corpus = Join-Path $CorpusDir "$corpusName.txt"
        if (-not (Test-Path $corpus)) { Write-Host "skipping missing corpus $corpusName" -ForegroundColor Yellow; continue }

        foreach ($configName in $Configs) {
            $cfg = $catalog[$configName]
            Write-Host ("-> {0,-16} {1,-18}" -f $corpusName, $configName) -NoNewline

            Clear-RuntimeEnvironment
            foreach ($k in $cfg.env.Keys) { Set-Item "Env:$k" $cfg.env[$k] }

            $tag = "ppl-$corpusName-$configName-b$Batch"
            $so  = Join-Path $OutDir "$tag.stdout.log"
            $se  = Join-Path $OutDir "$tag.stderr.log"

            $args = @(
                '-m', $Model, '-f', $corpus,
                '-c', '512', '-b', "$Batch", '-ub', "$Batch", '--chunks', "$Chunks",
                '-ngl', '999', '-ncmoe', "$($common.n_cpu_moe)", '-t', "$($cfg.threads)",
                '-fa', $common.flash_attention
            )

            $t0 = Get-Date
            $p = Start-Process -FilePath $exe -ArgumentList $args `
                               -RedirectStandardOutput $so -RedirectStandardError $se `
                               -PassThru -WindowStyle Hidden
            $p.WaitForExit()
            $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            Clear-RuntimeEnvironment

            $text = ((Get-Content $so -Raw -ErrorAction SilentlyContinue), (Get-Content $se -Raw -ErrorAction SilentlyContinue)) -join "`n"
            $m = [regex]::Match($text, 'Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)')

            $row = [pscustomobject]@{
                corpus       = $corpusName
                corpus_bytes = (Get-Item $corpus).Length
                config       = $configName
                threads      = $cfg.threads
                note         = $cfg.note
                chunks       = $Chunks
                batch        = $Batch
                ppl          = if ($m.Success) { [double]$m.Groups[1].Value } else { $null }
                stderr_est   = if ($m.Success) { [double]$m.Groups[2].Value } else { $null }
                seconds      = $elapsed
                exit_code    = $p.ExitCode
            }
            [void]$rows.Add($row)

            if ($m.Success) {
                Write-Host ("  PPL {0,9} +/- {1,-9}  {2,6} s" -f $row.ppl, $row.stderr_est, $elapsed) -ForegroundColor Green
            } else {
                Write-Host ("  no PPL parsed (exit {0})" -f $p.ExitCode) -ForegroundColor Red
            }
        }
    }
}
finally { Stop-StaleLlama; Clear-RuntimeEnvironment }

# ---- outputs --------------------------------------------------------------
$doc = [ordered]@{
    schema_version = '1.0.0'
    generated_utc  = (Get-Date).ToUniversalTime().ToString('o')
    chunks         = $Chunks
    batch          = $Batch
    environment    = Get-BenchmarkEnvironment -BinDir $BinDir -ModelPath $Model
    configs        = @($Configs | ForEach-Object { [ordered]@{ name = $_; threads = $catalog[$_].threads; env = $catalog[$_].env; note = $catalog[$_].note } })
    rows           = @($rows)
}
$doc | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir 'perplexity-matrix.json') -Encoding UTF8
$rows | Export-Csv (Join-Path $OutDir 'perplexity-matrix.csv') -NoTypeInformation -Encoding UTF8

Write-Host ''
Write-Host '=== matrix ===' -ForegroundColor Cyan
$byCorpus = $rows | Group-Object corpus
foreach ($g in $byCorpus) {
    Write-Host ""
    Write-Host ("{0} ({1} bytes, {2} chunks)" -f $g.Name, ($g.Group[0].corpus_bytes), $Chunks)
    $g.Group | Format-Table config, threads, ppl, stderr_est, seconds -AutoSize | Out-String | Write-Host
}
Write-Host "output: $OutDir"
