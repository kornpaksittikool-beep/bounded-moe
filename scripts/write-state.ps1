<#
.SYNOPSIS
    Regenerate PRODUCTIZATION_STATE.json from the current state of the workspace.

.DESCRIPTION
    The state file is a checkpoint, not a hand-maintained document. This script
    rebuilds it from what is actually on disk: which deliverables exist, which
    benchmark and test result sets have been produced, and the verified hashes of
    the protected baseline.

    Run it after each phase.
#>
[CmdletBinding()]
param(
    [string] $Repo,
    [string] $Phase  = 'G8',
    [string] $Status = 'COMPLETE',
    [string] $Next   = 'awaiting human approval for publication'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $Repo) { $Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }

$checkpoint = 'D:\llm-v0\autonomous-research\c1-checkpoints\f5-final'
$gguf       = 'D:\llm-v0\models\Qwen3.6-35B-A3B-Q4_K_M.gguf'

function Hash-Dir([string] $dir, [string[]] $ext) {
    $h = [ordered]@{}
    if (Test-Path $dir) {
        Get-ChildItem $dir -File | Where-Object { $_.Extension -in $ext } | Sort-Object Name |
            ForEach-Object { $h[$_.Name] = (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
    }
    $h
}

$deliverables = [ordered]@{}
foreach ($rel in @(
    'README.md','LICENSE','THIRD_PARTY_NOTICES.md','LICENSE_AUDIT.md','.gitignore',
    'PRODUCTIZATION_BASELINE.md','PRODUCTIZATION_LOG.md','PUBLIC_RELEASE_CHECKLIST.md',
    'PRODUCTIZATION_FINAL_REPORT.md','BUILD_ENVIRONMENT.json',
    'profiles\profiles.json','profiles\PROFILES.md',
    'benchmark\run-benchmark-suite.ps1','benchmark\run-build-ab.ps1',
    'benchmark\lib\BenchLib.psm1','benchmark\BENCHMARK_METHODOLOGY.md',
    'launcher\run-local-moe.ps1','launcher\USER_GUIDE.md',
    'scripts\build-clean.ps1','scripts\write-state.ps1',
    'tests\run-real-use.ps1','tests\run-quality-eval.ps1','tests\lib\ServerLib.psm1',
    'tests\tasks\realuse-tasks.ps1','tests\tasks\quality-checks.ps1',
    'patches\0001-external-expert-cache.patch',
    'docs\ARCHITECTURE.md','docs\PERFORMANCE.md','docs\CORRECTNESS.md',
    'docs\DESIGN_DECISIONS.md','docs\KNOWN_LIMITATIONS.md','docs\RESEARCH_HISTORY.md',
    'docs\BUILD_FROM_CLEAN.md','docs\CLEAN_REPRODUCTION.md','docs\QUALITY_REPORT.md',
    'docs\REAL_USE_VALIDATION.md','docs\STABILITY_REPORT.md'
)) {
    $p = Join-Path $Repo $rel
    $deliverables[$rel] = if (Test-Path $p) {
        [ordered]@{ present = $true; bytes = (Get-Item $p).Length; sha256 = (Get-FileHash $p -Algorithm SHA256).Hash }
    } else {
        [ordered]@{ present = $false }
    }
}

function Result-Sets([string] $dir) {
    if (-not (Test-Path $dir)) { return @() }
    @(Get-ChildItem $dir -Directory | Sort-Object Name | ForEach-Object {
        [ordered]@{
            name  = $_.Name
            files = @(Get-ChildItem $_.FullName -File | ForEach-Object { $_.Name })
            utc   = $_.LastWriteTimeUtc.ToString('o')
        }
    })
}

$state = [ordered]@{
    schema_version = '1.0.0'
    generated_utc  = (Get-Date).ToUniversalTime().ToString('o')
    phase          = $Phase
    status         = $Status
    next           = $Next
    workspace      = $Repo

    protected_baseline = [ordered]@{
        checkpoint_dir     = $checkpoint
        checkpoint_intact  = (Test-Path $checkpoint)
        checkpoint_sources = Hash-Dir $checkpoint @('.c', '.cpp', '.h')
        checkpoint_binaries = Hash-Dir (Join-Path $checkpoint 'bin') @('.dll', '.exe')
        gguf = if (Test-Path $gguf) {
            $i = Get-Item $gguf
            [ordered]@{ path = $i.FullName; bytes = $i.Length; last_write_utc = $i.LastWriteTimeUtc.ToString('o') }
        } else { $null }
        stale_llama_processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @('llama-cli.exe','llama-server.exe','llama-perplexity.exe') } |
            ForEach-Object { "$($_.Name):$($_.ProcessId)" })
    }

    clean_room = [ordered]@{
        source_present = (Test-Path (Join-Path $Repo 'clean-room\llama.cpp'))
        build_present  = (Test-Path (Join-Path $Repo 'clean-room\build\bin'))
        binaries       = Hash-Dir (Join-Path $Repo 'clean-room\build\bin') @('.dll', '.exe')
    }

    deliverables = $deliverables

    results = [ordered]@{
        benchmark = Result-Sets (Join-Path $Repo 'benchmark\results')
        tests     = Result-Sets (Join-Path $Repo 'tests\results')
    }
}

$out = Join-Path $Repo 'PRODUCTIZATION_STATE.json'
$state | ConvertTo-Json -Depth 10 | Set-Content $out -Encoding UTF8

$missing = @($deliverables.Keys | Where-Object { -not $deliverables[$_].present })
Write-Host "wrote $out"
Write-Host ("deliverables present: {0}/{1}" -f ($deliverables.Count - $missing.Count), $deliverables.Count)
if ($missing.Count) { Write-Host "missing:"; $missing | ForEach-Object { Write-Host "  $_" } }
Write-Host ("baseline checkpoint intact: {0}" -f $state.protected_baseline.checkpoint_intact)
Write-Host ("stale llama processes: {0}" -f @($state.protected_baseline.stale_llama_processes).Count)
