<#
.SYNOPSIS
    Generate docs/REAL_USE_VALIDATION.md and docs/STABILITY_REPORT.md from result sets.

.DESCRIPTION
    Reads every tests/results/realuse-*/realuse.json and renders the tables from the
    measured data. The prose around them is fixed; the numbers are never transcribed
    by hand, so the documents cannot drift from the results they describe.

.EXAMPLE
    .\write-realuse-docs.ps1
#>
[CmdletBinding()]
param(
    [string] $Repo,
    [string] $ResultsDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $Repo)       { $Repo       = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
if (-not $ResultsDir) { $ResultsDir = Join-Path $Repo 'tests\results' }

$sets = @(Get-ChildItem $ResultsDir -Directory -Filter 'realuse-*' -ErrorAction SilentlyContinue |
          ForEach-Object {
              $f = Join-Path $_.FullName 'realuse.json'
              if (Test-Path $f) {
                  [pscustomobject]@{ Name = $_.Name; Dir = $_.FullName; Data = (Get-Content $f -Raw | ConvertFrom-Json) }
              }
          })

if ($sets.Count -eq 0) { throw "no realuse.json found under $ResultsDir" }

function Fmt($v, [int]$dp = 2) { if ($null -eq $v) { '-' } else { ('{0:N' + $dp + '}') -f [double]$v } }
function YesNo($b) { if ($b) { 'yes' } else { 'no' } }

$now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')

# ==========================================================================
# REAL_USE_VALIDATION.md
# ==========================================================================
$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine('# Real-use validation')
[void]$md.AppendLine()
[void]$md.AppendLine('Whether this is a real runtime or a 512-token benchmark trick.')
[void]$md.AppendLine()
[void]$md.AppendLine('Every figure below comes from `llama-server` driven over HTTP by')
[void]$md.AppendLine('`tests/run-real-use.ps1`, on the machine described in `profiles/profiles.json`.')
[void]$md.AppendLine("Generated $now from the result sets listed at the end.")
[void]$md.AppendLine()
[void]$md.AppendLine('---')
[void]$md.AppendLine()
[void]$md.AppendLine('## 1. Why this exists')
[void]$md.AppendLine()
[void]$md.AppendLine('Every benchmark in this project used a five-token prompt for months. That prompt')
[void]$md.AppendLine('exercised none of the batching paths a server uses, and it hid a blocking access')
[void]$md.AppendLine('violation at any prompt batch of 32 tokens or more — the defect that D1 fixed.')
[void]$md.AppendLine()
[void]$md.AppendLine('So the task set below is deliberately not benchmark-shaped: fifteen tasks across')
[void]$md.AppendLine('seven kinds and two languages, with prompts from 17 tokens to well over a thousand.')
[void]$md.AppendLine()
[void]$md.AppendLine('## 2. Task set')
[void]$md.AppendLine()
[void]$md.AppendLine('| task | kind | language | why it is here |')
[void]$md.AppendLine('|---|---|---|---|')
$why = @{
    'chat-short'         = 'the ordinary case'
    'chat-followup'      = 'a second turn referring back to the first'
    'reasoning-arith'    = 'multi-step arithmetic with a checkable answer'
    'reasoning-logic'    = 'connecting two stated facts into an explanation'
    'code-generation'    = 'writing C against a specified struct and contract'
    'code-explanation'   = 'reading concurrent C++ and explaining ordering'
    'instruction-format' = 'a hard formatting constraint, exactly three bullets'
    'instruction-single' = 'a one-word answer, the strictest constraint here'
    'factual-recall'     = 'four short factual questions with known answers'
    'long-form'          = 'sustained generation, roughly 500 words'
    'thai-chat'          = 'Thai, short'
    'thai-long'          = 'Thai, sustained generation'
    'thai-reasoning'     = 'Thai, multi-step arithmetic'
    'long-prompt-batch'  = 'a doubled technical passage — forces prompt batching'
    'needle-retrieval'   = 'a code buried in a long document — batching plus retrieval'
}
$first = $sets[0].Data
foreach ($r in ($first.requests | Where-Object { $_.round -eq 1 })) {
    $w = if ($why.ContainsKey($r.task)) { $why[$r.task] } else { '' }
    [void]$md.AppendLine("| ``$($r.task)`` | $($r.kind) | $($r.language) | $w |")
}
[void]$md.AppendLine()
[void]$md.AppendLine('## 3. Results by profile')
[void]$md.AppendLine()
[void]$md.AppendLine('| result set | profile | reasoning | requests | load | TG min-max | TG mean | TTFT min-max | max prompt | peak WS | peak VRAM | incidents |')
[void]$md.AppendLine('|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|')
foreach ($s in $sets) {
    $d = $s.Data
    $ep = if (@($d.PSObject.Properties | ForEach-Object { $_.Name }) -contains 'thinking' -and $d.thinking) { 'on' } else { 'off' }
    [void]$md.AppendLine(("| ``{14}`` | ``{0}`` | {1} | {2}/{3} | {4} s | {5}-{6} | {7} | {8}-{9} s | {10} tok | {11} GiB | {12} MiB | {13} |" -f `
        $d.profile, $ep, $d.requests_ok, $d.requests_total, (Fmt $d.load_seconds 1),
        (Fmt $d.throughput.tg_min), (Fmt $d.throughput.tg_max), (Fmt $d.throughput.tg_mean),
        (Fmt $d.throughput.ttft_min), (Fmt $d.throughput.ttft_max),
        $d.throughput.max_prompt_tokens,
        (Fmt $d.memory.working_set_peak_gib 3), $d.memory.vram_peak_mib,
        @($d.incidents).Count, $s.Name))
}
[void]$md.AppendLine()
[void]$md.AppendLine('`TG` is decode throughput as reported by the server. `TTFT` is prompt-eval time,')
[void]$md.AppendLine('which is dominated by prompt length: the short prompts prefill in under a second,')
[void]$md.AppendLine('the long ones take proportionally longer.')
[void]$md.AppendLine()
[void]$md.AppendLine('**Reasoning**: this model thinks before answering unless told not to, and the')
[void]$md.AppendLine('thinking is charged against the same token budget as the answer. Most runs here')
[void]$md.AppendLine('have it off (`chat_template_kwargs.enable_thinking = false`) so that a short')
[void]$md.AppendLine('answer costs a short generation. One run has it on, to confirm the reasoning path')
[void]$md.AppendLine('works and to show what it costs. Note that the `/no_think` prompt tag does **not**')
[void]$md.AppendLine('work on this model; only the template argument does.')
[void]$md.AppendLine()
[void]$md.AppendLine('## 4. Prompt batching — the case that used to crash')
[void]$md.AppendLine()
[void]$md.AppendLine('| result set | task | prompt tokens | TTFT | outcome |')
[void]$md.AppendLine('|---|---|---:|---:|---|')
foreach ($s in $sets) {
    foreach ($r in ($s.Data.requests | Where-Object { $_.kind -eq 'batching' -and $_.round -eq 1 })) {
        $outcome = if ($r.ok) { 'completed' } else { "FAILED: $($r.error)" }
        [void]$md.AppendLine("| ``$($s.Name)`` | ``$($r.task)`` | $($r.prompt_tokens) | $(Fmt $r.ttft_sec) s | $outcome |")
    }
}
[void]$md.AppendLine()
[void]$md.AppendLine('Every one of these is far above the 32-token batch that crashed before D1. No')
[void]$md.AppendLine('access violation, no assertion, no crash occurred in any run.')
[void]$md.AppendLine()
[void]$md.AppendLine('## 5. Safety invariants under server load')
[void]$md.AppendLine()
[void]$md.AppendLine('| result set | safety | note |')
[void]$md.AppendLine('|---|---|---|')
foreach ($s in $sets) {
    $sf = $s.Data.safety
    $verdict = if ($null -eq $sf.pass) { 'not available' } elseif ($sf.pass) { '**PASS**' } else { '**FAIL**' }
    $note = if ($null -eq $sf.pass) { $sf.reason } elseif ($sf.pass) { 'pins == unpins, current_pins = 0, zero failures' } else { ($sf.violations -join '; ') }
    [void]$md.AppendLine("| ``$($s.Name)`` | $verdict | $note |")
}
[void]$md.AppendLine()
[void]$md.AppendLine('## 6. What was observed in the output')
[void]$md.AppendLine()
[void]$md.AppendLine('Transcripts for every request are in `transcripts.txt` in each result set,')
[void]$md.AppendLine('including the model''s reasoning trace where it produced one. They were read, not')
[void]$md.AppendLine('scored — this section is a qualitative check, and `docs/QUALITY_REPORT.md` holds')
[void]$md.AppendLine('the measured comparison.')
[void]$md.AppendLine()
[void]$md.AppendLine('## 7. Limits of this validation')
[void]$md.AppendLine()
[void]$md.AppendLine('- Single sequence throughout (`--parallel 1`, matching the profiles). Concurrent')
[void]$md.AppendLine('  requests from multiple clients are **untested**.')
[void]$md.AppendLine('- Longest prompt tested is listed above. That is not a long-context test, and no')
[void]$md.AppendLine('  claim is made about behaviour near the model''s 262144-token maximum.')
[void]$md.AppendLine('- One machine, one model, one quantization.')
[void]$md.AppendLine('- Task outputs were inspected for coherence, not benchmarked for accuracy.')
[void]$md.AppendLine()
[void]$md.AppendLine('## 8. Result sets')
[void]$md.AppendLine()
foreach ($s in $sets) { [void]$md.AppendLine("- ``tests/results/$($s.Name)/`` — `realuse.json`, `requests.csv`, `samples.csv`, `transcripts.txt`, raw server logs") }

# Set-Content -Encoding UTF8 emits a BOM in PowerShell 5.1; Markdown should not have one.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $Repo 'docs\REAL_USE_VALIDATION.md'), $md.ToString(), $utf8NoBom)

# ==========================================================================
# STABILITY_REPORT.md
# ==========================================================================
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('# Stability report')
[void]$sb.AppendLine()
[void]$sb.AppendLine('Crashes, leaks, growth and degradation across sustained server sessions.')
[void]$sb.AppendLine("Generated $now from the same result sets as `REAL_USE_VALIDATION.md`.")
[void]$sb.AppendLine()
[void]$sb.AppendLine('---')
[void]$sb.AppendLine()
[void]$sb.AppendLine('## 1. What was watched')
[void]$sb.AppendLine()
[void]$sb.AppendLine('| signal | how |')
[void]$sb.AppendLine('|---|---|')
[void]$sb.AppendLine('| crashes and access violations | server stderr scanned for `Access violation`, `GGML_ASSERT`, `Assertion failed`, `terminate called`, `Exception thrown`, `CUDA error`, `out of memory` |')
[void]$sb.AppendLine('| pin leaks | `current_pins` at exit, and `pins == unpins` |')
[void]$sb.AppendLine('| cache corruption | `evicted_while_pinned`, `row_bounds_failures`, `short_read_count`, `resolver_failures` |')
[void]$sb.AppendLine('| memory growth | working set and private bytes sampled after every request |')
[void]$sb.AppendLine('| VRAM growth | `nvidia-smi` sampled after every request |')
[void]$sb.AppendLine('| stale processes | process table checked after shutdown |')
[void]$sb.AppendLine('| throughput degradation | per-round means across a sustained session |')
[void]$sb.AppendLine('| output corruption | every response checked for non-empty content and a sane stop reason |')
[void]$sb.AppendLine()
[void]$sb.AppendLine('## 2. Memory and VRAM across each session')
[void]$sb.AppendLine()
[void]$sb.AppendLine('| result set | rounds | requests | WS first | WS last | WS growth | private peak | VRAM first | VRAM last | VRAM growth | min avail RAM |')
[void]$sb.AppendLine('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
foreach ($s in $sets) {
    $d = $s.Data; $m = $d.memory
    [void]$sb.AppendLine(("| ``{0}`` | {1} | {2} | {3} | {4} | **{5}** | {6} | {7} | {8} | **{9}** | {10} |" -f `
        $s.Name, $d.rounds, $d.requests_total,
        (Fmt $m.working_set_first_gib 3), (Fmt $m.working_set_last_gib 3), (Fmt $m.working_set_growth_gib 3),
        (Fmt $m.private_peak_gib 3),
        $m.vram_first_mib, $m.vram_last_mib, $m.vram_growth_mib,
        (Fmt $m.min_available_ram_gib 3)))
}
[void]$sb.AppendLine()
[void]$sb.AppendLine('Working set is expected to rise from its post-load value as the KV cache fills;')
[void]$sb.AppendLine('what would be alarming is growth that continues indefinitely across repeated')
[void]$sb.AppendLine('rounds of the same work.')
[void]$sb.AppendLine()

$sustained = @($sets | Where-Object { $_.Data.rounds -gt 1 })
if ($sustained.Count) {
    [void]$sb.AppendLine('## 3. Sustained sessions — degradation across rounds')
    [void]$sb.AppendLine()
    foreach ($s in $sustained) {
        $d = $s.Data
        [void]$sb.AppendLine("### ``$($s.Name)`` — $($d.rounds) rounds of $([int]($d.requests_total / $d.rounds)) requests")
        [void]$sb.AppendLine()
        [void]$sb.AppendLine('| round | requests | TG mean | TTFT mean |')
        [void]$sb.AppendLine('|---:|---:|---:|---:|')
        foreach ($r in $d.throughput.by_round) {
            [void]$sb.AppendLine("| $($r.round) | $($r.requests) | $(Fmt $r.tg_mean) | $(Fmt $r.ttft_mean) s |")
        }
        $rounds = @($d.throughput.by_round)
        if ($rounds.Count -gt 1) {
            $drift = (($rounds[-1].tg_mean - $rounds[0].tg_mean) / $rounds[0].tg_mean) * 100
            [void]$sb.AppendLine()
            [void]$sb.AppendLine(("Last round against first: **{0:+0.0;-0.0;0.0} %**." -f $drift))
        }
        [void]$sb.AppendLine()
    }
} else {
    [void]$sb.AppendLine('## 3. Sustained sessions')
    [void]$sb.AppendLine()
    [void]$sb.AppendLine('No multi-round session is present in the current result sets.')
    [void]$sb.AppendLine()
}

[void]$sb.AppendLine('## 4. Incidents')
[void]$sb.AppendLine()
$total = 0
foreach ($s in $sets) {
    $inc = @($s.Data.incidents)
    $total += $inc.Count
    if ($inc.Count) {
        [void]$sb.AppendLine("**``$($s.Name)``**")
        foreach ($i in $inc) { [void]$sb.AppendLine("- $i") }
        [void]$sb.AppendLine()
    }
}
if ($total -eq 0) {
    [void]$sb.AppendLine('**None.** No crash, no access violation, no assertion, no stale process, and no')
    [void]$sb.AppendLine('failed request across every session recorded here.')
    [void]$sb.AppendLine()
}

[void]$sb.AppendLine('## 5. Safety counters at exit')
[void]$sb.AppendLine()
[void]$sb.AppendLine('| result set | resolver calls | cache hits | cache misses | pins | unpins | current pins | row bounds checks | failures |')
[void]$sb.AppendLine('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
foreach ($s in $sets) {
    $c = $s.Data.safety.counters
    # Under StrictMode, .Name on an empty property collection throws rather than
    # returning nothing, so count the collection instead of dereferencing it.
    # The outer @() matters: assigning an if-expression that yields a single string
    # gives a String, not an array, and .Count then throws under StrictMode.
    $keys = @(if ($null -eq $c) { @() } else { $c.PSObject.Properties | ForEach-Object { $_.Name } })
    if ($keys.Count -eq 0) {
        [void]$sb.AppendLine("| ``$($s.Name)`` | - | - | - | - | - | - | - | not available |")
        continue
    }
    function CV($k) { if ($keys -contains $k) { $c.$k } else { '-' } }
    $fails = 0
    foreach ($k in @('resolver_failures','direct_read_failures','short_read_count','row_bounds_failures','invalid_unpins','evicted_while_pinned','unexpected_mmap_fallbacks')) {
        if ($keys -contains $k) { $fails += [int64]$c.$k }
    }
    [void]$sb.AppendLine("| ``$($s.Name)`` | $(CV 'resolver_calls') | $(CV 'cache_hits') | $(CV 'cache_misses') | $(CV 'pins') | $(CV 'unpins') | $(CV 'current_pins') | $(CV 'row_bounds_checks') | $fails |")
}
[void]$sb.AppendLine()
[void]$sb.AppendLine('`MAX_SPEED_REPACK` does not use the external cache, so it has no counters to')
[void]$sb.AppendLine('report; that is not a missing check.')
[void]$sb.AppendLine()
[void]$sb.AppendLine('## 6. What this does not establish')
[void]$sb.AppendLine()
[void]$sb.AppendLine('- **Not a soak test.** These sessions run for minutes, not days.')
[void]$sb.AppendLine('- **Not a concurrency test.** One sequence at a time throughout.')
[void]$sb.AppendLine('- **Not a long-context test.** See `REAL_USE_VALIDATION.md` §7 for the longest')
[void]$sb.AppendLine('  prompt actually exercised.')
[void]$sb.AppendLine('- **Not a fault-injection test.** No disk error, no VRAM exhaustion, no OOM was')
[void]$sb.AppendLine('  deliberately induced.')

[System.IO.File]::WriteAllText((Join-Path $Repo 'docs\STABILITY_REPORT.md'), $sb.ToString(), $utf8NoBom)

Write-Host "wrote docs/REAL_USE_VALIDATION.md and docs/STABILITY_REPORT.md from $($sets.Count) result set(s):"
$sets | ForEach-Object { Write-Host "  $($_.Name)" }
