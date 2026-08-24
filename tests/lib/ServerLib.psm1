<#
    ServerLib.psm1 - drive llama-server for validation work.

    Starting a server, waiting for it to be genuinely ready, issuing completions and
    reading the timings back, sampling process memory, and shutting down without
    leaving a stale process behind. Shared by the real-use and quality-evaluation
    scripts so both measure the same way.
#>

Set-StrictMode -Version Latest

function Start-LlamaServer {
    <#
      .SYNOPSIS
        Start llama-server for one profile and block until /health reports ok.
      .OUTPUTS
        A session object to pass to Invoke-LlamaCompletion and Stop-LlamaServer.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $Exe,
        [Parameter(Mandatory)] [string] $ModelPath,
        [Parameter(Mandatory)] [string] $OutDir,
        [Parameter(Mandatory)] [string] $Tag,
        [Parameter(Mandatory)] [hashtable] $EnvVars,
        [Parameter(Mandatory)] $Common,
        [int]    $Threads = 4,
        [int]    $Port    = 8137,
        [int]    $Context,
        [int]    $StartupTimeoutSec = 300
    )

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $stdout   = Join-Path $OutDir "$Tag.server.stdout.log"
    $stderr   = Join-Path $OutDir "$Tag.server.stderr.log"
    $counters = Join-Path $OutDir "$Tag.server.counters.txt"
    Remove-Item $stdout, $stderr, $counters -Force -ErrorAction SilentlyContinue

    if (-not $Context) { $Context = [int]$Common.context }

    Clear-RuntimeEnvironment
    foreach ($k in $EnvVars.Keys) { Set-Item "Env:$k" $EnvVars[$k] }
    if ($EnvVars.ContainsKey('B1B_EXTERNAL_EXPERT_STORAGE')) { $env:LLAMA_EXPERT_PERF_OUT = $counters }

    # --parallel must match the profile. llama-server otherwise allocates several
    # slots, each with its own KV cache, and the working set no longer corresponds
    # to the profile's published figure.
    $args = @(
        '-m', $ModelPath, '-ngl', '999', '-ncmoe', "$($Common.n_cpu_moe)", '-t', "$Threads",
        '-fa', $Common.flash_attention, '-ctk', $Common.kv_cache_type_k, '-ctv', $Common.kv_cache_type_v,
        '-b', "$($Common.batch)", '-ub', "$($Common.ubatch)", '-c', "$Context",
        '--parallel', "$($Common.parallel)",
        '--load-mode', $Common.load_mode, '--jinja',
        '--host', '127.0.0.1', '--port', "$Port"
    )

    $t0 = Get-Date
    $proc = Start-Process -FilePath $Exe -ArgumentList $args `
                          -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
                          -PassThru -WindowStyle Hidden

    $base  = "http://127.0.0.1:$Port"
    $ready = $false
    while (((Get-Date) - $t0).TotalSeconds -lt $StartupTimeoutSec) {
        if ($proc.HasExited) {
            throw "llama-server exited during startup with code $($proc.ExitCode). See $stderr"
        }
        try {
            $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5 -ErrorAction Stop
            if ($h.status -eq 'ok') { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 1000
    }
    if (-not $ready) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        throw "llama-server did not become healthy within $StartupTimeoutSec s. See $stderr"
    }

    [pscustomobject]@{
        Process      = $proc
        BaseUrl      = $base
        Tag          = $Tag
        Port         = $Port
        Context      = $Context
        EnvVars      = $EnvVars
        Args         = $args
        LoadSeconds  = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        StdoutPath   = $stdout
        StderrPath   = $stderr
        CountersPath = $counters
    }
}

function Invoke-LlamaCompletion {
    <#
      .SYNOPSIS
        One deterministic generation, with the server's own timings attached.

      .PARAMETER Endpoint
        'chat' posts to /v1/chat/completions, which applies the model's chat template.
        This is how a user actually reaches the model, and it lets the model stop on
        its own end-of-turn token instead of running to the token limit.

        'completion' posts a raw prompt to /completion with no template. Useful for
        measuring pure continuation, but a raw prompt has no stop condition, so every
        request runs to `MaxTokens` and instruction-following cannot be assessed.

      .DESCRIPTION
        `cache_prompt = $false` matters: prompt caching would make a repeated prompt
        skip prefill entirely and report a meaningless time to first token.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] $Session,
        [Parameter(Mandatory)] [string] $Prompt,
        [ValidateSet('chat', 'completion')]
        [string] $Endpoint   = 'chat',
        [int]    $MaxTokens  = 256,
        [int]    $Seed       = 42,
        [double] $Temperature = 0,
        [int]    $TimeoutSec = 600,
        # This model reasons before answering unless told not to, and the reasoning is
        # charged against the same token budget. Leaving it on makes short answers
        # expensive and can consume the whole budget before an answer is written.
        [bool]   $Thinking   = $false
    )

    if ($Endpoint -eq 'chat') {
        $uri  = "$($Session.BaseUrl)/v1/chat/completions"
        $payload = @{
            messages     = @(@{ role = 'user'; content = $Prompt })
            max_tokens   = $MaxTokens
            temperature  = $Temperature
            seed         = $Seed
            cache_prompt = $false
            stream       = $false
        }
        if (-not $Thinking) {
            # Verified against this model: `chat_template_kwargs.enable_thinking = false`
            # suppresses the reasoning pass. The `/no_think` tag does not.
            $payload.chat_template_kwargs = @{ enable_thinking = $false }
        }
        $body = $payload | ConvertTo-Json -Depth 6
    } else {
        $uri  = "$($Session.BaseUrl)/completion"
        $body = @{
            prompt       = $Prompt
            n_predict    = $MaxTokens
            temperature  = $Temperature
            seed         = $Seed
            cache_prompt = $false
            stream       = $false
        } | ConvertTo-Json -Depth 5
    }

    $t0 = Get-Date
    $result = [ordered]@{
        ok = $true; error = $null; content = ''; reasoning = ''
        endpoint = $Endpoint; thinking = $Thinking; truncated_in_reasoning = $false
        prompt_tokens = $null; predicted_tokens = $null
        ttft_sec = $null; prompt_toks = $null; tg_toks = $null
        stop_reason = $null; wall_sec = $null
    }

    try {
        $resp = Invoke-RestMethod -Uri $uri -Method Post `
                    -ContentType 'application/json; charset=utf-8' `
                    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
                    -TimeoutSec $TimeoutSec -ErrorAction Stop

        if ($Endpoint -eq 'chat') {
            $choice = @($resp.choices)[0]
            $result.content     = [string]$choice.message.content
            $result.stop_reason = [string]$choice.finish_reason
            # This model thinks before answering, and llama-server returns the thinking
            # in a separate field. A response can therefore have a full reasoning trace
            # and an empty answer when max_tokens is reached mid-thought - that is a
            # truncated answer, not a failed request, and the two must not be confused.
            if ($choice.message.PSObject.Properties.Name -contains 'reasoning_content') {
                $result.reasoning = [string]$choice.message.reasoning_content
            }
            if ($resp.PSObject.Properties.Name -contains 'usage' -and $resp.usage) {
                $result.prompt_tokens    = [int]$resp.usage.prompt_tokens
                $result.predicted_tokens = [int]$resp.usage.completion_tokens
            }
        } else {
            $result.content     = [string]$resp.content
            $result.stop_reason = if ($resp.PSObject.Properties.Name -contains 'stop_type') { [string]$resp.stop_type } else { $null }
        }

        # llama-server attaches its own timings to both endpoints. Fall back to wall
        # clock rather than reporting nothing if a future build stops doing so.
        $tm = if ($resp.PSObject.Properties.Name -contains 'timings') { $resp.timings } else { $null }
        if ($tm) {
            if ($null -eq $result.prompt_tokens)    { $result.prompt_tokens    = [int]$tm.prompt_n }
            if ($null -eq $result.predicted_tokens) { $result.predicted_tokens = [int]$tm.predicted_n }
            $result.ttft_sec    = [math]::Round([double]$tm.prompt_ms / 1000, 3)
            $result.prompt_toks = [math]::Round([double]$tm.prompt_per_second, 2)
            $result.tg_toks     = [math]::Round([double]$tm.predicted_per_second, 3)
        }

        if ($result.content.Length -eq 0) {
            if ($result.reasoning.Length -gt 0) {
                # The model produced a reasoning trace but ran out of budget before
                # writing an answer. The request succeeded; the answer is missing.
                $result.truncated_in_reasoning = $true
                $result.error = 'answer truncated: token budget consumed by reasoning'
            } else {
                # HTTP 200 with nothing at all is a failure.
                $result.ok = $false
                $result.error = 'empty response'
            }
        }
    } catch {
        $result.ok = $false
        $result.error = $_.Exception.Message
    }

    $result.wall_sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
    if ($null -eq $result.tg_toks -and $result.predicted_tokens -gt 0 -and $result.wall_sec -gt 0) {
        $result.tg_toks = [math]::Round($result.predicted_tokens / $result.wall_sec, 3)
    }
    [pscustomobject]$result
}

function Get-ServerSample {
    <#
      .SYNOPSIS
        One point-in-time memory and VRAM sample for a running server.
    #>
    param(
        [Parameter(Mandatory)] $Session,
        [string] $Phase = ''
    )

    $ws = 0L; $priv = 0L
    $q = Get-Process -Id $Session.Process.Id -ErrorAction SilentlyContinue
    if ($q) { $q.Refresh(); $ws = [int64]$q.WorkingSet64; $priv = [int64]$q.PrivateMemorySize64 }

    $vram = 0.0
    $g = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($g -match '^\s*([0-9.]+)') { $vram = [double]$Matches[1] }

    $os = Get-CimInstance Win32_OperatingSystem

    [pscustomobject]@{
        utc             = (Get-Date).ToUniversalTime().ToString('o')
        phase           = $Phase
        working_set_gib = [math]::Round($ws / 1GB, 3)
        private_gib     = [math]::Round($priv / 1GB, 3)
        vram_mib        = $vram
        avail_ram_gib   = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 3)
        alive           = (-not $Session.Process.HasExited)
    }
}

function Send-CtrlBreak {
    <#
      .SYNOPSIS
        Ask a console process to shut down the way Ctrl+C would.

      .DESCRIPTION
        `Stop-Process -Force` is TerminateProcess: no atexit handler runs, so
        llama-server never writes its safety-counter dump and a server run cannot be
        checked for pin leaks. Attaching to the target's console and raising
        CTRL_BREAK lets it exit through its normal path instead.

        The attach/detach sequence is done in a **child** PowerShell. AttachConsole
        requires FreeConsole first, and doing that in-process permanently detaches the
        harness from its own console - which was observed corrupting the caller's exit
        code. Isolating it in a child process contains the damage.

        Returns $true if the process exited within the timeout.
    #>
    param([int] $ProcessId, [int] $TimeoutMs = 30000)

    $child = @'
param([int] $Target)
Add-Type -Namespace W -Name C -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool AttachConsole(uint p);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool FreeConsole();
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetConsoleCtrlHandler(IntPtr h, bool a);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool GenerateConsoleCtrlEvent(uint e, uint g);
"@
[void][W.C]::FreeConsole()
if (-not [W.C]::AttachConsole([uint32]$Target)) { exit 1 }
[void][W.C]::SetConsoleCtrlHandler([IntPtr]::Zero, $true)
$ok = [W.C]::GenerateConsoleCtrlEvent(1, 0)
[void][W.C]::FreeConsole()
if ($ok) { exit 0 } else { exit 1 }
'@

    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ctrlbreak-{0}.ps1" -f [guid]::NewGuid())
    Set-Content -Path $tmp -Value $child -Encoding UTF8
    try {
        $p = Start-Process -FilePath 'powershell.exe' `
                           -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $tmp, '-Target', "$ProcessId") `
                           -PassThru -WindowStyle Hidden
        [void]$p.WaitForExit(15000)
        $sent = ($p.HasExited -and $p.ExitCode -eq 0)
    } catch {
        $sent = $false
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }

    if (-not $sent) { return $false }

    $deadline = (Get-Date).AddMilliseconds($TimeoutMs)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 250
    }
    $false
}

function Stop-LlamaServer {
    <#
      .SYNOPSIS
        Shut the server down and report anything alarming it left behind.
      .DESCRIPTION
        Tries a clean Ctrl+Break shutdown first so the safety-counter dump is written,
        and falls back to a forced kill. A forced kill is recorded as an incident,
        because it means the run has no counter evidence.
      .OUTPUTS
        Incident strings; empty means a clean shutdown.
    #>
    param([Parameter(Mandatory)] $Session, [int] $GracefulTimeoutMs = 30000)

    $incidents = New-Object System.Collections.ArrayList

    if ($Session.Process -and -not $Session.Process.HasExited) {
        $clean = Send-CtrlBreak -ProcessId $Session.Process.Id -TimeoutMs $GracefulTimeoutMs
        if (-not $clean) {
            Stop-Process -Id $Session.Process.Id -Force -ErrorAction SilentlyContinue
            try { [void]$Session.Process.WaitForExit(30000) } catch { }
        }
    }

    # Whether the counter dump exists matters; how the process ended does not. Report
    # only the fact that decides whether this run has safety evidence.
    if ($Session.CountersPath -and $Session.EnvVars.ContainsKey('B1B_EXTERNAL_EXPERT_STORAGE')) {
        Start-Sleep -Milliseconds 500
        $dump = Get-Item $Session.CountersPath -ErrorAction SilentlyContinue
        if (-not $dump -or $dump.Length -eq 0) {
            [void]$incidents.Add('no safety-counter dump was written; this run has no pin-leak evidence')
        }
    }

    Clear-RuntimeEnvironment

    $leftover = @(Get-ActiveLlamaProcess)
    if ($leftover.Count -gt 0) {
        [void]$incidents.Add("$($leftover.Count) stale llama process(es) after shutdown")
        Stop-StaleLlama
    }

    $stderrText = if (Test-Path $Session.StderrPath) { Get-Content $Session.StderrPath -Raw } else { '' }
    foreach ($pattern in @('Access violation', 'ACCESS_VIOLATION', 'Assertion failed',
                           'GGML_ASSERT', 'terminate called', 'Exception thrown',
                           'CUDA error', 'out of memory')) {
        if ($stderrText -match [regex]::Escape($pattern)) {
            [void]$incidents.Add("server stderr contains '$pattern'")
        }
    }

    @($incidents)
}

Export-ModuleMember -Function Start-LlamaServer, Invoke-LlamaCompletion, Get-ServerSample, Stop-LlamaServer, Send-CtrlBreak
