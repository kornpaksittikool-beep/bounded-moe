<#
    Objective checks for the quality evaluation.

    Each entry names a task from realuse-tasks.ps1 and a predicate over the model's
    output text. These are deliberately mechanical: they test whether a verifiable
    fact or a stated formatting constraint survived, not whether the prose is good.

    A check that both arms fail is still informative - it says the task was too hard
    for the model, not that one arm is worse. The report distinguishes the two.
#>

@(
    @{
        task  = 'instruction-single'
        name  = 'answers with the single word Tokyo'
        check = { param($t) $t.Trim().TrimEnd('.') -match '^\s*Tokyo\s*$' }
    }
    @{
        task  = 'instruction-single'
        name  = 'names Tokyo at all'
        check = { param($t) $t -match '(?i)\bTokyo\b' }
    }
    @{
        task  = 'instruction-format'
        name  = 'exactly three lines, each starting with "- "'
        check = {
            param($t)
            $lines = @($t -split "`r?`n" | Where-Object { $_.Trim().Length -gt 0 })
            ($lines.Count -eq 3) -and (@($lines | Where-Object { $_.Trim().StartsWith('- ') }).Count -eq 3)
        }
    }
    @{
        task  = 'instruction-format'
        name  = 'every bullet under twelve words'
        check = {
            param($t)
            $lines = @($t -split "`r?`n" | Where-Object { $_.Trim().StartsWith('- ') })
            ($lines.Count -gt 0) -and (@($lines | Where-Object { (($_ -replace '^\s*-\s*','') -split '\s+' | Where-Object { $_ }).Count -ge 12 }).Count -eq 0)
        }
    }
    @{
        task  = 'reasoning-arith'
        name  = 'hit rate 97.19 percent'
        check = { param($t) $t -match '97\.1[89]' }
    }
    @{
        task  = 'reasoning-arith'
        name  = 'total bytes read 75,037,999,104'
        check = { param($t) ($t -replace '[,\s_]','') -match '75037999104' }
    }
    @{
        task  = 'reasoning-arith'
        name  = 'bytes per token 146,558,592'
        check = { param($t) ($t -replace '[,\s_]','') -match '146558592' }
    }
    @{
        task  = 'thai-reasoning'
        name  = 'Thai arithmetic reaches 97.19 percent'
        check = { param($t) $t -match '97\.1[89]' }
    }
    @{
        task  = 'thai-reasoning'
        name  = 'Thai arithmetic reaches 75,037,999,104 bytes'
        check = { param($t) ($t -replace '[,\s_]','') -match '75037999104' }
    }
    @{
        task  = 'factual-recall'
        name  = 'SSD expanded correctly'
        check = { param($t) $t -match '(?i)solid[\s-]?state\s+drive' }
    }
    @{
        task  = 'factual-recall'
        name  = 'CUDA attributed to NVIDIA'
        check = { param($t) $t -match '(?i)\bNVIDIA\b' }
    }
    @{
        task  = 'factual-recall'
        name  = 'LRU expanded correctly'
        check = { param($t) $t -match '(?i)least[\s-]?recently[\s-]?used' }
    }
    @{
        task  = 'factual-recall'
        name  = 'mebibyte is 1048576 bytes'
        check = { param($t) ($t -replace '[,\s_]','') -match '1048576' }
    }
    @{
        task  = 'needle-retrieval'
        name  = 'retrieves the verification code from a long prompt'
        check = { param($t) $t -match '(?i)CACHE-SLOT-4417-ORANGE' }
    }
    @{
        task  = 'long-prompt-batch'
        name  = 'explains that workers all need the same bundle'
        check = { param($t) ($t -match '(?i)same\s+(expert\s+)?bundle|no\s+independent\s+work|nothing\s+to\s+overlap') }
    }
    @{
        task  = 'long-prompt-batch'
        name  = 'reports that the load itself was made faster'
        check = { param($t) ($t -match '(?i)(independent|three|concurrent).{0,40}(handle|read)|made\s+the\s+.{0,20}load.{0,20}faster|faster\s+instead') }
    }
    @{
        task  = 'code-generation'
        name  = 'defines lru_touch'
        check = { param($t) $t -match 'lru_touch\s*\(' }
    }
    @{
        task  = 'code-generation'
        name  = 'range-checks the index and returns -1'
        check = { param($t) ($t -match '(?s)return\s+-\s*1') -and ($t -match '(?i)(>=|>|<)\s*(len|length|n|count|size)') }
    }
    @{
        task  = 'code-explanation'
        name  = 'identifies the pin check as guarding against eviction or reuse'
        check = { param($t) $t -match '(?i)(pin|pinned).{0,80}(evict|in use|being used|reuse|reclaim)' }
    }
    @{
        task  = 'code-explanation'
        name  = 'mentions the generation or staleness check'
        check = { param($t) $t -match '(?i)(generation|stale|ABA)' }
    }
    @{
        task  = 'thai-chat'
        name  = 'answers in Thai script'
        check = { param($t) ([regex]::Matches($t, '[฀-๿]')).Count -gt 40 }
    }
    @{
        task  = 'thai-long'
        name  = 'answers in Thai script at length'
        check = { param($t) ([regex]::Matches($t, '[฀-๿]')).Count -gt 300 }
    }
    @{
        task  = 'thai-long'
        name  = 'covers privacy, cost and memory'
        check = {
            param($t)
            ($t -match 'ส่วนตัว|ความเป็นส่วนตัว') -and
            ($t -match 'ต้นทุน|ค่าใช้จ่าย|ราคา') -and
            ($t -match 'หน่วยความจำ|แรม|RAM')
        }
    }
    @{
        task  = 'long-form'
        name  = 'produces at least 350 words'
        check = { param($t) (($t -split '\s+' | Where-Object { $_ }).Count -ge 350) }
    }
    @{
        task  = 'long-form'
        name  = 'discusses where the trade stops paying off'
        check = { param($t) $t -match '(?i)(stops?\s+paying|diminish|no longer worth|breaks? down|limit|ceiling|point where)' }
    }
    @{
        task  = 'reasoning-logic'
        name  = 'connects the barrier or shared dependency to the failure to overlap'
        check = { param($t) $t -match '(?i)(same data|same bundle|all (the )?workers|barrier|critical path|serial|dependen)' }
    }
    @{
        task  = 'chat-followup'
        name  = 'distinguishes total from active parameters'
        check = { param($t) $t -match '(?i)(active|activated|only a (few|subset)).{0,120}(total|all|full)|(?i)(total|all|full).{0,120}(active|activated)' }
    }
)
