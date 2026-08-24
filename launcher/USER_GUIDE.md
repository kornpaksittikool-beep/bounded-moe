# User guide

How to run a 35B mixture-of-experts model on a Windows machine that does not have
enough RAM to hold it.

---

## 1. What you need

| | |
|---|---|
| OS | Windows 10 or 11, x64 |
| CPU | x86-64 with AVX2 |
| GPU | NVIDIA with **8 GiB or more** of VRAM, recent driver. The non-MoE layers are offloaded to it. |
| RAM | 4 GiB free for `LOW_MEMORY_EXACT`, up to 16 GiB free for the MAX_SPEED profiles |
| Disk | The GGUF, on an **NVMe SSD**. The low-memory profiles read expert weights from it on demand, continuously. |
| Model | A Qwen3.6-35B-A3B GGUF in Q4_K_M (about 19 GiB) |

Only the reference machine in `profiles/profiles.json` has actually been measured:
Intel i5-14400F, 32 GiB, RTX 4060 8 GiB, NVMe, Windows 11. Everything else is
untested. The launcher will tell you when your machine differs.

**A SATA SSD or a hard disk has not been tested and will be materially slower on the
low-memory profiles.** The whole design trades RAM for SSD reads.

---

## 2. Quick start

```powershell
.\launcher\run-local-moe.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf
```

That runs `LOW_MEMORY_EXACT`, the recommended starting profile: about 20 tok/s in
under 3.5 GiB of working set on the reference machine, producing output bit-identical
to the unmodified runtime.

As a server:

```powershell
.\launcher\run-local-moe.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf -Profile LOW_MEMORY_EXACT -Server
```

then open `http://127.0.0.1:8080`.

If you prefer double-clickable files, `launcher\` contains one `.cmd` per profile:

```
run-low-memory.cmd -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf
```

They are three-line wrappers; all the logic is in `run-local-moe.ps1`.

---

## 3. Choosing a profile

```powershell
.\launcher\run-local-moe.ps1 -List
```

| profile | working set | reference tok/s | output |
|---|---:|---:|---|
| `TINY_MEMORY_EXACT` | 1.44 GiB | 12.7 | bit-identical |
| `MIN_MEMORY_EXACT` | 2.94 GiB | 18.2 | bit-identical |
| `LOW_MEMORY_EXACT` | 3.44 GiB | 20.8 | bit-identical |
| `BALANCED_EXACT` | 4.94 GiB | 24.7 | bit-identical |
| `MAX_SPEED_EXACT` | 13.76 GiB | 32.3 | bit-identical |
| `MAX_SPEED_REPACK` | 13.60 GiB | 41.0 | deterministic, **not** bit-identical |

Throughput figures are reference measurements on the machine named above. They are
not guarantees for your hardware.

### How to pick

- **Start with `LOW_MEMORY_EXACT`.** It is the point the project was built to reach.
- **Machine feels starved, or you need RAM for other work** → `MIN_MEMORY_EXACT`,
  then `TINY_MEMORY_EXACT`. Each step down costs throughput and saves about 0.5 to
  1.5 GiB.
- **You have RAM to spare and want more speed** → `BALANCED_EXACT` at about 5 GiB is
  the knee of the curve. Beyond it you pay a lot of memory for a little speed.
- **You have 16 GiB+ free and want the fastest possible** → `MAX_SPEED_REPACK`, if
  you accept a token stream that is not bit-identical to the reference. If you need
  bit-identical output at maximum speed, `MAX_SPEED_EXACT`.

The memory rule for the low-memory profiles is simple:

```
working set ≈ 0.94 GiB + expert cache size
```

so `LLAMA_EXPERT_CACHE_MB=2560` gives about 3.44 GiB. That relationship held constant
from a 256 MiB cache upward in testing.

---

## 4. EXACT versus REPACK

Every profile ending in `_EXACT` produces a token stream **bit-identical** to
unmodified llama.cpp on the same model. Choosing between them changes only where
expert weights live and how fast they arrive — never what the model computes. You can
switch freely without wondering whether output changed.

`MAX_SPEED_REPACK` is different. It routes the CPU MoE weights through llama.cpp's
packed Q4_K GEMM kernels, which accumulate in a different order. The result:

- **Deterministic.** The same prompt and seed give the same output every time.
- **Thread-independent.** The output does not change with thread count.
- **Not bit-identical** to the EXACT reference. Under greedy decoding, tiny
  floating-point differences eventually change a token choice, and the texts diverge
  in wording from there.

What was measured about its quality: perplexity on four corpora (400 KB of C/C++,
two English prose sets of 460 KB and 356 KB, and 22 KB of Thai). Differences ranged
from −0.09 % to +0.27 %, all far inside their standard errors, with the sign changing
between corpora.

> No measurable quality degradation was observed on the tested English prose, C/C++
> code, and Thai perplexity corpora.

That is the whole claim. It is **not** a claim that quality is identical in general —
perplexity on four corpora does not cover reasoning accuracy, instruction following,
or long-context behaviour. See `docs/QUALITY_REPORT.md`.

---

## 5. What the launcher checks before starting

It refuses to start, and tells you why, when:

- the model file or the binary is missing
- a `llama-cli` or `llama-server` process is already running (a second instance makes
  every memory figure meaningless and can exhaust RAM)
- less RAM is free than the profile's expected working set
- your GPU has less VRAM than the profile peaked at on the reference machine

It warns, but continues, when:

- headroom is thin (free RAM under 1.25× the expected working set)
- no NVIDIA GPU is detected
- your CPU and GPU are not the validated reference pair — in which case the profile's
  throughput figure does not transfer
- you overrode `-Threads` or `-Context`, which invalidates the profile's reference
  numbers

`-Force` overrides the refusals. `-DryRun` prints the fully resolved configuration and
exits without starting anything, which is the fastest way to see exactly what a
profile expands to.

---

## 6. Common options

```powershell
# specific profile, as a server on a chosen port
.\run-local-moe.ps1 -Model <gguf> -Profile BALANCED_EXACT -Server -Port 9090

# see what a profile resolves to without running it
.\run-local-moe.ps1 -Model <gguf> -Profile MAX_SPEED_REPACK -DryRun

# one-shot prompt through the CLI
.\run-local-moe.ps1 -Model <gguf> -Prompt "Explain mixture-of-experts routing"

# pass anything else straight through to llama.cpp
.\run-local-moe.ps1 -Model <gguf> -ExtraArgs '--top-k','40','--temp','0.7'

# use binaries from a different build
.\run-local-moe.ps1 -Model <gguf> -BinDir D:\some\other\bin
```

`-BindAddress` sets the server bind address; it defaults to `127.0.0.1`, meaning the
server is reachable only from this machine. Change it only if you understand the
exposure — the server has no authentication.

---

## 7. Session logs

Every launch writes a log to `launcher\sessions\` recording the profile, the exact
binary, the model path, the environment variables, the full argument list, and the
machine it ran on. `-NoLog` disables it.

---

## 8. Troubleshooting

**"a model process is already running"**
A previous session did not exit. Close it, or:
```powershell
Get-Process llama-cli,llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Much slower than the reference figure**
In order of likelihood: the model is on a SATA SSD or a hard disk rather than NVMe;
another process is competing for RAM and the page cache; your CPU differs; you are
looking at the first (cold) run rather than warm steady state.

**"profile … peaked at N MiB of VRAM but this GPU reports only M MiB"**
The profiles assume an 8 GiB card with all non-MoE layers offloaded. Smaller cards
are untested. You can reduce `-ngl` through `-ExtraArgs`, but then you are outside
every profile's measured configuration.

**Out of memory partway through a long session**
Working set grows with context. The profile figures were measured at a 16384 context
with a short prompt. Long contexts add KV cache on top of the stated working set.

**Wrong or garbled output**
EXACT profiles should be bit-identical to unmodified llama.cpp. If they are not, that
is a correctness bug worth reporting — run
`benchmark\run-benchmark-suite.ps1 -Mode Quick` and include the resulting
`results.json`.

---

## 9. What is not supported

- Non-Windows platforms. The external cache uses Win32 overlapped file I/O directly.
- CPU-only operation. Untested; all profiles offload non-MoE layers to CUDA.
- Models other than Qwen3.6-35B-A3B Q4_K_M. The mechanism is not architecture-specific
  in principle, but nothing else has been measured.
- Multiple concurrent model instances.
- Quantization formats other than Q4_K for the expert tensors on the repack path.
