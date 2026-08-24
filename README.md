# SSD-backed MoE expert cache for llama.cpp

Run a 35B mixture-of-experts model on Windows in **3.4 GiB of working set**, at over
20 tokens per second, producing output **bit-identical** to unmodified llama.cpp.

---

## What this is

A modification to llama.cpp that stores mixture-of-experts weights outside the process
working set and materialises individual expert bundles on demand from the GGUF, through
a fixed-capacity cache whose size you choose.

The point is not that it is fast. The point is that **the memory is bounded and you pick
the bound**, and that picking it does not change what the model computes.

```
peak working set  ≈  0.94 GiB  +  expert cache size
```

Measured constant from a 256 MiB cache upward. Choose the cache, get the working set.

That figure is a 512-token `llama-cli` run — the benchmark configuration, and the one
directly comparable to the unmodified runtime measured the same way. A `llama-server`
session under real load settles about **1.95 GiB higher** and then stops growing. Both
numbers are in `profiles/PROFILES.md`; budget the server one if you are running a server.

## Why it exists

`Qwen3.6-35B-A3B` in Q4_K_M is a 19 GiB file. Offload the non-MoE layers to an 8 GiB
GPU and 12.66 GiB of expert weights remain on the CPU side — and they are almost
entirely idle, because only **8 of 256 experts per layer** are touched for any given
token.

Memory-mapping the file works, and unmodified llama.cpp does exactly that at 29.3 tok/s
in 13.7 GiB. But the residency decision then belongs to the OS: it is invisible,
unbounded, and competes with everything else on the machine. You cannot ask for "run in
3.4 GiB and leave the rest to my browser and my IDE."

This project makes that request expressible.

## Measured results

512-token generation, warm, context 16384, greedy, single sequence.

| profile | working set (CLI) | working set (server) | tok/s | output |
|---|---:|---:|---:|---|
| `TINY_MEMORY_EXACT` | **1.44 GiB** | ~3.4 GiB | 12.7 | bit-identical |
| `MIN_MEMORY_EXACT` | 2.94 GiB | ~4.9 GiB | 18.2 | bit-identical |
| `LOW_MEMORY_EXACT` | **3.44 GiB** | 5.42 GiB | **20.8** | bit-identical |
| `BALANCED_EXACT` | 4.94 GiB | 6.89 GiB | 24.7 | bit-identical |
| `MAX_SPEED_EXACT` | 13.76 GiB | 15.78 GiB | 32.3 | bit-identical |
| `MAX_SPEED_REPACK` | 13.60 GiB | 15.55 GiB | **41.0** | deterministic, not bit-identical |
| *unmodified llama.cpp, same model* | *13.69 GiB* | *—* | *29.3* | *reference* |

**Bit-identical** means the greedy 512-token trace hashes to
`414B86C6E75D7126C65D51EE96F144DC64E15562E30EEE574672FE0514ADC12B`, the same value the
unmodified runtime produces at the profile settings (`--temp 0`, context 16384,
`-b 256 -ub 256`). It is checked on every benchmark run, and a mismatch fails the suite.

This is a claim about the **token stream**, not about the underlying floating-point
arithmetic — those are not the same thing, and Phase G measured the difference directly.
See `docs/CORRECTNESS.md` §1 before relying on bit-identity outside greedy decoding at
these settings.

`MAX_SPEED_REPACK` is a separate configuration and is discussed below.

### The hardware these numbers come from

| | |
|---|---|
| CPU | Intel Core i5-14400F, 10 cores / 16 threads |
| RAM | 32 GiB |
| GPU | NVIDIA RTX 4060, 8 GiB, driver 591.86 |
| Storage | NVMe SSD |
| OS | Windows 11 Pro 10.0.26200 (25H2) |

**One machine. These are reference measurements, not guarantees.** On other hardware the
correctness properties still hold; the throughput will not.

Throughput is also sensitive to what else is running on the machine. Re-measured on a
busy desktop, the resident profiles came in up to 29 % lower while the smallest profile
came in 3 % *higher* — the same binaries, the same hashes, a different amount of GPU and
page-cache contention. `docs/CLEAN_REPRODUCTION.md` §5 has the numbers and the A/B that
established the cause.

## Quick start

You need Windows 10/11 x64, an AVX2 CPU, an NVIDIA GPU with 8 GiB or more, an NVMe SSD,
and a `Qwen3.6-35B-A3B` GGUF in Q4_K_M that **you obtain yourself**.

Build (about 25 minutes; see `docs/BUILD_FROM_CLEAN.md` for the full procedure):

```powershell
.\scripts\build-clean.ps1 -SourceDir <clean llama.cpp checkout> -BuildDir .\build -Fresh
```

Run:

```powershell
.\launcher\run-local-moe.ps1 -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf
```

As a server on `http://127.0.0.1:8080`:

```powershell
.\launcher\run-local-moe.ps1 -Model <gguf> -Profile LOW_MEMORY_EXACT -Server
```

See the profiles:

```powershell
.\launcher\run-local-moe.ps1 -List
```

The launcher checks that your machine can host the profile before starting, and warns
you when your hardware is not the one the numbers came from. `-DryRun` shows the fully
resolved command without running anything.

## Profiles

Start with `LOW_MEMORY_EXACT`. Move down for less memory, up for more speed.

Every profile ending in `_EXACT` produces the reference token stream exactly. Switching
between them changes only where expert weights live and how fast they arrive.

`MAX_SPEED_REPACK` is different. It routes the CPU MoE weights through llama.cpp's
packed Q4_K GEMM kernels, which accumulate in a different order, so under greedy decoding
the wording eventually diverges from the reference. It is **deterministic** and
**thread-independent**, and perplexity across four corpora — C/C++ code, two English
prose sets, and Thai — differed by between −0.09 % and +0.27 %, all far inside their
standard errors, with the sign changing between corpora.

> No measurable quality degradation was observed on the tested English prose, C/C++ code,
> and Thai perplexity corpora.

That is the whole claim. General quality equivalence is **not** claimed, and REPACK is
never described as lossless. `docs/QUALITY_REPORT.md` has the details.

Full table and exact settings: `profiles/PROFILES.md`.

## How it works, briefly

Expert weights are registered rather than allocated, and bound to a buffer that reserves
address space without committing memory. At `MUL_MAT_ID`, a resolver maps
`(layer, expert)` to a slot in a fixed-capacity slab, pinning it so it cannot be evicted
while a worker is reading it. On a miss, the expert's three planes are read
**concurrently on three independent file handles** — because the Windows kernel
serialises reads that share a file object, a design using one duplicated handle cannot
overlap at all. That single change was worth +16.6 %.

Full detail, including the cost accounting and the ceiling: `docs/ARCHITECTURE.md`.

## Limitations

- **Windows only.** The cache uses Win32 file I/O directly.
- **One model tested.** `Qwen3.6-35B-A3B` Q4_K_M and nothing else.
- **NVMe assumed.** The low-memory profiles read 405 MiB of expert bundles per token. A
  SATA SSD or hard disk is untested and will be materially slower.
- **8 GiB VRAM assumed.** Smaller cards are untested; CPU-only is untested.
- **Context 16384.** Every memory figure in the table above is at that context with a
  short prompt. A follow-on research pass validated 65536/131072/262144 (the model's
  native maximum) with RAM still bounded - see `docs/LONG_CONTEXT.md`. It is not yet a
  default profile.
- **Single sequence.** Parallel serving is untested.
- **The low-memory path is copy bound** and near its architectural ceiling — 78 % of the
  bound where the copy is free. Removing the copy means mapping, which means losing the
  bounded footprint. The two goals are in direct opposition.

The full list, including throughput ceilings, historical figures that should not be
trusted, and a diagnostic counter that reports double: `docs/KNOWN_LIMITATIONS.md`.

## Reproducing this

Everything is reproducible from a pristine upstream checkout:

```powershell
# 1. clean checkout of upstream llama.cpp at 3e7344670, then
git -C <src> apply .\patches\0001-external-expert-cache.patch
# copy patches\new-files\ over <src>

# 2. build
.\scripts\build-clean.ps1 -SourceDir <src> -BuildDir .\build -Fresh

# 3. verify
.\benchmark\run-benchmark-suite.ps1 -Mode Full -Model <gguf> -BinDir .\build\bin
```

A correct build reproduces the EXACT and REPACK hashes and passes every safety invariant.
It will **not** reproduce the binaries bit-for-bit — MSVC embeds timestamps, PDB GUIDs
and `__FILE__` paths — and reproduction is verified functionally instead.

`benchmark/BENCHMARK_METHODOLOGY.md` explains what is measured and why, including the
one thing most likely to give you wrong numbers: run ordering. Interleaving profiles with
different memory footprints produces results that are stable, repeatable, and wrong.

## Documentation

| document | contents |
|---|---|
| `docs/ARCHITECTURE.md` | the problem, the design, where the time goes, the ceiling |
| `docs/CORRECTNESS.md` | what is guaranteed, the runtime invariants, the race that was rejected |
| `docs/PERFORMANCE.md` | the frontier, the cost accounting, what moves it and what does not |
| `docs/QUALITY_REPORT.md` | EXACT vs REPACK, measured |
| `docs/RESEARCH_HISTORY.md` | every experiment, including the failures and the methodology errors |
| `docs/LONG_CONTEXT.md` | pushing context from 16384 to the model's native 262144 with RAM still bounded - researched and reproducible, not yet a default |
| `docs/DESIGN_DECISIONS.md` | each decision, its alternatives, and the evidence |
| `docs/KNOWN_LIMITATIONS.md` | everything this does not do or cannot claim |
| `docs/BUILD_FROM_CLEAN.md` | the build, its traps, and what the clean build proved |
| `docs/CLEAN_REPRODUCTION.md` | the clean-room reproduction result |
| `docs/REAL_USE_VALIDATION.md`, `docs/STABILITY_REPORT.md` | server behaviour under real workloads |
| `launcher/USER_GUIDE.md` | running it |
| `profiles/PROFILES.md` | choosing a profile |

`docs/RESEARCH_HISTORY.md` is worth reading even if you never build this. Most of what
the project learned came from experiments that failed, and three predictions that missed
by 5x, 2x, and a sign flip.

## Licence and attribution

This work modifies **llama.cpp / ggml**, MIT licensed, © 2023-2026 The ggml authors.
Base commit `3e7344670adf63ce28527a4d42f2d71eca27c41e`.

All vendored third-party components are MIT, BSD-2-Clause or public domain. None is
copyleft. Notices: `THIRD_PARTY_NOTICES.md`. Audit: `LICENSE_AUDIT.md`.

**Model weights are not covered by this licence and are not distributed here.** Obtain
the model from its original source and comply with its terms.
