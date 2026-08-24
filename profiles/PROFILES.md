# Profiles

Six supported configurations. Each one is a validated point on a measured
memory/throughput curve, not a preset somebody guessed at.

`profiles.json` is the machine-readable source of truth; the launcher and the benchmark
harness both read it. This file is the human explanation.

---

## The one rule you need

```
peak working set  ≈  0.94 GiB  +  expert cache size          (llama-cli, 512-token run)
                  ≈  0.94 GiB  +  expert cache size  +  1.95 GiB   (llama-server session)
```

Measured constant from a 256 MiB cache upward. **You choose the cache size, you get the
working set.** That relationship is the whole low-memory story.

**Two numbers, because there are two situations.** The per-profile figures below are the
peak working set of a 512-token `llama-cli` generation from a short prompt — the
benchmark configuration, and the one directly comparable to the 13.69 GiB the unmodified
runtime uses when measured the same way.

A `llama-server` session answering varied real requests at the same 16384 context settles
about **1.95 GiB higher**, because it touches the whole cache slab and fills far more KV
cache. Measured on four profiles: +1.98, +1.95, +1.91, +1.95 GiB — constant, and
independent of cache size.

It **plateaus**. In a three-round sustained session the working set rose during round 1
and then stopped: rounds 2 and 3 ended 0.01 GiB apart, and VRAM was flat to within 6 MiB.
See `docs/STABILITY_REPORT.md`.

**If you are running a server, budget the profile figure plus 2 GiB.**

---

## The table

512-token generation, warm, context 16384, greedy, single sequence.

| profile | class | WS (CLI) | WS (server) | reference tok/s | `-t` | external layers | cache MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `TINY_MEMORY_EXACT` | EXACT | 1.44 GiB | ~3.4 GiB* | 12.7 | 4 | 30 | 512 |
| `MIN_MEMORY_EXACT` | EXACT | 2.94 GiB | ~4.9 GiB* | 18.2 | 4 | 30 | 2048 |
| `LOW_MEMORY_EXACT` | EXACT | 3.44 GiB | **5.42 GiB** | 20.8 | 4 | 30 | 2560 |
| `BALANCED_EXACT` | EXACT | 4.94 GiB | **6.89 GiB** | 24.7 | 4 | 30 | 4096 |
| `MAX_SPEED_EXACT` | EXACT | 13.76 GiB | **15.78 GiB** | 32.3 | 4 | 2 | 1024 |
| `MAX_SPEED_REPACK` | REPACK | 13.60 GiB | **15.55 GiB** | 41.0 | 12 | — | — |

Bold server figures are measured. `*` is the CLI figure plus the measured +1.95 GiB
overhead — those two profiles were not run under a server.

For reference, unmodified llama.cpp on the same machine and model: **29.3 tok/s at
13.69 GiB**.

> **Every throughput figure above is a reference measurement on one machine** — Intel
> i5-14400F, 32 GiB, RTX 4060 8 GiB, NVMe SSD, Windows 11 Pro 10.0.26200. They are not
> guarantees. On different hardware the correctness properties still hold; the
> throughput will not.

---

## Choosing one

**Start with `LOW_MEMORY_EXACT`.** Over 20 tok/s in under 3.5 GiB with bit-identical
output is the point this project was built to reach.

Then adjust in one direction:

```
need more RAM back                          want more speed
◄──────────────────────────────────────────────────────────►
TINY (1.4)   MIN (2.9)   LOW (3.4)   BALANCED (4.9)   MAX_SPEED (13.8)
   12.7        18.2        20.8         24.7             32.3 / 41.0
```

- **Below `LOW`** each step saves 0.5–1.5 GiB and costs throughput roughly
  proportionally. `TINY` at 1.44 GiB still beats what this project's own research
  session started from, on less than half the memory.
- **`BALANCED` at ~5 GiB is the knee.** Past it you pay a lot of memory for a little
  speed — 5120 MiB and 6144 MiB caches were measured and are strictly dominated.
- **The MAX_SPEED profiles are a different regime.** They keep 28 of 30 MoE layers
  resident (or all of them, for REPACK), which is where the ~14 GiB goes. Use them only
  if that RAM is genuinely spare.

---

## EXACT versus REPACK

**EXACT** — five of the six. **Greedy token stream** bit-identical to unmodified
llama.cpp: `414B86C6E75D7126…`, at the profile settings (`--temp 0`, context 16384,
`-b 256 -ub 256`). Switching between EXACT profiles cannot change the text the model
produces under those settings.

This is a claim about the token stream, not about the underlying floating-point
arithmetic. Phase G measured perplexity — which reads logits directly rather than only
their argmax — across EXACT configurations that all produce the identical token trace,
and found small differences (well inside measurement noise) between the resident and
external-cache paths. See `docs/CORRECTNESS.md` §1 for the measurement and its scope.

**REPACK** — `MAX_SPEED_REPACK` only. Routes CPU MoE weights through llama.cpp's packed
Q4_K GEMM kernels, which accumulate in a different order. Verified **deterministic** and
**thread-independent** (hash `1D63259E5842DFE3…`), but **not** bit-identical to the
reference. Under greedy decoding the wording eventually diverges.

Quality evidence — perplexity, identical corpora, run back to back:

| corpus | chunks | EXACT | REPACK | delta |
|---|---:|---:|---:|---:|
| C/C++ code, 400 KB | 40 | 1.6462 | **1.6448** | −0.09 % |
| English prose, 460 KB | 40 | 4.8065 | 4.8196 | +0.27 % |
| English prose, 356 KB | 24 | 5.9473 | 5.9626 | +0.26 % |
| Thai prose, 22 KB | 7 | 5.7578 | **5.7568** | −0.02 % |

Every difference is far inside its standard error and the **sign changes between
corpora** — repack is nominally better on code and Thai, nominally worse on English.
That is what noise looks like, not degradation.

> Supported claim: **no measurable quality degradation was observed on the tested
> English prose, C/C++ code, and Thai perplexity corpora.**
>
> Not claimed: that quality is identical in general. Perplexity on four corpora does not
> cover reasoning, instruction following, or long-context behaviour.

REPACK is never described as lossless.

---

## What every profile shares

| parameter | value |
|---|---|
| GPU layers | 999 (all non-MoE layers offloaded) |
| CPU MoE layers | 30 |
| context | 16384 |
| batch / ubatch | 256 / 256 |
| flash attention | on |
| KV cache K / V | `q8_0` / `q8_0` |
| parallel sequences | 1 |
| load mode | `none` |

Only these differ: thread count, whether external expert storage is on, how many layers
use it, cache size, repack on or off.

---

## Exact settings

### `TINY_MEMORY_EXACT`
```
B1B_EXTERNAL_EXPERT_STORAGE=1   B1B_EXTERNAL_MAX_LAYERS=30   LLAMA_EXPERT_CACHE_MB=512
-t 4
```
Smallest footprint. Every miss is an SSD read, so throughput tracks NVMe latency
directly.

### `MIN_MEMORY_EXACT`
```
B1B_EXTERNAL_EXPERT_STORAGE=1   B1B_EXTERNAL_MAX_LAYERS=30   LLAMA_EXPERT_CACHE_MB=2048
-t 4
```
A good default for a 16 GiB machine.

### `LOW_MEMORY_EXACT`
```
B1B_EXTERNAL_EXPERT_STORAGE=1   B1B_EXTERNAL_MAX_LAYERS=30   LLAMA_EXPERT_CACHE_MB=2560
-t 4
```
The headline point. Recommended starting profile.

### `BALANCED_EXACT`
```
B1B_EXTERNAL_EXPERT_STORAGE=1   B1B_EXTERNAL_MAX_LAYERS=30   LLAMA_EXPERT_CACHE_MB=4096
-t 4
```
The knee of the curve.

### `MAX_SPEED_EXACT`
```
B1B_EXTERNAL_EXPERT_STORAGE=1   B1B_EXTERNAL_MAX_LAYERS=2    LLAMA_EXPERT_CACHE_MB=1024
-t 4
```
Only the first two MoE layers are external; the other 28 are resident. Capped near
32 tok/s by an upstream `vec_dot` thread-scaling characteristic that the unmodified
resident path shares — more threads make it worse, not better.

### `MAX_SPEED_REPACK`
```
B1B_MOE_REPACK=1
-t 12
```
No external cache at all, so no low-memory story: all 30 CPU MoE layers are resident.
The packed kernels are the one path that scales past 4 threads, which is why this is the
only profile at `-t 12`.

---

## Intermediate points

Any cache size works; these were measured explicitly and sit on the same line:

| cache MiB | working set | tok/s |
|---:|---:|---:|
| 1024 | 1.94 GiB | 14.9 |
| 1536 | 2.44 GiB | 16.0 |
| 3072 | 3.94 GiB | 22.5 |

```powershell
.\launcher\run-local-moe.ps1 -Model <gguf> -Profile LOW_MEMORY_EXACT `
    -ExtraArgs @() # then set LLAMA_EXPERT_CACHE_MB yourself for a custom point
```

For a genuinely custom cache size, copy a profile block in `profiles.json` and change
`LLAMA_EXPERT_CACHE_MB` and `expected_peak_working_set_gib` together. The launcher's
resource check uses the latter.

---

## Configurations that were rejected

| configuration | why |
|---|---|
| `LOW_MEMORY_REPACK` | Implemented and measured. Packed kernels are worth **+4.7 %** on the external path against +30 % on the resident path — not enough to leave the EXACT class |
| cache 5120 / 6144 MiB | Dominated by 4096 MiB: throughput inside noise, 1–2 GiB more memory |
| `-t 12` on any plain (non-repack) path | Dominated by the same configuration at `-t 4`, by up to 2.17x |
| CPU affinity masks (`-C`, `--cpu-strict`) | Every mask measured worse than none |

---

## Long-context profiles (research, opt-in, not EXACT)

Three more profiles push context from 16384 out to the model's native maximum. They are
not part of the six-profile table above - different context, different correctness
class, and not the default. See `docs/LONG_CONTEXT.md` for the full account (root cause
of the `-ncmoe 40` RAM regression that was fixed to make these possible, the performance
tuning pass, and what "not EXACT" means here precisely).

| profile | context | `-ncmoe` | cache | TG (512 tok, confirmed) | Peak WS | Peak VRAM | VRAM margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LONG_CONTEXT_LOW_RAM_64K` | 65536 | 32 | 3072 MiB | 17.283 +- 0.427 | 3.969 GiB | 7374 MiB | 814 MiB |
| `LONG_CONTEXT_LOW_RAM_128K` | 131072 | 35 | 3072 MiB | 16.414 +- 0.501 | 4.002 GiB | 6977 MiB | 1211 MiB |
| `LONG_CONTEXT_LOW_RAM_256K` | 262144 | 37 | 3072 MiB | 14.870 +- 0.171 | 4.071 GiB | 7405-7413 MiB | ~780 MiB |

Each context's `-ncmoe`/cache was found independently (not carried over from another
context) - a real per-context Pareto search, not an assumption. `256K` is the model's
`n_ctx_train`, its architectural ceiling.

**Correctness class: `LONG_CONTEXT_LOW_RAM`, not `EXACT`.** Every configuration here uses
`-ncmoe > 30`, which was found (mid-way through this research) to diverge from the EXACT
reference token stream at a fixed point (token 208 of this project's benchmark prompt),
deterministically, with coherent (not corrupted) output on both sides of the divergence.
`-ncmoe <= 30` - every profile in the table above - is unaffected. See
`docs/CORRECTNESS.md` and `docs/LONG_CONTEXT.md` section 3 for the full mechanism and
what is and is not proven about its cause.

**A note on VRAM margin.** These figures come from `nvidia-smi`, which was found (also
mid-way through this research) to under-report the *logical* buffer sizes llama.cpp's own
allocator requests by roughly 1.9 GiB at 256K context, for reasons not fully explained -
candidate causes include Windows/WDDM memory virtualization reporting only resident pages.
Directly tested with a genuine ~44,000-token prompt run to completion (not a short
benchmark): VRAM reached its steady value within 30 seconds of load and stayed flat for
the remaining ~14.5 minutes of real processing - it does not climb progressively as the
KV cache fills. Treat the VRAM margin figures above as measured and reproducible, not as
a fully-understood accounting - see `docs/LONG_CONTEXT.md` section 4b.

Run exactly like the six profiles above - no extra flags:

```powershell
.\launcher\run-local-moe.ps1 -Model <gguf> -Profile LONG_CONTEXT_LOW_RAM_256K -Server -Port 8080
```

---

## `REFERENCE_CONTROL`

`profiles.json` contains a seventh entry, hidden from `-List`. It is unmodified
llama.cpp — no external cache, no repack — used as the comparison baseline in benchmarks.
29.3 tok/s at 13.69 GiB. It is not a user profile.
