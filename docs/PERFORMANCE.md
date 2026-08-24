# Performance

The validated frontier, where the time goes, what moves it, and what does not.

Read `BENCHMARK_METHODOLOGY.md` first if you intend to reproduce any of this. Run
ordering alone can produce a 29 % error that looks perfectly stable.

---

## 1. The validated Pareto frontier

512-token generation, warm, same-session, context 16384, `-b 256 -ub 256`,
`--seed 42 --temp 0`, single sequence. Same model, weights, quantization, routing and
active expert count in every row.

| TG | peak WS | peak private | min avail | VRAM | quality | `-t` | ext layers | cache | repack |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 12.661 | **1.442 GiB** | 8.465 | 21.95 | 7565 | EXACT | 4 | 30 | 512 | off |
| 14.888 | 1.942 GiB | 8.966 | 21.41 | 7565 | EXACT | 4 | 30 | 1024 | off |
| 16.049 | 2.444 GiB | 9.469 | 20.02 | 7565 | EXACT | 4 | 30 | 1536 | off |
| 18.184 | 2.942 GiB | 9.968 | 17.82 | 7565 | EXACT | 4 | 30 | 2048 | off |
| **20.785** | **3.443 GiB** | 10.471 | 17.40 | 7565 | EXACT | 4 | 30 | 2560 | off |
| 22.510 | 3.943 GiB | 10.970 | 16.82 | 7565 | EXACT | 4 | 30 | 3072 | off |
| 24.673 | 4.943 GiB | 11.973 | 15.82 | 7565 | EXACT | 4 | 30 | 4096 | off |
| 29.346 | 13.687 GiB | 20.821 | 8.16 | 7776 | EXACT | 4 | 0 | — | off |
| 32.268 | 13.755 GiB | 20.883 | 8.11 | 7707 | EXACT | 4 | 2 | 1024 | off |
| **40.974** | 13.601 GiB | 20.900 | 8.41 | 7592 | REPACK | 12 | 0 | — | on |

The 29.346 row is unmodified llama.cpp — the control, not a target.

**Every figure is a reference measurement on one machine**: Intel i5-14400F, 32 GiB,
RTX 4060 8 GiB, NVMe SSD, Windows 11 Pro 10.0.26200.

### Dominated points, removed

- caches of 5120 MiB (24.475 at 5.944 GiB) and 6144 MiB (24.498 at 6.942 GiB) are
  dominated by 4096 MiB — throughput inside noise, 1–2 GiB more memory
- everything measured at `-t 12` on the plain path is dominated by `-t 4`

---

## 2. The memory model

```
peak working set  =  0.938 GiB  +  cache size
```

Measured constant from a 256 MiB cache upward. The fixed term is resident non-expert
weights, the KV cache at a 16384 context, and runtime overhead.

This is the entire user-facing story of the low-memory curve, and it is why a 512 MiB
cache reaches 12.661 tok/s in **1.442 GiB** — faster than the point the research session
started from, on less than half the memory.

Note the gap between working set and private bytes: 1.44 GiB against 8.47 GiB at the
smallest point. The external buffer **reserves** the full expert address space without
committing it. Quoting only private bytes would make the design look like it saves
nothing; quoting only working set without saying so would be incomplete. Both are in the
table.

---

## 3. Where the time goes

`LOW_MEMORY_EXACT`, 512 tokens, 20.785 tok/s:

| item | seconds | share |
|---|---:|---:|
| generation wall | 24.63 | 100 % |
| bundle copy — `ReadFile`, 3 readers overlapped | ~5.84 | 23.7 % |
| row validation, after F5 | ~0.05 | 0.2 % |
| resolver, pin, unpin | small | — |
| compute and everything else | ~18.7 | 76 % |

Per token the run moves **405 MiB** of expert bundles: 30 layers × 8 active experts ×
1,769,472 bytes = 424,673,280. At a 2560 MiB cache that is **57.8 GB written into the
slab per 512-token run**, and the same read out of the page cache.

### The ceiling

If the copy and its side effects were free, this point would run at **26.7 tok/s**.
Measured: 20.785 — **78 % of that bound**.

The remaining cost is not removable overhead. It is the copy itself, in two forms: its
direct `ReadFile` time, and the memory bandwidth its traffic consumes competing with the
Q4_K dot products. Both are properties of *being* a cache that materialises copies.

Closing the rest means not copying, which means mapping, which means giving the residency
decision back to the OS and losing the bounded footprint. **The two goals are in direct
opposition.**

---

## 4. What moved the numbers

| change | effect | why |
|---|---:|---|
| C1a per-thread registry cache | 4,522,464 locked lookups → 3,922 | removed a global exclusive mutex plus 3 heap allocations, ~80 `strcmp`s, 3 hash lookups and a map write **per call** |
| C2a cheaper row validation | +15.7 % | 433 M calls/run stopped parsing the tensor name and stopped hitting a contended global counter |
| C7 independent file handles | +16.6 % | the Windows kernel serialises reads sharing a file object |
| E4 per-layer external cap | opens the entire curve | residency becomes a per-layer choice |
| **E6 `-t 4` instead of `-t 12`** | **2.17x** | avoids hyperthread siblings and E-cores; `ggml_barrier` waits on the slowest worker |
| E14 repack buffer type | +30 %, opt-in | `-ncmoe` had been forcing the plain CPU buffer type |
| F5 per-range validation hoist | +4.6 % | 391 M validator calls → 10 M |

C1a, C2a and C7 together were **+107 %** while changing *nothing* about behaviour:
resolver calls, pins, unpins, cache hits and misses were identical to the baseline
afterwards, and so was the token stream.

### Thread count

512 tokens, `B1B_EXTERNAL_MAX_LAYERS=1`, cache 1024 MiB, identical hash at every point:

| threads | 3 | **4** | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TG | 31.36 | **32.19** | 31.68 | 30.11 | 26.74 | 25.29 | 22.13 | 19.77 | 16.96 | 14.81 |

The i5-14400F is 6 P-cores with SMT plus 4 E-cores. Every barrier waits on the slowest
worker, and past four workers Windows starts placing them on siblings and E-cores.

Affinity masks made it **worse** in every configuration tested; `--cpu-strict 1` was
worst of all.

The repack path is the exception — packed kernels scale to 12 threads, which is why
`MAX_SPEED_REPACK` is the only profile at `-t 12`.

---

## 5. What did not move the numbers

Recorded because the negative results are the useful part.

| attempt | predicted | measured |
|---|---:|---:|
| F3 packed Q4_K kernels for the external cache | ~30 % | **+4.7 %** |
| F5 removing 381 M validator calls | ~9 % | **+4.6 %** |
| F6 removing 1.1 M resolver, pin and unpin calls | ~10 % | **−2.4 %** |
| B2e speculative async prefetch | positive | **−3.02 %** |
| C5 staggered per-worker expert order | positive | **neutral** |
| C1b lock-free unpin via pin token | positive | **−2.3 %** |
| B4 shared-gate unpin | positive | **−6.6 %** |
| E11 CPU affinity masks | positive | **worse at every mask** |
| E13 `--poll` | positive | **no effect** |
| C7b 6 readers instead of 3 | positive | **no measurable gain** |
| C2b compile out per-row diagnostics | positive | **−1.0 %** |

**Per-call overhead in the external path is already small, and call frequency is a poor
proxy for cost.** F6 removed three quarters of the resolver traffic and still lost,
because one extra `ggml_barrier` per `MUL_MAT_ID` costs more than the calls it saved —
and there are 46,080 of those per 512-token run.

### The load cannot be hidden

Three independent attempts, one structural reason: during decode **all workers need the
same expert bundle before any of them can compute with it**, and there is no independent
work left to overlap against. The load is on the critical path by construction.

What worked instead was making the unavoidable load faster.

### The cache cannot usefully be enlarged

The slab and the Windows page cache compete for the same RAM, and the slab is the worse
of the two because it also pays the copy. Past the balance point, halving the miss count
raises per-read latency about 2.5x.

---

## 6. Measured I/O scaling

`c7_read_scaling.exe`, 589,824-byte reads at expert-strided offsets, warm:

| threads | one shared handle | independent handles |
|---:|---:|---:|
| 1 | 7.10 GB/s | 6.77 GB/s |
| 2 | 6.60 | 10.29 |
| 3 | 6.63 | 12.74 |
| 4 | 6.60 | 14.50 |
| 6 | 6.21 | 16.27 |
| 8 | 6.40 | **16.70** |
| 12 | 6.23 | 16.24 |

Two facts, both load-bearing. Reads sharing a file object **do not scale at all** — this
is why the original `DuplicateHandle` design could never overlap. And independent handles
saturate near **16.5 GB/s**, about 2.4x one thread.

For reference, full expert streaming would be 405 MiB/token, needing 8.5 GB/s at
20 tok/s. Pinned-memory PCIe measured 12.74 GB/s host-to-device.

---

## 7. Throughput is sensitive to machine state

This matters more than it usually does, because the low-memory path depends on the
Windows page cache holding GGUF pages, and the GPU is shared with the desktop.

During Phase G the same profiles were re-measured on a busy desktop — around thirty
processes holding GPU contexts, including a browser, several Electron applications and an
animated wallpaper. Results, grouped ordering, 3 runs each:

| profile | frontier | Phase G re-measurement | delta |
|---|---:|---:|---:|
| `TINY_MEMORY_EXACT` | 12.661 | 13.041 | **+3.0 %** |
| `MIN_MEMORY_EXACT` | 18.184 | 16.862 | −7.3 % |
| `LOW_MEMORY_EXACT` | 20.785 | 18.017 | −13.3 % |
| `BALANCED_EXACT` | 24.673 | 19.170 | −22.3 % |
| `MAX_SPEED_EXACT` | 32.268 | 22.982 | −28.8 % |
| `MAX_SPEED_REPACK` | 40.974 | 30.321 | −26.0 % |

Standard deviations were 0.009 to 0.805 — the low figures were **highly repeatable**.

The shortfall grows monotonically with how much the profile depends on resident memory
and GPU throughput, and the smallest profile is *faster* than its reference. That is the
signature of contention for the page cache and the GPU, not of a slower build — and a
direct A/B against the original validated binaries confirmed it: see
`CLEAN_REPRODUCTION.md` §5.

**Practical consequence.** If you reproduce these numbers, close the browser and any
GPU-accelerated desktop applications first. On a loaded desktop, expect the low-memory
profiles to hold up best and the resident profiles to suffer most.

---

## 8. Targets met and not met

| tier | requirement | status |
|---|---|---|
| Tier 1 | >= 20 tok/s at <= 3.5 GiB | **MET** — 20.785 at 3.443 GiB, EXACT |
| Tier 2 | >= 22 tok/s at <= 3.5 GiB | not met — 22.510 needs 3.943 GiB |
| Tier 3 | >= 25 tok/s at <= 4 GiB | not met — 24.673 needs 4.943 GiB |
| Exceptional | >= 30 tok/s at <= 6 GiB | not met |

Working set is 0.94 GiB plus the cache, so a 3.5 GiB budget caps the cache near 2.56 GiB.
Tiers 2 and 3 are out of reach by cache sizing alone. Reaching them needs either a
smaller fixed overhead or a better hit rate at fixed capacity — and expert locality was
measured as only moderate, with tiering rejected on migration and thrash grounds over
1,198,800 routed selections.

---

## 9. Open performance questions

- **Why four workers beat six on a six P-core part.** Memory bandwidth saturating at four
  threads, or barrier spin-waiting competing with the CUDA driver. `--poll` tested the
  second and found nothing.
- **Why the plain `vec_dot` path caps near four threads while the packed path scales to
  twelve.** Resolving this would lift `MAX_SPEED_EXACT`. It is an upstream kernel
  characteristic the unmodified resident path shares.
- **In-band prefetch with perfect information.** `matrix_row_counts` already names every
  active expert for the layer at the top of `MUL_MAT_ID`, so helpers could load experts
  1..7 while the main path computes expert 0. Blocked behind the C4 precondition: pin
  release must reference the exact slot rather than resolving it by `(layer, expert)`.
