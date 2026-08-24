# Research history

A record of what was tried, what it measured, and what was decided. Failed experiments
are here in the same detail as successful ones, because most of what this project
learned came from the failures.

Nothing in this file has been rewritten to look better in hindsight.

---

## Summary of the arc

| stage | throughput | note |
|---|---:|---|
| starting point of the C-series | 5.46 tok/s | re-measured control |
| after C1a + C2a + C7 | 11.31 tok/s | +107 %, behaviour provably unchanged |
| after D1 + E4 + E6 + E14 | 40.97 tok/s | +227 % over the session start |
| F-series low-memory frontier | 20.79 tok/s at 3.44 GiB | Tier 1 met, EXACT |
| F-series close-out | 12.66 tok/s at 1.44 GiB | frontier extended downward |

The two largest single wins were **not in the expert cache**: the project had been
benchmarking at a near-worst thread count, and `-ncmoe` was bypassing llama.cpp's packed
Q4_K kernels entirely.

---

## Kept changes

| id | change | effect | why it worked |
|---|---|---:|---|
| **C1a** | per-thread expert-registry lookup cache, epoch-guarded | locked lookups 4,522,464 → 3,922; 512-token mean 8.35 | removed a global exclusive mutex, three heap allocations, ~80 `strcmp`s, three hash lookups and a map write **per resolver call** |
| **C2a** | cheaper per-row validation | 512-token mean 9.66 (+15.7 %) | 433 M calls/run stopped parsing the tensor name and stopped hitting a contended global counter |
| **C7** | parallel plane reads on independent file handles | 512-token mean 11.27 (+16.6 %) | the Windows kernel serialises reads sharing a file object; independent handles reach 16.5 GB/s |
| **D1** | non-host external buffer type | fixes the batch >= 32 access violation, performance-neutral | stops the CUDA scheduler copying out of reserved, uncommitted address space |
| **E4** | `B1B_EXTERNAL_MAX_LAYERS` + per-tensor indirection | opens the whole memory/throughput curve | residency becomes a per-layer choice instead of all-or-nothing |
| **E6** | `-t 4` instead of `-t 12` | **2.17x** | avoids hyperthread siblings and E-cores; `ggml_barrier` waits on the slowest worker |
| **E14** | repack buffer type for `-ncmoe` weights | +30 %, opt-in | `-ncmoe` had been forcing the plain CPU buffer type, bypassing packed Q4_K GEMM |
| **F5** | per-range validation hoist | +4.6 % interleaved | 391 M validator calls reduced to 10 M |

The most important property of C1a, C2a and C7 is that **none of them changed behaviour
at all**. Resolver calls, pins, unpins, cache hits and cache misses were identical to
the baseline afterwards, and so was the token stream. A combined +107 % came entirely
from removing per-call cost and from fixing the I/O shape.

---

## Rejected: correctness

### C4 — unlocked concurrent cache-miss loads

Reserve a slot under the locks, release the writer gate, run the three `ReadFile` calls
unlocked, republish under the gate. An in-flight key set plus a condition variable
prevented duplicate loads.

+3.0 % at 128 tokens warm, exact hashes on all four runs. But combined with C5 — which
is what finally made loads actually overlap — output became **non-deterministic**:
repeated 16-token runs gave different hashes and truncated generations.

The race is most likely slot-identity ambiguity between reservation and publication:
unpin resolves a slot by `(layer, expert)` through the direct map, and during the
unlocked window a slot carries its new layer and expert while not yet valid.

**Rolled back.** Precondition for any retry: pin release must reference the exact slot,
not resolve it by `(layer, expert)`.

### E1 — spin-before-block in the plane reader helpers

Replaced the `has_job` / `done` flags with sequence counters plus spin, and made
`notify_all` conditional on a blocked flag. Hypothesis: ~1.1 s of the 8.5 s bundle wall
is condition-variable wake latency.

**Intermittent hang.** No crash — the process simply stopped making progress. Rolled
back rather than debugged: maximum upside was about 3 % while a 21 % bottleneck was
still open. If retried, keep the unconditional `notify_all` and add only the
caller-side spin.

---

## Rejected: measured slower

### B4 / C1b — removing the unpin lock (two independent attempts)

**B4** replaced the unpin mutex and exclusive gate with a shared gate: 6.51 → 6.08.

**C1b** had the resolver return a `(slot, generation)` pin token so unpin became a
single `pin.fetch_sub`. It was fully correct — pins equalled unpins (237,024), current
pins 0, invalid unpins 0, hash matched — and still 2.3 % slower.

Two different implementations, both correct, both slower. **"The cache locks are the
bottleneck" is disproven.** C1a's large win came from removing the mutex *plus* string
hashing, heap allocation and a map write per call — not from removing a lock as such.

### F6 — shared expert resolve across workers

Worker 0 resolves and pins every active expert once before the existing barrier and
publishes pointers through a shared array; one extra barrier at the end releases the
pins.

Correct: exact hashes, `resolver_calls` fell 4x (1,507,488 → 376,872 per 512 tokens),
all safety counters clean.

**−4.5 %** on back-to-back 512-token runs (20.615 → 19.685). Removing three quarters of
the resolver calls saved less than the one extra `ggml_barrier` per `MUL_MAT_ID` cost —
there are 46,080 of those per 512-token run.

A 128-token screen had suggested +3.6 %, but that comparison spanned about forty minutes
of machine drift.

### B2e — speculative async prefetch

QD1 regressed 128-token TG by 3.02 %; QD8 regressed the 16-token screen by 7.03 %; QD2
progressed so slowly it was stopped at 62/128 tokens. Exact output and safety PASS
throughout.

Do not retry the same foreground-wait plus single-exclusive-cache-loader architecture
without a materially different publication or overlap design.

### C5 — staggered per-worker expert order

Each worker starts the expert loop at a different active expert so misses overlap.
Correct, and fast at short lengths (16-token TG 12.4 against about 7.8) — but that is
warm-up, not steady state. At 512 tokens: 9.626 against a 9.663 control. **Neutral.**

Together C4 and C5 established the important negative result: during decode the expert
loads **cannot** be hidden. All workers need the same bundle before any can compute with
it, and there is no independent work to overlap against.

### E11 — CPU affinity masks

Every mask made things worse; the `--cpu-strict 1` variants were worst of all.

| config | TG |
|---|---:|
| `-t 4`, no mask | **32.19** |
| `-t 4`, `-C 55 --cpu-strict 1` | 28.69 |
| `-t 6`, no mask | 30.11 |
| `-t 6`, `-C 555 --cpu-strict 1` | 26.54 |

The Windows scheduler places these threads better than llama.cpp's affinity path does.

### Others rejected on measurement

- **C2b** compile out remaining per-row diagnostics: −1.0 %
- **B4** relaxed atomic ordering on hot diagnostic counters: 6.26 against 6.51
- **C7b** 6-way read split instead of 3: no measurable gain
- **E13** `--poll`: no effect at any thread count
- **B3** additional full GPU MoE layers: one layer neutral after the hot-path fixes, a
  second regressed 2.7 %

---

## Rejected: refuted by arithmetic or offline analysis

### B2c — tiered expert residency

Offline analysis over **1,198,800 routed selections**. No policy simultaneously met
coverage >= 80 %, migration <= 150 MB/token, and thrash <= 10 %. Coverage can reach about
83 %, but low-migration candidates thrash and high-coverage candidates blow the migration
budget. Runtime prototype: **NO-GO**.

### F1 / F3 — packed Q4_K kernels for the external cache

The hypothesis was strong: `-ncmoe` weights gained about 30 % from packed Q4_K GEMM, so
the external cache should gain similarly.

The plumbing worked — 2-token and 16-token gates produced exact reference hashes, so the
packed kernels do consume cached planes correctly. Measured with a 12288 MiB cache,
where per-admission repack cost is largely amortised away:

| config | TG | class |
|---|---:|---|
| external repack on, `-t 4` | 24.666 | REPACK |
| external repack off, `-t 4` | 23.561 | EXACT |
| external repack on, `-t 12` | 24.059 | REPACK |
| external repack off, `-t 12` | 14.876 | EXACT |

**+4.7 %, not +30 %.** The Q4_K arithmetic is not where the external path spends its
time. At a realistic 2560 MiB cache the per-admission repack costs 14.5 s per run against
a 1 s compute saving. An offline pre-packed backing file would remove the repack cost but
still only buy ~4.7 %, at the price of 12.7 GiB of disk, a file format and a generator.

Rolled back completely; every source file was diffed against the `e14-repack` checkpoint
and the 512-token hash is `414B86C6E75D7126` again.

One useful side observation: with packed kernels the external path stopped losing
throughput at high thread counts (24.06 at `-t 12` against 14.88 without), further
evidence that the plain path's thread ceiling belongs to the `vec_dot` kernel rather
than to the cache.

---

## Three predictions, three misses

| experiment | what it removed | predicted | **measured** |
|---|---|---:|---:|
| F3 packed Q4_K kernels | Q4_K arithmetic cost | ~30 % | **+4.7 %** |
| F5 per-range validation | 381 M validator calls | ~9 % | **+4.6 %** |
| F6 shared resolve | 1.1 M resolver, pin and unpin calls | ~10 % | **−2.4 %** |

The consistent lesson: **per-call overhead in the external path is already small, and
call frequency is a poor proxy for cost.**

F5 is worth spelling out. The estimate behind it was that
`ggml_expert_validate_cached_row` cost about 1.96 s per 512 tokens, at roughly 5 ns
across 391 million calls. Removing 99.2 % of those calls moved the 128-token warm figure
from 20.688 to 21.043 — so the true aggregate cost was about 0.4 s, roughly **1 ns per
call**, an error of 5x. C2a had already stripped the expensive parts of that function.
The change was kept because it is sound and positive, but the reasoning that produced it
was wrong.

---

## Methodology errors found and corrected

### The machine drifts about 5 %

Larger than several of the effects still on the table. Sequential A/B gave F5 anywhere
from 0 % to +4.3 % against a true +4.6 %, and F6 −4.5 % against a true −2.4 %. Small
comparisons must alternate builds run by run.

### Interleaving across dissimilar footprints is also wrong

Found during Phase G. Interleaving is correct for an A/B of *comparable* configurations.
It is wrong for measuring absolute throughput across profiles with very different memory
footprints: alternating a 1.4 GiB profile with a 13.8 GiB one makes each run evict the
other's page-cache state, so every profile measures its own cold-start cost.

Directly observed: interleaved, `MAX_SPEED_EXACT` read 22.9 tok/s against a grouped
32.3, and `BALANCED_EXACT` 19.0 against 24.7. Correctness hashes were unaffected.

The benchmark harness now groups by default and warns when `-Interleave` is used across
profiles whose working sets differ by more than 2x.

### "Production mmap" was never mmap

The benchmark passed `--load-mode mmap -lm none`. `-lm` is an **alias** for
`--load-mode`, so the last occurrence won and the effective mode was always `none`.
Every historical "production mmap" label in the project notes is wrong. Both arms of
every A/B used identical flags, so the comparisons remain valid.

### The five-token prompt hid a crash

Every benchmark used a five-token prompt until real-use validation began. That exercised
none of the batching paths a server uses, and hid a blocking access violation at batch
>= 32. See `CORRECTNESS.md` section 6.

### A control that could not be reproduced

An early "production mmap control" of 29.2 tok/s was recorded. It re-measured at 13.95,
and later at 14.81 interleaved. The original should be treated as an artifact of a
differently warmed machine state. It was corrected in place rather than quietly dropped,
and the corrected value changed the interpretation of an entire series.

### A non-atomic counter lost a third of its increments

The old global row counter was non-atomic and lost about a third of its increments to a
data race. C2a's per-thread counter is exact. Any row-validation figure recorded before
C2a is an undercount.

---

## Open questions

- **Why four workers beat six on a six P-core part.** If the workload were purely compute
  bound, six should win. Two candidates remain: memory bandwidth saturating at four
  threads, or barrier spin-waiting competing with the CUDA driver while the GPU handles
  its ten MoE layers. `--poll` tested the second and found no effect.
- **Why the plain `vec_dot` path caps near 4 threads while the packed path scales to 12.**
  Resolving this would lift `MAX_SPEED_EXACT`. It is an upstream kernel characteristic
  that the unmodified resident path shares, not a property of this project's code.
- **In-band prefetch with perfect information.** At the top of each `MUL_MAT_ID`,
  `matrix_row_counts` already names every active expert for that layer, so helper threads
  could load experts 1..7 while the main path computes expert 0. This is strictly better
  informed than B2e's cross-layer speculation. It requires concurrent loading into the
  cache, which is exactly what broke C4, and must not be attempted before the C4
  precondition is met.
- **Why byte-identical input produces logits that differ in low bits.** Found in Phase G:
  EXACT configurations that read the identical expert plane bytes from the cache slab
  produce the same greedy token stream (verified directly, repeatedly) but measurably
  different perplexity — meaning different logits — depending on cache size and how many
  layers are external. Thread count was ruled out (`plain-t12` and `plain-t4` on the
  resident path agreed to five significant figures). The external-cache path itself is
  implicated, but the mechanism was not traced past that. It never changes the token
  stream at the tested settings, so it does not compromise the EXACT guarantee as stated,
  but the guarantee is now stated narrowly (token stream, not arithmetic) rather than
  broadly, precisely because of this finding. See `CORRECTNESS.md` §1.

---

## Where the line was drawn

Credible measured hypotheses for the low-memory frontier are exhausted. The path is
**copy bound**, and the copy is architectural: a cache that materialises copies pays
both the `ReadFile` time and the memory bandwidth its traffic consumes. Removing that
means not copying, which means mapping, which means giving the residency decision back
to the OS and losing the bounded footprint.

`MAX_SPEED_EXACT` is capped near 32 tok/s by an upstream `vec_dot` thread-scaling
characteristic that the unmodified resident path shares. The kernel that fixes it is
repack, which leaves the EXACT class.

Both ceilings are documented in `KNOWN_LIMITATIONS.md`.
