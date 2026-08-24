# Design decisions

Each entry: the decision, the alternatives, and the evidence that settled it.

---

## D1 — Bound the memory instead of letting the OS decide

**Decision.** Expert weights live in a fixed-size slab whose capacity the user chooses,
not in the page cache.

**Alternative.** `mmap` the GGUF and let Windows page. This is what unmodified llama.cpp
does and it works well — 29.3 tok/s at 13.7 GiB of working set on this machine.

**Why not.** The OS residency decision is invisible and unbounded. You cannot ask for
"run in 3.4 GiB". A user who wants to keep an IDE and a browser open has no lever.

**What it costs.** The copy. Every cache miss materialises 1.77 MB, and at
`LOW_MEMORY_EXACT` the run moves 405 MiB per token. That copy is 23.7 % of wall time,
and the memory bandwidth it consumes costs more on top. `ARCHITECTURE.md` section 7.

**The trade is explicit and unavoidable.** Not copying means mapping, which means giving
residency back to the OS. The two goals are in direct opposition, and this project chose
the bounded side.

---

## D2 — Never modify the GGUF

**Decision.** The model file is opened read-only. There is no repacked sidecar, no
index, no derived artifact, no in-place rewrite.

**Alternative considered and costed.** An offline pre-packed backing file storing expert
planes in the repack layout, removing the per-admission repack cost entirely.

**Why not.** F3 measured what it would actually buy: **+4.7 %**, not the +30 % predicted.
The price would have been 12.7 GiB of extra disk, a file format, a generator, and cache
plumbing to keep the two in sync. The arithmetic killed it before implementation.

**The general principle.** A user's model file is not the project's to mutate. Any design
that requires rewriting it must clear a much higher bar than one that does not, and this
one never came close.

---

## D3 — EXACT is the default, REPACK is opt-in

**Decision.** Five of six profiles are bit-identical to the reference. The fastest one is
not, and says so at every point of contact: profile name, `profiles.json`, launcher
banner, benchmark summary, user guide.

**Alternative.** Ship REPACK as the default because it is 27 % faster.

**Why not.** "Same model, faster" and "same model, different output" are different
products. A user who benchmarks this against unmodified llama.cpp and gets different text
needs to have been told in advance, not to discover it.

**What REPACK actually is.** Upstream llama.cpp's packed Q4_K GEMM kernels, which
`-ncmoe` had been bypassing. Same weights, same quantization, same routing, same active
expert count — a different accumulation order. Deterministic and thread-independent.
Never described as lossless.

---

## D4 — Per-layer external residency, not all-or-nothing

**Decision.** `B1B_EXTERNAL_MAX_LAYERS=N` puts layers `0..N-1` in the external cache and
leaves the rest on the ordinary loader path.

**What it required.** The indirection decision had to become **per tensor**
(`ggml_expert_external_is_tensor(src0)`) instead of a global environment flag. Without
that, unregistered layers still entered the resolver and tripped
`GGML_ASSERT(cached_ptr != NULL)`.

**Why it matters.** This one knob turns a single operating point into the entire Pareto
curve. `MAX_SPEED_EXACT` is just `N=2`. The relationship is close to linear in the number
of external layers:

```
1/TG = 0.04994 + n * 0.00105
```

n=1 predicts 19.61 (measured 19.61–19.77), n=2 predicts 19.22 (measured 19.22), n=4
predicts 18.47 (measured 18.47).

---

## D5 — Three independent file handles, not one duplicated handle

**Decision.** Open three genuinely independent handles to the GGUF, one per plane kind,
and issue a bundle's three plane reads concurrently.

**What was there before.** A single handle, duplicated. This cannot overlap: the Windows
kernel serialises reads that share a file object.

**Evidence.** A standalone probe, 589,824-byte reads at expert-strided offsets, warm:

| threads | shared handle | independent handles |
|---:|---:|---:|
| 1 | 7.10 GB/s | 6.77 GB/s |
| 4 | 6.60 | 14.50 |
| 8 | 6.40 | **16.70** |
| 12 | 6.23 | 16.24 |

Worth **+16.6 %** in the real path.

**Why three and not more.** C7b widened to six readers and measured nothing. A 1.77 MB
bundle is already past the useful part of that curve.

---

## D6 — Keep the locks; remove the per-call work

**Decision.** The cache keeps its mutex, its shared read gate and its exclusive writer
gate. What was removed is the *work done per call*: string hashing, heap allocation, a
map write, tensor-name parsing, and a contended global counter.

**Alternative, attempted twice.** Remove the unpin lock. B4 used a shared gate
(6.51 → 6.08). C1b used a `(slot, generation)` pin token so unpin became a single
`fetch_sub` — fully correct, and 2.3 % slower.

**Conclusion.** Two independent correct implementations both lost throughput.
**"The cache locks are the bottleneck" is disproven.** C1a's large win came from removing
the mutex *together with* everything else per call, not from removing a lock as such.

---

## D7 — Do not try to hide the load

**Decision.** No prefetch. Make the unavoidable load faster instead.

**Three attempts, all failed:**

- B2e speculative cross-layer prefetch: −3.02 %
- C4 unlocked concurrent loads: +3.0 % alone, a real race once loads actually overlapped
- C5 staggered per-worker expert order: exactly neutral at 512 tokens

**Why they had to fail.** During decode, all workers need the same expert bundle before
any of them can compute with it, and there is no independent work left to overlap
against. The load is on the critical path by construction.

**What worked instead.** D5 — three concurrent plane reads within a single load.

One idea remains open and is not the same as B2e: in-band prefetch with *perfect*
information, since `matrix_row_counts` already names every active expert for the layer at
the top of `MUL_MAT_ID`. It requires concurrent loading into the cache, which is exactly
what broke C4, and must not be attempted before pin release references the exact slot
rather than resolving it by `(layer, expert)`.

---

## D8 — `-t 4` as the default for every EXACT profile

**Decision.** Four threads, not the twelve the project used for months.

**Evidence.** 512 tokens, identical hash at every point:

| threads | 3 | **4** | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| TG | 31.36 | **32.19** | 30.11 | 25.29 | 19.77 | 14.81 |

**2.17x.** The i5-14400F is 6 P-cores with SMT plus 4 E-cores. `ggml_barrier` at every
graph node means the slowest worker sets the pace; hyperthread siblings and E-cores are
the stragglers.

**Not fixed by affinity.** Every `-C` mask tested made things worse, and `--cpu-strict 1`
was worst. The Windows scheduler places these threads better than llama.cpp's affinity
path.

**REPACK is the exception** — packed kernels scale to 12 threads, which is why that one
profile uses 12.

**Honest consequence.** Every absolute number recorded before this discovery was limited
by thread scheduling rather than by the expert cache. Relative comparisons survived
because both arms always used the same thread count.

---

## D9 — Non-host external buffer type

**Decision.** The external expert buffer does not advertise itself as host memory.

**What went wrong when it did.** The CUDA scheduler saw host-resident weights, decided it
could offload the operation, and copied expert weights out of reserved but uncommitted
address space. Any batch of 32 or more tokens took an access violation.

**Alternative.** `--no-op-offload`, which avoids it completely and costs nothing at
decode. Rejected as a workaround: it requires every user to know a flag, and forgetting
it crashes.

**Note.** The pre-external baseline crashed identically, so this was a latent property of
the advertised buffer type rather than something the cache introduced.

---

## D10 — Bounds-check every row in the shipping configuration

**Decision.** `row_bounds_checks` runs on every row in the configuration that ships —
about 10.35 million checks per 512-token run — and is not compiled out.

**Alternative.** C2b compiled out the remaining per-row diagnostics. It measured
**−1.0 %**: slower, not faster.

**Why it stands.** The check turns a stale or corrupt pointer into a counted failure
instead of a wild read into a 2.5 GiB slab, and it demonstrably costs nothing. C2a had
already removed the expensive part of that path — the tensor-name parse and the contended
counter.

---

## D11 — Benchmark ordering is a correctness property of the harness

**Decision.** The harness groups repeats by profile **by default** and interleaves only
when asked, with a warning when the profiles' footprints differ by more than 2x.

**Why both orderings exist.** The machine drifts about 5 % over tens of minutes, larger
than several effects worth resolving, so a small A/B between comparable configurations
must alternate run by run. Sequential A/B gave F5 anywhere from 0 % to +4.3 % against a
true +4.6 %.

**Why grouping is the default.** Interleaving profiles with very different memory
footprints makes each run evict the others' page-cache state, so every profile measures
its own cold-start cost. Measured directly during Phase G: interleaved,
`MAX_SPEED_EXACT` read 22.9 tok/s against a grouped 32.3. Correctness hashes were
unaffected.

Two orderings, two different questions. Getting this wrong produces numbers that are
stable, repeatable, and wrong.

---

## D12 — Identify the model by its head, not by a full hash

**Decision.** The harness records the model's size, modification time, and the SHA-256 of
its **first 16 MiB**.

**Why not the whole file.** Hashing 19 GiB evicts the Windows page cache, which is
exactly the state a warm measurement depends on. The measurement instrument would change
the measurement.

**Why the head is enough.** It covers the GGUF header, metadata and tensor index — what
identifies the model. Size and modification time cover the rest.

---

## D13 — Profiles carry reference measurements, not promises

**Decision.** `profiles.json` records `tg_reference_toks` with the exact machine it was
measured on, and every surface that displays it says so.

**Why.** The launcher compares the running machine against the validated reference pair
and warns when they differ. A throughput figure measured on one i5-14400F with one NVMe
SSD is a reference point. Presenting it as an expectation for unknown hardware would be
a claim the project has no evidence for.
