# Correctness

What is guaranteed, how it is checked, and what a failure looks like.

---

## 1. The central claim, stated precisely

> Every profile whose name ends in `_EXACT` produces a **greedy token stream
> bit-identical** to unmodified llama.cpp running the same model, **at the profile
> settings**: `--temp 0`, `--seed 42`, `-b 256 -ub 256`, context 16384.

This is not an assertion about intent. It is checked on every benchmark run, and the
check is a hash comparison that either matches or does not. It held on 24 measured runs
during Phase G alone, across five profiles and two independent builds.

### What it does not say

**It does not say the logits are bit-identical.** They are not.

Perplexity reads the model's logits directly; greedy decoding reads only their argmax.
Phase G measured perplexity across configurations that all produce the identical token
trace, and they do not agree:

| configuration | PPL, english-356k, 24 chunks, b512 | vs plain |
|---|---:|---:|
| plain `vec_dot`, resident, `-t 12` | **5.9473** | — |
| plain `vec_dot`, resident, `-t 4` | **5.9473** | **0.000 %** |
| external expert cache, 2 layers, `-t 4` | 5.9748 | +0.463 % |

The first two rows are the load-bearing ones. **Thread count changes nothing** — the
same value to five significant figures. So the third row's difference is not a threading
artefact; it is the external cache path itself producing floating-point results that
differ from the resident path in low bits.

At the profile batch size the difference persists and changes sign, and it also depends
on how many layers are external:

| configuration | PPL, english-356k, 24 chunks, **b256** |
|---|---:|
| plain `vec_dot`, resident, `-t 4` | 5.9417 |
| external cache, 2 layers (`MAX_SPEED_EXACT`) | 5.9359 |
| external cache, 30 layers (`LOW_MEMORY_EXACT`) | 5.9562 |

Every one of these sits inside the corpus's own ±0.201 standard error, so none is a
quality difference. But they are not zero, and zero is what "bit-identical arithmetic"
would predict.

### What this means in practice

- **For greedy decoding at the profile settings — the case that is measured — output is
  identical.** That is directly verified, not inferred, and it is what the EXACT class
  is for.
- **Outside those settings the guarantee has not been established.** At a non-zero
  temperature, with a different sampler, or at a different batch size, two EXACT profiles
  could in principle diverge. Nothing measured says they will; nothing measured says they
  will not.
- **The mechanism was not traced.** The bytes fed to `vec_dot` are the same bytes; why
  the results differ is an open question recorded in `RESEARCH_HISTORY.md`, not a
  resolved one.

The earlier, looser phrasing of this claim — "the token stream is bit-identical, so
nothing changes" — was true about the thing it measured and wrong about the thing it
implied. This section is the corrected version.

| class | 512-token token-trace SHA-256 |
|---|---|
| EXACT | `414B86C6E75D7126C65D51EE96F144DC64E15562E30EEE574672FE0514ADC12B` |
| REPACK | `1D63259E5842DFE3...` — deterministic, thread-independent, **not** equal to EXACT |

The reference is a 512-token greedy generation at `--seed 42 --temp 0`, context 16384,
`-b 256 -ub 256`, prompt `Explain_MoE_routing`. Any other length produces a different
trace document; the hash is only meaningful at matching parameters.

---

## 2. Why the token stream comes out identical

The cache changes where the bytes come from, not what they are. An expert plane read
from the GGUF into the cache slab is byte-for-byte the same 589,824 bytes a memory-mapped
page would have exposed, and `vec_dot` receives the same layout and the same values.

That is why the greedy token stream matches. It is **not** sufficient to conclude the
arithmetic is identical — section 1 shows it is not — but a difference confined to low
bits never reaches the argmax over 512 tokens, and the hash check confirms that
empirically on every run rather than assuming it.

The stronger evidence is behavioural. Every kept optimisation was verified to leave the
cache's *observable behaviour* unchanged, not merely to produce the same tokens. After
C1a, C2a and C7 — a combined +107 % — the resolver call count, pin count, unpin count,
cache-hit count and cache-miss count were **identical to the baseline**, along with the
token stream. The gain came entirely from removing per-call cost on paths executed
4.5 million and 433 million times per run, and from discovering that the Windows kernel
serialises reads sharing a file object.

So: the *cache logic* is provably inert with respect to what gets computed. What is not
established is that routing an expert plane through the cache produces floating-point
results identical to the last bit, and Phase G's perplexity measurements say it does
not.

---

## 3. Runtime invariants

The cache maintains counters that encode its own safety properties. They are dumped at
exit to the path in `LLAMA_EXPERT_PERF_OUT`. Every one of these must be **zero**:

| counter | what a non-zero value means |
|---|---|
| `resolver_failures` | the resolver could not produce a valid pointer |
| `direct_read_failures` | a `ReadFile` on the GGUF failed |
| `short_read_count` | a read returned fewer bytes than a full plane |
| `current_pins` | a pin was leaked; some slot is permanently unevictable |
| `invalid_unpins` | an unpin arrived for a slot that was not pinned |
| `evicted_while_pinned` | **the serious one** — a slot was reused while a worker was still reading it |
| `row_bounds_failures` | a computed row range fell outside the plane extent |
| `row_offset_overflows`, `row_end_overflows` | address arithmetic overflowed |
| `geometry_invariant_failures` | slab layout arithmetic did not hold |
| `resolved_pointer_range_failures` | a resolved pointer fell outside the slab |
| `unexpected_mmap_fallbacks` | a registered tensor was served from the mapped path |
| `unexpected_mmap_expert_bindings` | a registered tensor got bound to a mapped buffer |
| `external_buffer_failures` | external buffer creation or binding failed |
| `vec_dot_reserved_pointer_violations` | `vec_dot` received a reserved, uncommitted pointer |
| `invalid_ids` | an expert id outside `[0, 256)` reached the resolver |

And one balance condition:

```
pins == unpins        and        current_pins == 0 at exit
```

A representative clean 512-token run at `LOW_MEMORY_EXACT`:

```
resolver_calls          = 1507488
cache_hits              = 1465081
cache_misses            =   42407
pins                    = 1507488
unpins                  = 1507488
current_pins            =       0
evicted_while_pinned    =       0
row_bounds_checks       = 10350272
row_bounds_failures     =       0
short_read_count        =       0
direct_read_failures    =       0
```

Ten million row-bounds checks, zero failures. The bounds check is not a debug assertion
that was compiled out — it runs in the shipping configuration on every row.

---

## 4. How the pin protects the pointer

A resolved pointer stays valid for the whole `vec_dot` because eviction requires
`pin == 0`.

The eviction scan in `cache_resolve()` will not select a slot with a non-zero pin. While
pinned, the slot's contents, `layer`, `expert` and `valid` flag are all stable. The
resolved pointer is `slab.data() + slot * bundle + plane_offset`, and the slab is
allocated once at startup and never reallocated.

The direct slot map stores `(generation, slot + 1)`. On a hit the resolver re-checks
`generation`, `valid`, `layer` and `expert` **before** pinning, under the shared read
gate. A stale map entry is therefore detected rather than followed.

This is why two separate attempts to remove the unpin lock were rejected even though
both were *correct*: they were slower, and the existing design was never unsafe. See
`RESEARCH_HISTORY.md`, entries B4 and C1b.

---

## 5. The race that was found and rejected

**C4** made cache-miss loads run unlocked: reserve a slot under the locks
(`valid=false, loading=true, pin=1`), release the writer gate, run the three `ReadFile`
calls unlocked, then republish under the gate. An in-flight key set and a condition
variable prevented two threads loading the same expert.

On its own it measured +3.0 % with exact token hashes on all four runs.

Combined with **C5** — which is what finally made loads actually overlap — output became
**non-deterministic**: repeated 16-token runs produced different token hashes and
truncated generations.

C4 contains a real race that the in-order access pattern had been hiding. The most
likely mechanism is slot-identity ambiguity between reservation and publication: unpin
resolves a slot by `(layer, expert)` through the direct map, and during the unlocked
window a slot carries its *new* layer and expert while not yet valid.

Both were rolled back. The recorded precondition for ever retrying concurrent loads is
that **pin release must reference the exact slot rather than resolving it by
`(layer, expert)`**.

This is documented rather than quietly dropped because it is the one place where a
plausible optimisation was genuinely unsafe, and the reason is worth carrying forward.

---

## 6. The D1 defect

Batches of 32 or more tokens crashed the external path with an access violation.

The external buffer advertised itself as **host memory**. The CUDA scheduler concluded
it could offload the operation and copied expert weights out of reserved,
uncommitted address space.

- The pre-external baseline crashed identically, so this was a latent property of the
  advertised buffer type, not something the cache introduced.
- The mapped path was unaffected.
- All benchmarking to that point had used a five-token prompt, which is why it survived
  undetected.

**D1** changed the external buffer to a non-host buffer type. The scheduler no longer
attempts the copy. The fix was performance-neutral and is present in every profile.

The real lesson is about test coverage, not about buffer types. `docs/REAL_USE_VALIDATION.md`
and the batch >= 32 case in `tests/` exist because of this.

---

## 7. What the harness enforces per run

`benchmark/run-benchmark-suite.ps1` marks a run **invalid** unless all of:

- exit code 0
- stop reason `COMPLETED` — not a timeout, not resource pressure
- emitted token count equals requested token count
- **at most one `llama-*` process alive for the entire run**
- every safety counter zero and `pins == unpins`
- a parsable throughput line

and marks the *suite* failed if an EXACT profile's 512-token hash is not the reference.

The single-process rule deserves emphasis. Two model instances alive at once makes every
memory figure meaningless and is the most common way a benchmark harness quietly lies.
The harness refuses to start if a model process already exists, polls throughout, and
re-checks afterwards.

Invalid runs are written to `results.json` with their reason. Nothing is dropped.

---

## 8. REPACK: what is and is not claimed

`MAX_SPEED_REPACK` routes CPU MoE weights through llama.cpp's packed Q4_K GEMM kernels.
These accumulate in a different order, so results differ in the last bits and, under
greedy decoding, the token stream eventually diverges.

**Verified:**

- **Deterministic** — repeating the configuration reproduces `1D63259E5842DFE3` exactly.
- **Thread-independent** — `repack, external limit=2` gave `87C19214B464CCAD` at both
  `-t 4` and `-t 6`.
- **No measurable perplexity difference** across four corpora — see `QUALITY_REPORT.md`.

**Not claimed:** that quality is identical in general.

This is the same class of change as choosing a different SIMD kernel, and it is upstream
llama.cpp functionality rather than a project invention. It must nevertheless always be
reported as a **separate configuration** and never compared against hash-identical
results as if it were the same benchmark.

It is never called "lossless".

---

## 9. Reproducing the correctness check

```powershell
.\benchmark\run-benchmark-suite.ps1 `
    -Mode Full `
    -Model D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf `
    -Profiles LOW_MEMORY_EXACT
```

Look for, in `SUMMARY.md`:

- `hash` column showing `414B86C6E75D7126 = REF`
- `safety` column showing `PASS`

On different hardware, **these two must still hold**. Throughput will not.
