# Benchmark methodology

This document describes exactly what `run-benchmark-suite.ps1` measures and why. It
is written for someone who has never seen this project before and wants to decide
whether to believe its numbers.

---

## 1. What is being measured

**Token generation throughput (TG)** in tokens per second, during single-sequence
greedy decoding, reported by the runtime itself.

Alongside it, on every run:

| quantity | source | why it is here |
|---|---|---|
| Prompt throughput (PP) | runtime | prompt processing is a different regime from decode |
| Peak working set | `Process.WorkingSet64`, polled | the headline memory number |
| Peak private bytes | `Process.PrivateMemorySize64`, polled | reserved-but-uncommitted address space differs sharply from working set on the external path, and quoting only one of them is misleading |
| Minimum available system RAM | `Win32_OperatingSystem.FreePhysicalMemory`, polled | catches a configuration that technically fits but starves the rest of the machine |
| Peak VRAM | `nvidia-smi`, polled | the 8 GiB card is close to full in every profile |
| Token-trace SHA-256 | runtime trace file | correctness |
| Safety counters | runtime counter dump | correctness |
| Wall time, process CPU time | harness | sanity check against TG |

---

## 2. Correctness is checked on every run, not separately

Each run writes a token trace. Its SHA-256 is compared across runs and against the
reference recorded in `profiles/profiles.json`.

- **EXACT** profiles must produce
  `414B86C6E75D7126C65D51EE96F144DC64E15562E30EEE574672FE0514ADC12B`.
  A different hash is a correctness failure, not a performance observation.
- **REPACK** profiles must produce a *stable* hash across repeats. Its value differs
  from the EXACT reference by design; instability would not be.

The external expert cache also maintains counters that encode its own invariants.
The harness fails a run if any of these is non-zero:

```
resolver_failures                 direct_read_failures
short_read_count                  current_pins
row_bounds_failures               invalid_unpins
evicted_while_pinned              unexpected_mmap_fallbacks
geometry_invariant_failures       resolved_pointer_range_failures
invalid_ids                       external_buffer_failures
unexpected_mmap_expert_bindings   vec_dot_reserved_pointer_violations
row_offset_overflows              row_end_overflows
```

and additionally requires `pins == unpins`. `current_pins = 0` at exit plus balanced
pin/unpin counts is the evidence that the cache never leaked a reference and never
evicted a bundle that a worker was still reading.

A run that produces a fast number and a violated invariant is reported as **invalid**
and excluded from the aggregate. It is still written to `results.json`; nothing is
hidden.

---

## 3. Why a run can be rejected

`Test-RunValid` requires all of:

- process exit code 0
- stop reason `COMPLETED` (not timeout, not resource pressure)
- emitted token count equals requested token count
- at most one `llama-*` process alive for the whole run
- no safety-invariant violation
- a parsable throughput line

The single-process rule matters more than it looks. Two model instances alive at once
makes every memory figure meaningless, and it is the most common way a benchmark
harness quietly lies. The harness refuses to start if a model process already exists,
and re-checks after every run.

---

## 4. Warm, same-session measurement

The first run after a cold boot reads 19 GiB through the Windows page cache for the
first time and is not representative of steady state. Every reported figure is taken
**warm**: a short warm-up run precedes the measured runs, and its result is discarded.

`-NoWarmup` disables this. Do not use it for numbers you intend to publish.

---

## 5. Run ordering is a correctness property of the harness

There are two valid orderings and they answer different questions. Choosing the wrong
one produces numbers that are stable, repeatable, and wrong.

### GROUPED — the default

All repeats of a profile run consecutively:

```
warmup(A) A r1 A r2 A r3   warmup(B) B r1 B r2 B r3   ...
```

This is correct for measuring the **absolute throughput of profiles with different
memory footprints**. Each profile's own working set stays warm in the Windows page cache
across its repeats, which is the steady state a user actually experiences.

### INTERLEAVED — `-Interleave`

```
warmup(A) warmup(B)   A r1 B r1   A r2 B r2   A r3 B r3
```

The reference machine drifts by roughly **5 %** over tens of minutes — thermal
behaviour, background work, page-cache state. Several effects this project measured are
smaller than that. Running all repeats of A and then all of B attributes drift to the
configuration. During the research phase this produced real errors: one change measured
anywhere from 0 % to +4.3 % depending on run order against a true +4.6 %, and another
measured −4.5 % against a true −2.4 %.

**Use `-Interleave` when comparing two comparable configurations and the difference is
under about 5 %.** A sequential A/B cannot resolve it on this hardware.

### Do not interleave across dissimilar footprints

Alternating a 1.4 GiB profile with a 13.8 GiB one makes each run evict the other's
page-cache state, so every profile measures its own cold-start cost instead of its steady
state.

Measured directly, five profiles, three repeats, 512 tokens:

| profile | interleaved | grouped | reference |
|---|---:|---:|---:|
| `TINY_MEMORY_EXACT` | 12.17 | see results | 12.661 |
| `BALANCED_EXACT` | 19.00 | see results | 24.673 |
| `MAX_SPEED_EXACT` | 22.98 | see results | 32.268 |

The interleaved figures were highly *repeatable* — TINY read 12.169 / 12.174 across
repeats — which is what makes this failure mode dangerous. Low variance is not evidence
of a valid measurement. Correctness hashes were unaffected throughout.

The harness warns when `-Interleave` is used across profiles whose expected working sets
differ by more than 2x.

---

## 6. Fixed generation parameters

Identical across every profile and every run:

| parameter | value |
|---|---|
| context | 16384 |
| batch / ubatch | 256 / 256 |
| flash attention | on |
| KV cache type (K and V) | `q8_0` |
| GPU layers | 999 (all non-MoE layers offloaded) |
| CPU MoE layers | 30 |
| parallel sequences | 1 |
| seed | 42 |
| temperature | 0 (greedy) |
| load mode | `none` |

Only these vary between profiles: **thread count**, **whether external expert storage
is on**, **how many layers use it**, **the cache size**, and **whether repack is on**.

### A note on `--load-mode`

The research benchmarks passed `--load-mode mmap -lm none`. `-lm` is an *alias* for
`--load-mode`, so the last occurrence won and the effective mode was always `none`.
Every historical label of "production mmap" in this project's notes is therefore
wrong, though both arms of every A/B used identical flags so the comparisons remain
valid. This harness spells it `--load-mode none`, which is what actually happened.

---

## 7. Quick versus Full

| | Quick | Full |
|---|---|---|
| tokens per run | 128 | 512 |
| repeats | 1 | 3 |
| ordering | grouped | grouped (add `-Interleave` for a same-footprint A/B) |
| warm-up | yes | yes |
| intended use | "does this build work and produce the right hash" | published numbers |
| rough duration, 5 profiles | minutes | tens of minutes |

---

## 8. What the harness records about the machine

Written into `results.json` on every run so a result can never be separated from the
machine that produced it:

- CPU model, physical and logical core count, max clock
- total RAM, free RAM at capture
- GPU name, driver version, total VRAM
- **media type and bus type of the volume holding the model** — the external cache is
  an SSD-read path, so NVMe versus SATA changes the result materially
- Windows caption, version, display version, UBR
- SHA-256 of every `.exe` and `.dll` in the binary directory under test
- model path, size, modification time, and the SHA-256 of its first 16 MiB
- any other resident AI runtime it can see (`ollama`, `lmstudio`, and similar)

### Why the model is not fully hashed

Hashing 19 GiB evicts the Windows page cache, which is precisely the state the warm
measurement depends on. The head hash covers the GGUF header, metadata and tensor
index, which is what identifies the model; size and modification time cover the rest.
The GGUF is opened read-only and is never written by anything in this repository.

---

## 9. Environment hygiene

Before every run the harness clears **every** environment variable the runtime reads
— all 40 of them, not just the ones the profile sets. An inherited
`LLAMA_EXPERT_CACHE_MB` from an earlier shell would otherwise silently change a result
with no trace in the output.

---

## 10. Reproducing a published number

```powershell
.\benchmark\run-benchmark-suite.ps1 `
    -Mode Full `
    -Model  D:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf `
    -BinDir D:\path\to\bin `
    -Profiles LOW_MEMORY_EXACT
```

Expect: the EXACT hash, `safety = PASS`, and a TG mean within the profile's stated
tolerance **on hardware equivalent to the reference machine**. On different hardware,
expect the hash and the safety result to hold and the throughput not to.

---

## 11. What this methodology does not establish

- **Throughput on other hardware.** Every figure is one machine.
- **Long-context behaviour.** All benchmark runs use a short prompt at a 16384 context.
  Long-context behaviour is covered separately in `docs/STABILITY_REPORT.md`, to the
  extent it was actually tested.
- **Server-side latency.** TG here is CLI decode throughput. Time to first token under
  a real server workload is measured in the real-use validation, not here.
- **Quality.** Hash equality proves the EXACT profiles compute the reference result.
  It says nothing about REPACK, which is covered in `docs/QUALITY_REPORT.md`.
