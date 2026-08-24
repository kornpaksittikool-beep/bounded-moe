# Stability report

Crashes, leaks, growth and degradation across sustained server sessions.
Generated 2026-08-24 from the same result sets as REAL_USE_VALIDATION.md.

---

## 1. What was watched

| signal | how |
|---|---|
| crashes and access violations | server stderr scanned for `Access violation`, `GGML_ASSERT`, `Assertion failed`, `terminate called`, `Exception thrown`, `CUDA error`, `out of memory` |
| pin leaks | `current_pins` at exit, and `pins == unpins` |
| cache corruption | `evicted_while_pinned`, `row_bounds_failures`, `short_read_count`, `resolver_failures` |
| memory growth | working set and private bytes sampled after every request |
| VRAM growth | `nvidia-smi` sampled after every request |
| stale processes | process table checked after shutdown |
| throughput degradation | per-round means across a sustained session |
| output corruption | every response checked for non-empty content and a sane stop reason |

## 2. Memory and VRAM across each session

| result set | rounds | requests | WS first | WS last | WS growth | private peak | VRAM first | VRAM last | VRAM growth | min avail RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `realuse-BALANCED_EXACT` | 1 | 15 | 4.941 | 6.889 | **1.948** | 13.883 | 7686 | 7680 | **-6** | 13.159 |
| `realuse-LOW_MEMORY_EXACT` | 3 | 45 | 3.441 | 5.421 | **1.980** | 12.575 | 7549 | 7721 | **172** | 13.920 |
| `realuse-LOW_MEMORY_EXACT-thinking` | 1 | 15 | 3.445 | 5.479 | **2.034** | 12.497 | 7634 | 7662 | **28** | 14.996 |
| `realuse-MAX_SPEED_EXACT` | 1 | 15 | 13.872 | 15.780 | **1.908** | 22.871 | 7711 | 7712 | **1** | 5.166 |
| `realuse-MAX_SPEED_REPACK` | 1 | 15 | 13.598 | 15.552 | **1.954** | 22.589 | 7631 | 7670 | **39** | 5.482 |

Working set is expected to rise from its post-load value as the KV cache fills;
what would be alarming is growth that continues indefinitely across repeated
rounds of the same work.

## 3. Sustained sessions — degradation across rounds

### `realuse-LOW_MEMORY_EXACT` — 3 rounds of 15 requests

| round | requests | TG mean | TTFT mean |
|---:|---:|---:|---:|
| 1 | 15 | 16.68 | 3.53 s |
| 2 | 15 | 16.60 | 3.18 s |
| 3 | 15 | 16.66 | 3.14 s |

Last round against first: **-0.1 %**.

## 4. Incidents

**None.** No crash, no access violation, no assertion, no stale process, and no
failed request across every session recorded here.

## 5. Safety counters at exit

| result set | resolver calls | cache hits | cache misses | pins | unpins | current pins | row bounds checks | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `realuse-BALANCED_EXACT` | 21698292 | 21380022 | 318270 | 21698292 | 21698292 | 0 | 172222640 | 0 |
| `realuse-LOW_MEMORY_EXACT` | 65073108 | 63468302 | 1604806 | 65073108 | 65073108 | 0 | 516490576 | 0 |
| `realuse-LOW_MEMORY_EXACT-thinking` | 41471016 | 40394121 | 1076895 | 41471016 | 41471016 | 0 | 303583104 | 0 |
| `realuse-MAX_SPEED_EXACT` | 1470564 | 1470052 | 512 | 1470564 | 1470564 | 0 | 11947088 | 0 |
| `realuse-MAX_SPEED_REPACK` | - | - | - | - | - | - | - | not available |

`MAX_SPEED_REPACK` does not use the external cache, so it has no counters to
report; that is not a missing check.

## 6. What this does not establish

- **Not a soak test.** These sessions run for minutes, not days.
- **Not a concurrency test.** One sequence at a time throughout.
- **Not a long-context test.** See `REAL_USE_VALIDATION.md` §7 for the longest
  prompt actually exercised.
- **Not a fault-injection test.** No disk error, no VRAM exhaustion, no OOM was
  deliberately induced.
