# Clean-room reproduction

**Question:** do the published results depend on accidental state in the development
tree, or can they be reproduced from a pristine upstream checkout plus a patch?

**Answer:** they reproduce. Correctness reproduces exactly. Throughput reproduces to
within the machine's own state, which was measured and turned out to be the dominant
variable on the day.

Performed 2026-08-24. Procedure in `BUILD_FROM_CLEAN.md`; environment in
`../BUILD_ENVIRONMENT.json`.

---

## 1. Method

Nothing was reused from the development tree except the git objects for one upstream
commit and the exported patch.

```
git init                      → 3444 files, 0 dirty
git fetch --depth=1 3e7344670
core.autocrlf = false, core.eol = lf
git apply patches/0001-external-expert-cache.patch      → exit 0
copy patches/new-files/                                 → 4 files
build with the recorded configuration                   → 476/476 targets, 0 errors
```

The development tree, the `f5-final` checkpoint and the validated binaries were read
and never written.

---

## 2. Source verification

| check | result |
|---|---|
| `f5-final` checkpoint files byte-identical in the clean tree | **9 / 9** |
| Other changed files identical after line-ending normalisation | **10 / 10** |
| Patch applies cleanly | yes, `--check` exit 0 |
| Files modified after apply | 19 tracked + 4 new — matches the recorded change set |

The nine checkpoint files are the ones the F-series touched, including
`ggml-expert-indirection.cpp` and `ggml-cpu.c` where the cache and its consumption live.
They hash **identical**, not equivalent.

**Later update (2026-08-24, same day):** two of these nine —
`ggml-expert-indirection.h` and `ggml-expert-indirection.cpp` — subsequently had SPDX
licence headers added *in the clean-room tree only* (never in `f5-final` or the research
dev tree), as the last item on `PUBLIC_RELEASE_CHECKLIST.md`. That was followed
immediately by an incremental rebuild and a fresh 512-token verification confirming both
the EXACT and REPACK hashes were unaffected — see `PRODUCTIZATION_BASELINE.md` §3–4 for
the post-header hashes and `benchmark/results/g1-spdx-reverify/` for the confirming run.
The 9/9 figure above describes the state at the moment G1 first ran; it is not
retroactively false, but it no longer describes the clean-room tree's current content
for those two files.

The other ten differ from the development tree only in line endings — the development
tree carries CRLF for those files, the clean-room tree is uniformly LF. Verified by
comparing hashes of the normalised text.

`tools/completion/completion.cpp`, which `git status` lists as modified in the
development tree, has no semantic change and correctly produces no hunk. That is the one
place where the development tree carried state the patch does not — and it turned out to
be nothing.

---

## 3. Build verification

Build options diffed against the reference build. **Two differences, both benign:**

| option | reference | clean-room | effect |
|---|---|---|---|
| `LLAMA_BUILD_EXAMPLES` | ON | OFF | fewer extra targets built |
| `LLAMA_BUILD_TESTS` | ON | OFF | fewer extra targets built |

Everything else — build type, shared libs, all `GGML_CUDA_*`, `GGML_NATIVE`, the compiler
and linker flags including the four project defines — is identical.

### Binaries are not bit-identical, and that was expected

| binary | reference | clean-room | delta |
|---|---:|---:|---:|
| `ggml-cpu.dll` | 1,561,600 | 1,561,600 | 0 |
| `ggml-base.dll` | 5,763,072 | 5,763,072 | 0 |
| `ggml.dll` | 358,400 | 358,400 | 0 |
| `llama-cli.exe` | 37,376 | 37,376 | 0 |
| `llama-cli-impl.dll` | 3,178,496 | 3,178,496 | 0 |
| `llama-common.dll` | 13,656,064 | 13,656,064 | 0 |
| `llama-server.exe` | 36,352 | 36,352 | 0 |
| `llama-server-impl.dll` | 18,955,264 | 18,955,264 | 0 |
| `llama-perplexity.exe` | 36,352 | 36,352 | 0 |
| `llama-perplexity-impl.dll` | 505,856 | 505,856 | 0 |
| `ggml-cuda.dll` | 160,925,184 | 160,927,744 | +2,560 |
| `llama.dll` | 4,983,296 | 4,986,368 | +3,072 |
| `mtmd.dll` | 3,478,528 | 3,479,040 | +512 |

Ten of thirteen match to the byte. MSVC embeds `__FILE__` strings containing the absolute
source path, and the clean-room path is 16 characters longer; section alignment is
512 bytes, so only the three binaries whose accumulated strings crossed a boundary grew.
Contents differ in all thirteen because MSVC also embeds a build timestamp and a PDB GUID.

**Bit-reproducible binaries were never a goal.** Reproduction is verified functionally.

---

## 4. Correctness reproduction — PASS

Two full suites, 512 tokens, 3 runs per profile, on the clean-room binaries.

| profile | class | runs | token-trace SHA-256 | vs reference | safety |
|---|---|---:|---|---|---|
| `TINY_MEMORY_EXACT` | EXACT | 3 | `414B86C6E75D7126…` | **MATCHES** | PASS |
| `MIN_MEMORY_EXACT` | EXACT | 3 | `414B86C6E75D7126…` | **MATCHES** | PASS |
| `LOW_MEMORY_EXACT` | EXACT | 3 | `414B86C6E75D7126…` | **MATCHES** | PASS |
| `BALANCED_EXACT` | EXACT | 3 | `414B86C6E75D7126…` | **MATCHES** | PASS |
| `MAX_SPEED_EXACT` | EXACT | 3 | `414B86C6E75D7126…` | **MATCHES** | PASS |
| `MAX_SPEED_REPACK` | REPACK | 3 | `1D63259E5842DFE3…` | **MATCHES** | PASS |

Every hash was stable across all three repeats. Every safety invariant held on every run:
`pins == unpins`, `current_pins = 0`, zero read failures, zero bounds failures, zero
unexpected mmap fallbacks. No access violation, no crash, no stale process.

### Memory reproduction — exact

| profile | frontier peak WS | clean-room peak WS |
|---|---:|---:|
| `TINY_MEMORY_EXACT` | 1.442 GiB | 1.443 GiB |
| `MIN_MEMORY_EXACT` | 2.942 GiB | 2.943 GiB |
| `LOW_MEMORY_EXACT` | 3.443 GiB | 3.444 GiB |
| `BALANCED_EXACT` | 4.943 GiB | 4.943 GiB |
| `MAX_SPEED_EXACT` | 13.755 GiB | 13.870 GiB |
| `MAX_SPEED_REPACK` | 13.601 GiB | 13.601 GiB |

Within a millibyte-scale rounding difference everywhere except `MAX_SPEED_EXACT`, which
is 0.115 GiB higher — under 1 %.

---

## 5. Throughput reproduction — investigated, cause identified

The clean-room throughput came in below the frontier, and the shortfall grew with
footprint.

| profile | frontier | clean-room, grouped | delta | sd |
|---|---:|---:|---:|---:|
| `TINY_MEMORY_EXACT` | 12.661 | **13.041** | **+3.0 %** | 0.009 |
| `MIN_MEMORY_EXACT` | 18.184 | 16.862 | −7.3 % | 0.040 |
| `LOW_MEMORY_EXACT` | 20.785 | 18.017 | −13.3 % | 0.039 |
| `BALANCED_EXACT` | 24.673 | 19.170 | −22.3 % | 0.127 |
| `MAX_SPEED_EXACT` | 32.268 | 22.982 | −28.8 % | 0.043 |
| `MAX_SPEED_REPACK` | 40.974 | 30.321 | −26.0 % | 0.805 |

This was investigated rather than reported as noise. Two causes were found; only the
second matters.

### Cause 1 — a harness bug of my own making (fixed)

The first Full-mode suite interleaved profiles run by run. Alternating a 1.4 GiB profile
with a 13.8 GiB one makes each run evict the others' page-cache state, so every profile
measures its own cold-start cost. Grouping recovered `TINY_MEMORY_EXACT` from 12.145 to
13.041 — from 4 % below the frontier to 3 % above it.

It did **not** recover the larger profiles, so this was only part of the story. Full
account in `BENCHMARK_METHODOLOGY.md` §5.

### Cause 2 — machine state, proved by A/B against the original binaries

The decisive experiment: run the **original validated binaries** and the clean-room
binaries alternately, run by run, on the same profile, under the same conditions. Same
footprint in both arms, so interleaving is the correct ordering here.

**`LOW_MEMORY_EXACT`, 3 repeats each, interleaved:**

| arm | runs | mean | min | max | sd | peak WS | hash | safety |
|---|---:|---:|---:|---:|---:|---:|---|---|
| reference binaries | 3 | 17.358 | 16.456 | 17.820 | 0.781 | 3.443 | `414B86C6…` | PASS |
| clean-room binaries | 3 | **17.840** | 17.792 | 17.868 | 0.042 | 3.444 | `414B86C6…` | PASS |

Token hash identical between arms. Clean-room **+2.78 %**.

**`MAX_SPEED_EXACT`, 3 repeats each, interleaved:**

| arm | runs | mean | min | max | sd | peak WS | hash | safety |
|---|---:|---:|---:|---:|---:|---:|---|---|
| reference binaries | 3 | 22.998 | 22.986 | 23.011 | 0.013 | 13.870 | `414B86C6…` | PASS |
| clean-room binaries | 3 | **23.214** | 23.128 | 23.383 | 0.146 | 13.870 | `414B86C6…` | PASS |

Token hash identical between arms. Clean-room **+0.94 %**.

**The original validated binaries, which produced 20.785 and 32.268 when the frontier was
recorded, measure 17.358 and 22.998 today.** The clean-room build matches them or is
marginally faster, on both profiles, with identical token hashes. The build is not the
variable — the machine is.

Note also how tight the reference arm is at `MAX_SPEED_EXACT`: sd 0.013 across three
runs, 29 % below its own published figure. A stable number is not a correct one.

### What changed about the machine

The desktop was carrying roughly thirty processes holding GPU contexts during Phase G —
Chrome, several Electron applications, Discord, Steam, an animated wallpaper running a
video decoder, and two other AI desktop clients. The GPU runs ten MoE layers plus all
attention for every profile, so contention there is directly on the critical path, and
the low-memory path additionally depends on the Windows page cache holding GGUF pages.

The pattern fits precisely: the shortfall grows monotonically with how much a profile
depends on resident memory and GPU throughput, and the **smallest** profile — the one
that spends the largest share of its time waiting on SSD reads it issues itself — is
*faster* than its reference.

The desktop applications were left running. Closing another person's applications to
improve a benchmark number is not a change to make unattended, and the A/B had already
answered the question the reproduction needed to answer.

---

## 6. Verdict

| criterion | result |
|---|---|
| Clean build from a pristine tree | **PASS** |
| Source byte-identical to the checkpoint | **PASS** — 9/9 |
| EXACT token hash reproduced | **PASS** — 5 profiles × 3 runs |
| REPACK token hash reproduced | **PASS** — 3 runs |
| REPACK determinism | **PASS** — stable across repeats |
| Safety invariants | **PASS** — every run |
| Memory reproduced | **PASS** — within 1 % |
| No access violation, crash or stale process | **PASS** |
| Throughput consistent with the frontier | **PASS with a documented caveat** — equal to the original binaries measured side by side; both below the frontier by an amount attributable to machine load |

**G1 PASS.** The results do not depend on development-tree state.

The honest form of the throughput conclusion: *the clean build performs identically to
the validated build*. It is **not**: *the clean build reproduces the published
throughput figures on any machine state*. The frontier numbers stand as measurements
taken under the conditions recorded with them, and this exercise showed how much those
conditions matter — up to 29 % on the resident profiles.

---

## 7. Raw data

| result set | contents |
|---|---|
| `benchmark/results/g1-clean-reproduction/` | first suite, interleaved — retained because it documents the ordering error |
| `benchmark/results/g1-clean-reproduction-grouped/` | second suite, grouped, 6 profiles × 3 runs |
| `benchmark/results/build-ab-LOW_MEMORY_EXACT/` | reference vs clean-room binaries, interleaved |
| `benchmark/results/build-ab-MAX_SPEED_EXACT/` | reference vs clean-room binaries, interleaved |

Each contains `results.json` (including every safety counter), `runs.csv`, `SUMMARY.md`
and a `raw/` directory with stdout, stderr, counter dump and token trace per run.
