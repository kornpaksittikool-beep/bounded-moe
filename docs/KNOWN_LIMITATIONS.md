# Known limitations

Everything this project does not do, has not tested, or cannot claim.

---

## 1. Hardware

**One machine has been measured.** Intel i5-14400F, 32 GiB RAM, RTX 4060 8 GiB, NVMe
SSD, Windows 11 Pro 10.0.26200. Every throughput figure anywhere in this repository
comes from it.

| assumption | status |
|---|---|
| x86-64 with AVX2 | required (the repack path needs it; the plain path is untested without) |
| NVIDIA GPU with >= 8 GiB VRAM | **required and untested below 8 GiB.** All profiles offload non-MoE layers to CUDA and peak at 7.5–7.8 GiB |
| NVMe SSD holding the GGUF | **strongly assumed.** The low-memory profiles read expert bundles from it continuously — 405 MiB per token. A SATA SSD or hard disk is untested and will be materially slower |
| AMD CPU | untested |
| CPU-only operation | untested; not a supported configuration |
| Non-NVIDIA GPU (ROCm, Vulkan, Metal) | untested |
| Linux, macOS | **not supported.** The external cache uses Win32 file I/O directly |

The `-t 4` recommendation is specific to a hybrid P-core/E-core Intel part. On a
homogeneous CPU the optimum will be different and nobody has measured where.

---

## 2. Model

Only **Qwen3.6-35B-A3B in Q4_K_M** has been run.

The mechanism is not architecture-specific in principle: it needs MoE expert tensors
addressed by `(layer, kind, expert)` with a fixed stride, which is the ordinary GGUF
layout. But nothing else has been measured, and the profiles' memory figures are
derived from this model's geometry (256 experts per layer, 8 active, 144 MiB per expert
tensor).

The repack path additionally requires **Q4_K** expert tensors with `ne[1] % 8 == 0`.
Other quantizations will silently fall back to the plain path.

---

## 3. Throughput ceilings

Both are real, both are understood, and neither is going to move without a different
architecture.

### The low-memory path is copy bound

At `LOW_MEMORY_EXACT`, the bundle copy is 23.7 % of wall time, and the memory bandwidth
its traffic consumes costs more on top. If the copy and its side effects were free, the
point would run at **26.7 tok/s**; it runs at 20.785, which is 78 % of that bound.

Removing the copy means mapping the weights, which means giving the residency decision
back to the OS and losing the bounded footprint. **The two goals are in direct
opposition.** See `ARCHITECTURE.md` section 7.

Three attempts to hide the load (B2e, C4, C5) all failed, and for the same structural
reason: during decode all workers need the same bundle before any can compute with it.

### `MAX_SPEED_EXACT` is capped near 32 tok/s

By an upstream `vec_dot` thread-scaling characteristic that the **unmodified resident
path shares**. It is not a property of this project's code. The kernel that fixes it is
repack, which leaves the EXACT class.

### The cache cannot usefully be made bigger

The slab and the Windows page cache compete for the same RAM, and the slab is the worse
of the two because it also pays the copy. Past the balance point, halving the miss count
raises per-read latency about 2.5x. Caches of 5120 and 6144 MiB were measured and are
**dominated** by 4096 MiB: throughput inside noise, 1–2 GiB more memory.

---

## 4. Memory targets that were not reached

Working set is `0.938 GiB + cache size`. A 3.5 GiB budget therefore caps the cache at
about 2.56 GiB.

| target | status |
|---|---|
| >= 20 tok/s at <= 3.5 GiB | **met** — 20.785 at 3.443 GiB, EXACT |
| >= 22 tok/s at <= 3.5 GiB | **not met** — 22.510 needs 3.943 GiB |
| >= 25 tok/s at <= 4 GiB | **not met** — 24.673 needs 4.943 GiB |
| >= 30 tok/s at <= 6 GiB | **not met** |

Reaching the unmet tiers requires either a smaller fixed overhead or a better hit rate at
fixed capacity. Expert locality was measured as only moderate, and tiering was rejected
on migration and thrash grounds over 1,198,800 routed selections, so a replacement-policy
change is unlikely to find the extra hit rate required.

There is no `LOW_MEMORY_REPACK`. It was implemented and measured at **+4.7 %** on the
external path against the +30 % it gives resident weights — not enough to justify leaving
the EXACT class.

---

## 5. "EXACT" is a claim about the token stream, not the arithmetic

Every `_EXACT` profile is verified to produce a **greedy token stream bit-identical** to
unmodified llama.cpp, at the profile settings (`--temp 0`, context 16384, `-b 256 -ub
256`). That is checked directly, on every benchmark run, and it has held on every
measured run to date.

It is **not** a claim that the floating-point arithmetic is identical. Phase G measured
perplexity — which reads logits directly rather than only their argmax — across EXACT
configurations that read byte-identical input from the cache and found small, non-zero
differences depending on cache size and how many layers are external. Thread count was
ruled out as the cause. The mechanism was not traced further; see
`docs/RESEARCH_HISTORY.md`, open questions.

**Practical consequence:** the bit-identity guarantee is established for greedy decoding
at the profile settings, which is what the project measures and what most users will run.
It has not been established for non-zero temperature, other samplers, or other batch
sizes — two EXACT profiles could in principle diverge there, and nothing has measured
whether they do.

See `docs/CORRECTNESS.md` §1 for the measurements.

---

## 6. What the quality evidence does and does not cover

For `MAX_SPEED_REPACK` the supported statement is exactly:

> No measurable quality degradation was observed on the tested English prose, C/C++
> code, and Thai perplexity corpora.

**Not covered by that evidence:**

- reasoning accuracy beyond the arithmetic/logic tasks in the Phase G battery
- instruction following beyond the formatting constraints tested
- long-context behaviour
- languages other than English and Thai
- any standard task-level benchmark (MMLU, HumanEval, or similar)

Perplexity was measured across **three distinct sources** (English documentation, C/C++
code, Thai prose), not four independent corpora as earlier notes stated — two of the four
files used turned out to share a byte-identical prefix, so at the chunk counts used they
were reading the same text twice. Phase G corrected this by adding a genuinely distinct
English sample and re-running the matched comparison; see `QUALITY_REPORT.md` §3.

Phase G also ran a 27-check task battery through `llama-server` comparing EXACT and
REPACK directly: 5 of 15 tasks produced character-for-character identical output between
the two kernels, 9 were textually different but agreed on every objective check, and 1
check disagreed on a task whose answer was truncated by the token budget in both arms.
That is qualitative-plus-mechanical evidence, not a benchmark score.

**General quality equivalence is not claimed.** REPACK is never called "lossless".

See `QUALITY_REPORT.md` for the full results and what was and was not measured in
Phase G.

---

## 7. Memory figures depend on how the model is driven

The per-profile working set is measured from a **512-token `llama-cli` run with a short
prompt**. That is the benchmark configuration, and it is what makes the number comparable
to the 13.687 GiB the unmodified runtime uses measured the same way.

**A `llama-server` session uses about 1.95 GiB more.** Measured across four profiles:
+1.98, +1.95, +1.91, +1.95 GiB — constant, independent of cache size. The server touches
the whole cache slab and fills far more KV cache than a short benchmark does.

It **plateaus rather than growing without bound**: in a three-round sustained session the
working set rose during round 1 and then stopped, with rounds 2 and 3 ending 0.01 GiB
apart and VRAM flat to within 6 MiB. See `STABILITY_REPORT.md`.

Both numbers appear in `profiles/profiles.json` and `profiles/PROFILES.md`. Quoting only
the smaller one would understate what a server needs by two gigabytes.

---

## 8. Context length

Every figure in `profiles/profiles.json` and the profile table is at a **16384** context.
The model's GGUF metadata declares support up to 262144.

Memory scales with context: the KV cache is inside the 0.938 GiB fixed term at 16384 and
will grow beyond it at larger contexts. **No profile's memory figure is valid at a context
other than 16384.**

Long-context behaviour at the default `-ncmoe 30` placement is tested only to the extent
recorded in `STABILITY_REPORT.md`. Do not read the profile table as a claim about
262144-token contexts.

A separate research pass (`docs/LONG_CONTEXT.md`) found and fixed a hardcoded 30-layer
capacity in the external-expert cache that made `-ncmoe 40` regress to unbounded RAM
instead of extending the bounded design, then independently measured (not inferred) a
tuned `-ncmoe`/cache Pareto point at each of 65536, 131072, and 262144 (the model's
native ceiling): Peak Working Set 3.97-4.07 GiB, VRAM margin 780 MiB-1.2 GiB, real
retrieval correctness at up to ~245,000 tokens (94% of the 262144 window, spanning
English, Thai, and code content), and a clean 11-request `llama-server` session -
available as opt-in profiles (`LONG_CONTEXT_LOW_RAM_64K/128K/256K` in
`profiles/profiles.json`), not defaults.

**These are not EXACT.** Retested at 512 tokens (this project's own EXACT standard, not
the 128 tokens the class was first checked at), every `-ncmoe > 30` configuration
diverges from the EXACT reference token stream starting at the same token index,
deterministically, with every safety counter clean and the output coherent before and
after - see `docs/LONG_CONTEXT.md` section 3 for the full account, including what is and
is not confirmed about the cause. `-ncmoe <= 30` (every other profile in this file) is
unaffected and was regression-tested byte-for-byte.

**`nvidia-smi`'s VRAM figure under-reports what llama.cpp's own allocator logs as the
logical buffer size, by roughly 1.9 GiB at 262144 context, for a reason not fully
explained** (candidate cause: Windows/WDDM memory virtualization reporting only resident
pages). Directly tested with a genuine ~44,000-token prompt run to completion: VRAM
reached its steady reading within 30 seconds of load and stayed flat for the remaining
~14.5 minutes of real processing - it does not climb progressively as the KV cache fills,
which rules out a "lazily-committed buffer that eventually catches up" explanation. This
means `-ncmoe 30` itself does not actually exceed VRAM at 262144 context (contrary to an
earlier prediction in this research, based on the uncorrected logical-size formula) - but
it is still measurably slower and uses more RAM than the tuned `-ncmoe 37` point, so the
tuned profiles remain the recommendation regardless. See `docs/LONG_CONTEXT.md` section
4b.

Also open: Thai/code/multi-turn long-context retrieval specifically (only English/Thai/
code *content* was tested, not a dedicated long-context test of each independently), and
a per-`-ncmoe`-value cache hit-rate accounting pass.

---

## 9. Concurrency and serving

- **Single sequence.** All profiles use `-np 1`. Multiple parallel sequences are
  untested, and the expert cache's hit rate under interleaved requests from different
  contexts is unknown.
- **One model instance.** Two instances on one machine will contend for both the page
  cache and VRAM. The launcher and the benchmark harness both refuse to start when
  another is running.
- **No authentication on the server.** `llama-server` binds to `127.0.0.1` by default
  here for that reason. Changing `-BindAddress` exposes an unauthenticated endpoint.

---

## 10. Build and reproducibility

- **Binaries are not bit-reproducible.** MSVC embeds build timestamps, PDB GUIDs and
  `__FILE__` paths, so a rebuild from identical sources produces different hashes.
  Reproduction is verified **functionally**, by token-trace hash and safety counters,
  not by binary equality.
- **`GGML_NATIVE=ON`** means the compiler selects ISA extensions from the build host. A
  binary built on one machine is not guaranteed to run on an older one.
- **The web UI is fetched from the network at build time.** `LLAMA_BUILD_UI=ON` pulls
  prebuilt assets from a Hugging Face bucket. An offline build must pass
  `-DLLAMA_BUILD_UI=OFF`, which removes the browser UI; the HTTP API is unaffected.
- **The profiling defines are part of the measured configuration.**
  `B1S_PROFILE`, `B1T_PROFILE`, `B1W_PROFILE` and `B1R4_COMPILE_OUT_DIAGNOSTICS` were
  enabled in every published measurement. A build without them has not been benchmarked
  and its numbers are not comparable.

---

## 11. Diagnostic counters that are wrong

`external_logical_bytes` and `external_reserved_bytes` report exactly **twice** the
physical expert-weight size, because the loader materialises two ggml contexts holding
duplicate tensor objects and the accounting hook runs for both.

The physical figures are 90 expert tensors, 144 MiB each, 12.66 GiB total. Independently
confirmed by `full_bundle_bytes_per_token = 424,673,280 = 240 x 1,769,472`.

The **safety** counters are per-event and unaffected.

---

## 12. Historical figures that should not be trusted

Recorded here so nobody rediscovers them in old notes and treats them as data:

- **"Production mmap" labels.** The benchmark passed `--load-mode mmap -lm none`; `-lm`
  is an alias for `--load-mode`, so the effective mode was always `none`. Every such
  label is wrong. Comparisons remain valid because both arms used identical flags.
- **A 29.2 tok/s "production control".** Could not be reproduced; re-measured at 13.95
  and later 14.81 interleaved. Treat the original as an artifact of a differently warmed
  machine.
- **Any absolute number recorded before the `-t 4` discovery.** All were taken at `-t 12`
  and are limited by thread scheduling, not by the expert cache.
- **Any row-validation count recorded before C2a.** The global counter was non-atomic and
  lost about a third of its increments to a data race.

---

## 13. Not implemented

- Tiered or preloaded expert residency (rejected on measured migration and thrash)
- Speculative prefetch (three attempts, all negative)
- Lock-free unpin (two correct implementations, both slower)
- Offline pre-packed expert backing file (costed at +4.7 % for 12.7 GiB of disk)
- Any GUI beyond the CLI, the PowerShell launcher, and llama.cpp's own web UI
- Automatic profile selection from detected hardware — the launcher warns, it does not
  choose for you
