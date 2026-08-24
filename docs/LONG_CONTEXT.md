# Long context: 16384 to the model's native 262144

Everything in the rest of this repository is measured at context 16384. This document
covers a follow-on research pass that pushes the same bounded-memory design out to the
model's full native `n_ctx_train` (262144) while keeping RAM bounded, and reports what
that took, what it costs, and what is and is not validated yet.

**Status: researched and reproducible from this repository, not yet a default.** The
three profiles below are opt-in flags on top of the existing binary, not new defaults.
Building them requires the updated `ggml-expert-indirection.cpp` in `patches/new-files/`
(already current in this repo as of this document) - a build from an older checkout of
this repo's `patches/` will not have the fix described below.

**These profiles are not EXACT.** `-ncmoe` values above 30 measurably diverge from the
EXACT reference token stream past a few hundred tokens (still coherent, not corrupted -
see section 3). An earlier version of this document claimed bit-identity; that was
checked only at 128 tokens and did not hold at this project's own 512-token EXACT
standard. Corrected here.

---

## 1. Why context scales the way it does here

The model (`qwen35moe` architecture, Qwen3.6-35B-A3B) is a **hybrid**: of 40 transformer
layers, only 10 use conventional GQA attention with a KV cache; the other 30 are Gated
DeltaNet (linear/recurrent attention) layers whose state is **fixed-size, independent of
context** (62.81 MiB total, confirmed from `llama-cli -v`'s own
`llama_memory_recurrent: CUDA0 RS buffer size` line, constant from 16384 through 262144).

The KV cache itself, at this project's `-ctk q8_0 -ctv q8_0` default, costs exactly
**10.625 KiB/token** across the 10 attention layers - derived from the GGUF's own
`attention.head_count_kv=2`, `attention.key_length=256`, and confirmed to the byte against
`llama-cli -v`'s `llama_kv_cache: size = N MiB` line at four context sizes with zero
residual error. A second, smaller VRAM term - the CUDA compute/graph scratch buffer
(`sched_reserve`) - also scales linearly at 2.5 KiB/token, which is not obvious from the
`-fa on` flag alone and was found only by reading the load-time log directly.

Net: **VRAM scales with context (roughly 13.125 KiB/token plus a fixed weights term);
CPU RAM does not** - it is set entirely by the external expert cache's own budget. This
is the property that makes a bounded-RAM long-context mode possible at all: the resource
that grows with context (VRAM) and the resource this project bounds (RAM) are different
resources.

## 2. Why raising `-ncmoe` past 30 used to cost RAM instead of saving it

Passing `-ncmoe 40` (externalising all 40 MoE layers instead of 30) correctly moves the
extra 10 layers off the GPU - `-ncmoe` is upstream llama.cpp, unmodified, and does its
job. But the *external expert cache* that is supposed to bound where those layers'
weights end up (see `ARCHITECTURE.md`) had a **hardcoded 30-layer capacity** left over
from when it was built and never exercised past `B1B_EXTERNAL_MAX_LAYERS=30` - every
profile in this repository uses `-ncmoe <= 30`. Layers 30-39 were silently rejected by
the cache's registration step (`ggml_expert_storage_register`, in
`ggml-expert-indirection.cpp`) and fell back to an ordinary, fully-resident CPU buffer -
correct output, wrong memory class, and exactly why `-ncmoe 40` used to cost 4.3 GiB of
RAM instead of saving VRAM.

The fix is eight integer-literal edits (`30` -> `40`) across the registration gate, its
paired validator, and two secondary lookup helpers, all in the same file. Nothing about
the resolver, pin/unpin, cache eviction, the non-host buffer type, or the parallel-read
path needed to change - none of it was ever capped. Verified with
`EXPERT_REGISTRY_VALIDATE` (a diagnostic this project already prints at every load):
`cpu_layers=30, result=PASS` before the fix and after it at `-ncmoe 30` (byte-identical
regression), `cpu_layers=40, result=PASS` at `-ncmoe 40` after the fix.

## 3. Measured results

Hardware: the same one machine as everywhere else in this repository (i5-14400F, 32 GiB
RAM, RTX 4060 8 GiB, NVMe SSD). Flags: `-ngl 999 -ncmoe 40 -t 4 -fa on -ctk q8_0 -ctv q8_0
-b 256 -ub 256 --load-mode none`, `B1B_EXTERNAL_MAX_LAYERS=40`. Greedy, `--temp 0`,
128-token generation, single run per point (not the alternating-build protocol this
project uses below ~5 % effect sizes - the cache-size effect here is 30-40 % across the
range tested, well above that floor).

| profile | context | cache | TG tok/s | Peak WS | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| `LONG_CONTEXT_LOW_RAM_64K` | 65536 | 2048 MiB | 10.4 | 2.96 GiB | 3888 MiB |
| `LONG_CONTEXT_LOW_RAM_128K` | 131072 | 2048 MiB | 10.9 | 2.99 GiB | 4736 MiB |
| `LONG_CONTEXT_LOW_RAM_256K` | 262144 | 2048 MiB | **13.5** | **3.06 GiB** | 6316 MiB |

262144 is the model's full `n_ctx_train` - its architectural ceiling. No RoPE scaling is
in play anywhere in this table; every context tested is inside the model's native range.

A lower-RAM point exists at every context (`cache=1024 MiB`: 8.7/9.0/11.5 tok/s at
1.96/1.99/2.06 GiB WS respectively), and a faster one at 262144 (`cache=3072 MiB`:
14.9 tok/s at 4.06 GiB WS). 2048 MiB is the recommended default: it clears a 12 tok/s
throughput bar at every context tested while staying inside a 4 GiB working-set budget by
a wide margin at 64K/128K and a comfortable margin at 256K.

**Correctness - corrected.** An earlier version of this document claimed every
configuration above reproduces the identical greedy token-stream hash to the 16384
baseline. That was true at the short generation length (128 tokens) it was checked at,
and false at the 512-token length this project's own EXACT claim is defined against.
Retested at 512 tokens: **`-ncmoe` values above 30 (every profile in this document uses
`-ncmoe 40`) diverge from the EXACT reference starting at token 208** of this specific
benchmark prompt/seed, and stay diverged (a different but still coherent continuation)
for the rest of the generation. This reproduces exactly - `-ncmoe 31` diverges at the
identical token index as `-ncmoe 40` - and is deterministic across repeated runs; every
safety counter (`pins==unpins`, `current_pins=0`, `resolver_failures=0`,
`short_read_count=0`) stays clean throughout, and the generated text itself is coherent
before and after the divergence point, not garbled. The most likely explanation is the
same one this project's own `KNOWN_LIMITATIONS.md` §5 already documents for the existing
EXACT profiles at their tested settings - small, non-zero logit differences depending on
how many layers are external - now visible as an actual token flip because `-ncmoe > 30`
had never been tested at a long enough generation to hit one of the model's near-tied
greedy decisions. **This has not been proven to be that mechanism specifically** - it is
the best-supported explanation given the evidence collected (deterministic, coherent,
counter-clean, and the divergence point is `-ncmoe`-magnitude-independent), not a
confirmed root cause.

**Practical consequence: these profiles are not EXACT.** They are the same weights, same
routing, same active expert count, same resolver/cache mechanism as the EXACT profiles -
nothing about *what* is computed changes - but the token stream is not proven bit-
identical past a few hundred tokens, so they must not be labelled EXACT. See
`profiles/profiles.json`'s `LONG_CONTEXT_LOW_RAM` correctness class for the precise
wording now used.

A needle-in-haystack retrieval test (~44,000 tokens of synthetic briefing document, two
facts planted early and two-thirds through) recovered both facts verbatim with correct
section attribution - this remains valid evidence against silent truncation/corruption,
independent of the EXACT-vs-not-EXACT question above (a different, non-reference
continuation can still correctly retrieve facts from its own context). `llama-server` was
validated separately at the 256K profile: 4.08 s load, five real requests (chat, code, a
reasoning question, a ~4000-token document, a multi-turn follow-up) all completed, Peak WS
3.75 GiB, Peak VRAM 6672 MiB, clean shutdown, zero stale processes - none of that is
affected by the correctness-class correction either.

## 4. Building and running

The updated `ggml-expert-indirection.cpp`/`.h` are already current in
`patches/new-files/` - build exactly as described in `BUILD_FROM_CLEAN.md`, no extra
steps. To run:

```powershell
llama-cli.exe -m Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 999 -ncmoe 40 -t 4 -fa on `
  -ctk q8_0 -ctv q8_0 -b 256 -ub 256 -c 262144 -np 1 --jinja
```

with environment `B1B_EXTERNAL_EXPERT_STORAGE=1`, `B1B_EXTERNAL_MAX_LAYERS=40`,
`LLAMA_EXPERT_CACHE_MB=2048`. `-c 65536` or `-c 131072` for the shorter-context profiles.
None of the existing `TINY/LOW/MIN/BALANCED/MAX_SPEED_EXACT/REPACK` profiles or their
flags change - this is purely additive.

## 5. What is not yet validated

- **Retrieval depth.** The 44,000-token retrieval test is real evidence against silent
  truncation or corruption, but it is not a test of the full 262144-token window - a
  haystack that large would take on the order of an hour of prompt processing at this
  hardware's real throughput (~40-65 tok/s on realistic prompts) and was not run.
- **Thai, code, and multi-turn long-context retrieval specifically.** The server
  validation exercises code and multi-turn briefly, at short context. A long-context
  version of each (a fact planted deep in a large code file, or accumulated across many
  turns toward 64K+ tokens) has not been run.
- **Cache hit-rate / SSD-traffic accounting for the 40-layer path.** This project's
  existing loads/token and MiB/token figures (`ARCHITECTURE.md`, `PERFORMANCE.md`) are
  for the 30-layer configuration. The cache's own mechanics did not change, but a
  dedicated 40-layer accounting pass has not been re-run.
- **An unexplained pattern:** TG rises with context at a fixed cache size (e.g. 8.7 tok/s
  at 64K, 11.5 tok/s at 256K, both at `cache=1024`). The most likely explanation is
  ordinary run-to-run drift (this project's own documented ~5 % machine variance) rather
  than a causal effect - expert routing for a short, fixed generation should not depend
  on how large the allocated-but-mostly-empty KV buffer is - but it was not resolved with
  an alternating-build measurement.
- **This is a research result on top of the shipped mechanism, not a new default.** No
  profile in `profiles/profiles.json` has been added for these settings; the numbers
  above come from direct CLI/server runs, matching how every other figure in this
  repository was first established before entering the profile table.
