# Real-use validation

Whether this is a real runtime or a 512-token benchmark trick.

Every figure below comes from `llama-server` driven over HTTP by
`tests/run-real-use.ps1`, on the machine described in `profiles/profiles.json`.
Generated 2026-08-24 from the result sets listed at the end.

---

## 1. Why this exists

Every benchmark in this project used a five-token prompt for months. That prompt
exercised none of the batching paths a server uses, and it hid a blocking access
violation at any prompt batch of 32 tokens or more — the defect that D1 fixed.

So the task set below is deliberately not benchmark-shaped: fifteen tasks across
seven kinds and two languages, with prompts from 17 tokens to well over a thousand.

## 2. Task set

| task | kind | language | why it is here |
|---|---|---|---|
| `chat-short` | chat | en | the ordinary case |
| `chat-followup` | chat | en | a second turn referring back to the first |
| `reasoning-arith` | reasoning | en | multi-step arithmetic with a checkable answer |
| `reasoning-logic` | reasoning | en | connecting two stated facts into an explanation |
| `code-generation` | code | en | writing C against a specified struct and contract |
| `code-explanation` | code | en | reading concurrent C++ and explaining ordering |
| `instruction-format` | instruction | en | a hard formatting constraint, exactly three bullets |
| `instruction-single` | instruction | en | a one-word answer, the strictest constraint here |
| `factual-recall` | factual | en | four short factual questions with known answers |
| `long-form` | writing | en | sustained generation, roughly 500 words |
| `thai-chat` | chat | th | Thai, short |
| `thai-long` | writing | th | Thai, sustained generation |
| `thai-reasoning` | reasoning | th | Thai, multi-step arithmetic |
| `long-prompt-batch` | batching | en | a doubled technical passage — forces prompt batching |
| `needle-retrieval` | batching | en | a code buried in a long document — batching plus retrieval |

## 3. Results by profile

| result set | profile | reasoning | requests | load | TG min-max | TG mean | TTFT min-max | max prompt | peak WS | peak VRAM | incidents |
|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `realuse-BALANCED_EXACT` | `BALANCED_EXACT` | off | 15/15 | 7.1 s | 12.47-21.10 | 18.50 | 0.84-16.80 s | 1234 tok | 6.889 GiB | 7686 MiB | 0 |
| `realuse-LOW_MEMORY_EXACT` | `LOW_MEMORY_EXACT` | off | 45/45 | 4.6 s | 10.72-18.89 | 16.65 | 0.92-19.68 s | 1234 tok | 5.557 GiB | 7725 MiB | 0 |
| `realuse-LOW_MEMORY_EXACT-thinking` | `LOW_MEMORY_EXACT` | on | 15/15 | 5.6 s | 15.87-17.88 | 16.96 | 0.90-16.62 s | 1232 tok | 5.479 GiB | 7662 MiB | 0 |
| `realuse-MAX_SPEED_EXACT` | `MAX_SPEED_EXACT` | off | 15/15 | 17.7 s | 13.82-23.69 | 22.50 | 0.47-6.40 s | 1234 tok | 15.780 GiB | 7723 MiB | 0 |
| `realuse-MAX_SPEED_REPACK` | `MAX_SPEED_REPACK` | off | 15/15 | 18.8 s | 14.49-32.23 | 29.67 | 0.39-7.07 s | 1234 tok | 15.552 GiB | 7670 MiB | 0 |

`TG` is decode throughput as reported by the server. `TTFT` is prompt-eval time,
which is dominated by prompt length: the short prompts prefill in under a second,
the long ones take proportionally longer.

**Reasoning**: this model thinks before answering unless told not to, and the
thinking is charged against the same token budget as the answer. Most runs here
have it off (`chat_template_kwargs.enable_thinking = false`) so that a short
answer costs a short generation. One run has it on, to confirm the reasoning path
works and to show what it costs. Note that the `/no_think` prompt tag does **not**
work on this model; only the template argument does.

## 4. Prompt batching — the case that used to crash

| result set | task | prompt tokens | TTFT | outcome |
|---|---|---:|---:|---|
| `realuse-BALANCED_EXACT` | `long-prompt-batch` | 819 | 11.49 s | completed |
| `realuse-BALANCED_EXACT` | `needle-retrieval` | 1234 | 16.80 s | completed |
| `realuse-LOW_MEMORY_EXACT` | `long-prompt-batch` | 819 | 13.31 s | completed |
| `realuse-LOW_MEMORY_EXACT` | `needle-retrieval` | 1234 | 19.68 s | completed |
| `realuse-LOW_MEMORY_EXACT-thinking` | `long-prompt-batch` | 817 | 11.45 s | completed |
| `realuse-LOW_MEMORY_EXACT-thinking` | `needle-retrieval` | 1232 | 16.62 s | completed |
| `realuse-MAX_SPEED_EXACT` | `long-prompt-batch` | 819 | 4.38 s | completed |
| `realuse-MAX_SPEED_EXACT` | `needle-retrieval` | 1234 | 6.40 s | completed |
| `realuse-MAX_SPEED_REPACK` | `long-prompt-batch` | 819 | 4.82 s | completed |
| `realuse-MAX_SPEED_REPACK` | `needle-retrieval` | 1234 | 7.07 s | completed |

Every one of these is far above the 32-token batch that crashed before D1. No
access violation, no assertion, no crash occurred in any run.

## 5. Safety invariants under server load

| result set | safety | note |
|---|---|---|
| `realuse-BALANCED_EXACT` | **PASS** | pins == unpins, current_pins = 0, zero failures |
| `realuse-LOW_MEMORY_EXACT` | **PASS** | pins == unpins, current_pins = 0, zero failures |
| `realuse-LOW_MEMORY_EXACT-thinking` | **PASS** | pins == unpins, current_pins = 0, zero failures |
| `realuse-MAX_SPEED_EXACT` | **PASS** | pins == unpins, current_pins = 0, zero failures |
| `realuse-MAX_SPEED_REPACK` | not available | no counter file (non-external run) |

## 6. What was observed in the output

Transcripts for every request are in `transcripts.txt` in each result set,
including the model's reasoning trace where it produced one. They were read, not
scored — this section is a qualitative check, and `docs/QUALITY_REPORT.md` holds
the measured comparison.

## 7. Limits of this validation

- Single sequence throughout (`--parallel 1`, matching the profiles). Concurrent
  requests from multiple clients are **untested**.
- Longest prompt tested is listed above. That is not a long-context test, and no
  claim is made about behaviour near the model's 262144-token maximum.
- One machine, one model, one quantization.
- Task outputs were inspected for coherence, not benchmarked for accuracy.

## 8. Result sets

- `tests/results/realuse-BALANCED_EXACT/` — ealuse.json, equests.csv, samples.csv, 	ranscripts.txt, raw server logs
- `tests/results/realuse-LOW_MEMORY_EXACT/` — ealuse.json, equests.csv, samples.csv, 	ranscripts.txt, raw server logs
- `tests/results/realuse-LOW_MEMORY_EXACT-thinking/` — ealuse.json, equests.csv, samples.csv, 	ranscripts.txt, raw server logs
- `tests/results/realuse-MAX_SPEED_EXACT/` — ealuse.json, equests.csv, samples.csv, 	ranscripts.txt, raw server logs
- `tests/results/realuse-MAX_SPEED_REPACK/` — ealuse.json, equests.csv, samples.csv, 	ranscripts.txt, raw server logs
