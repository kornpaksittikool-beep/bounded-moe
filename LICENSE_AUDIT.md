# License audit

Audit performed **2026-08-24** against the licence files present in the clean-room
source tree at `productization-v1/clean-room/llama.cpp`, checked out from upstream
llama.cpp commit `3e7344670adf63ce28527a4d42f2d71eca27c41e`.

This is an engineering audit of what the licence files in the tree say. **It is not
legal advice.** Items that cannot be settled from the files present are marked
**REQUIRES HUMAN LICENSE REVIEW**.

**2026-08-24, later same day — correction:** section 5 (model weights) was updated after
a more thorough re-scan of the GGUF's own metadata found an embedded licence field the
first pass missed, and after fetching and confirming the licence it points to. The model
licence question is now resolved (Apache 2.0). See section 5 for the correction and what
it changed.

---

## 1. Three separate things, three separate licences

The most common licensing mistake in this kind of project is treating the runtime and
the model as one artifact. They are not.

| layer | what it is | licence |
|---|---|---|
| **Runtime** | llama.cpp / ggml plus this project's modifications | MIT (see section 2) |
| **Third-party** | libraries vendored inside llama.cpp | MIT, BSD-2-Clause, public domain (see section 4) |
| **Model weights** | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | Apache License 2.0, © Alibaba Cloud — **a separate licence from the runtime's** (see section 5) |

**The model weights do not inherit the runtime licence.** Nothing in this repository
distributes weights, and nothing should.

---

## 2. Runtime — llama.cpp / ggml

```
MIT License
Copyright (c) 2023-2026 The ggml authors
```

Full text at `clean-room/llama.cpp/LICENSE`.

MIT permits use, modification and redistribution, including in modified form, provided
the copyright notice and permission notice are preserved.

**Obligation:** ship the upstream `LICENSE` file and its copyright notice with any
distribution of this project's source or binaries. Satisfied by `LICENSE` and
`THIRD_PARTY_NOTICES.md` at the repository root.

### This project's modifications

19 modified files (~1,340 inserted lines) and 4 new source files, listed in
`PRODUCTIZATION_BASELINE.md` section 4. They are modifications to an MIT-licensed work.
Under MIT there is no obligation to release them, and no obstacle to releasing them.

**Recommendation:** release under the same MIT terms, preserving the upstream notice.
This is the least surprising outcome for anyone reading the diff, and it is what the
repository root `LICENSE` currently expresses.

**No upstream file had its licence header removed or altered.** The modifications are
in-place edits to existing files plus new files; every upstream header remains.

### New files authored by this project

| file | note |
|---|---|
| `ggml/include/ggml-expert-indirection.h` | original, SPDX header added |
| `ggml/src/ggml-expert-indirection.cpp` | original, SPDX header added |
| `ggml/include/ggml-routing-trace.h` | original, SPDX header added |
| `ggml/src/ggml-routing-trace.cpp` | original, SPDX header added |

**Done.** Each now opens with:

```
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 kornpaksittikool-beep
```

matching the copyright holder in the repository's `LICENSE`. Added to the `patches/
new-files/` shipped copies and the clean-room build tree (never to `f5-final` or the
research dev tree, both left untouched); an incremental rebuild and full 512-token
reverification confirmed the EXACT and REPACK hashes were unaffected — see
`PRODUCTIZATION_BASELINE.md` §3–4 and `benchmark/results/g1-spdx-reverify/`.

---

## 3. Productization artifacts authored in Phase G

Everything under `benchmark/`, `launcher/`, `profiles/`, `scripts/`, `docs/`, `tests/`
and the root Markdown files was written for this project and contains no copied
third-party code.

`benchmark/run-benchmark-suite.ps1` and `benchmark/lib/BenchLib.psm1` reimplement the
measurement approach used by the research-phase scripts (also this project's own work);
no external snippets were incorporated.

**No external code snippets from Stack Overflow, blogs, or other repositories were
incorporated into any Phase G artifact.**

---

## 4. Third-party libraries vendored inside llama.cpp

All under `clean-room/llama.cpp/vendor/` and `clean-room/llama.cpp/licenses/`.

| component | path | licence | licence file present |
|---|---|---|---|
| cpp-httplib | `vendor/cpp-httplib/` | MIT, © 2017 yhirose | yes — `vendor/cpp-httplib/LICENSE` |
| nlohmann/json | `vendor/nlohmann/` | MIT, © 2013-2025 Niels Lohmann | yes — `licenses/LICENSE-jsonhpp` |
| miniaudio | `vendor/miniaudio/` | dual: public domain **or** MIT-0, at the user's choice | in-file, at end of `miniaudio.h` |
| sheredom/subprocess.h | `vendor/sheredom/` | Unlicense (public domain) | in-file header |
| stb_image | `vendor/stb/` | dual: MIT **or** public domain (Unlicense), at the user's choice | in-file, at end of `stb_image.h` |
| xxHash | `vendor/hash/xxhash/` | BSD 2-Clause, © 2012-2021 Yann Collet | yes — `vendor/hash/xxhash/LICENSE` |
| rotate-bits | `vendor/hash/rotate-bits/` | MIT, © 2021 William Casarin | yes — `vendor/hash/rotate-bits/LICENSE.md` |
| SHA-256 | `vendor/hash/sha256/` | public domain, Igor Pavlov, 2010 | yes — `vendor/hash/sha256/LICENSE` |
| SHA-1 | `vendor/hash/sha1/` | public domain, Steve Reid | in-file header |
| MurmurHash3 (inside miniaudio) | `vendor/miniaudio/miniaudio.h` | public domain | in-file note |

**Assessment:** every vendored component is MIT, BSD-2-Clause, or public domain. All are
permissive and none is copyleft. All are compatible with MIT redistribution.

**Obligation:** preserve each copyright and permission notice. The MIT and BSD-2-Clause
components require the notice to accompany binary distribution too. Satisfied by
`THIRD_PARTY_NOTICES.md`.

**Not audited:** this project does not vendor or redistribute these itself — they arrive
with the llama.cpp checkout. A binary release would embed cpp-httplib (server),
nlohmann/json, and the hash libraries; `stb_image` and `miniaudio` only if multimodal
targets are built (they are not, in the documented build).

---

## 5. Model weights — RESOLVED: Apache License 2.0

`Qwen3.6-35B-A3B-Q4_K_M.gguf`, 20,419,565,568 bytes.

**Correction to an earlier version of this audit.** The first pass of this section
reported no locally-determinable licence, based on a scan of the GGUF metadata that
checked only a fixed set of expected keys (architecture, expert count, and similar) and
missed the licence field because it was not on that list. A full, unfiltered scan of
every key in the GGUF's metadata found:

```
general.license      = apache-2.0
general.license.link = https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE
```

This was verified, not taken on faith: the linked page was fetched directly and confirms
**Apache License, Version 2.0**, copyright Alibaba Cloud (2026). The model's own card was
also checked and states no additional usage restrictions, acceptable-use policy, or
commercial terms beyond standard Apache 2.0 practice — no separate "Qwen License
Agreement" applies to this model.

Provenance: shell history on this workspace shows the file was fetched via `hf download`
from either `unsloth/Qwen3.6-35B-A3B-GGUF` or `ggml-org/Qwen3.6-35B-A3B-GGUF` (both
attempted against the same local path in immediate succession; which one's transfer
actually completed could not be determined from local artefacts alone). Both are
community GGUF re-quantizations of the same upstream weights,
`Qwen/Qwen3.6-35B-A3B`, and neither quantizer repo adds terms beyond the upstream
Apache 2.0 licence — GGUF conversion does not create a new work under a new licence.

**Consequences, now resolved:**

- **Apache 2.0 permits use, reproduction, modification, and redistribution** of the
  licensed work, including in derivative or object-code (here, quantized-weight) form,
  provided the copyright and licence notice are preserved and any changes are noted.
  Benchmarking, quoting architecture metadata, and naming the model in documentation are
  all ordinary uses under this licence.
- **The runtime's MIT licence is still separate from the model's Apache 2.0 licence** —
  this project's code and Alibaba Cloud's model weights are two different works under two
  different (both permissive) licences, and neither licenses the other.
- **This repository still must not distribute the weights.** `.gitignore` excludes
  `*.gguf`. Apache 2.0 does not change that decision — the weights are 19+ GiB, entirely
  separate from what a source-and-documentation repository should carry, and users are
  expected to obtain them from Hugging Face directly.
- The linked upstream `LICENSE` and its copyright notice should be reproduced in
  `THIRD_PARTY_NOTICES.md` when this project is made public, the same way the runtime's
  own MIT notice is. Done — see `THIRD_PARTY_NOTICES.md` section 3.

**What is not claimed:** that this resolves every model licence question a downstream
user might have. If the actual file in `D:\llm-v0\models\` did not come from one of the
two repos in the shell history — for instance if it was later replaced by hand — this
provenance chain would not apply to it. Re-verifying `general.license` directly from the
GGUF you actually have (as done here) is cheap and is the authoritative check, not this
document.

---

## 6. Build-time dependencies (not redistributed)

| tool | licence status |
|---|---|
| MSVC 14.44 (`cl.exe`, `link.exe`) | Microsoft proprietary — **not redistributed**; the user supplies it |
| Windows SDK 10.0.26100 | Microsoft proprietary — not redistributed |
| CUDA Toolkit 12.4 / `nvcc` | NVIDIA proprietary — not redistributed |
| CMake 3.31.6, Ninja 1.12.1 | BSD-3-Clause and Apache-2.0 respectively — not redistributed |

This repository ships **no compiled binaries** and no redistributable runtime components.
It ships source, patches, scripts and documentation, and instructs the user to build.

> **REQUIRES HUMAN LICENSE REVIEW if binaries are ever published:** a CUDA-enabled binary
> release may need to bundle NVIDIA runtime redistributables, which carry their own EULA
> and attribution requirements. This audit covers a source-only release.

---

## 7. Network-fetched build asset

`LLAMA_BUILD_UI=ON` (upstream default) makes CMake download prebuilt web-UI assets from
the Hugging Face bucket `ggml-org/llama-ui` at configure time.

- The assets are **not** stored in this repository.
- They are upstream llama.cpp's own artifacts and fall under llama.cpp's MIT licence.
- `-DLLAMA_BUILD_UI=OFF` produces a fully offline build without them; only the browser UI
  is lost, not the HTTP API.

Documented in `docs/BUILD_FROM_CLEAN.md` and `docs/KNOWN_LIMITATIONS.md`.

---

## 8. Findings summary

| # | finding | severity | status |
|---|---|---|---|
| 1 | Runtime is MIT; modifications may be released under MIT | — | clear |
| 2 | All 10 vendored components permissive, none copyleft | — | clear |
| 3 | Four new source files lack SPDX/copyright headers | low | **CLEAR** — headers added and rebuild re-verified |
| 4 | Model licence: Apache License 2.0, Alibaba Cloud | resolved this phase; missed on first scan, found and verified on re-scan | **CLEAR** |
| 5 | Binary release would pull in NVIDIA redistributables | medium | **REQUIRES HUMAN LICENSE REVIEW** if binaries are published |
| 6 | Web UI is fetched from the network at build time | low | documented, and switchable off |

**No copyleft obligation was found. No licence notice was found to have been removed or
altered. No incompatible combination was found.**

---

## 9. What this audit did not do

- Section 5 covers the model licence found in the GGUF's own metadata and verified against
  its upstream source; it does not extend to any other model, nor to a model file that may
  have replaced the one audited here.
- Did not perform automated licence scanning of every one of the 3,444 upstream files;
  it relied on the licence files and vendor directories the project ships.
- Did not assess patent, trademark, or export-control questions.
- Did not evaluate whether publishing benchmark numbers for a named third-party model
  requires that model owner's permission.
- Did not review any obligations attached to the account or platform used to publish.

All of these need a person.
