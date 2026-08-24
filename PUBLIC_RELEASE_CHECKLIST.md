# Public release checklist

Status as of the end of Phase G. **Nothing has been published.** No repository was
created, no remote was added, no commit was pushed, no release was cut, and no
sponsorship or donation mechanism was enabled. All of that requires explicit approval.

Legend: **DONE** · **ACTION** (must be done before publishing) · **BLOCKER** (needs a
human decision) · **N/A**

---

## 1. Correctness and reproducibility

| # | item | status |
|---|---|---|
| 1.1 | Clean-room source tree builds from a pristine upstream checkout plus a patch | **DONE** |
| 1.2 | Clean-room sources verified byte-identical to the `f5-final` checkpoint | **DONE** — 9/9 at first check; 7/9 after SPDX headers were deliberately added to 2 files (3.6), rebuild-and-reverified |
| 1.3 | Remaining changed files verified semantically identical (line-ending only) | **DONE** |
| 1.4 | Clean build reproduces the EXACT 512-token hash `414B86C6E75D7126…` | **DONE** |
| 1.5 | Clean build reproduces the REPACK hash `1D63259E5842DFE3…` | **DONE** |
| 1.6 | Safety invariants pass on every measured run (`pins == unpins`, `current_pins = 0`) | **DONE** |
| 1.7 | No access violation, no crash, no stale process across all runs | **DONE** |
| 1.8 | Throughput reproduction on the larger-footprint profiles | see `docs/CLEAN_REPRODUCTION.md` |
| 1.9 | Build environment recorded machine-readably | **DONE** — `BUILD_ENVIRONMENT.json` |

## 2. Baseline protection

| # | item | status |
|---|---|---|
| 2.1 | `f5-final` checkpoint unmodified | **DONE** — hashes re-verified |
| 2.2 | Validated binaries never overwritten | **DONE** — Phase G wrote only to `productization-v1/` |
| 2.3 | GGUF unmodified, opened read-only | **DONE** — timestamp and head hash unchanged |
| 2.4 | Restore procedure documented | **DONE** — `PRODUCTIZATION_BASELINE.md` §10 |

## 3. Licensing

| # | item | status |
|---|---|---|
| 3.1 | Upstream llama.cpp MIT notice preserved | **DONE** |
| 3.2 | All vendored third-party notices catalogued | **DONE** — `THIRD_PARTY_NOTICES.md` |
| 3.3 | No copyleft obligation found | **DONE** |
| 3.4 | Runtime / model / third-party licences kept distinct | **DONE** |
| 3.5 | `LICENSE` copyright holder filled in | **DONE** — set to `kornpaksittikool-beep` |
| 3.6 | SPDX headers added to the four new source files | **DONE** — added, rebuilt, and 512-token re-verified; see note below |
| 3.7 | Model weights licence reviewed | **DONE** — Apache License 2.0, © Alibaba Cloud, verified against the source; see `LICENSE_AUDIT.md` §5 |
| 3.8 | Permission to name the model and publish measurements against it | **DONE** — Apache 2.0 permits this; no additional restriction found on the model card |
| 3.9 | NVIDIA redistributable terms | **N/A** for a source-only release; becomes an **ACTION** if binaries are ever published |

## 4. Publication safety scan

| # | item | status |
|---|---|---|
| 4.1 | No API keys, tokens, passwords or credentials | **DONE** — scanned, zero hits |
| 4.2 | No email addresses | **DONE** — zero hits |
| 4.3 | No `C:\Users\<name>` paths | **DONE** — zero hits |
| 4.4 | No GGUF or other model weights | **DONE** — `.gitignore` excludes `*.gguf` and friends |
| 4.5 | No compiled binaries | **DONE** — `.gitignore` excludes `*.dll`, `*.exe`, `*.pdb` |
| 4.6 | No crash dumps or profiler traces | **DONE** — none exist; `.gitignore` covers them |
| 4.7 | No raw benchmark logs (they carry local paths and PIDs) | **DONE** — `benchmark/results/` and `tests/results/` ignored |
| 4.8 | No launcher session logs | **DONE** — `launcher/sessions/` ignored |
| 4.9 | Machine name / workspace path leakage | **ACTION** — `PRODUCTIZATION_BASELINE.md` and `LICENSE_AUDIT.md` cite `D:\llm-v0\…`; see 6.2 |
| 4.10 | Evaluation corpora excluded | **DONE** — `tests/corpora/` added to `.gitignore`; `docs/QUALITY_REPORT.md` describes how to assemble equivalents |

## 5. Documentation

| # | item | status |
|---|---|---|
| 5.1 | `README.md` — what, why, results, hardware, quick start, limits, reproduction | **DONE** |
| 5.2 | `docs/ARCHITECTURE.md` | **DONE** |
| 5.3 | `docs/PERFORMANCE.md` | **DONE** |
| 5.4 | `docs/CORRECTNESS.md` | **DONE** |
| 5.5 | `docs/QUALITY_REPORT.md` | **DONE** |
| 5.6 | `docs/KNOWN_LIMITATIONS.md` | **DONE** |
| 5.7 | `docs/RESEARCH_HISTORY.md`, including failures and negative results | **DONE** |
| 5.8 | `docs/DESIGN_DECISIONS.md` | **DONE** |
| 5.9 | `docs/BUILD_FROM_CLEAN.md` | **DONE** |
| 5.10 | `docs/CLEAN_REPRODUCTION.md` | **DONE** |
| 5.11 | `benchmark/BENCHMARK_METHODOLOGY.md` | **DONE** |
| 5.12 | `launcher/USER_GUIDE.md` | **DONE** |
| 5.13 | `profiles/PROFILES.md` | **DONE** |
| 5.14 | `docs/REAL_USE_VALIDATION.md`, `docs/STABILITY_REPORT.md` | **DONE** |
| 5.15 | No sensational claims; REPACK never called "lossless" | **DONE** |
| 5.16 | Every throughput figure labelled as a reference measurement on one machine | **DONE** |

## 6. Repository hygiene

| # | item | status |
|---|---|---|
| 6.1 | `.gitignore` covering weights, binaries, logs, dumps, secrets | **DONE** |
| 6.2 | Decide which internal documents ship publicly | **ACTION** — `PRODUCTIZATION_BASELINE.md`, `PRODUCTIZATION_STATE.json` and `PRODUCTIZATION_LOG.md` are internal records that point at the private research workspace; recommend excluding them |
| 6.3 | `clean-room/llama.cpp/` excluded (it is upstream's tree, reconstructed by `patches/`) | **DONE** |
| 6.4 | Repository initialised as git | **ACTION** — not yet a git repository |
| 6.5 | Curated benchmark results force-added if they are to be published | **ACTION** — `SUMMARY.md` / `results.csv` only, never `raw/` |

## 7. Things deliberately NOT done

| item | status |
|---|---|
| Create a GitHub (or any) repository | **NOT DONE** — requires approval |
| Add a git remote | **NOT DONE** — requires approval |
| Push any commit | **NOT DONE** — requires approval |
| Publish a release or upload artifacts | **NOT DONE** — requires approval |
| Enable sponsorship or donations | **NOT DONE** — requires approval |
| Distribute model weights | **NOT DONE** — must never be done |

---

### Note on 3.6 - why the SPDX headers were deferred, then added as their own step

`ggml-expert-indirection.{h,cpp}` and `ggml-routing-trace.{h,cpp}` carried no licence
header for most of Phase G. Adding one was correct before publication, but it was
deliberately not done immediately: two of those files are among the nine that Phase G
verified byte-identical to the `f5-final` checkpoint, and that verification was the
strongest evidence that the clean-room tree matched the validated tree. Editing them
mid-phase would have broken that check with no way to tell header noise from a real
divergence.

Done, as its own step, later in Phase G:

```powershell
.\benchmark\run-benchmark-suite.ps1 -Mode Full -Model <gguf> -Profiles LOW_MEMORY_EXACT,MAX_SPEED_REPACK
```

Headers were added only to the `productization-v1/clean-room/llama.cpp/` build tree and
the shipped `patches/new-files/` copies - never to `f5-final` or the research dev tree,
both of which remain untouched and byte-identical to each other. An incremental rebuild
followed, then the command above confirmed both the EXACT and REPACK 512-token hashes
were unaffected. See `PRODUCTIZATION_BASELINE.md` section 3-4 for the post-header hashes
and `benchmark/results/g1-spdx-reverify/` for the confirming run.

`patches/0001-external-expert-cache.patch` itself did not need regenerating: it is a
`git diff` over the 19 pre-existing tracked files, and the new files ship separately via
`patches/new-files/`, which is what changed.

---

## Blockers before any public release

**None remaining.** Both items previously listed here are resolved:

1. **Copyright holder — `LICENSE`.** Set to `kornpaksittikool-beep`.
2. **Model licence.** The GGUF's own metadata names Apache License 2.0 (Alibaba Cloud),
   verified against the licence text at its source rather than taken on faith. See
   `LICENSE_AUDIT.md` §5 for the full correction — the first audit pass missed this field
   on an incomplete metadata scan.

Everything else on this list is either done or a mechanical action.

---

## Recommended order

All licensing and correctness items (sections 1-4) are now done. What remains is purely
repository mechanics:

1. Decide what to do with the internal records (6.2).
2. `git init`, commit, review `git status` against `.gitignore` one more time (6.4).
3. Force-add only the curated benchmark summaries you intend to publish (6.5).
4. **Then** ask for approval to create a repository and push.
