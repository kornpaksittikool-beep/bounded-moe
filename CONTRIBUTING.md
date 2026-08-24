# Contributing to bounded-moe

Thanks for helping improve bounded-moe.

This project changes performance-sensitive llama.cpp / ggml execution paths. Correctness and reproducibility matter more than an attractive single benchmark result.

## Before opening a pull request

1. Start from the repository's documented upstream llama.cpp base and follow `docs/BUILD_FROM_CLEAN.md`.
2. Keep changes focused and reversible. Avoid unrelated formatting or refactors in performance experiments.
3. Build from a clean state when the change affects runtime code.
4. Run the relevant benchmark/correctness gates described in `benchmark/BENCHMARK_METHODOLOGY.md`.
5. For `_EXACT` profiles, verify the validated greedy token-stream reference hash. A hash mismatch is a correctness failure unless the PR explicitly proposes and justifies changing that contract.
6. Report safety counters and crashes. Do not hide failed experiments or unstable runs.
7. Compare performance using warm, controlled runs. When comparing binaries, prefer back-to-back or grouped methodology appropriate to the footprint; see the benchmark methodology for the known page-cache ordering trap.

## Pull requests

Please include:

- what problem the change solves;
- the hypothesis or design rationale;
- exact build and benchmark commands;
- hardware, model, profile, and cache size;
- before/after results with multiple runs where practical;
- correctness/reference-hash result;
- peak RAM / working set and VRAM when relevant;
- any regressions, failed variants, or known limitations.

Performance improvements that change model behavior must be clearly labeled. Do not describe numerical/logit equivalence as bit-exact unless it has actually been demonstrated. See `docs/CORRECTNESS.md` and `docs/QUALITY_REPORT.md`.

## Issues and Discussions

Use **Issues** for reproducible bugs, crashes, build failures, and compatibility problems.

Use **Discussions → Benchmarks & Hardware** for normal benchmark submissions, hardware results, questions, and comparisons that do not represent a bug.

## Scope

The initial validated platform is deliberately narrow. Check `docs/KNOWN_LIMITATIONS.md` before assuming support for another OS, model, GPU, storage device, context size, or serving configuration.

## Licensing

By contributing, you agree that your contribution may be distributed under the repository's MIT license. Do not submit model weights, proprietary code, secrets, or third-party material that you do not have permission to redistribute.
