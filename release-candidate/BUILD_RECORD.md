# Release-Candidate Clean Build Record

## Source

- Base upstream commit: `3e7344670adf63ce28527a4d42f2d71eca27c41e` (2026-08-19, matches
  `README.md`'s stated base and the `bounded-moe` v0.1.0 patch base)
- Cloned fresh from the pristine local mirror (`llama.cpp-source-10488`, clean status
  verified before cloning) into `release-candidate/llama.cpp` - not derived from, or
  sharing any state with, `clean-room/llama.cpp` (which was hand-edited directly during
  Phase I, bypassing the patch process).
- Patch applied: `patches/0001-external-expert-cache.patch`
  SHA-256 `1c9c428d3ddbdf630b0500abd945ccb58005618b5f2a1f01b9db0cdef2aa21c4`
  (`git apply --check` passed cleanly against this exact base commit)
- New files copied from `patches/new-files/` (includes the Phase I hard-cap fix,
  `ggml-expert-indirection.cpp` SHA-256
  `3b9c8c9cc0f37455cbfb4bb10c75aedced11f5d5782ae91f3fa39dd65a9d39ec`)
- Verified byte-for-byte: `release-candidate/llama.cpp/ggml/src/ggml-expert-indirection.cpp`
  diffs at zero bytes against `patches/new-files/ggml/src/ggml-expert-indirection.cpp`,
  and contains exactly the 8 intended `30`->`40` edits (verified by grep, matches the
  Phase I audit in `I2_FULL_EXTERNALIZATION_DESIGN.md` exactly).

## Build

Same configuration as `clean-room` (`scripts/build-clean.ps1`, unmodified): Ninja,
RelWithDebInfo, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_NATIVE=ON`,
CUDA 12.4, MSVC 2022 Build Tools, the project's compile defines
(`B1R4_COMPILE_OUT_DIAGNOSTICS=1 B1S_PROFILE=1 B1T_PROFILE=1 B1U_PROFILE=0 B1W_PROFILE=1`).
Fresh CMake configure (`-Fresh`), not incremental. `BUILD_OK`, no errors.

## Binaries

| file | SHA-256 |
|---|---|
| `llama-cli.exe` | `60AF92498937B291936D8E09B019A444D7B3463AED8EF867E0CB42381DD3811A` |
| `llama-server.exe` | `D5159077BB0566BCD4F2CA168E1891DE1AF6C256AACB28363F8F90D48CEC9D28` |
| `ggml-cpu.dll` | `060F93F57946B3AF486781FCDE2EEDAF61CB4FAB23E9A04C3623400C3168B825` |
| `ggml-cuda.dll` | `11ABBC363FF51F874C122A2F231B420E6CD8EC8A8D357F6AC06B8FB98C51A64E` |
| `ggml-base.dll` | `FFD052C1A17795B975A8DA46A4FCFCA572C068F96EFC57718142128F67862C79` |
| `llama-cli-impl.dll` | `F50F45369BA0B62D0E05077B58B4D4658741F2499BAD9E1DFE94717B0EA854D5` |
| `llama-common.dll` | `6D8D4876BD178FB10489E9625F3939BA6CEF0A041299C4F21B46D21C57711E7C` |
| `llama-server-impl.dll` | `EEBD43015012B1E62C15CD6AC7CAC0561496FC43C1969B6FB56D8D73BD193D59` |

Different bytes from every other binary built this session (`build-windbg-ninja`,
`clean-room/build`, `build-context-research-i2`) - expected, per `KNOWN_LIMITATIONS.md`
§10: MSVC embeds build timestamps/PDB GUIDs/`__FILE__` paths, so identical source never
produces identical binaries. Reproduction is verified functionally (regression status
below and behaviorally against the reference token hash), not by binary equality.

## Regression status: PASS

`LOW_MEMORY_EXACT` settings (`-c 16384 -ncmoe 30 cache=2560 -t 4`, 512 tokens, greedy)
against this exact binary reproduce the reference hash
`414B86C6E75D7126C65D51EE96F144DC64E15562E30EEE574672FE0514ADC12B` exactly, and
`EXPERT_REGISTRY_VALIDATE` reports `cpu_layers=30, result=PASS` - byte-for-byte identical
behavior to every other binary built from this source line this session. See
`autonomous-research/long-context-perf-research/LONG_CONTEXT_FINAL_TABLE.md` for the full
long-context measurement table produced from this build.
