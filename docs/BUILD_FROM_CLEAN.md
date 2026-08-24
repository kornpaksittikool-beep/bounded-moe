# Building from a clean tree

How to reproduce the runtime from nothing but this repository, an upstream llama.cpp
checkout, and a Windows toolchain.

Everything here was executed end to end on 2026-08-24; the result is recorded in
`CLEAN_REPRODUCTION.md` and `BUILD_ENVIRONMENT.json`.

---

## 1. What you need

| tool | version used | note |
|---|---|---|
| Visual Studio 2022 Build Tools | MSVC 14.44.35207 (`cl.exe` 14.44.35228.0) | the C++ workload; the full IDE is not required |
| Windows SDK | 10.0.26100.0 | comes with the Build Tools |
| CUDA Toolkit | 12.4 (`nvcc` V12.4.131) | must be installed before configuring |
| CMake | 3.31.6-msvc6 | bundled with the Build Tools |
| Ninja | 1.12.1 | bundled with the Build Tools |
| Git | 2.55.0 | |

Bundled CMake and Ninja live at:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
```

Neither is on `PATH` by default. `scripts/build-clean.ps1` uses them by absolute path.

Disk: about 12 GiB for the build tree. Time: roughly 25 minutes on the reference
machine, almost all of it compiling CUDA kernels.

---

## 2. Get a pristine source tree

Do **not** reuse a development tree. The point of this procedure is to prove that the
result does not depend on accidental working-tree state.

```powershell
$src = 'D:\work\clean-room\llama.cpp'
New-Item -ItemType Directory -Force -Path $src | Out-Null

git init -q $src
git -C $src config core.autocrlf false
git -C $src config core.eol lf
git -C $src remote add origin https://github.com/ggml-org/llama.cpp
git -C $src fetch -q --depth=1 origin 3e7344670adf63ce28527a4d42f2d71eca27c41e
git -C $src checkout -q FETCH_HEAD
```

Verify it is pristine — 3444 tracked files, nothing modified:

```powershell
git -C $src ls-files | Measure-Object          # 3444
git -C $src status --porcelain=v1 | Measure-Object  # 0
```

### Why `core.autocrlf false`

The patch is generated with git's line-ending normalisation. Checking out with
`autocrlf=true` gives CRLF working files, and the patch then applies against different
bytes. It still compiles, but the source no longer hashes equal to the reference
checkpoint, which removes the strongest available verification step. With `false`, all
nine checkpoint files hash **identical**.

---

## 3. Apply the change set

```powershell
git -C $src apply --check .\patches\0001-external-expert-cache.patch
git -C $src apply         .\patches\0001-external-expert-cache.patch

Get-ChildItem .\patches\new-files -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring((Resolve-Path .\patches\new-files).Path.Length + 1)
    $dst = Join-Path $src $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    Copy-Item $_.FullName $dst -Force
}
```

`git apply --check` must report exit code 0 before you apply. In PowerShell, check
`$LASTEXITCODE` rather than `$?` — git writes progress to stderr, which makes `$?` false
even on success.

Afterwards `git -C $src status --porcelain=v1` should list **19 modified** files and
**4 untracked** new ones.

`patches/new-files/` ships the four new files with SPDX license headers already applied
(`// SPDX-License-Identifier: MIT`, `// SPDX-FileCopyrightText: 2026
kornpaksittikool-beep`), added after the initial clean-room verification and reconfirmed
not to change program behaviour — see `docs/CLEAN_REPRODUCTION.md` §2. Copying them as
shown above gives you the headers automatically; nothing extra to do.

---

## 4. Build

```powershell
.\scripts\build-clean.ps1 `
    -SourceDir D:\work\clean-room\llama.cpp `
    -BuildDir  D:\work\clean-room\build `
    -Fresh
```

The script writes only into `-BuildDir` and never touches the source tree.

### The exact configuration it applies

```
-G Ninja
-DCMAKE_BUILD_TYPE=RelWithDebInfo
-DBUILD_SHARED_LIBS=ON
-DGGML_CUDA=ON  -DGGML_CUDA_FA=ON  -DGGML_CUDA_GRAPHS=ON
-DGGML_NATIVE=ON
-DGGML_BLAS=OFF
-DLLAMA_BUILD_TESTS=OFF  -DLLAMA_BUILD_EXAMPLES=OFF
-DCMAKE_C_FLAGS=<common>  -DCMAKE_CXX_FLAGS=<common>
```

where `<common>` is

```
/Zi /O2 /Ob1 /DNDEBUG
/DB1R4_COMPILE_OUT_DIAGNOSTICS=1
/DB1S_PROFILE=1 /DB1T_PROFILE=1 /DB1U_PROFILE=0 /DB1W_PROFILE=1
```

Targets built: `llama-cli`, `llama-server`, `llama-perplexity`.

### The profiling defines are not optional

`B1S_PROFILE`, `B1T_PROFILE`, `B1W_PROFILE` and `B1R4_COMPILE_OUT_DIAGNOSTICS` were
enabled in **every** published measurement. A build without them has never been
benchmarked, and its numbers would not be comparable to anything in this repository.
They are part of the measured configuration, not debug scaffolding to strip.

### Two mechanical traps

**`vcvars64.bat` needs `vswhere.exe` on `PATH`.** It invokes it by bare name, and the VS
Installer directory is not on `PATH` by default. The script prepends
`C:\Program Files (x86)\Microsoft Visual Studio\Installer`.

**Do not pipe the build through PowerShell.** In Windows PowerShell 5.1, redirecting a
native command's stderr (`*>&1`, `2>&1`) wraps ordinary cmake status lines in
`NativeCommandError` records and corrupts the exit code — a successful build reports
failure. The script has the batch file own its redirection and reads only the exit code.

---

## 5. Offline builds

Upstream defaults to `LLAMA_BUILD_UI=ON`, which makes CMake **download** prebuilt web-UI
assets from the Hugging Face bucket `ggml-org/llama-ui` at configure time. On a machine
without network access this stalls.

```powershell
.\scripts\build-clean.ps1 -SourceDir <src> -BuildDir <build> -Fresh `
    -ExtraCMakeArgs '-DLLAMA_BUILD_UI=OFF'
```

You lose only the browser UI. The HTTP API, the CLI, and every profile are unaffected.

---

## 6. Verify

```powershell
.\benchmark\run-benchmark-suite.ps1 `
    -Mode Full -Model <path to gguf> `
    -BinDir D:\work\clean-room\build\bin `
    -Profiles LOW_MEMORY_EXACT,MAX_SPEED_REPACK
```

A correct build produces, in `SUMMARY.md`:

- `LOW_MEMORY_EXACT` → hash `414B86C6E75D7126… = REF`
- `MAX_SPEED_REPACK` → hash `1D63259E5842DFE3… = REF`
- `safety` = `PASS` on both

**Those two properties are the reproduction criterion.** Throughput is machine state.

---

## 7. Binaries are not bit-reproducible, and that is expected

MSVC embeds a build timestamp, a PDB GUID, and `__FILE__` strings that contain the
absolute source path. A rebuild from identical sources therefore produces different
hashes, and a longer source path produces slightly larger binaries.

Observed against the reference build, with an identical option set:

| binary | reference | clean-room | size |
|---|---:|---:|---|
| `ggml-cpu.dll` | 1,561,600 | 1,561,600 | identical |
| `ggml-base.dll` | 5,763,072 | 5,763,072 | identical |
| `llama-cli-impl.dll` | 3,178,496 | 3,178,496 | identical |
| `llama-common.dll` | 13,656,064 | 13,656,064 | identical |
| `llama-server-impl.dll` | 18,955,264 | 18,955,264 | identical |
| `ggml-cuda.dll` | 160,925,184 | 160,927,744 | +2,560 |
| `llama.dll` | 4,983,296 | 4,986,368 | +3,072 |
| `mtmd.dll` | 3,478,528 | 3,479,040 | +512 |

Ten of thirteen match to the byte. The three that grew are the ones whose accumulated
`__FILE__` strings pushed a section past a 512-byte alignment boundary — the clean-room
path is 16 characters longer than the reference path.

**Reproduction is verified functionally, by token-trace hash and safety counters, not by
binary equality.** If you need bit-reproducible binaries, that is a separate project
requiring `/Brepro`, deterministic PDB paths, and a pinned toolchain.

---

## 8. What the clean build proved

| check | result |
|---|---|
| Pristine checkout + patch applies cleanly | yes, exit 0 |
| All 9 `f5-final` checkpoint sources byte-identical | **9 / 9** at first check; 7/9 after SPDX headers were deliberately added to 2 files in the clean-room tree only, see `docs/CLEAN_REPRODUCTION.md` §2 |
| Remaining 10 changed files semantically identical | yes — differ only in line endings |
| Build option set identical to the reference build | yes, apart from `LLAMA_BUILD_EXAMPLES` / `LLAMA_BUILD_TESTS` |
| EXACT 512-token hash reproduced | yes, on 4 profiles × 3 runs |
| REPACK 512-token hash reproduced | yes, 3 runs |
| Safety invariants | PASS on every run |

Details and the throughput discussion are in `CLEAN_REPRODUCTION.md`.
