# Third-party notices

This project is a set of modifications to **llama.cpp / ggml**, plus original scripts and
documentation. It builds against several third-party libraries that llama.cpp vendors.

The notices below are reproduced to satisfy the attribution requirements of those
licences. See `LICENSE_AUDIT.md` for the full audit.

**The model weights are not covered by any licence in this file.** See section 3.

---

## 1. llama.cpp / ggml

Upstream: https://github.com/ggml-org/llama.cpp
Base commit: `3e7344670adf63ce28527a4d42f2d71eca27c41e`

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The complete `AUTHORS` file and every upstream source header are preserved unmodified in
the source tree.

---

## 2. Libraries vendored by llama.cpp

These ship inside the llama.cpp checkout. They are listed here because a binary built
from this project may embed them.

| component | upstream | licence |
|---|---|---|
| cpp-httplib | https://github.com/yhirose/cpp-httplib | MIT, © 2017 yhirose |
| nlohmann/json | https://github.com/nlohmann/json | MIT, © 2013-2025 Niels Lohmann |
| xxHash | https://github.com/Cyan4973/xxHash | BSD 2-Clause, © 2012-2021 Yann Collet |
| rotate-bits | https://github.com/jb55/rotate-bits.h | MIT, © 2021 William Casarin |
| SHA-256 | Igor Pavlov, 2010 | Public domain |
| SHA-1 | Steve Reid | Public domain |
| MurmurHash3 | Peter Scott / Austin Appleby | Public domain |
| miniaudio | https://github.com/mackron/miniaudio | Public domain **or** MIT-0, at your choice |
| stb_image | https://github.com/nothings/stb | Public domain (Unlicense) **or** MIT, at your choice |
| sheredom/subprocess.h | https://github.com/sheredom/subprocess.h | Unlicense (public domain) |

Full licence texts live in the source tree at:

```
clean-room/llama.cpp/LICENSE
clean-room/llama.cpp/licenses/LICENSE-jsonhpp
clean-room/llama.cpp/vendor/cpp-httplib/LICENSE
clean-room/llama.cpp/vendor/hash/rotate-bits/LICENSE.md
clean-room/llama.cpp/vendor/hash/sha256/LICENSE
clean-room/llama.cpp/vendor/hash/xxhash/LICENSE
clean-room/llama.cpp/vendor/miniaudio/miniaudio.h        (end of file)
clean-room/llama.cpp/vendor/stb/stb_image.h              (end of file)
clean-room/llama.cpp/vendor/sheredom/subprocess.h        (file header)
clean-room/llama.cpp/vendor/hash/sha1/sha1.c             (file header)
```

Every component above is permissive. **None is copyleft.** All are compatible with MIT
redistribution.

The documented build (`-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF`, targets
`llama-cli`, `llama-server`, `llama-perplexity`) embeds cpp-httplib, nlohmann/json and
the hash libraries. `stb_image` and `miniaudio` are used by multimodal targets that this
build does not produce.

---

## 3. Model weights — separate licence, not granted here

**Nothing in this repository licenses, contains, or distributes model weights.**

`Qwen3.6-35B-A3B-Q4_K_M.gguf` is a third-party artifact that you must obtain yourself,
from https://huggingface.co/Qwen/Qwen3.6-35B-A3B or a GGUF re-quantization of it. Its
licence is **not** the MIT licence above and does not follow from it — it is a separate
work under its own licence, reproduced below for reference:

```
Apache License, Version 2.0
Copyright 2026 Alibaba Cloud
```

Full text: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE

This was confirmed two ways: the GGUF's own metadata embeds
`general.license = apache-2.0` with a link to the file above, and that file was fetched
directly and its contents verified. The model card states no additional usage
restrictions, acceptable-use policy, or commercial terms beyond standard Apache 2.0
practice.

**Obtain the model from its original source and comply with its terms** — in particular,
preserve the copyright notice and licence text if you redistribute the weights or a
derivative of them (this repository does not; see `.gitignore`).

Full audit: `LICENSE_AUDIT.md` section 5.

---

## 4. Build tools — used, not redistributed

The build requires Microsoft Visual C++ Build Tools, the Windows SDK, the NVIDIA CUDA
Toolkit, CMake and Ninja. These are **not** redistributed by this project; you install
them yourself under their own terms.

If compiled binaries are ever published, a CUDA-enabled build may require bundling NVIDIA
runtime redistributables under their EULA. That is out of scope for this source-only
repository and is flagged in `LICENSE_AUDIT.md` section 6.

---

## 5. Build-time network asset

With upstream's default `LLAMA_BUILD_UI=ON`, CMake downloads prebuilt web-UI assets from
the Hugging Face bucket `ggml-org/llama-ui`. Those assets are llama.cpp's own artifacts
under the MIT licence in section 1, and are not stored in this repository. Build with
`-DLLAMA_BUILD_UI=OFF` for a fully offline build.
