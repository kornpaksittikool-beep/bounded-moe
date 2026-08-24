// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 kornpaksittikool-beep
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "ggml.h"
GGML_API void ggml_routing_trace_set_enabled(int enabled);
GGML_API void ggml_routing_trace_reset(void);
GGML_API void ggml_routing_trace_dump(const char * path);
GGML_API int  ggml_routing_trace_is_enabled(void);
GGML_API uint64_t ggml_routing_trace_parse_failures(void);
GGML_API void ggml_routing_trace_record(const char * tensor_name, const int32_t * ids, int n_ids);
#ifdef __cplusplus
}
#endif
