// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 kornpaksittikool-beep
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API int ggml_expert_indirection_enabled(void);
GGML_API int ggml_expert_verify_only_enabled(void);
GGML_API int ggml_expert_cached_bounds_enabled(void);
GGML_API int ggml_expert_cached_compute_enabled(void);
GGML_API int ggml_expert_b1b_detached_enabled(void);
GGML_API int ggml_expert_external_storage_enabled(void);
GGML_API int ggml_expert_placement_diagnostics_enabled(void);
GGML_API int ggml_expert_external_is_tensor(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_tensor(size_t logical_bytes, size_t reserved_bytes, size_t committed_bytes);
GGML_API void ggml_expert_external_note_placement(const struct ggml_expert_storage_info * info,
        int cpu_candidate, int mmap_candidate, int external_assigned,
        const char * buft_name, const char * device_name);
GGML_API void ggml_expert_external_note_get(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_set(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_copy(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_clear(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_view(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_validation(const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_note_failure(const char * reason, const struct ggml_tensor * tensor);
GGML_API void ggml_expert_external_report(void);
GGML_API void ggml_expert_external_memory_checkpoint(const char * stage);
GGML_API void ggml_expert_note_original_pointer_use(void);
GGML_API void ggml_expert_note_cached_pointer_use(void);
GGML_API void ggml_expert_note_external_vec_dot(const void * vx, const void * reserved_begin, size_t reserved_size);
GGML_API void ggml_expert_note_unexpected_mmap_fallback(void);
GGML_API int ggml_expert_profile_enabled(void);
GGML_API void ggml_expert_note_mul_mat_id_time(uint64_t elapsed_us);
GGML_API int ggml_expert_mmid_profile_enabled(void);
GGML_API uint64_t ggml_expert_mmid_now_ns(void);
GGML_API void ggml_expert_mmid_note_call(uint64_t elapsed_ns);
GGML_API void ggml_expert_mmid_add_region(int region, uint64_t elapsed_ns);
GGML_API int ggml_expert_mmid_vecdot_sample(void);
GGML_API int ggml_expert_mmid_sample_ratio(uint32_t ratio);
GGML_API void ggml_expert_mmid_note_vecdot(uint64_t elapsed_ns);
GGML_API int ggml_expert_b1k_profile_enabled(void);
GGML_API int ggml_expert_b1k_sample(void);
GGML_API void ggml_expert_b1k_add(int region, uint64_t elapsed_ns);
GGML_API void ggml_expert_b1k_count(int region);
GGML_API void ggml_expert_b1l_add(int region, uint64_t elapsed_ns);
GGML_API void ggml_expert_b1l_branch(int branch, int taken);
GGML_API void ggml_expert_b1m_note_external_storage_eval(void);
GGML_API int ggml_expert_b1n_sample(void);
GGML_API int ggml_expert_b1n_profile_enabled(void);
GGML_API void ggml_expert_b1n_add(int region, uint64_t elapsed_ns);
GGML_API void ggml_expert_b1n_note_sampled_call(int region);
GGML_API int ggml_expert_validate_cached_row(const struct ggml_tensor * tensor, int expert_id,
        const char * original_plane, const char * cached_plane, int64_t ir0,
        size_t nb01, size_t nb02, size_t required_bytes, const char * vec_dot_name);
GGML_API const char * ggml_expert_resolve_ptr(const struct ggml_tensor * src0, int expert_id, size_t expert_stride);
GGML_API void ggml_expert_indirection_dump(const char * path);

struct ggml_expert_storage_info {
    const struct ggml_tensor * tensor;
    const char * name;
    int layer;
    int kind;
    uint32_t file_index;
    uint64_t os_handle;
    uint64_t absolute_offset;
    uint64_t plane_stride;
    uint64_t plane_size;
    uint64_t tensor_size;
    int ggml_type;
    uint32_t n_experts;
    int cpu_override;
};
GGML_API void ggml_expert_storage_register(const struct ggml_expert_storage_info * info);
GGML_API int ggml_expert_storage_registry_validate(void);
GGML_API int ggml_expert_storage_lookup(const struct ggml_tensor * tensor, struct ggml_expert_storage_info * out);
GGML_API void ggml_expert_cache_unpin(const struct ggml_tensor * tensor, int expert_id);
// B2d: opt-in, low-overhead route/load/first-use timing probe.
GGML_API int  ggml_expert_b2d_enabled(void);
GGML_API int  ggml_expert_b2d_note_route(const struct ggml_tensor * tensor, const int32_t * expert_ids, int n_ids);
GGML_API void ggml_expert_b2d_note_first_use(int layer_id, int expert_id);

// B1s: coarse wall-clock profiling only; inactive unless compiled with B1S_PROFILE.
GGML_API int ggml_b1s_profile_enabled(void);
GGML_API uint64_t ggml_b1s_mmid_enter(int worker);
GGML_API void ggml_b1s_mmid_exit(int worker, uint64_t start_ns);
GGML_API uint64_t ggml_b1s_region_begin(void);
GGML_API void ggml_b1s_region_end(int worker, int region, uint64_t start_ns);
GGML_API int ggml_b1t_profile_enabled(void);
GGML_API uint64_t ggml_b1t_phase_begin(void);
GGML_API void ggml_b1t_phase_end(int worker, int phase, uint64_t start_ns);
GGML_API void ggml_b1t_total_end(int worker, uint64_t start_ns);

// B1w: low-overhead worker-state sampling; inactive unless compiled and enabled at runtime.
enum ggml_expert_b1w_state {
    GGML_EXPERT_B1W_OUTSIDE_MMID = 0,
    GGML_EXPERT_B1W_MMID_SETUP = 1,
    GGML_EXPERT_B1W_MMID_OTHER = 2,
    GGML_EXPERT_B1W_TASK_ACQUIRE = 3,
    GGML_EXPERT_B1W_SCHEDULER = 4,
    GGML_EXPERT_B1W_WAIT_IDLE = 5,
    GGML_EXPERT_B1W_CHUNK_ACQUIRE = 6,
    GGML_EXPERT_B1W_CHUNK_PREPARE = 7,
    GGML_EXPERT_B1W_RESOLVER = 8,
    GGML_EXPERT_B1W_EXTERNAL_COMPUTE = 9,
    GGML_EXPERT_B1W_ACCUMULATION = 10,
    GGML_EXPERT_B1W_UNPIN_RELEASE = 11,
    GGML_EXPERT_B1W_CHUNK_COMPLETE = 12,
    GGML_EXPERT_B1W_BARRIER_WAIT = 13,
    GGML_EXPERT_B1W_LOCK_WAIT = 14,
    GGML_EXPERT_B1W_IO_WAIT = 15,
    GGML_EXPERT_B1W_GPU_WAIT = 16,
    GGML_EXPERT_B1W_UNKNOWN = 17,
    GGML_EXPERT_B1W_MMID_RANGE_A = 18,
    GGML_EXPERT_B1W_MMID_RANGE_B = 19,
    GGML_EXPERT_B1W_MMID_RANGE_C = 20,
    GGML_EXPERT_B1W_MMID_RANGE_D = 21,
    GGML_EXPERT_B1W_STATE_COUNT = 22,
};
GGML_API int ggml_expert_b1w_enabled(void);
GGML_API void ggml_expert_b1w_start(int worker_count);
GGML_API void ggml_expert_b1w_register_worker(int worker);
GGML_API void ggml_expert_b1w_set_state(int worker, int state);
GGML_API void ggml_expert_b1w_set_mmid_active(int worker, int active);
GGML_API void ggml_expert_b1w_shutdown(void);

#ifdef __cplusplus
}
#endif
