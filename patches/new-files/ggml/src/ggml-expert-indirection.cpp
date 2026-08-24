// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 kornpaksittikool-beep
#include "ggml-expert-indirection.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <deque>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <shared_mutex>
#include <condition_variable>
#include <limits>
#include <memory>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <unistd.h>
#endif

#ifndef B1R4_COMPILE_OUT_DIAGNOSTICS
#define B1R4_COMPILE_OUT_DIAGNOSTICS 0
#endif
#ifndef B1S_PROFILE
#define B1S_PROFILE 0
#endif
#ifndef B1T_PROFILE
#define B1T_PROFILE 0
#endif
#ifndef B1U_PROFILE
#define B1U_PROFILE 0
#endif
#ifndef B1W_PROFILE
#define B1W_PROFILE 0
#endif

#if B1R4_COMPILE_OUT_DIAGNOSTICS
#undef B1S_PROFILE
#undef B1T_PROFILE
#undef B1U_PROFILE
#undef B1W_PROFILE
#define B1S_PROFILE 0
#define B1T_PROFILE 0
#define B1U_PROFILE 0
#define B1W_PROFILE 0
#endif

// B2d timing observability. Disabled unless B2D_TIMING_PATH is set. The probe
// stores integer keys and QPC ticks only; it never writes from the hot path.
static constexpr size_t B2D_MAX_EVENTS = 65536;
static constexpr size_t B2D_KEY_COUNT = 40 * 256;
struct b2d_event {
    uint64_t route = 0;
    uint64_t resolve = 0;
    uint64_t load_start = 0;
    uint64_t load_end = 0;
    std::atomic<uint64_t> first_use{0};
    int32_t layer = -1;
    int32_t expert = -1;
    int32_t kind = -1;
    uint32_t bytes = 0;
};
static b2d_event g_b2d_events[B2D_MAX_EVENTS];
static std::atomic<uint64_t> g_b2d_route[B2D_KEY_COUNT] = {};
static std::atomic<uint32_t> g_b2d_pending[B2D_KEY_COUNT] = {};
static std::atomic<uint64_t> g_b2d_count{0};
static std::atomic<uint64_t> g_b2d_dropped{0};
static bool g_b2d_enabled = false;
static std::string g_b2d_path;
static uint64_t g_b2d_freq = 1000000000ULL;

static void b2d_init() {
    const char * path = std::getenv("B2D_TIMING_PATH");
    g_b2d_enabled = path && path[0];
    if (g_b2d_enabled) g_b2d_path = path;
#ifdef _WIN32
    LARGE_INTEGER f = {};
    if (QueryPerformanceFrequency(&f) && f.QuadPart > 0) g_b2d_freq = (uint64_t) f.QuadPart;
#endif
}

static uint64_t b2d_now() {
#ifdef _WIN32
    LARGE_INTEGER v = {};
    QueryPerformanceCounter(&v);
    return (uint64_t) v.QuadPart;
#else
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

static int b2d_key(int layer, int expert) {
    return layer >= 0 && layer < 40 && expert >= 0 && expert < 256 ? layer * 256 + expert : -1;
}

extern "C" int ggml_expert_b2d_enabled(void) {
    static const bool initialized = []() { b2d_init(); return true; }();
    (void) initialized;
    return g_b2d_enabled ? 1 : 0;
}

extern "C" int ggml_expert_b2d_note_route(const ggml_tensor * tensor, const int32_t * expert_ids, int n_ids) {
    if (!ggml_expert_b2d_enabled() || !tensor || !expert_ids || n_ids <= 0) return -1;
    ggml_expert_storage_info info = {};
    if (!ggml_expert_storage_lookup(tensor, &info)) return -1;
    const uint64_t now = b2d_now();
    for (int i = 0; i < n_ids; ++i) {
        const int key = b2d_key(info.layer, expert_ids[i]);
        if (key >= 0) g_b2d_route[(size_t) key].store(now, std::memory_order_release);
    }
    return info.layer;
}

static int b2d_begin_miss(int layer, int expert, int kind, uint32_t bytes, uint64_t resolve, uint64_t load_start) {
    if (!ggml_expert_b2d_enabled()) return -1;
    const uint64_t index = g_b2d_count.fetch_add(1, std::memory_order_relaxed);
    if (index >= B2D_MAX_EVENTS) { g_b2d_dropped.fetch_add(1, std::memory_order_relaxed); return -1; }
    b2d_event & e = g_b2d_events[index];
    const int key = b2d_key(layer, expert);
    e.route = key >= 0 ? g_b2d_route[(size_t) key].load(std::memory_order_acquire) : 0;
    e.resolve = resolve;
    e.load_start = load_start;
    e.layer = layer;
    e.expert = expert;
    e.kind = kind;
    e.bytes = bytes;
    e.first_use.store(0, std::memory_order_relaxed);
    return (int) index;
}

static void b2d_end_miss(int index, int layer, int expert, uint64_t load_end) {
    if (index < 0 || (size_t) index >= B2D_MAX_EVENTS) return;
    g_b2d_events[(size_t) index].load_end = load_end;
    const int key = b2d_key(layer, expert);
    if (key >= 0) g_b2d_pending[(size_t) key].store((uint32_t) index + 1, std::memory_order_release);
}

extern "C" void ggml_expert_b2d_note_first_use(int layer_id, int expert_id) {
    if (!ggml_expert_b2d_enabled()) return;
    const int key = b2d_key(layer_id, expert_id);
    if (key < 0) return;
    const uint32_t token = g_b2d_pending[(size_t) key].exchange(0, std::memory_order_acq_rel);
    if (!token) return;
    const size_t index = (size_t) token - 1;
    if (index >= B2D_MAX_EVENTS) return;
    uint64_t expected = 0;
    g_b2d_events[index].first_use.compare_exchange_strong(expected, b2d_now(), std::memory_order_release, std::memory_order_relaxed);
}

static double b2d_us(uint64_t a, uint64_t b) {
    return a && b && b >= a ? 1000000.0 * (double) (b - a) / (double) g_b2d_freq : -1.0;
}

static void b2d_dump() {
    if (!ggml_expert_b2d_enabled() || g_b2d_path.empty()) return;
    FILE * f = nullptr;
    fopen_s(&f, g_b2d_path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "index,layer,expert,kind,bytes,route_to_resolve_us,route_to_load_start_us,route_to_first_use_us,load_us,load_end_to_first_use_us,resolve_to_first_use_us,route_ticks,resolve_ticks,load_start_ticks,load_end_ticks,first_use_ticks\n");
    const uint64_t count = std::min<uint64_t>(g_b2d_count.load(std::memory_order_relaxed), B2D_MAX_EVENTS);
    for (uint64_t i = 0; i < count; ++i) {
        const b2d_event & e = g_b2d_events[i];
        const uint64_t first = e.first_use.load(std::memory_order_acquire);
        std::fprintf(f, "%llu,%d,%d,%d,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%llu,%llu,%llu,%llu\n",
            (unsigned long long) i, e.layer, e.expert, e.kind, e.bytes,
            b2d_us(e.route, e.resolve), b2d_us(e.route, e.load_start), b2d_us(e.route, first),
            b2d_us(e.load_start, e.load_end), b2d_us(e.load_end, first), b2d_us(e.resolve, first),
            (unsigned long long) e.route, (unsigned long long) e.resolve,
            (unsigned long long) e.load_start, (unsigned long long) e.load_end, (unsigned long long) first);
    }
    std::fprintf(f, "#count=%llu,dropped=%llu,qpc_frequency=%llu\n",
        (unsigned long long) count, (unsigned long long) g_b2d_dropped.load(), (unsigned long long) g_b2d_freq);
    std::fclose(f);
}

struct b2d_exit_dumper { ~b2d_exit_dumper() { b2d_dump(); } };
static b2d_exit_dumper g_b2d_exit_dumper;

#if B1W_PROFILE
static constexpr int B1W_MAX_WORKERS = 64;
static std::atomic<int> g_b1w_worker_state[B1W_MAX_WORKERS] = {};
static std::atomic<bool> g_b1w_worker_registered[B1W_MAX_WORKERS] = {};
static std::atomic<bool> g_b1w_mmid_active[B1W_MAX_WORKERS] = {};
static std::atomic<uint64_t> g_b1w_registered_workers{0};
static std::atomic<uint64_t> g_b1w_sample_ticks{0};
static std::atomic<uint64_t> g_b1w_worker_state_reads{0};
static std::atomic<uint64_t> g_b1w_valid_state_reads{0};
static std::atomic<uint64_t> g_b1w_mmid_active_state_reads{0};
static std::atomic<uint64_t> g_b1w_state_samples[GGML_EXPERT_B1W_STATE_COUNT] = {};
static std::atomic<uint64_t> g_b1w_mmid_samples[GGML_EXPERT_B1W_STATE_COUNT] = {};
static std::atomic<bool> g_b1w_started{false};
static std::atomic<bool> g_b1w_stop{false};
static std::once_flag g_b1w_start_once;
static std::thread g_b1w_sampler;

static const char * b1w_state_name(int state) {
    static const char * names[GGML_EXPERT_B1W_STATE_COUNT] = {
        "outside_mmid", "mmid_setup", "mmid_other", "task_acquire", "scheduler", "wait_idle",
        "chunk_acquire", "chunk_prepare", "resolver", "external_compute", "accumulation",
        "unpin_release", "chunk_complete", "barrier_wait", "lock_wait", "io_wait", "gpu_wait", "unknown",
        "mmid_range_a", "mmid_range_b", "mmid_range_c", "mmid_range_d",
    };
    return state >= 0 && state < GGML_EXPERT_B1W_STATE_COUNT ? names[state] : "invalid";
}

static bool b1w_runtime_enabled() {
    static const bool enabled = []() {
        const char * p = std::getenv("LLAMA_EXPERT_B1W");
        return p && p[0] == '1';
    }();
    return enabled;
}

static void b1w_sampler_loop() {
    while (!g_b1w_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (g_b1w_stop.load(std::memory_order_relaxed)) break;
        g_b1w_sample_ticks.fetch_add(1, std::memory_order_relaxed);
        for (int w = 0; w < B1W_MAX_WORKERS; ++w) {
            if (!g_b1w_worker_registered[w].load(std::memory_order_relaxed)) continue;
            g_b1w_worker_state_reads.fetch_add(1, std::memory_order_relaxed);
            const int state = g_b1w_worker_state[w].load(std::memory_order_relaxed);
            if (state < 0 || state >= GGML_EXPERT_B1W_STATE_COUNT) continue;
            g_b1w_valid_state_reads.fetch_add(1, std::memory_order_relaxed);
            g_b1w_state_samples[state].fetch_add(1, std::memory_order_relaxed);
            if (g_b1w_mmid_active[w].load(std::memory_order_relaxed)) {
                g_b1w_mmid_active_state_reads.fetch_add(1, std::memory_order_relaxed);
                g_b1w_mmid_samples[state].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

static void b1w_write_stats(FILE * f) {
    if (!f) return;
    const uint64_t reads = g_b1w_worker_state_reads.load(std::memory_order_relaxed);
    const uint64_t valid = g_b1w_valid_state_reads.load(std::memory_order_relaxed);
    std::fprintf(f, "b1w_enabled=1\nb1w_registered_workers=%llu\nb1w_sample_ticks=%llu\nb1w_worker_state_reads=%llu\nb1w_valid_state_reads=%llu\nb1w_mmid_active_state_reads=%llu\nb1w_valid_sample_rate_pct=%.3f\n",
        (unsigned long long) g_b1w_registered_workers.load(),
        (unsigned long long) g_b1w_sample_ticks.load(),
        (unsigned long long) reads,
        (unsigned long long) valid,
        (unsigned long long) g_b1w_mmid_active_state_reads.load(),
        reads ? (100.0 * (double) valid / (double) reads) : 0.0);
    for (int s = 0; s < GGML_EXPERT_B1W_STATE_COUNT; ++s) {
        std::fprintf(f, "b1w_state_%s_samples=%llu\nb1w_mmid_%s_samples=%llu\n", b1w_state_name(s),
            (unsigned long long) g_b1w_state_samples[s].load(), b1w_state_name(s),
            (unsigned long long) g_b1w_mmid_samples[s].load());
    }
}
#endif

#if B1S_PROFILE
static constexpr int B1S_MAX_WORKERS = 64;
static constexpr int B1S_REGION_RESOLVER = 0;
static constexpr int B1S_REGION_COMPUTE = 1;
static constexpr int B1S_REGION_UNPIN = 2;
static std::atomic<uint32_t> g_b1s_active_mmid{0};
static std::atomic<uint64_t> g_b1s_mmid_union_start{0}, g_b1s_mmid_wall_ns{0};
static std::atomic<uint64_t> g_b1s_mmid_aggregate_ns{0};
static std::atomic<uint64_t> g_b1s_region_ns[3] = {};
static std::atomic<uint64_t> g_b1s_worker_mmid_ns[B1S_MAX_WORKERS] = {};
static std::atomic<uint64_t> g_b1s_worker_region_ns[B1S_MAX_WORKERS][3] = {};
static uint64_t b1s_now_ns() { return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#endif

#if B1T_PROFILE
static constexpr int B1T_MAX_PHASES = 8;
static std::atomic<uint64_t> g_b1t_worker_phase_ns[64][B1T_MAX_PHASES] = {};
static std::atomic<uint64_t> g_b1t_worker_total_ns[64] = {};
static uint64_t b1t_now_ns() { return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#endif

#if B1T_PROFILE
extern "C" int ggml_b1t_profile_enabled(void) { return 1; }
extern "C" uint64_t ggml_b1t_phase_begin(void) { return b1t_now_ns(); }
extern "C" void ggml_b1t_phase_end(int worker, int phase, uint64_t start_ns) {
    if (worker >= 0 && worker < 64 && phase >= 0 && phase < B1T_MAX_PHASES) {
        g_b1t_worker_phase_ns[worker][phase].fetch_add(b1t_now_ns() - start_ns, std::memory_order_relaxed);
    }
}
extern "C" void ggml_b1t_total_end(int worker, uint64_t start_ns) {
    if (worker >= 0 && worker < 64) g_b1t_worker_total_ns[worker].fetch_add(b1t_now_ns() - start_ns, std::memory_order_relaxed);
}
#else
extern "C" int ggml_b1t_profile_enabled(void) { return 0; }
extern "C" uint64_t ggml_b1t_phase_begin(void) { return 0; }
extern "C" void ggml_b1t_phase_end(int worker, int phase, uint64_t start_ns) { (void) worker; (void) phase; (void) start_ns; }
extern "C" void ggml_b1t_total_end(int worker, uint64_t start_ns) { (void) worker; (void) start_ns; }
#endif

#if B1W_PROFILE
extern "C" int ggml_expert_b1w_enabled(void) { return b1w_runtime_enabled() ? 1 : 0; }
extern "C" void ggml_expert_b1w_start(int worker_count) {
    (void) worker_count;
    if (!b1w_runtime_enabled()) return;
    std::call_once(g_b1w_start_once, []() {
        for (int w = 0; w < B1W_MAX_WORKERS; ++w) g_b1w_worker_state[w].store(GGML_EXPERT_B1W_OUTSIDE_MMID, std::memory_order_relaxed);
        g_b1w_stop.store(false, std::memory_order_relaxed);
        g_b1w_started.store(true, std::memory_order_release);
        g_b1w_sampler = std::thread(b1w_sampler_loop);
    });
}
extern "C" void ggml_expert_b1w_register_worker(int worker) {
    if (!g_b1w_started.load(std::memory_order_acquire) || worker < 0 || worker >= B1W_MAX_WORKERS) return;
    bool expected = false;
    if (g_b1w_worker_registered[worker].compare_exchange_strong(expected, true, std::memory_order_release, std::memory_order_relaxed)) {
        g_b1w_registered_workers.fetch_add(1, std::memory_order_relaxed);
    }
}
extern "C" void ggml_expert_b1w_set_state(int worker, int state) {
    if (!g_b1w_started.load(std::memory_order_acquire) || worker < 0 || worker >= B1W_MAX_WORKERS) return;
    g_b1w_worker_state[worker].store(state, std::memory_order_relaxed);
}
extern "C" void ggml_expert_b1w_set_mmid_active(int worker, int active) {
    if (!g_b1w_started.load(std::memory_order_acquire) || worker < 0 || worker >= B1W_MAX_WORKERS) return;
    g_b1w_mmid_active[worker].store(active != 0, std::memory_order_relaxed);
}
extern "C" void ggml_expert_b1w_shutdown(void) {
    if (!g_b1w_started.exchange(false, std::memory_order_acq_rel)) return;
    g_b1w_stop.store(true, std::memory_order_release);
    if (g_b1w_sampler.joinable()) g_b1w_sampler.join();
}
#else
extern "C" int ggml_expert_b1w_enabled(void) { return 0; }
extern "C" void ggml_expert_b1w_start(int worker_count) { (void) worker_count; }
extern "C" void ggml_expert_b1w_register_worker(int worker) { (void) worker; }
extern "C" void ggml_expert_b1w_set_state(int worker, int state) { (void) worker; (void) state; }
extern "C" void ggml_expert_b1w_set_mmid_active(int worker, int active) { (void) worker; (void) active; }
extern "C" void ggml_expert_b1w_shutdown(void) {}
#endif

static std::atomic<uint64_t> g_calls{0}, g_mismatches{0}, g_invalid{0}, g_metadata_failures{0};
static std::atomic<uint64_t> g_profile_resolver_samples{0}, g_profile_resolver_us{0};
static std::atomic<uint64_t> g_profile_hit_samples{0}, g_profile_hit_us{0};
static std::atomic<uint64_t> g_profile_miss_samples{0}, g_profile_miss_us{0};
static std::atomic<uint64_t> g_profile_mul_mat_calls{0}, g_profile_mul_mat_us{0};
static std::atomic<uint64_t> g_mmid_external_calls{0}, g_mmid_external_total_ns{0};
static std::atomic<uint64_t> g_mmid_region_ns[7] = {};
static std::atomic<uint64_t> g_mmid_vecdot_samples{0}, g_mmid_vecdot_sample_ns{0};
static thread_local uint32_t g_mmid_vecdot_local_seq = 0;
static thread_local uint32_t g_mmid_region_local_seq = 0;
static thread_local uint32_t g_b1k_local_seq = 0;
static std::atomic<uint64_t> g_b1k_ns[8] = {};
static std::atomic<uint64_t> g_b1k_calls[8] = {};
static std::atomic<uint64_t> g_b1l_ns[4] = {};
static std::atomic<uint64_t> g_b1l_calls[4] = {};
static std::atomic<uint64_t> g_b1l_branch_taken[4] = {};
static std::atomic<uint64_t> g_b1l_branch_not_taken[4] = {};
static std::atomic<uint64_t> g_b1m_external_storage_evals{0};
static std::atomic<uint64_t> g_b1n_ns[8] = {};
static std::atomic<uint64_t> g_b1n_sample_calls[8] = {};
static thread_local uint32_t g_b1n_local_seq = 0;
static std::atomic<uint64_t> g_b1p_resolver_samples{0}, g_b1p_hit_samples{0}, g_b1p_miss_samples{0};
static std::atomic<uint64_t> g_b1p_resolver_ns{0}, g_b1p_registry_ns{0}, g_b1p_metadata_ns{0};
static std::atomic<uint64_t> g_b1p_cache_hit_ns{0}, g_b1p_shared_gate_ns{0}, g_b1p_direct_map_ns{0};
static std::atomic<uint64_t> g_b1p_slot_validation_ns{0}, g_b1p_pin_ns{0}, g_b1p_hit_return_ns{0};
static std::atomic<uint64_t> g_b1p_miss_lock_ns{0}, g_b1p_lru_ns{0}, g_b1p_backing_read_ns{0};
static std::atomic<uint64_t> g_b1p_publication_ns{0};
static constexpr uint64_t B1P_SAMPLE_MASK = 4095;
static uint64_t g_b1n_timer_read_ns = 0;
static uint64_t g_b1n_empty_scope_ns = 0;
static uint64_t lock_now_ns();
static void b1n_calibrate_timers() {
    static std::once_flag once;
    std::call_once(once, []() {
        constexpr int n = 100000;
        volatile uint64_t sink = 0;
        uint64_t t0 = lock_now_ns();
        for (int i = 0; i < n; ++i) sink ^= lock_now_ns();
        uint64_t t1 = lock_now_ns();
        g_b1n_timer_read_ns = (t1 - t0) / (uint64_t) n;
        t0 = lock_now_ns();
        for (int i = 0; i < n; ++i) { const uint64_t a = lock_now_ns(); const uint64_t b = lock_now_ns(); sink ^= b - a; }
        t1 = lock_now_ns();
        g_b1n_empty_scope_ns = (t1 - t0) / (uint64_t) n;
        (void) sink;
    });
}
static thread_local bool g_profile_current_sample = false;
static thread_local int64_t g_profile_current_start_us = 0;
static bool ggml_expert_profile_env() {
#if B1R4_COMPILE_OUT_DIAGNOSTICS
    return false;
#else
    const char * p = std::getenv("LLAMA_EXPERT_TIMING"); return p && p[0] == '1';
#endif
}
static bool ggml_expert_b1p_profile_env() {
#if B1R4_COMPILE_OUT_DIAGNOSTICS
    return false;
#else
    static const bool enabled = []() { const char * p = std::getenv("LLAMA_EXPERT_B1P"); return p && p[0] == '1'; }(); return enabled;
#endif
}
static bool ggml_expert_lock_profile_env() {
#if B1R4_COMPILE_OUT_DIAGNOSTICS
    return false;
#else
    const char * p = std::getenv("LLAMA_EXPERT_LOCK_PROFILE"); return p && p[0] == '1';
#endif
}
static uint64_t lock_now_ns() { return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint64_t lock_contention_threshold_ns() {
    static const uint64_t threshold = []() -> uint64_t { const char * p = std::getenv("LLAMA_EXPERT_LOCK_THRESHOLD_NS"); if (!p) return 1000; char * e=nullptr; const unsigned long long v=std::strtoull(p,&e,10); return e && *e=='\0' ? (uint64_t)v : 1000; }();
    return threshold;
}
static void atomic_max_u64(std::atomic<uint64_t> & dst, uint64_t value) { uint64_t old=dst.load(std::memory_order_relaxed); while(old<value && !dst.compare_exchange_weak(old,value,std::memory_order_relaxed)) {} }
enum class cache_lock_path : uint32_t { unknown, hit, miss, unpin };
static std::atomic<uint64_t> g_lock_acquisitions{0}, g_lock_wait_ns{0}, g_lock_max_wait_ns{0}, g_lock_critical_count{0}, g_lock_critical_ns{0}, g_lock_max_critical_ns{0}, g_lock_contended{0};
static std::atomic<uint64_t> g_hit_lock_acquisitions{0}, g_hit_lock_wait_ns{0}, g_hit_lock_critical_ns{0};
static std::atomic<uint64_t> g_miss_lock_acquisitions{0}, g_miss_lock_wait_ns{0}, g_miss_lock_critical_ns{0};
static std::atomic<uint64_t> g_unpin_lock_acquisitions{0}, g_unpin_lock_wait_ns{0}, g_unpin_lock_critical_ns{0};
static std::atomic<uint64_t> g_lockless_hit_attempts{0}, g_lockless_hit_successes{0};
static std::atomic<uint64_t> g_lockless_validation_failures{0}, g_lockless_generation_failures{0}, g_lockless_state_failures{0};
static std::atomic<uint64_t> g_lockless_post_pin_revalidation_failures{0}, g_lockless_hit_retries{0}, g_lockless_to_locked_fallbacks{0};
struct profiled_cache_lock {
    std::unique_lock<std::mutex> lock;
    const bool profiled;
    cache_lock_path path = cache_lock_path::unknown;
    uint64_t wait_start = 0, wait_ns = 0, critical_start = 0;
    explicit profiled_cache_lock(std::mutex & mutex, cache_lock_path p = cache_lock_path::unknown) : lock(mutex, std::defer_lock), profiled(ggml_expert_lock_profile_env()), path(p) {
        if (profiled) wait_start = lock_now_ns();
        lock.lock();
        if (profiled) { const uint64_t acquired = lock_now_ns(); wait_ns = acquired - wait_start; critical_start = acquired; }
    }
    void set_path(cache_lock_path p) { path = p; }
    ~profiled_cache_lock() {
        if (!profiled) return;
        const uint64_t critical_ns = lock_now_ns() - critical_start;
        lock.unlock();
        ++g_lock_acquisitions; g_lock_wait_ns += wait_ns; ++g_lock_critical_count; g_lock_critical_ns += critical_ns;
        atomic_max_u64(g_lock_max_wait_ns, wait_ns); atomic_max_u64(g_lock_max_critical_ns, critical_ns);
        if (wait_ns > lock_contention_threshold_ns()) ++g_lock_contended;
        if (path == cache_lock_path::hit) { ++g_hit_lock_acquisitions; g_hit_lock_wait_ns += wait_ns; g_hit_lock_critical_ns += critical_ns; }
        else if (path == cache_lock_path::miss) { ++g_miss_lock_acquisitions; g_miss_lock_wait_ns += wait_ns; g_miss_lock_critical_ns += critical_ns; }
        else if (path == cache_lock_path::unpin) { ++g_unpin_lock_acquisitions; g_unpin_lock_wait_ns += wait_ns; g_unpin_lock_critical_ns += critical_ns; }
    }
};
static std::atomic<uint64_t> g_registry_hits{0}, g_registry_misses{0};
static std::set<int> g_layers;
static std::set<std::string> g_kinds;
static std::mutex g_mutex;
static std::string g_path;
static std::unordered_map<const ggml_tensor *, ggml_expert_storage_info> g_storage;
static std::unordered_map<std::string, ggml_expert_storage_info> g_storage_by_name;
static std::unordered_map<std::string, std::string> g_storage_name_owners;
static std::unordered_map<std::string, const ggml_tensor *> g_runtime_by_name;
static std::mutex g_storage_mutex;
static uint64_t g_registry_duplicates = 0;
static bool parse_metadata(const char * name, int & layer, const char * & kind);

// C1a: per-thread cache for the identity hit path of ggml_expert_storage_lookup.
// Only identity hits are cached; they just copy the registry entry and have no
// side effects. Every registry change bumps g_registry_epoch under g_storage_mutex,
// and a thread with a stale epoch drops its entries and takes the locked path.
// Cached payloads live in g_registry_pool and are never freed, so they cannot dangle.
static std::atomic<uint64_t> g_registry_epoch{1};
static std::deque<ggml_expert_storage_info> g_registry_pool;   // stable addresses; guarded by g_storage_mutex
static std::unordered_map<const ggml_tensor *, const ggml_expert_storage_info *> g_registry_pool_index;
static bool same_storage_info(const ggml_expert_storage_info & a, const ggml_expert_storage_info & b) {
    return a.tensor == b.tensor && a.name == b.name && a.layer == b.layer && a.kind == b.kind &&
        a.file_index == b.file_index && a.os_handle == b.os_handle && a.absolute_offset == b.absolute_offset &&
        a.plane_stride == b.plane_stride && a.plane_size == b.plane_size && a.tensor_size == b.tensor_size &&
        a.ggml_type == b.ggml_type && a.n_experts == b.n_experts && a.cpu_override == b.cpu_override;
}
// Returns a stable immutable copy of info. Pooled entries are never mutated or freed,
// so a cached pointer stays valid even if the registry later changes. Needs g_storage_mutex.
static const ggml_expert_storage_info * registry_pool_intern(const ggml_tensor * tensor, const ggml_expert_storage_info & info) {
    auto it = g_registry_pool_index.find(tensor);
    if (it != g_registry_pool_index.end() && same_storage_info(*it->second, info)) return it->second;
    g_registry_pool.push_back(info);
    const ggml_expert_storage_info * stable = &g_registry_pool.back();
    g_registry_pool_index[tensor] = stable;
    return stable;
}
static constexpr size_t IDENTITY_CACHE_SLOTS = 1024;           // power of two
static constexpr size_t IDENTITY_CACHE_MASK  = IDENTITY_CACHE_SLOTS - 1;
struct identity_cache_entry {
    const ggml_tensor * key = nullptr;
    const ggml_expert_storage_info * info = nullptr;
};
static thread_local identity_cache_entry g_identity_cache[IDENTITY_CACHE_SLOTS];
static thread_local uint64_t g_identity_cache_epoch = 0;
static inline size_t identity_cache_index(const ggml_tensor * tensor) {
    uint64_t h = (uint64_t) (uintptr_t) tensor;
    h *= UINT64_C(0x9E3779B97F4A7C15);
    return (size_t) (h >> 40) & IDENTITY_CACHE_MASK;
}
// Per-thread hot counters.  Registered once per thread so the report can sum them
// without putting a contended atomic on the hot path.
struct hot_counters {
    uint64_t identity_fast_hits = 0;
    uint64_t identity_locked_lookups = 0;
    uint64_t identity_cache_flushes = 0;
    uint64_t row_bounds_checks = 0;
};
static std::mutex g_hot_counter_mutex;
static std::vector<hot_counters *> g_hot_counter_blocks;
static hot_counters g_hot_counter_retired;   // sums from threads that have exited
// g_tl_counters is a POD so the hot path reads it with no thread_local init guard.
// The registrar is only touched on the slow path, which always runs first on a thread.
static thread_local hot_counters g_tl_counters;
struct hot_counter_registrar {
    hot_counter_registrar() {
        std::lock_guard<std::mutex> lock(g_hot_counter_mutex);
        g_hot_counter_blocks.push_back(&g_tl_counters);
    }
    ~hot_counter_registrar() {
        std::lock_guard<std::mutex> lock(g_hot_counter_mutex);
        g_hot_counter_retired.identity_fast_hits      += g_tl_counters.identity_fast_hits;
        g_hot_counter_retired.identity_locked_lookups += g_tl_counters.identity_locked_lookups;
        g_hot_counter_retired.identity_cache_flushes  += g_tl_counters.identity_cache_flushes;
        g_hot_counter_retired.row_bounds_checks       += g_tl_counters.row_bounds_checks;
        for (auto it = g_hot_counter_blocks.begin(); it != g_hot_counter_blocks.end(); ++it) {
            if (*it == &g_tl_counters) { g_hot_counter_blocks.erase(it); break; }
        }
    }
};
// Function-local thread_local: guaranteed to init on first use and destruct at thread exit.
static void hot_counters_register_this_thread() {
    static thread_local hot_counter_registrar registrar;
    (void) registrar;
}
static hot_counters hot_counter_totals() {
    std::lock_guard<std::mutex> lock(g_hot_counter_mutex);
    hot_counters total = g_hot_counter_retired;
    for (const hot_counters * block : g_hot_counter_blocks) {
        total.identity_fast_hits      += block->identity_fast_hits;
        total.identity_locked_lookups += block->identity_locked_lookups;
        total.identity_cache_flushes  += block->identity_cache_flushes;
        total.row_bounds_checks       += block->row_bounds_checks;
    }
    return total;
}
// Must be called with g_storage_mutex held.
static void registry_mutated(void) {
    g_registry_epoch.fetch_add(1, std::memory_order_release);
}
// Publish a settled lookup result into this thread's cache. Needs g_storage_mutex.
static void identity_cache_publish(const ggml_tensor * tensor, const ggml_expert_storage_info & info) {
    const ggml_expert_storage_info * stable = registry_pool_intern(tensor, info);
    const size_t i = identity_cache_index(tensor);
    const size_t j = (i + 1) & IDENTITY_CACHE_MASK;
    const size_t target = (g_identity_cache[i].key == nullptr || g_identity_cache[i].key == tensor) ? i : j;
    g_identity_cache[target].key = tensor;
    g_identity_cache[target].info = stable;
}

extern "C" void ggml_expert_storage_context_invalidated(const void * mem_buffer, size_t mem_size, const char * reason) {
    if (!mem_buffer || mem_size == 0) return;
    const uintptr_t begin = (uintptr_t) mem_buffer;
    if (mem_size > UINTPTR_MAX - begin) return;
    const uintptr_t end = begin + mem_size;
    std::lock_guard<std::mutex> lock(g_storage_mutex);
    size_t removed_identity = 0;
    size_t removed_runtime = 0;
    for (auto it = g_storage.begin(); it != g_storage.end();) {
        const uintptr_t p = (uintptr_t) it->first;
        if (p >= begin && p < end) {
            it = g_storage.erase(it);
            ++removed_identity;
        } else {
            ++it;
        }
    }
    for (auto it = g_runtime_by_name.begin(); it != g_runtime_by_name.end();) {
        const uintptr_t p = (uintptr_t) it->second;
        if (p >= begin && p < end) {
            it = g_runtime_by_name.erase(it);
            ++removed_runtime;
        } else {
            ++it;
        }
    }
    if (removed_identity || removed_runtime) {
        registry_mutated();
        std::fprintf(stderr,
            "B1B_RUNTIME_CONTEXT_INVALIDATED reason=%s mem_buffer=%p mem_size=%zu removed_identity=%zu removed_runtime=%zu remaining_runtime=%zu\n",
            reason ? reason : "unknown", mem_buffer, mem_size, removed_identity, removed_runtime, g_runtime_by_name.size());
        std::fflush(stderr);
    }
}
static constexpr uint64_t CACHE_SLOT_WRITE_BIT = UINT64_C(1) << 63;
struct cache_slot {
    int layer=-1, expert=-1;
    std::atomic<int> pin{0};
    std::atomic<uint64_t> last{0};
    std::atomic<uint64_t> generation{1};
    std::atomic<uint64_t> readers{0};
    bool valid=false;
    cache_slot() = default;
    cache_slot(const cache_slot & other) : layer(other.layer), expert(other.expert), pin(other.pin.load()), last(other.last.load()), generation(other.generation.load()), readers(other.readers.load()), valid(other.valid) {}
    cache_slot & operator=(const cache_slot & other) { layer=other.layer; expert=other.expert; pin.store(other.pin.load()); last.store(other.last.load()); generation.store(other.generation.load()); readers.store(other.readers.load()); valid=other.valid; return *this; }
};
static std::mutex g_cache_mutex;
static std::shared_mutex g_cache_read_gate;
static std::vector<uint8_t> g_cache_slab;
static std::vector<cache_slot> g_cache_slots;
// Direct map entries are immutable publication tokens for the lock-light hit path.
// Low 32 bits contain slot+1; high 32 bits contain the slot generation.
static std::unique_ptr<std::atomic<uint64_t>[]> g_direct_slot_map;
static constexpr size_t g_direct_slot_map_count = 256 * 256;
static std::atomic<uint64_t> g_direct_lookup_hits{0}, g_direct_lookup_misses{0};
static std::atomic<uint64_t> g_stale_mapping_detections{0};
static uint64_t g_mapping_invalidations = 0, g_mapping_publications = 0, g_fallback_scans = 0;
static std::atomic<uint64_t> g_cache_clock{0}, g_cache_hits{0}, g_cache_pins{0}, g_cached_returns{0};
static uint64_t g_cache_misses=0, g_cache_loads=0, g_cache_evictions=0, g_cache_bytes=0, g_cache_peak=0, g_cache_unpins=0, g_cache_bad_unpins=0, g_cache_pinned_evictions=0;
static uint64_t g_normal_returns=0, g_direct_read_failures=0, g_resolver_failures=0;
static std::atomic<uint64_t> g_resolver_success{0}, g_lookup_return_count{0}, g_external_vec_dot_calls{0}, g_reserved_pointer_violations{0};
static uint64_t g_read_ops=0, g_read_bytes=0, g_read_latency_total_us=0, g_read_latency_min_us=UINT64_MAX, g_read_latency_max_us=0;
static uint64_t & g_read_latency_count = g_read_ops;
static uint64_t g_kind_ops[3] = {}, g_kind_bytes[3] = {}, g_kind_latency_total_us[3] = {}, g_kind_latency_min_us[3] = { UINT64_MAX, UINT64_MAX, UINT64_MAX }, g_kind_latency_max_us[3] = {}, g_short_reads = 0;
static std::string g_perf_path;
static size_t g_cache_bundle=0, g_cache_gate=0, g_cache_up=0, g_cache_down=0;
static std::atomic<bool> g_cache_ready{false};
static uint64_t g_geometry_failures=0, g_pointer_range_failures=0;
static uint64_t g_row_bounds_checks=0, g_row_bounds_failures=0, g_row_offset_overflows=0, g_row_end_overflows=0;
static uint64_t g_original_pointer_uses=0, g_cached_pointer_uses=0, g_unexpected_mmap_fallbacks=0;

static bool g_row_failure_recorded=false;
static std::atomic<uint64_t> g_external_tensors{0}, g_external_logical_bytes{0}, g_external_reserved_bytes{0}, g_external_committed_bytes{0};
static std::atomic<uint64_t> g_external_get_attempts{0}, g_external_set_attempts{0}, g_external_copy_attempts{0}, g_external_clear_attempts{0};
static std::atomic<uint64_t> g_external_view_attempts{0}, g_external_validation_attempts{0}, g_external_buffer_failures{0}, g_unexpected_mmap_expert_bindings{0};
static std::atomic<uint64_t> g_placement_registered{0}, g_placement_cpu_candidates{0}, g_placement_gpu_candidates{0};
static std::atomic<uint64_t> g_external_expected{0}, g_external_created{0}, g_cpu_mmap_bound{0}, g_gpu_mmap_bound{0};
static std::atomic<uint64_t> g_unexpected_cpu_mmap{0}, g_unexpected_external_gpu{0};
static std::atomic<uint64_t> g_placement_kind[3] = {};
static bool b1b_sample_milestone(uint64_t n) {
    return n <= 10 || n == 100 || n == 1000 || n == 10000 || n == 100000 || n == 1000000 ||
        (n && (n & (n - 1)) == 0);
}
static bool b1b_sample_lookup_return() {
    return b1b_sample_milestone(g_lookup_return_count.fetch_add(1, std::memory_order_relaxed) + 1);
}
struct row_bounds_failure { int layer=-1, expert=-1, slot=-1, pin=-1; int64_t ir0=0; size_t nb01=0, nb02=0, ne0=0, ne1=0, ne2=0, row_offset=0, required=0; uintptr_t plane_begin=0, plane_end=0, cached_ptr=0, dot_ptr=0, dot_end=0; const char * vec_dot=nullptr; };
static row_bounds_failure g_row_failure;
// Breadcrumb state must be per-thread.  The validation path is concurrent;
// process-global breadcrumb fields can otherwise report another thread's
// stage and make the crash location misleading.
static thread_local uint64_t g_b1b_breadcrumb_seq = 0;
static thread_local uint32_t g_b1b_breadcrumb_stage = 0, g_b1b_breadcrumb_substage = 0;
static thread_local uint32_t g_b1b_breadcrumb_layer = UINT32_MAX, g_b1b_breadcrumb_kind = UINT32_MAX, g_b1b_breadcrumb_expert = UINT32_MAX;
static thread_local uintptr_t g_b1b_breadcrumb_tensor = 0;
static std::atomic<uintptr_t> g_b1b_prov[10] = {};
static std::atomic<uintptr_t> g_b1b_expected_tensor{0}, g_b1b_expected_buffer{0};
static void b1b_prov(uint32_t slot, uintptr_t value) {
    if (slot < 10) g_b1b_prov[slot].store(value, std::memory_order_relaxed);
}
extern "C" void ggml_expert_b1b_prov(uint32_t slot, uintptr_t value) {
    b1b_prov(slot, value);
}
struct b1b_buffer_record { uintptr_t ptr = 0; uintptr_t tensor = 0; uint32_t id = 0; bool freed = false; };
static b1b_buffer_record g_b1b_buffers[128] = {};
static std::atomic<uint32_t> g_b1b_buffer_next_id{1};
static void b1b_breadcrumb(uint32_t stage, uint32_t substage = 0, int layer = -1, int kind = -1,
        int expert = -1, const ggml_tensor * tensor = nullptr) {
    g_b1b_breadcrumb_stage = stage;
    g_b1b_breadcrumb_substage = substage;
    g_b1b_breadcrumb_layer = layer < 0 ? UINT32_MAX : (uint32_t) layer;
    g_b1b_breadcrumb_kind = kind < 0 ? UINT32_MAX : (uint32_t) kind;
    g_b1b_breadcrumb_expert = expert < 0 ? UINT32_MAX : (uint32_t) expert;
    g_b1b_breadcrumb_tensor = (uintptr_t) tensor;
    ++g_b1b_breadcrumb_seq;
}
extern "C" void ggml_expert_b1b_helper_breadcrumb(uint32_t stage, uint32_t substage) {
    b1b_breadcrumb(stage, substage);
}
extern "C" void ggml_expert_b1b_helper_breadcrumb_ptr(uint32_t stage, uint32_t substage, uintptr_t ptr) {
    b1b_breadcrumb(stage, substage, -1, -1, -1, (const ggml_tensor *) ptr);
}
extern "C" uint32_t ggml_expert_b1b_buffer_created(uintptr_t ptr, uintptr_t tensor) {
    if (!ptr) return 0;
    if (!g_b1b_expected_buffer.load(std::memory_order_relaxed)) {
        g_b1b_expected_buffer.store(ptr, std::memory_order_relaxed);
    }
    for (auto & r : g_b1b_buffers) if (r.ptr == ptr) { r.tensor = tensor; r.freed = false; return r.id; }
    for (auto & r : g_b1b_buffers) if (!r.ptr) {
        r.ptr = ptr; r.tensor = tensor; r.id = g_b1b_buffer_next_id.fetch_add(1); r.freed = false; return r.id;
    }
    return 0;
}
extern "C" void ggml_expert_b1b_buffer_freed(uintptr_t ptr) {
    if (!ptr) return;
    for (auto & r : g_b1b_buffers) if (r.ptr == ptr) { r.freed = true; return; }
}
extern "C" uint32_t ggml_expert_b1b_buffer_state(uintptr_t ptr) {
    if (!ptr) return 0;
    for (const auto & r : g_b1b_buffers) if (r.ptr == ptr) return r.freed ? 2u : 1u;
    return 0;
}
enum class b1b_ring_event : uint32_t {
    lookup_enter, identity_find_begin, identity_find_end, name_find_begin, name_find_end,
    lookup_not_found, lookup_found, metadata_begin, metadata_end, validation_dispatch_begin,
    validate_enter, external_check_begin, external_check_end, external_skip_content,
    validation_note_begin, validation_note_end, validate_return, lookup_return, load_stage,
    post_meta_begin, post_meta_tensor_state_begin, post_meta_tensor_state_end,
    post_meta_runtime_size_begin, post_meta_runtime_size_end,
    post_meta_placement_begin, post_meta_placement_end,
    post_meta_validation_begin, post_meta_validation_end,
    post_meta_result_begin, post_meta_result_end, post_meta_before_return,
};
struct b1b_ring_record {
    std::atomic<uint64_t> published{0};
    uint64_t sequence=0;
    uint32_t thread_id=0;
    b1b_ring_event event=b1b_ring_event::load_stage;
    const ggml_tensor * tensor=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    const void * data=nullptr;
    int layer=-1, kind=-1, expert=-1;
    int external=0;
    uint64_t name_hash=0;
};
static constexpr size_t B1B_RING_CAPACITY = 128;
static b1b_ring_record g_b1b_ring[B1B_RING_CAPACITY];
static std::atomic<uint64_t> g_b1b_ring_next{0};
static uint64_t b1b_ring_hash(const char * name) {
    uint64_t h = 1469598103934665603ull;
    if (name) while (*name) { h ^= (unsigned char) *name++; h *= 1099511628211ull; }
    return h;
}
static void b1b_ring_record_event(b1b_ring_event event, const ggml_tensor * tensor = nullptr,
        ggml_backend_buffer_t buffer = nullptr, const void * data = nullptr,
        int layer = -1, int kind = -1, int expert = -1, int external = 0, const char * name = nullptr) {
    const uint64_t sequence = g_b1b_ring_next.fetch_add(1, std::memory_order_relaxed);
    b1b_ring_record & r = g_b1b_ring[sequence % B1B_RING_CAPACITY];
    r.published.store(0, std::memory_order_relaxed);
    r.sequence = sequence;
#ifdef _WIN32
    r.thread_id = (uint32_t) GetCurrentThreadId();
#else
    r.thread_id = 0;
#endif
    r.event = event; r.tensor = tensor; r.buffer = buffer; r.data = data;
    r.layer = layer; r.kind = kind; r.expert = expert; r.external = external; r.name_hash = b1b_ring_hash(name);
    r.published.store(sequence + 1, std::memory_order_release);
}
#define b1b_ring_record_event(...) ((void) 0)
#ifdef _WIN32
static LONG g_b1b_veh_entered = 0;
static std::atomic<uint64_t> g_b1b_veh_count{0}, g_b1b_av_count{0}, g_b1b_cpp_eh_count{0};
static void b1b_veh_write(const char * text) {
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    if (!h || h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(h, text, (DWORD) std::strlen(text), &written, nullptr);
    FlushFileBuffers(h);
}
static void b1b_dump_breadcrumb(DWORD code, ULONG_PTR access, ULONG_PTR target, void * address) {
    char line[1536] = {};
    _snprintf_s(line, sizeof(line), _TRUNCATE,
        "B1B_BREADCRUMB pid=%lu tid=%lu code=0x%08lX access=%s target=%p exception_address=%p seq=%llu stage=%lu substage=%lu layer=%lu kind=%lu expert=%lu tensor=%p P1=%p P2=%p P3=%p P4=%p P5=%p P6=%p P7=%p P8=%p P9=%p P10=%p expected_tensor=%p expected_buffer=%p\n",
        (unsigned long) GetCurrentProcessId(), (unsigned long) GetCurrentThreadId(),
        (unsigned long) code, access == 0 ? "READ" : access == 1 ? "WRITE" : access == 8 ? "EXECUTE" : "UNKNOWN",
        (void *) target, address,
        (unsigned long long) g_b1b_breadcrumb_seq,
        (unsigned long) g_b1b_breadcrumb_stage,
        (unsigned long) g_b1b_breadcrumb_substage,
        (unsigned long) g_b1b_breadcrumb_layer,
        (unsigned long) g_b1b_breadcrumb_kind,
        (unsigned long) g_b1b_breadcrumb_expert,
        (void *) g_b1b_breadcrumb_tensor,
        (void *) g_b1b_prov[0].load(), (void *) g_b1b_prov[1].load(), (void *) g_b1b_prov[2].load(),
        (void *) g_b1b_prov[3].load(), (void *) g_b1b_prov[4].load(), (void *) g_b1b_prov[5].load(),
        (void *) g_b1b_prov[6].load(), (void *) g_b1b_prov[7].load(), (void *) g_b1b_prov[8].load(),
        (void *) g_b1b_prov[9].load(), (void *) g_b1b_expected_tensor.load(), (void *) g_b1b_expected_buffer.load());
    b1b_veh_write(line);
}
static const char * b1b_ring_event_name(b1b_ring_event event) {
    switch (event) {
        case b1b_ring_event::lookup_enter: return "LOOKUP_ENTER";
        case b1b_ring_event::identity_find_begin: return "IDENTITY_FIND_BEGIN";
        case b1b_ring_event::identity_find_end: return "IDENTITY_FIND_END";
        case b1b_ring_event::name_find_begin: return "NAME_FIND_BEGIN";
        case b1b_ring_event::name_find_end: return "NAME_FIND_END";
        case b1b_ring_event::lookup_not_found: return "LOOKUP_NOT_FOUND";
        case b1b_ring_event::lookup_found: return "LOOKUP_FOUND";
        case b1b_ring_event::metadata_begin: return "METADATA_BEGIN";
        case b1b_ring_event::metadata_end: return "METADATA_END";
        case b1b_ring_event::validation_dispatch_begin: return "VALIDATION_DISPATCH_BEGIN";
        case b1b_ring_event::validate_enter: return "VALIDATE_ENTER";
        case b1b_ring_event::external_check_begin: return "EXTERNAL_CHECK_BEGIN";
        case b1b_ring_event::external_check_end: return "EXTERNAL_CHECK_END";
        case b1b_ring_event::external_skip_content: return "EXTERNAL_SKIP_CONTENT";
        case b1b_ring_event::validation_note_begin: return "VALIDATION_NOTE_BEGIN";
        case b1b_ring_event::validation_note_end: return "VALIDATION_NOTE_END";
        case b1b_ring_event::validate_return: return "VALIDATE_RETURN";
        case b1b_ring_event::lookup_return: return "LOOKUP_RETURN";
        case b1b_ring_event::load_stage: return "LOAD_STAGE";
        case b1b_ring_event::post_meta_begin: return "POST_META_BEGIN";
        case b1b_ring_event::post_meta_tensor_state_begin: return "POST_META_TENSOR_STATE_BEGIN";
        case b1b_ring_event::post_meta_tensor_state_end: return "POST_META_TENSOR_STATE_END";
        case b1b_ring_event::post_meta_runtime_size_begin: return "POST_META_RUNTIME_SIZE_BEGIN";
        case b1b_ring_event::post_meta_runtime_size_end: return "POST_META_RUNTIME_SIZE_END";
        case b1b_ring_event::post_meta_placement_begin: return "POST_META_PLACEMENT_BEGIN";
        case b1b_ring_event::post_meta_placement_end: return "POST_META_PLACEMENT_END";
        case b1b_ring_event::post_meta_validation_begin: return "POST_META_VALIDATION_BEGIN";
        case b1b_ring_event::post_meta_validation_end: return "POST_META_VALIDATION_END";
        case b1b_ring_event::post_meta_result_begin: return "POST_META_RESULT_BEGIN";
        case b1b_ring_event::post_meta_result_end: return "POST_META_RESULT_END";
        case b1b_ring_event::post_meta_before_return: return "POST_META_BEFORE_RETURN";
    }
    return "UNKNOWN";
}
static void b1b_dump_ring() {
    const uint64_t next = g_b1b_ring_next.load(std::memory_order_acquire);
    const uint64_t first = next > B1B_RING_CAPACITY ? next - B1B_RING_CAPACITY : 0;
    b1b_veh_write("B1B_RING_BEGIN\n");
    for (uint64_t sequence = first; sequence < next; ++sequence) {
        const b1b_ring_record & r = g_b1b_ring[sequence % B1B_RING_CAPACITY];
        if (r.published.load(std::memory_order_acquire) != sequence + 1) continue;
        char line[384] = {};
        _snprintf_s(line, sizeof(line), _TRUNCATE,
            "B1B_RING seq=%llu tid=%lu event=%s tensor=%p buffer=%p data=%p layer=%d kind=%d expert=%d external=%d name_hash=0x%llX\n",
            (unsigned long long) r.sequence, (unsigned long) r.thread_id, b1b_ring_event_name(r.event),
            (const void *) r.tensor, (void *) r.buffer, r.data, r.layer, r.kind, r.expert, r.external,
            (unsigned long long) r.name_hash);
        b1b_veh_write(line);
    }
    b1b_veh_write("B1B_RING_END\n");
}
static LONG CALLBACK b1b_vectored_exception_reporter(EXCEPTION_POINTERS * ep) {
    if (!ep || !ep->ExceptionRecord) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    ++g_b1b_veh_count;
    if (code == EXCEPTION_ACCESS_VIOLATION) ++g_b1b_av_count;
    if (code == 0xE06D7363) ++g_b1b_cpp_eh_count;
    if (code != EXCEPTION_ACCESS_VIOLATION && InterlockedExchange(&g_b1b_veh_entered, 1) != 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const ULONG_PTR * info = ep->ExceptionRecord->ExceptionInformation;
    const ULONG_PTR access = code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2 ? info[0] : 3;
    const ULONG_PTR target = code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2 ? info[1] : 0;
    b1b_dump_breadcrumb(code, access, target, ep->ExceptionRecord->ExceptionAddress);
    char line[256] = {};
    _snprintf_s(line, sizeof(line), _TRUNCATE,
        "B1B_VEH pid=%lu tid=%lu code=0x%08lX access=%s target=%p exception_address=%p\n",
        (unsigned long) GetCurrentProcessId(), (unsigned long) GetCurrentThreadId(),
        (unsigned long) code, access == 0 ? "READ" : access == 1 ? "WRITE" : access == 8 ? "EXECUTE" : "UNKNOWN",
        (void *) target, ep->ExceptionRecord->ExceptionAddress);
    b1b_veh_write(line);
    return EXCEPTION_CONTINUE_SEARCH;
}
static LONG WINAPI b1b_exception_reporter(EXCEPTION_POINTERS * ep) {
    if (ep && ep->ExceptionRecord) {
        const DWORD code = ep->ExceptionRecord->ExceptionCode;
        const ULONG_PTR * info = ep->ExceptionRecord->ExceptionInformation;
        const ULONG_PTR access = code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2 ? info[0] : 3;
        const ULONG_PTR target = code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2 ? info[1] : 0;
        std::fprintf(stderr, "B1B_EXCEPTION code=0x%08lX access=%s target=%p exception_address=%p thread_id=%lu\n",
            (unsigned long) code, access == 0 ? "READ" : access == 1 ? "WRITE" : access == 8 ? "EXECUTE" : "UNKNOWN",
            (void *) target, ep->ExceptionRecord->ExceptionAddress, (unsigned long) GetCurrentThreadId());
        std::fflush(stderr);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
static const bool g_b1b_exception_reporter_installed = []() {
    AddVectoredExceptionHandler(1, b1b_vectored_exception_reporter);
    SetUnhandledExceptionFilter(b1b_exception_reporter);
    return true;
}();
#endif
static bool cache_on() {
    static const bool enabled = []() {
        const char * p = std::getenv("LLAMA_EXPERT_CACHE");
        if (p && p[0] == '1') return true;
        const char * external = std::getenv("B1B_EXTERNAL_EXPERT_STORAGE");
        return external && external[0] == '1';
    }();
    return enabled;
}
static bool b1b_lookup_logging_on() {
    static const bool enabled = []() {
        const char * p = std::getenv("B1B_LOOKUP_LOGGING");
        return p && p[0] == '1';
    }();
    return enabled;
}
#define B1B_LOOKUP_LOG(...) do { if (b1b_lookup_logging_on()) { std::fprintf(stderr, __VA_ARGS__); } } while (0)
#define B1B_LOOKUP_FLUSH() do { if (b1b_lookup_logging_on()) { std::fflush(stderr); } } while (0)
static bool pointer_trace_on() {
    static const bool enabled = []() {
        const char * p = std::getenv("B1B_TRACE_POINTER_FLOW");
        return p && p[0] == '1';
    }();
    return enabled;
}
static bool pointer_trace_take_enabled() { static std::atomic<uint64_t> n{0}; return n.fetch_add(1, std::memory_order_relaxed) < 96; }
static void cache_fail(const char * stage, int layer, int expert) { std::fprintf(stderr, "CACHE_FAIL stage=%s layer=%d expert=%d\n", stage, layer, expert); }
static bool safe_add(size_t a, size_t b, size_t & out) { if (b > SIZE_MAX - a) return false; out = a + b; return true; }
static bool safe_mul(size_t a, size_t b, size_t & out) { if (a && b > SIZE_MAX / a) return false; out = a * b; return true; }
static bool safe_addr_add(uintptr_t a, size_t b, uintptr_t & out) { if (b > UINTPTR_MAX - a) return false; out = a + b; return true; }
static int direct_slot_key(int layer, int expert) {
    return layer >= 0 && layer < 256 && expert >= 0 && expert < 256 ? layer * 256 + expert : -1;
}
static void direct_slot_invalidate(int layer, int expert, int slot) {
    const int key = direct_slot_key(layer, expert);
    if (key >= 0 && g_direct_slot_map) {
        const uint64_t token = g_direct_slot_map[(size_t) key].load(std::memory_order_acquire);
        const int mapped = token ? (int)((token & UINT64_C(0xffffffff)) - 1) : -1;
        if (mapped != slot) return;
        g_direct_slot_map[(size_t) key].store(0, std::memory_order_release);
        ++g_mapping_invalidations;
    }
}
static void direct_slot_publish(int layer, int expert, int slot) {
    const int key = direct_slot_key(layer, expert);
    if (key >= 0 && g_direct_slot_map && slot >= 0 && (size_t) slot < g_cache_slots.size()) {
        const uint64_t generation = g_cache_slots[(size_t) slot].generation.load(std::memory_order_acquire);
        const uint64_t token = (generation << 32) | (uint64_t)(slot + 1);
        g_direct_slot_map[(size_t) key].store(token, std::memory_order_release);
        ++g_mapping_publications;
    }
}
static bool cache_slot_read_enter(cache_slot & slot) {
    uint64_t state = slot.readers.load(std::memory_order_acquire);
    for (;;) {
        if (state & CACHE_SLOT_WRITE_BIT) return false;
        if (slot.readers.compare_exchange_weak(state, state + 1, std::memory_order_acquire, std::memory_order_relaxed)) return true;
    }
}
static void cache_slot_read_exit(cache_slot & slot) { slot.readers.fetch_sub(1, std::memory_order_release); }
static bool cache_slot_write_try_enter(cache_slot & slot) {
    uint64_t expected = 0;
    return slot.readers.compare_exchange_strong(expected, CACHE_SLOT_WRITE_BIT, std::memory_order_acquire, std::memory_order_relaxed);
}
static void cache_slot_write_exit(cache_slot & slot) { slot.readers.store(0, std::memory_order_release); }
static int direct_token_slot(uint64_t token) { return token ? (int)((token & UINT64_C(0xffffffff)) - 1) : -1; }
static uint64_t direct_token_generation(uint64_t token) { return token >> 32; }
static bool read_plane(const ggml_expert_storage_info & info, int expert, uint8_t * dst, int kind) {
#ifdef _WIN32
    LARGE_INTEGER fq={}, begin={}, end={}; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&begin);
    HANDLE dup=INVALID_HANDLE_VALUE; if(!DuplicateHandle(GetCurrentProcess(),(HANDLE)(uintptr_t)info.os_handle,GetCurrentProcess(),&dup,0,FALSE,DUPLICATE_SAME_ACCESS)) return false;
    LARGE_INTEGER pos; pos.QuadPart=(LONGLONG)(info.absolute_offset+(uint64_t)expert*info.plane_stride); bool ok=SetFilePointerEx(dup,pos,nullptr,FILE_BEGIN); DWORD got=0; const DWORD requested=(DWORD)info.plane_size; if(ok) ok=ReadFile(dup,dst,requested,&got,nullptr)&&got==requested; CloseHandle(dup); QueryPerformanceCounter(&end); uint64_t us=uint64_t((end.QuadPart-begin.QuadPart)*1000000/fq.QuadPart); ++g_read_ops; g_read_bytes+=got; g_read_latency_total_us+=us; g_read_latency_min_us=std::min(g_read_latency_min_us,us); g_read_latency_max_us=std::max(g_read_latency_max_us,us); if(kind>=0&&kind<3){++g_kind_ops[kind];g_kind_bytes[kind]+=got;g_kind_latency_total_us[kind]+=us;g_kind_latency_min_us[kind]=std::min(g_kind_latency_min_us[kind],us);g_kind_latency_max_us[kind]=std::max(g_kind_latency_max_us[kind],us);} if(got!=requested) ++g_short_reads; return ok;
#else
    return false;
#endif
}

// C7: the three planes of a bundle are read in parallel on independent file handles.
// Concurrent reads that share one file object are serialised by the kernel (measured:
// 7.10 GB/s on 1 thread, 6.23 GB/s on 12). Separate handles scale: 6.77 GB/s on 1 thread
// to 16.70 GB/s on 8. Each reader therefore owns a handle opened by CreateFile.
// This runs inside the existing writer gate, so it adds no new sharing of cache state.
static std::mutex g_read_stats_mutex;   // reads now run concurrently, so their stats need a lock

static bool read_plane_handle(HANDLE h, const ggml_expert_storage_info & info, int expert, uint8_t * dst, int kind) {
#ifdef _WIN32
    if (h == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER fq={}, begin={}, end={}; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&begin);
    const uint64_t offset = info.absolute_offset + (uint64_t) expert * info.plane_stride;
    OVERLAPPED ov = {};
    ov.Offset = (DWORD) (offset & 0xFFFFFFFFull);
    ov.OffsetHigh = (DWORD) (offset >> 32);
    DWORD got = 0;
    const DWORD requested = (DWORD) info.plane_size;
    const bool ok = ReadFile(h, dst, requested, &got, &ov) && got == requested;
    QueryPerformanceCounter(&end);
    const uint64_t us = uint64_t((end.QuadPart-begin.QuadPart)*1000000/fq.QuadPart);
    std::lock_guard<std::mutex> stats_lock(g_read_stats_mutex);
    ++g_read_ops; g_read_bytes+=got; g_read_latency_total_us+=us;
    g_read_latency_min_us=std::min(g_read_latency_min_us,us); g_read_latency_max_us=std::max(g_read_latency_max_us,us);
    if(kind>=0&&kind<3){++g_kind_ops[kind];g_kind_bytes[kind]+=got;g_kind_latency_total_us[kind]+=us;g_kind_latency_min_us[kind]=std::min(g_kind_latency_min_us[kind],us);g_kind_latency_max_us[kind]=std::max(g_kind_latency_max_us[kind],us);}
    if(got!=requested) ++g_short_reads;
    return ok;
#else
    (void) h; (void) info; (void) expert; (void) dst; (void) kind; return false;
#endif
}

#ifdef _WIN32
struct plane_job {
    const ggml_expert_storage_info * info = nullptr;
    int expert = 0;
    uint8_t * dst = nullptr;
    int kind = -1;
    bool ok = false;
};
// One helper per extra plane. The caller reads one plane itself, so two helpers cover a
// three plane bundle. Helpers only ever run while the caller waits, inside the gate.
struct plane_helper {
    HANDLE handle = INVALID_HANDLE_VALUE;
    std::mutex m;
    std::condition_variable cv;
    plane_job job;
    bool has_job = false;
    bool done = true;
    bool quit = false;

    void loop() {
        for (;;) {
            plane_job local;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&]{ return has_job || quit; });
                if (quit) return;
                local = job;
            }
            const bool ok = read_plane_handle(handle, *local.info, local.expert, local.dst, local.kind);
            {
                std::lock_guard<std::mutex> lock(m);
                job.ok = ok;
                has_job = false;
                done = true;
            }
            cv.notify_all();
        }
    }
    void submit(const plane_job & j) {
        {
            std::lock_guard<std::mutex> lock(m);
            job = j; has_job = true; done = false;
        }
        cv.notify_one();
    }
    bool wait() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]{ return done; });
        return job.ok;
    }
};

static HANDLE g_bundle_handle = INVALID_HANDLE_VALUE;   // used by the calling thread
static plane_helper g_plane_helpers[2];
static bool g_bundle_readers_ready = false;
static uint64_t g_parallel_bundle_loads = 0;

// Called with g_cache_mutex held. Opens independent file objects so plane reads can
// actually run in parallel; falls back to the original serial path if anything fails.
static bool ensure_bundle_readers(const ggml_expert_storage_info & info) {
    if (g_bundle_readers_ready) return g_bundle_handle != INVALID_HANDLE_VALUE;
    g_bundle_readers_ready = true;
    wchar_t path[32768];
    const DWORD n = GetFinalPathNameByHandleW((HANDLE)(uintptr_t) info.os_handle, path, 32767, FILE_NAME_NORMALIZED);
    if (n == 0 || n >= 32767) { std::fprintf(stderr, "C7_BUNDLE_READER_PATH_FAILED err=%lu\n", GetLastError()); return false; }
    path[n] = 0;
    auto open_one = [&]() {
        return CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    };
    g_bundle_handle = open_one();
    if (g_bundle_handle == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "C7_BUNDLE_READER_OPEN_FAILED err=%lu\n", GetLastError()); return false; }
    for (auto & helper : g_plane_helpers) {
        helper.handle = open_one();
        if (helper.handle == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "C7_BUNDLE_HELPER_OPEN_FAILED err=%lu\n", GetLastError()); CloseHandle(g_bundle_handle); g_bundle_handle = INVALID_HANDLE_VALUE; return false; }
        std::thread([&helper]{ helper.loop(); }).detach();   // detached: never joinable at exit
    }
    std::fprintf(stderr, "C7_BUNDLE_READERS_READY helpers=2\n");
    return true;
}
#endif

// Reads the gate/up/down planes of one expert. Uses parallel handles when available.
static bool read_bundle(const ggml_expert_storage_info infos[3], int expert, uint8_t * base, size_t gate_bytes, size_t up_bytes) {
#ifdef _WIN32
    if (ensure_bundle_readers(infos[0])) {
        plane_job j1; j1.info=&infos[1]; j1.expert=expert; j1.dst=base+gate_bytes;            j1.kind=1;
        plane_job j2; j2.info=&infos[2]; j2.expert=expert; j2.dst=base+gate_bytes+up_bytes;   j2.kind=2;
        g_plane_helpers[0].submit(j1);
        g_plane_helpers[1].submit(j2);
        const bool ok0 = read_plane_handle(g_bundle_handle, infos[0], expert, base, 0);
        const bool ok1 = g_plane_helpers[0].wait();
        const bool ok2 = g_plane_helpers[1].wait();
        ++g_parallel_bundle_loads;
        return ok0 && ok1 && ok2;
    }
#endif
    return read_plane(infos[0],expert,base,0) && read_plane(infos[1],expert,base+gate_bytes,1) && read_plane(infos[2],expert,base+gate_bytes+up_bytes,2);
}

static bool ensure_cache_layout() {
    if (g_cache_ready.load(std::memory_order_acquire)) return true;
    ggml_expert_storage_info a={},b={},c={};
    for(const auto & x:g_storage_by_name){ if(x.second.layer==0){ if(x.second.kind==0)a=x.second; if(x.second.kind==1)b=x.second; if(x.second.kind==2)c=x.second; } }
    if(!a.plane_size||!b.plane_size||!c.plane_size) return false;
    g_cache_gate=a.plane_size; g_cache_up=b.plane_size; g_cache_down=c.plane_size;
    size_t gate_up=0; if (!safe_add(g_cache_gate, g_cache_up, gate_up) || !safe_add(gate_up, g_cache_down, g_cache_bundle)) { ++g_geometry_failures; return false; }
    size_t budget = 256ull*1024*1024;
    if (const char * p = std::getenv("LLAMA_EXPERT_CACHE_MB")) {
        char * end = nullptr;
        const unsigned long long mb = std::strtoull(p, &end, 10);
        if (end != p && *end == '\0' && mb > 0 && mb <= SIZE_MAX / (1024ull*1024ull)) {
            budget = (size_t) mb * 1024ull * 1024ull;
        }
    }
    size_t slab_bytes=0; if (!g_cache_bundle || !safe_mul(budget / g_cache_bundle, g_cache_bundle, slab_bytes)) { ++g_geometry_failures; return false; }
    const size_t n=budget/g_cache_bundle; if(!n) return false;
    g_cache_slab.resize(slab_bytes); g_cache_slots.resize(n);
    if (!g_direct_slot_map) {
        g_direct_slot_map = std::make_unique<std::atomic<uint64_t>[]>(g_direct_slot_map_count);
        for (size_t i = 0; i < g_direct_slot_map_count; ++i) g_direct_slot_map[i].store(0, std::memory_order_relaxed);
    }
    const uintptr_t slab_begin=(uintptr_t)g_cache_slab.data(); uintptr_t slab_end=0;
    if (!safe_addr_add(slab_begin, slab_bytes, slab_end)) { ++g_geometry_failures; return false; }
    bool geometry_ok=true;
    for (size_t i=0; i<n; ++i) {
        size_t slot_off=0, slot_end_off=0; uintptr_t slot_begin=0, slot_end=0;
        if (!safe_mul(i, g_cache_bundle, slot_off) || !safe_add(slot_off, g_cache_bundle, slot_end_off) ||
            !safe_addr_add(slab_begin, slot_off, slot_begin) || !safe_addr_add(slab_begin, slot_end_off, slot_end) ||
            slot_begin < slab_begin || slot_end > slab_end || slot_begin >= slot_end) { geometry_ok=false; break; }
        size_t up_off=0, down_off=0, down_end_off=0;
        if (g_cache_gate > g_cache_bundle || !safe_add(g_cache_gate, g_cache_up, up_off) ||
            !safe_add(up_off, g_cache_down, down_end_off) || down_end_off > g_cache_bundle ||
            !safe_add(g_cache_gate, g_cache_up, down_off) || down_off > g_cache_bundle ||
            g_cache_up > g_cache_bundle - g_cache_gate || g_cache_down > g_cache_bundle - down_off) { geometry_ok=false; break; }
    }
    if (!geometry_ok) { ++g_geometry_failures; std::fprintf(stderr, "CACHE_GEOMETRY_FAIL slot geometry\n"); return false; }
    if (const char * p = std::getenv("LLAMA_EXPERT_GEOMETRY_OUT")) {
        FILE * f = nullptr;
#ifdef _WIN32
        fopen_s(&f, p, "w");
#else
        f = std::fopen(p, "w");
#endif
        if (f) {
            std::fprintf(f, "requested_budget_bytes=%zu\ncache_gate_bytes=%zu\ncache_up_bytes=%zu\ncache_down_bytes=%zu\ncache_bundle_bytes=%zu\nslot_stride_bytes=%zu\nslot_count=%zu\nactual_capacity_bytes=%zu\nslab_allocated_bytes=%zu\nslab_begin=0x%llx\nslab_end=0x%llx\nslot_metadata_size=%zu\nslot_metadata_bytes=%zu\nslot_alignment_requirement=%zu\nslab_alignment_mod_16=%llu\nslab_alignment_mod_32=%llu\noffset_gate=0\noffset_up=%zu\noffset_down=%zu\ngeometry_invariant_failures=%llu\n", budget, g_cache_gate, g_cache_up, g_cache_down, g_cache_bundle, g_cache_bundle, n, slab_bytes, slab_bytes, (unsigned long long)slab_begin, (unsigned long long)slab_end, sizeof(cache_slot), n*sizeof(cache_slot), alignof(cache_slot), (unsigned long long)(slab_begin%16), (unsigned long long)(slab_begin%32), g_cache_gate, g_cache_gate+g_cache_up, (unsigned long long)g_geometry_failures);
            for (const auto & x : g_storage_by_name) {
                if (x.second.layer == 0 || x.second.layer == 1 || x.second.layer == 15 || x.second.layer == 29) {
                    std::fprintf(f, "registry name=%s layer=%d kind=%d plane_size=%llu plane_stride=%llu tensor_size=%llu\n", x.first.c_str(), x.second.layer, x.second.kind, (unsigned long long) x.second.plane_size, (unsigned long long) x.second.plane_stride, (unsigned long long) x.second.tensor_size);
                }
            }
            std::fclose(f);
        }
    }
    g_cache_ready.store(true, std::memory_order_release);
    return true;
}
static const char * cache_slot_ptr(int slot, int layer, int expert, const char * kind) {
    if (slot < 0 || (size_t) slot >= g_cache_slots.size() || !g_cache_slots[slot].valid ||
        g_cache_slots[slot].layer != layer || g_cache_slots[slot].expert != expert) {
        ++g_pointer_range_failures; cache_fail("pointer_slot", layer, expert); return nullptr;
    }
    const size_t kind_off = kind[0] == 'g' ? 0 : kind[0] == 'u' ? g_cache_gate : g_cache_gate + g_cache_up;
    const size_t kind_size = kind[0] == 'g' ? g_cache_gate : kind[0] == 'u' ? g_cache_up : g_cache_down;
    size_t slot_off=0, ptr_off=0, ptr_end_off=0; uintptr_t begin=0, end=0, ptr=0, ptr_end=0;
    if (!safe_mul((size_t) slot, g_cache_bundle, slot_off) || !safe_add(slot_off, kind_off, ptr_off) ||
        !safe_add(ptr_off, kind_size, ptr_end_off) || ptr_end_off > g_cache_slab.size() ||
        !safe_addr_add((uintptr_t) g_cache_slab.data(), slot_off, begin) ||
        !safe_addr_add((uintptr_t) g_cache_slab.data(), g_cache_slab.size(), end) ||
        !safe_addr_add((uintptr_t) g_cache_slab.data(), ptr_off, ptr) ||
        !safe_addr_add((uintptr_t) g_cache_slab.data(), ptr_end_off, ptr_end) ||
        ptr < begin || ptr_end > end || ptr >= ptr_end) {
        ++g_pointer_range_failures; cache_fail("pointer_range", layer, expert); return nullptr;
    }
    return (const char *) ptr;
}

static const char * cache_resolve(const ggml_expert_storage_info * storage, int expert, size_t stride, bool b1p_sample, bool b2d_sample, uint64_t b2d_resolve) {
    const uint64_t b1p_cache_start = b1p_sample ? lock_now_ns() : 0;
    if (!storage || expert < 0 || expert >= 256) { cache_fail("storage", storage ? storage->layer : -1, expert); return nullptr; }
    const int layer = storage->layer;
    const char * kind = storage->kind == 0 ? "gate" : storage->kind == 1 ? "up" : storage->kind == 2 ? "down" : nullptr;
    if (!kind) { cache_fail("storage_kind", layer, expert); return nullptr; }
    if (!g_cache_ready.load(std::memory_order_acquire)) { profiled_cache_lock init_lock(g_cache_mutex); std::unique_lock<std::shared_mutex> init_gate(g_cache_read_gate); if (!ensure_cache_layout()) { cache_fail("layout",layer,expert); return nullptr; } }
    // B1i-A read side: shared gate permits concurrent hits while the writer
    // gate remains exclusive for miss/load/eviction.  Holding this gate across
    // pin acquisition prevents reuse until the returned pointer is pinned.
    const uint64_t b1p_gate_start = b1p_sample ? lock_now_ns() : 0;
    std::shared_lock<std::shared_mutex> read_gate(g_cache_read_gate);
    if (b1p_sample) g_b1p_shared_gate_ns += lock_now_ns() - b1p_gate_start;
    ++g_lockless_hit_attempts;
    int slot=-1;
    const int key = direct_slot_key(layer, expert);
    const uint64_t b1p_direct_start = b1p_sample ? lock_now_ns() : 0;
    const uint64_t b1p_validation_start = b1p_sample ? lock_now_ns() : 0;
    if (key >= 0) {
        const uint64_t token = g_direct_slot_map ? g_direct_slot_map[(size_t) key].load(std::memory_order_acquire) : 0;
        const int mapped = direct_token_slot(token);
        const bool generation_ok = mapped >= 0 && (size_t) mapped < g_cache_slots.size() && direct_token_generation(token) == g_cache_slots[(size_t) mapped].generation.load(std::memory_order_acquire);
        if (mapped >= 0 && (size_t) mapped < g_cache_slots.size() && generation_ok && g_cache_slots[(size_t) mapped].valid &&
            g_cache_slots[(size_t) mapped].layer == layer && g_cache_slots[(size_t) mapped].expert == expert) {
            slot = mapped;
            ++g_direct_lookup_hits;
        } else {
            if (mapped >= 0 && !generation_ok) ++g_lockless_generation_failures;
            if (mapped >= 0) { ++g_stale_mapping_detections; if (g_direct_slot_map) g_direct_slot_map[(size_t) key].store(0, std::memory_order_release); }
            ++g_direct_lookup_misses;
        }
    } else {
        ++g_direct_lookup_misses;
    }
    if (b1p_sample) { g_b1p_slot_validation_ns += lock_now_ns() - b1p_validation_start; g_b1p_direct_map_ns += lock_now_ns() - b1p_direct_start; }
    if(slot>=0){++g_lockless_hit_successes;++g_cache_hits; const uint64_t b1p_pin_start = b1p_sample ? lock_now_ns() : 0; g_cache_slots[slot].last.store(++g_cache_clock,std::memory_order_relaxed);g_cache_slots[slot].pin.fetch_add(1,std::memory_order_acq_rel);++g_cache_pins;++g_cached_returns; if (b1p_sample) { ++g_b1p_hit_samples; g_b1p_pin_ns += lock_now_ns() - b1p_pin_start; } if(g_profile_current_sample){++g_profile_hit_samples;g_profile_hit_us+=std::max<int64_t>(0,ggml_time_us()-g_profile_current_start_us);} const uint64_t b1p_return_start = b1p_sample ? lock_now_ns() : 0; const char * result = cache_slot_ptr(slot,layer,expert,kind); if (b1p_sample) { g_b1p_hit_return_ns += lock_now_ns() - b1p_return_start; g_b1p_cache_hit_ns += lock_now_ns() - b1p_cache_start; } return result;}
    ++g_lockless_validation_failures; ++g_lockless_hit_retries; ++g_lockless_to_locked_fallbacks;
    if (b1p_sample) ++g_b1p_miss_samples;
    const uint64_t b1p_miss_lock_start = b1p_sample ? lock_now_ns() : 0;
    read_gate.unlock();
    profiled_cache_lock lock(g_cache_mutex); std::unique_lock<std::shared_mutex> write_gate(g_cache_read_gate); if(!ensure_cache_layout()){cache_fail("layout",layer,expert);return nullptr;}
    if (b1p_sample) g_b1p_miss_lock_ns += lock_now_ns() - b1p_miss_lock_start;
    slot=-1;
    if (key >= 0) {
        const uint64_t token = g_direct_slot_map ? g_direct_slot_map[(size_t) key].load(std::memory_order_acquire) : 0;
        const int mapped = direct_token_slot(token);
        if (mapped >= 0 && (size_t) mapped < g_cache_slots.size() && g_cache_slots[(size_t) mapped].valid && g_cache_slots[(size_t) mapped].layer == layer && g_cache_slots[(size_t) mapped].expert == expert) slot=mapped;
    }
    if(slot>=0){lock.set_path(cache_lock_path::hit);++g_cache_hits;g_cache_slots[slot].last.store(++g_cache_clock,std::memory_order_relaxed);g_cache_slots[slot].pin.fetch_add(1,std::memory_order_acq_rel);++g_cache_pins;++g_cached_returns;if(g_profile_current_sample){++g_profile_hit_samples;g_profile_hit_us+=std::max<int64_t>(0,ggml_time_us()-g_profile_current_start_us);}return cache_slot_ptr(slot,layer,expert,kind);}
    const uint64_t b1p_lru_start = b1p_sample ? lock_now_ns() : 0;
    uint64_t oldest=UINT64_MAX; for(int i=0;i<(int)g_cache_slots.size();++i){if(!g_cache_slots[i].valid){slot=i;break;}if(g_cache_slots[i].pin.load(std::memory_order_acquire)==0&&g_cache_slots[i].last.load(std::memory_order_relaxed)<oldest){oldest=g_cache_slots[i].last.load(std::memory_order_relaxed);slot=i;}}
    if (b1p_sample) g_b1p_lru_ns += lock_now_ns() - b1p_lru_start;
    lock.set_path(cache_lock_path::miss); ++g_cache_misses; if(slot<0){++g_cache_pinned_evictions;cache_fail("no_slot",layer,expert);return nullptr;} if(g_cache_slots[slot].valid)++g_cache_evictions;
    ggml_expert_storage_info infos[3]={}; char n[128]; const char * ss[3]={"gate","up","down"}; for(int k=0;k<3;++k){std::snprintf(n,sizeof(n),"blk.%d.ffn_%s_exps.weight",layer,ss[k]);auto it=g_storage_by_name.find(n);if(it==g_storage_by_name.end()){cache_fail("storage",layer,expert);return nullptr;}infos[k]=it->second;}
    uint8_t * base=g_cache_slab.data()+slot*g_cache_bundle; const uint64_t b1p_read_start = b1p_sample ? lock_now_ns() : 0; const uint64_t b2d_load_start = b2d_sample ? b2d_now() : 0; const int b2d_index = b2d_sample ? b2d_begin_miss(layer, expert, storage->kind, (uint32_t) g_cache_bundle, b2d_resolve, b2d_load_start) : -1; if(!read_bundle(infos,expert,base,g_cache_gate,g_cache_up)){++g_direct_read_failures;cache_fail("read_bundle",layer,expert);return nullptr;} if (b1p_sample) g_b1p_backing_read_ns += lock_now_ns() - b1p_read_start; if (b2d_index >= 0) b2d_end_miss(b2d_index, layer, expert, b2d_now());
    if (g_cache_slots[slot].valid) direct_slot_invalidate(g_cache_slots[slot].layer, g_cache_slots[slot].expert, slot);
    const uint64_t b1p_publish_start = b1p_sample ? lock_now_ns() : 0; g_cache_slots[slot].layer=layer; g_cache_slots[slot].expert=expert; g_cache_slots[slot].last.store(++g_cache_clock,std::memory_order_relaxed); g_cache_slots[slot].pin.store(1,std::memory_order_release); g_cache_slots[slot].valid=true; g_cache_slots[slot].generation.fetch_add(1,std::memory_order_release); direct_slot_publish(layer, expert, slot); ++g_cache_loads;g_cache_bytes+=g_cache_bundle;g_cache_peak=std::max(g_cache_peak,(uint64_t)std::count_if(g_cache_slots.begin(),g_cache_slots.end(),[](const cache_slot&s){return s.valid;}));++g_cache_pins;++g_cached_returns; if (b1p_sample) g_b1p_publication_ns += lock_now_ns() - b1p_publish_start; if(g_profile_current_sample){++g_profile_miss_samples;g_profile_miss_us+=std::max<int64_t>(0,ggml_time_us()-g_profile_current_start_us);}return cache_slot_ptr(slot,layer,expert,kind);
}
extern "C" void ggml_expert_cache_unpin(const ggml_tensor * tensor, int expert) {
    if (!cache_on()) return;
    int layer = -1;
    const char * kind = nullptr;
    if (!parse_metadata(ggml_get_name(tensor), layer, kind)) return;
    profiled_cache_lock lock(g_cache_mutex, cache_lock_path::unpin);
    std::unique_lock<std::shared_mutex> write_gate(g_cache_read_gate);
    int slot = -1;
    const int key = direct_slot_key(layer, expert);
    if (key >= 0) {
        const uint64_t token = g_direct_slot_map ? g_direct_slot_map[(size_t) key].load(std::memory_order_acquire) : 0;
        const int mapped = direct_token_slot(token);
        if (mapped >= 0 && (size_t) mapped < g_cache_slots.size() &&
            g_cache_slots[(size_t) mapped].valid &&
            g_cache_slots[(size_t) mapped].layer == layer &&
            g_cache_slots[(size_t) mapped].expert == expert) {
            slot = mapped;
        } else if (mapped >= 0) {
            ++g_stale_mapping_detections;
            if (g_direct_slot_map) g_direct_slot_map[(size_t) key].store(0, std::memory_order_release);
        }
    }
    if (slot < 0) {
        ++g_fallback_scans;
        for (int i = 0; i < (int) g_cache_slots.size(); ++i) {
            if (g_cache_slots[(size_t) i].valid &&
                g_cache_slots[(size_t) i].layer == layer &&
                g_cache_slots[(size_t) i].expert == expert) {
                slot = i;
                break;
            }
        }
    }
    if (slot >= 0) {
        cache_slot & slot_ref = g_cache_slots[(size_t) slot];
        if (slot_ref.pin.load(std::memory_order_acquire) > 0) {
            slot_ref.pin.fetch_sub(1, std::memory_order_acq_rel);
            ++g_cache_unpins;
        } else {
            ++g_cache_bad_unpins;
        }
        return;
    }
}
static std::once_flag g_validation_once;
static std::atomic<bool> g_validation_started{false};
static std::atomic<bool> g_validation_completed{false};
static uint64_t g_validation_pass = 0, g_validation_fail = 0, g_validation_bytes = 0;
static bool g_validated[40][3][256] = {};
static std::mutex g_validation_mutex;
static thread_local bool g_trace_validation_current = false;

static void validate_current(const ggml_tensor * tensor, const ggml_expert_storage_info & info, int expert_id) {
    if (g_trace_validation_current) {
        std::fprintf(stderr,
            "B1B_VALIDATE_CALL_STATE tensor=%p field=%p buffer=%p data=%p view_src=%p name=%s\\n",
            (const void *) tensor, tensor ? (const void *) &tensor->buffer : nullptr,
            tensor ? (void *) tensor->buffer : nullptr, tensor ? tensor->data : nullptr,
            tensor ? (const void *) tensor->view_src : nullptr,
            tensor ? ggml_get_name(tensor) : "<null>");
        std::fflush(stderr);
    }
    b1b_prov(4, (uintptr_t) tensor);
    b1b_breadcrumb(160, 0, info.layer, info.kind, expert_id, tensor);
    b1b_breadcrumb(160, 10, info.layer, info.kind, expert_id, tensor);
    b1b_breadcrumb(160, 11, info.layer, info.kind, expert_id, tensor);
    b1b_breadcrumb(160, 12, info.layer, info.kind, expert_id, tensor);
    b1b_breadcrumb(160, 13, info.layer, info.kind, expert_id, tensor);
    static bool validation_loop_entered = false;
    if (!validation_loop_entered) {
        validation_loop_entered = true;
    }
    b1b_breadcrumb(160, 14, info.layer, info.kind, expert_id, tensor);
    const ggml_expert_storage_info info_copy = info;
    b1b_breadcrumb(160, 15, info.layer, info.kind, expert_id, tensor);
    const ggml_tensor * tensor_ptr = tensor;
    b1b_prov(5, (uintptr_t) tensor_ptr);
    b1b_breadcrumb(160, 16, info.layer, info.kind, expert_id, tensor);
    ggml_backend_buffer_t buffer_ptr = tensor_ptr ? tensor_ptr->buffer : nullptr;
    b1b_prov(3, (uintptr_t) buffer_ptr);
    b1b_prov(5, (uintptr_t) buffer_ptr);
    b1b_breadcrumb(160, 17, info.layer, info.kind, expert_id, tensor);
    void * external_context = nullptr;
    void * external_base = nullptr;
    size_t external_reserved = 0;
    b1b_prov(6, (uintptr_t) buffer_ptr);
    b1b_breadcrumb(160, 20, info.layer, info.kind, expert_id, tensor);
    const bool buffer_external = buffer_ptr &&
        ggml_backend_buffer_external_expert_details(buffer_ptr, &external_context, &external_base, &external_reserved);
    b1b_breadcrumb(160, 21, info.layer, info.kind, expert_id, tensor);
    b1b_breadcrumb(170, buffer_external ? 1 : 0, info_copy.layer, info_copy.kind, expert_id, tensor);
    b1b_breadcrumb(160, 22, info.layer, info.kind, expert_id, tensor);
    if (info_copy.layer < 0 || info_copy.layer >= 40 || info_copy.kind < 0 || info_copy.kind >= 3) {
        b1b_breadcrumb(160, 23, info.layer, info.kind, expert_id, tensor);
        return;
    }
    b1b_breadcrumb(160, 24, info.layer, info.kind, expert_id, tensor);
    if (buffer_external) {
        b1b_breadcrumb(160, 30, info.layer, info.kind, expert_id, tensor);
        b1b_breadcrumb(160, 31, info.layer, info.kind, expert_id, tensor);
        ggml_expert_external_note_validation(tensor);
        b1b_breadcrumb(160, 32, info.layer, info.kind, expert_id, tensor);
        b1b_breadcrumb(180, 0, info_copy.layer, info_copy.kind, expert_id, tensor);
        return;
    }
    b1b_breadcrumb(160, 40, info.layer, info.kind, expert_id, tensor);
    if (!tensor || !tensor->data) {
        b1b_breadcrumb(160, 41, info.layer, info.kind, expert_id, tensor);
        return;
    }
    static const int wanted[][2] = {{0,0},{0,255},{15,127},{29,0},{29,255}};
    b1b_breadcrumb(160, 42, info.layer, info.kind, expert_id, tensor);
    bool selected = false;
    for (const auto & x : wanted) if (x[0] == info.layer && x[1] == expert_id) selected = true;
    b1b_breadcrumb(160, 43, info.layer, info.kind, expert_id, tensor);
    if (!selected) return;
    b1b_breadcrumb(160, 44, info.layer, info.kind, expert_id, tensor);
    std::lock_guard<std::mutex> lock(g_validation_mutex);
    b1b_breadcrumb(160, 45, info.layer, info.kind, expert_id, tensor);
    if (g_validated[info.layer][info.kind][expert_id]) return;
    b1b_breadcrumb(160, 46, info.layer, info.kind, expert_id, tensor);
    const uint64_t file_offset = info.absolute_offset + (uint64_t) expert_id * info.plane_stride;
    b1b_breadcrumb(160, 47, info.layer, info.kind, expert_id, tensor);
    std::vector<uint8_t> direct(info.plane_size);
    b1b_breadcrumb(160, 48, info.layer, info.kind, expert_id, tensor);
    bool ok = false;
#ifdef _WIN32
    HANDLE dup = INVALID_HANDLE_VALUE;
    b1b_breadcrumb(160, 50, info.layer, info.kind, expert_id, tensor);
    ok = DuplicateHandle(GetCurrentProcess(), (HANDLE)(uintptr_t)info.os_handle, GetCurrentProcess(), &dup, 0, FALSE, DUPLICATE_SAME_ACCESS);
    b1b_breadcrumb(160, 51, info.layer, info.kind, expert_id, tensor);
    LARGE_INTEGER pos; pos.QuadPart = (LONGLONG) file_offset;
    b1b_breadcrumb(160, 52, info.layer, info.kind, expert_id, tensor);
    if (ok) ok = SetFilePointerEx(dup, pos, nullptr, FILE_BEGIN);
    b1b_breadcrumb(160, 53, info.layer, info.kind, expert_id, tensor);
    DWORD got = 0;
    BOOL read_ok = FALSE;
    b1b_breadcrumb(160, 54, info.layer, info.kind, expert_id, tensor);
    if (ok) read_ok = ReadFile(dup, direct.data(), (DWORD) direct.size(), &got, nullptr);
    b1b_breadcrumb(160, 55, info.layer, info.kind, expert_id, tensor);
    ok = ok && read_ok && got == direct.size();
    b1b_breadcrumb(160, 56, info.layer, info.kind, expert_id, tensor);
    if (dup != INVALID_HANDLE_VALUE) CloseHandle(dup);
    b1b_breadcrumb(160, 57, info.layer, info.kind, expert_id, tensor);
#else
    const int fd = dup(info.file_id);
    ok = fd >= 0 && lseek(fd, (off_t) file_offset, SEEK_SET) >= 0;
    if (ok) ok = read(fd, direct.data(), direct.size()) == (ssize_t) direct.size();
    if (fd >= 0) close(fd);
#endif
    b1b_breadcrumb(160, 58, info.layer, info.kind, expert_id, tensor);
    if (!ok) {
        ++g_validation_fail;
        g_validated[info.layer][info.kind][expert_id] = true;
        b1b_breadcrumb(160, 59, info.layer, info.kind, expert_id, tensor);
        return;
    }
    b1b_breadcrumb(160, 60, info.layer, info.kind, expert_id, tensor);
    const uint8_t * mapped = (const uint8_t *) tensor->data + (size_t)expert_id * info.plane_stride;
    b1b_breadcrumb(160, 61, info.layer, info.kind, expert_id, tensor);
    const bool mapped_external = ggml_backend_buffer_external_expert_details(tensor->buffer, &external_context, &external_base, &external_reserved);
    b1b_breadcrumb(160, 62, info.layer, info.kind, expert_id, tensor);
    const uintptr_t comparison_addr = (uintptr_t) mapped;
    const uintptr_t external_begin = (uintptr_t) external_base;
    const uintptr_t external_end = external_begin && external_reserved <= UINTPTR_MAX - external_begin ? external_begin + external_reserved : 0;
    const bool comparison_in_external_range = external_begin && external_end > external_begin && comparison_addr >= external_begin && comparison_addr < external_end;
    b1b_breadcrumb(160, 63, info.layer, info.kind, expert_id, tensor);
    if (mapped_external) {
        ++g_validation_fail;
        g_validated[info.layer][info.kind][expert_id] = true;
        b1b_breadcrumb(160, 64, info.layer, info.kind, expert_id, tensor);
        return;
    }
    b1b_breadcrumb(160, 65, info.layer, info.kind, expert_id, tensor);
    const int cmp = std::memcmp(direct.data(), mapped, info.plane_size);
    b1b_breadcrumb(160, 66, info.layer, info.kind, expert_id, tensor);
    if (cmp == 0) { ++g_validation_pass; g_validation_bytes += info.plane_size; }
    else ++g_validation_fail;
    g_validated[info.layer][info.kind][expert_id] = true;
    b1b_breadcrumb(160, 67, info.layer, info.kind, expert_id, tensor);
}

/* legacy bulk validator is intentionally unused; validation is performed against runtime tensors below */
static void validate_direct_storage() {
    return;
/*
    const int samples[][2] = {{0,0},{0,255},{15,127},{29,0},{29,255}};
    for (const auto & sample : samples) {
        for (int kind = 0; kind < 3; ++kind) {
            const char * suffix = kind == 0 ? "gate" : kind == 1 ? "up" : "down";
            char name[128];
            std::snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", sample[0], suffix);
            ggml_expert_storage_info info = {};
            const ggml_tensor * tensor = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_storage_mutex);
                auto it = g_storage_by_name.find(name);
                if (it != g_storage_by_name.end()) { info = it->second; tensor = it->second.tensor; }
            }
            if (!tensor || info.os_handle == 0 || info.plane_size == 0) { ++g_validation_fail; continue; }
            std::vector<uint8_t> direct(info.plane_size);
            const int fd = _dup(info.file_id);
            bool ok = fd >= 0;
            if (ok) ok = _lseeki64(fd, (__int64)(info.absolute_offset + (uint64_t)sample[1] * info.plane_stride), SEEK_SET) >= 0;
            if (ok) ok = _read(fd, direct.data(), (unsigned)direct.size()) == (int)direct.size();
            if (fd >= 0) _close(fd);
            const uint8_t * mapped = (const uint8_t *) tensor->data + (size_t)sample[1] * info.plane_stride;
            if (ok && std::memcmp(direct.data(), mapped, info.plane_size) == 0) { ++g_validation_pass; g_validation_bytes += info.plane_size; }
            else { ++g_validation_fail; }
        }
    }
*/
}

static bool parse_expert_registry_name(const char * name, int & layer, int & kind) {
    if (!name) return false;
    char extra = 0;
    if (std::sscanf(name, "blk.%d.ffn_gate_exps.weight%c", &layer, &extra) == 1 && std::string(name) == "blk." + std::to_string(layer) + ".ffn_gate_exps.weight") { kind = 0; return true; }
    if (std::sscanf(name, "blk.%d.ffn_up_exps.weight%c", &layer, &extra) == 1 && std::string(name) == "blk." + std::to_string(layer) + ".ffn_up_exps.weight") { kind = 1; return true; }
    if (std::sscanf(name, "blk.%d.ffn_down_exps.weight%c", &layer, &extra) == 1 && std::string(name) == "blk." + std::to_string(layer) + ".ffn_down_exps.weight") { kind = 2; return true; }
    return false;
}
extern "C" void ggml_expert_storage_register(const ggml_expert_storage_info * info) {
    if (!info || !info->name) return;
    int parsed_layer = -1, parsed_kind = -1;
    if (!parse_expert_registry_name(info->name, parsed_layer, parsed_kind) || parsed_layer < 0 || parsed_layer >= 40 || info->layer != parsed_layer || info->kind != parsed_kind) return;
    std::lock_guard<std::mutex> lock(g_storage_mutex);
    auto existing = g_storage_by_name.find(info->name);
    if (existing != g_storage_by_name.end()) {
        const auto & old = existing->second;
        if (old.layer != info->layer || old.kind != info->kind || old.file_index != info->file_index || old.absolute_offset != info->absolute_offset || old.plane_stride != info->plane_stride || old.plane_size != info->plane_size || old.tensor_size != info->tensor_size || old.n_experts != info->n_experts) ++g_registry_duplicates;
        return;
    }
    // The tensor name returned by ggml is owned by the runtime tensor/context.
    // Registry entries outlive the first metadata context, so never retain the
    // caller's raw name pointer.
    auto owner = g_storage_name_owners.emplace(info->name, std::string(info->name)).first;
    ggml_expert_storage_info stable = *info;
    stable.name = owner->second.c_str();
    g_storage[info->tensor] = stable;
    g_storage_by_name[owner->first] = stable;
    registry_mutated();
}
extern "C" int ggml_expert_storage_registry_validate(void) {
    std::lock_guard<std::mutex> lock(g_storage_mutex);
    int counts[3] = {}, invalid = 0;
    std::set<std::pair<int, int>> keys;
    for (const auto & item : g_storage_by_name) {
        const auto & x = item.second;
        int layer = -1, kind = -1;
        parse_expert_registry_name(item.first.c_str(), layer, kind);
        if (layer < 0 || layer >= 40 || kind < 0 || x.layer != layer || x.kind != kind || x.plane_size != 589824 || x.n_experts != 256 || !keys.emplace(layer, kind).second) { ++invalid; continue; }
        ++counts[kind];
    }
    const int cpu_layers = counts[0];
    bool contiguous = cpu_layers >= 1 && cpu_layers <= 40;
    for (int layer = 0; contiguous && layer < cpu_layers; ++layer) {
        for (int kind = 0; kind < 3; ++kind) contiguous = keys.count({layer, kind}) == 1;
    }
    const bool ok = invalid == 0 && g_registry_duplicates == 0 && contiguous &&
        counts[1] == cpu_layers && counts[2] == cpu_layers &&
        g_storage_by_name.size() == (size_t) cpu_layers * 3;
    std::fprintf(stderr, "EXPERT_REGISTRY_VALIDATE total=%zu cpu_layers=%d gate=%d up=%d down=%d contiguous=%d invalid=%d duplicates=%llu result=%s\n", g_storage_by_name.size(), cpu_layers, counts[0], counts[1], counts[2], contiguous ? 1 : 0, invalid, (unsigned long long) g_registry_duplicates, ok ? "PASS" : "FAIL");
    return ok ? 1 : 0;
}

extern "C" int ggml_expert_storage_lookup(const ggml_tensor * tensor, ggml_expert_storage_info * out) {
    // C1a fast path: serve a known identity hit without taking g_storage_mutex.
    if (tensor && out) {
        const uint64_t epoch = g_registry_epoch.load(std::memory_order_acquire);
        if (g_identity_cache_epoch != epoch) {
            std::memset(g_identity_cache, 0, sizeof(g_identity_cache));
            g_identity_cache_epoch = epoch;
            ++g_tl_counters.identity_cache_flushes;
        } else {
            const size_t i = identity_cache_index(tensor);
            const size_t j = (i + 1) & IDENTITY_CACHE_MASK;
            const identity_cache_entry & a = g_identity_cache[i];
            if (a.key == tensor) { *out = *a.info; ++g_tl_counters.identity_fast_hits; return 1; }
            const identity_cache_entry & b = g_identity_cache[j];
            if (b.key == tensor) { *out = *b.info; ++g_tl_counters.identity_fast_hits; return 1; }
        }
    }
    hot_counters_register_this_thread();
    ++g_tl_counters.identity_locked_lookups;
    b1b_breadcrumb(100, 0, -1, -1, -1, tensor);
    b1b_ring_record_event(b1b_ring_event::lookup_enter, tensor, tensor ? tensor->buffer : nullptr,
        tensor ? tensor->data : nullptr, -1, -1, -1, 0, nullptr);
    B1B_LOOKUP_LOG("B1B_LOOKUP_ENTER tensor=%p out=%p\n", (const void *) tensor, (void *) out);
    B1B_LOOKUP_FLUSH();
    if (!tensor || !out) {
        B1B_LOOKUP_LOG("B1B_LOOKUP_ARG_INVALID tensor=%p out=%p\n", (const void *) tensor, (void *) out);
        B1B_LOOKUP_FLUSH();
        return 0;
    }
    B1B_LOOKUP_LOG("B1B_LOOKUP_ARG_VALID tensor=%p data=%p buffer=%p view_src=%p type=%d ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n",
        (const void *) tensor, tensor->data, (void *) tensor->buffer, (void *) tensor->view_src, (int) tensor->type,
        (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    B1B_LOOKUP_FLUSH();
    const char * tensor_name = ggml_get_name(tensor);
    const bool trace_output_norm = tensor_name && std::strcmp(tensor_name, "output_norm.weight") == 0;
    if (trace_output_norm) {
        std::fprintf(stderr, "OUTPUT_NORM_LOOKUP_ENTER tensor=%p name=%s data=%p buffer=%p view_src=%p storage=%zu by_name=%zu\n",
            (const void *) tensor, tensor_name, tensor->data, (void *) tensor->buffer, (void *) tensor->view_src,
            g_storage.size(), g_storage_by_name.size());
        std::fflush(stderr);
    }
    if (tensor_name && std::strcmp(tensor_name, "blk.0.ffn_gate_exps.weight") == 0) {
        void * context = nullptr, * base = nullptr;
        size_t reserved = 0;
        const bool external = ggml_backend_buffer_external_expert_details(tensor->buffer, &context, &base, &reserved);
        std::fprintf(stderr, "B1B_LIFECYCLE stage=lookup_entry tensor=%p name=%s data=%p buffer=%p buffer_base=%p buffer_size=%zu view_src=%p view_offs=%zu external=%d external_context=%p external_base=%p external_reserved=%zu registry_result=unknown mode=%d\n",
            (const void *) tensor, tensor_name, tensor->data, (void *) tensor->buffer,
            tensor->buffer ? ggml_backend_buffer_get_base(tensor->buffer) : nullptr,
            tensor->buffer ? ggml_backend_buffer_get_size(tensor->buffer) : 0,
            (void *) tensor->view_src, tensor->view_offs, external ? 1 : 0, context, base, reserved,
            ggml_expert_external_storage_enabled());
        std::fflush(stderr);
    }
    B1B_LOOKUP_LOG("B1B_LOOKUP_NAME_READ tensor=%p name_ptr=%p name=%s external=%d\n",
        (const void *) tensor, (const void *) tensor_name, tensor_name ? tensor_name : "<null>",
        tensor->buffer && ggml_backend_buffer_is_external_expert(tensor->buffer) ? 1 : 0);
    B1B_LOOKUP_FLUSH();
    std::lock_guard<std::mutex> lock(g_storage_mutex);
    B1B_LOOKUP_LOG("B1B_LOOKUP_REGISTRY_BEGIN tensor=%p name=%s storage=%zu by_name=%zu\n",
        (const void *) tensor, tensor_name ? tensor_name : "<null>", g_storage.size(), g_storage_by_name.size());
    B1B_LOOKUP_FLUSH();
    b1b_ring_record_event(b1b_ring_event::identity_find_begin, tensor, tensor->buffer, tensor->data,
        -1, -1, -1, 0, tensor_name);
    if (trace_output_norm) {
        std::fprintf(stderr, "ON_IDENTITY_BEGIN container=%p size=%zu key=%p\n", (const void *) &g_storage, g_storage.size(), (const void *) tensor);
        std::fflush(stderr);
        std::fprintf(stderr, "ON_IDENTITY_FIND_BEGIN\n");
        std::fflush(stderr);
    }
    auto it = g_storage.find(tensor);
    b1b_breadcrumb(110, it != g_storage.end() ? 1 : 0, -1, -1, -1, tensor);
    b1b_ring_record_event(b1b_ring_event::identity_find_end, tensor, tensor->buffer, tensor->data,
        -1, -1, -1, 0, tensor_name);
    if (trace_output_norm) {
        std::fprintf(stderr, "ON_IDENTITY_FIND_END\n");
        std::fflush(stderr);
        std::fprintf(stderr, "ON_IDENTITY_RESULT found=%d at_end=%d\n", it != g_storage.end() ? 1 : 0, it == g_storage.end() ? 1 : 0);
        std::fflush(stderr);
        std::fprintf(stderr, "ON_NAME_BEGIN container=%p size=%zu key_ptr=%p key_len=%zu\n",
            (const void *) &g_storage_by_name, g_storage_by_name.size(), (const void *) tensor_name,
            tensor_name ? std::strlen(tensor_name) : 0);
        std::fflush(stderr);
        std::fprintf(stderr, "ON_NAME_KEY_READY key=%s\n", tensor_name ? tensor_name : "<null>");
        std::fflush(stderr);
        std::fprintf(stderr, "ON_NAME_FIND_BEGIN\n");
        std::fflush(stderr);
    }
    if (it != g_storage.end()) {
        b1b_ring_record_event(b1b_ring_event::lookup_found, tensor, tensor->buffer, tensor->data,
            it->second.layer, it->second.kind, -1, 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::metadata_begin, tensor, tensor->buffer, tensor->data,
            it->second.layer, it->second.kind, -1, 0, tensor_name);
        B1B_LOOKUP_LOG("B1B_LOOKUP_REGISTRY_FOUND identity tensor=%p layer=%d kind=%d\n", (const void *) tensor, it->second.layer, it->second.kind);
        B1B_LOOKUP_LOG("B1B_LOOKUP_METADATA_READ_BEGIN source=identity\n");
        *out = it->second;
        b1b_ring_record_event(b1b_ring_event::metadata_end, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        B1B_LOOKUP_LOG("B1B_LOOKUP_METADATA_READ_END name=%s layer=%d kind=%d\n", out->name ? out->name : "<null>", out->layer, out->kind);
        if (b1b_sample_lookup_return()) { B1B_LOOKUP_LOG("B1B_LOOKUP_RETURN result=1 source=identity\n"); B1B_LOOKUP_FLUSH(); }
        b1b_ring_record_event(b1b_ring_event::lookup_return, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        identity_cache_publish(tensor, it->second);
        return 1;
    }
    B1B_LOOKUP_LOG("B1B_LOOKUP_IDENTITY_NOT_FOUND tensor=%p\n", (const void *) tensor);
    B1B_LOOKUP_FLUSH();
    auto it_name = g_storage_by_name.find(tensor_name);
    b1b_breadcrumb(120, it_name != g_storage_by_name.end() ? 1 : 0, -1, -1, -1, tensor);
    b1b_ring_record_event(b1b_ring_event::name_find_end, tensor, tensor->buffer, tensor->data,
        -1, -1, -1, 0, tensor_name);
    if (trace_output_norm) {
        std::fprintf(stderr, "ON_NAME_FIND_END\n");
        std::fflush(stderr);
        std::fprintf(stderr, "ON_NAME_RESULT found=%d at_end=%d\n", it_name != g_storage_by_name.end() ? 1 : 0, it_name == g_storage_by_name.end() ? 1 : 0);
        std::fflush(stderr);
    }
    if (it_name == g_storage_by_name.end()) {
        b1b_ring_record_event(b1b_ring_event::lookup_not_found, tensor, tensor->buffer, tensor->data,
            -1, -1, -1, 0, tensor_name);
        if (trace_output_norm) {
            std::fprintf(stderr, "ON_EXPECTED_NONEXPERT_NOT_FOUND\n");
            std::fflush(stderr);
            std::fprintf(stderr, "ON_RETURN_PREP result=0\n");
            std::fflush(stderr);
            std::fprintf(stderr, "ON_RETURN\n");
            std::fflush(stderr);
        }
        B1B_LOOKUP_LOG("B1B_LOOKUP_REGISTRY_NOT_FOUND name=%s\n", tensor_name ? tensor_name : "<null>");
        B1B_LOOKUP_FLUSH();
        b1b_ring_record_event(b1b_ring_event::lookup_return, tensor, tensor->buffer, tensor->data,
            -1, -1, -1, 0, tensor_name);
        return 0;
    }
    b1b_ring_record_event(b1b_ring_event::lookup_found, tensor, tensor->buffer, tensor->data,
        it_name->second.layer, it_name->second.kind, -1, 0, tensor_name);
    b1b_ring_record_event(b1b_ring_event::metadata_begin, tensor, tensor->buffer, tensor->data,
        it_name->second.layer, it_name->second.kind, -1, 0, tensor_name);
    B1B_LOOKUP_LOG("B1B_LOOKUP_REGISTRY_FOUND name=%s layer=%d kind=%d\n", tensor_name, it_name->second.layer, it_name->second.kind);
    if (g_runtime_by_name.size() >= g_storage_by_name.size() - 1) {
        ggml_expert_external_memory_checkpoint("before_runtime_map_insert");
    }
    // Several model contexts can contain tensors with the same name.  Keep
    // the tensor that is actually bound to storage; a later metadata/context
    // tensor may have the same name but no buffer/data and must not replace it.
    auto runtime_existing = g_runtime_by_name.find(tensor_name);
    const bool current_bound = tensor->buffer != nullptr && tensor->data != nullptr;
    const bool existing_bound = runtime_existing != g_runtime_by_name.end() &&
        runtime_existing->second != nullptr &&
        runtime_existing->second->buffer != nullptr &&
        runtime_existing->second->data != nullptr;
    // Skip the write when it would store the value that is already there. The old code
    // rewrote the same entry on every lookup, which kept bumping the registry epoch.
    const bool runtime_map_unchanged = runtime_existing != g_runtime_by_name.end() && runtime_existing->second == tensor;
    if (!runtime_map_unchanged && (runtime_existing == g_runtime_by_name.end() || current_bound || !existing_bound)) {
        B1B_LOOKUP_LOG(
            "B1B_RUNTIME_MAP_UPDATE name=%s old=%p old_bound=%d new=%p new_bound=%d\\n",
            tensor_name, runtime_existing == g_runtime_by_name.end() ? nullptr : (void *) runtime_existing->second,
            existing_bound ? 1 : 0, (const void *) tensor, current_bound ? 1 : 0);
        g_runtime_by_name[tensor_name] = tensor;
        registry_mutated();
    } else {
        B1B_LOOKUP_LOG(
            "B1B_RUNTIME_MAP_KEEP name=%s kept=%p kept_bound=1 rejected=%p rejected_bound=0\\n",
            tensor_name, (const void *) runtime_existing->second, (const void *) tensor);
    }
    if (g_runtime_by_name.size() >= g_storage_by_name.size() - 1) {
        ggml_expert_external_memory_checkpoint("after_runtime_map_insert");
    }
    B1B_LOOKUP_LOG("B1B_LOOKUP_METADATA_READ_BEGIN source=name runtime=%zu\n", g_runtime_by_name.size());
    *out = it_name->second;
    b1b_breadcrumb(130, 0, out->layer, out->kind, -1, tensor);
    b1b_ring_record_event(b1b_ring_event::metadata_end, tensor, tensor->buffer, tensor->data,
        out->layer, out->kind, -1, 0, tensor_name);
    if (g_runtime_by_name.size() == g_storage_by_name.size()) {
        ggml_expert_external_memory_checkpoint("after_validation_metadata_copy");
    }
    B1B_LOOKUP_LOG("B1B_LOOKUP_METADATA_READ_END name=%s layer=%d kind=%d\n", out->name ? out->name : "<null>", out->layer, out->kind);
    const bool placement_only = ggml_expert_placement_diagnostics_enabled() && !ggml_expert_external_storage_enabled();
    const bool trace_post_meta = tensor_name && std::strcmp(tensor_name, "blk.0.ffn_down_exps.weight") == 0 &&
        out->layer == 0 && out->kind == 2;
    if (trace_post_meta) {
        b1b_ring_record_event(b1b_ring_event::post_meta_begin, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, tensor->buffer && ggml_backend_buffer_is_external_expert(tensor->buffer) ? 1 : 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::post_meta_tensor_state_begin, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::post_meta_tensor_state_end, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::post_meta_runtime_size_begin, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
    }
    const size_t runtime_size_post_meta = g_runtime_by_name.size();
    const size_t storage_size_post_meta = g_storage_by_name.size();
    if (trace_post_meta) {
        b1b_ring_record_event(b1b_ring_event::post_meta_runtime_size_end, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::post_meta_placement_begin, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, placement_only ? 1 : 0, tensor_name);
    }
    const bool validation_ready_post_meta = runtime_size_post_meta == storage_size_post_meta && !placement_only;
    b1b_breadcrumb(140, validation_ready_post_meta ? 1 : 0, out->layer, out->kind, -1, tensor);
    if (trace_post_meta) {
        b1b_ring_record_event(b1b_ring_event::post_meta_placement_end, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, validation_ready_post_meta ? 1 : 0, tensor_name);
    }
    B1B_LOOKUP_LOG("B1B_LOOKUP_POST_METADATA placement_only=%d runtime=%zu storage=%zu\n", placement_only ? 1 : 0, g_runtime_by_name.size(), g_storage_by_name.size());
    B1B_LOOKUP_FLUSH();
    // Validation is a diagnostic sample, not a per-lookup operation.  Multiple
    // runtime tensor instances share the same expert names, so without this
    // one-shot gate the sample loop repeats for every name-based lookup.
    const bool run_validation = validation_ready_post_meta &&
        !g_validation_started.exchange(true, std::memory_order_acq_rel);
    if (run_validation) {
        b1b_breadcrumb(150, 0, out->layer, out->kind, -1, tensor);
        if (trace_post_meta) {
            b1b_ring_record_event(b1b_ring_event::post_meta_validation_begin, tensor, tensor->buffer, tensor->data,
                out->layer, out->kind, -1, 0, tensor_name);
        }
        static const int samples[][2] = {{0,0},{0,255},{15,127},{29,0},{29,255}};
        static bool raw_boundary_used = false;
        ggml_expert_external_memory_checkpoint("before_validation_begin");
        b1b_breadcrumb(1800, 10);
        B1B_LOOKUP_LOG("B1B_LOOKUP_VALIDATION_BEGIN\n");
        B1B_LOOKUP_FLUSH();
        b1b_breadcrumb(1800, 11);
        B1B_LOOKUP_LOG("VALIDATION_DISPATCH_BEGIN\n");
        B1B_LOOKUP_FLUSH();
        for (const auto & s : samples) for (int k = 0; k < 3; ++k) {
            b1b_breadcrumb(1800, 12, s[0], k, s[1]);
            const char * suf = k == 0 ? "gate" : k == 1 ? "up" : "down";
            b1b_breadcrumb(1800, 13, s[0], k, s[1]);
            B1B_LOOKUP_LOG("VALIDATION_SAMPLE_SLOT_BEGIN layer=%d expert=%d kind=%d\n", s[0], s[1], k);
            B1B_LOOKUP_FLUSH();
            char n[128];
            b1b_breadcrumb(1800, 14, s[0], k, s[1]);
            std::snprintf(n, sizeof(n), "blk.%d.ffn_%s_exps.weight", s[0], suf);
            b1b_breadcrumb(1800, 15, s[0], k, s[1]);
            B1B_LOOKUP_LOG("VALIDATION_SAMPLE_NAME_READY name=%s\n", n);
            B1B_LOOKUP_FLUSH();
            b1b_breadcrumb(1800, 16, s[0], k, s[1]);
            auto si = g_storage_by_name.find(n); auto ri = g_runtime_by_name.find(n);
            b1b_breadcrumb(1800, 17, s[0], k, s[1]);
            B1B_LOOKUP_LOG("VALIDATION_SAMPLE_LOOKUP_END storage_found=%d runtime_found=%d\n",
                si != g_storage_by_name.end() ? 1 : 0, ri != g_runtime_by_name.end() ? 1 : 0);
            B1B_LOOKUP_FLUSH();
            if (si != g_storage_by_name.end() && ri != g_runtime_by_name.end()) {
                b1b_breadcrumb(1800, 18, s[0], k, s[1]);
                if (s[0] == 0 && k == 0 && s[1] == 0) {
                    b1b_prov(0, (uintptr_t) ri->second);
                    g_b1b_expected_tensor.store((uintptr_t) ri->second, std::memory_order_relaxed);
                    b1b_prov(1, (uintptr_t) ri->second->buffer);
                }
                b1b_breadcrumb(1800, 181, s[0], k, s[1]);
                const bool trace_this_validation = k == 0 && s[0] == 0 && s[1] == 0;
                const bool raw_probe = trace_this_validation && !raw_boundary_used;
                if (raw_probe) raw_boundary_used = true;
                b1b_breadcrumb(1800, 182, s[0], k, s[1]);
                const ggml_tensor * call_tensor = nullptr;
                b1b_breadcrumb(1800, 184, s[0], k, s[1]);
                try {
                    call_tensor = ri->second;
                    b1b_prov(2, (uintptr_t) call_tensor);
                    b1b_breadcrumb(1800, 185, s[0], k, s[1]);
                    const ggml_expert_storage_info call_info = si->second;
                    b1b_breadcrumb(1800, 186, s[0], k, s[1]);
                    const int call_expert = s[1];
                    b1b_breadcrumb(1800, 187, s[0], k, s[1]);
                    g_trace_validation_current = trace_this_validation;
                    b1b_breadcrumb(1800, 188, s[0], k, s[1]);
                    if (trace_this_validation) {
                        B1B_LOOKUP_LOG(
                            "B1B_VALIDATE_DISPATCH_STATE tensor=%p field=%p buffer=%p data=%p view_src=%p name=%s\\n",
                            (const void *) call_tensor, (const void *) &call_tensor->buffer,
                            (void *) call_tensor->buffer, call_tensor->data,
                            (const void *) call_tensor->view_src, ggml_get_name(call_tensor));
                        B1B_LOOKUP_FLUSH();
                    }
                    validate_current(call_tensor, call_info, call_expert);
                    b1b_breadcrumb(1800, 193, s[0], k, s[1]);
                    g_trace_validation_current = false;
                } catch (const std::exception & e) {
                    g_trace_validation_current = false;
                    if (trace_this_validation) { std::fprintf(stderr, "CALLSITE_STD_EXCEPTION what=%s\n", e.what()); std::fflush(stderr); }
                    throw;
                } catch (...) {
                    g_trace_validation_current = false;
                    if (trace_this_validation) { std::fprintf(stderr, "CALLSITE_UNKNOWN_CPP_EXCEPTION\n"); std::fflush(stderr); }
                    throw;
                }
                b1b_breadcrumb(1800, 194, s[0], k, s[1]);
            }
        }
        B1B_LOOKUP_LOG("B1B_LOOKUP_VALIDATION_END\n");
        g_validation_completed.store(true, std::memory_order_release);
        if (trace_post_meta) {
            b1b_ring_record_event(b1b_ring_event::post_meta_validation_end, tensor, tensor->buffer, tensor->data,
                out->layer, out->kind, -1, 0, tensor_name);
        }
    }
    if (trace_post_meta) {
        b1b_ring_record_event(b1b_ring_event::post_meta_result_begin, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
    }
    if (b1b_sample_lookup_return()) { B1B_LOOKUP_LOG("B1B_LOOKUP_RETURN result=1 source=name\n"); B1B_LOOKUP_FLUSH(); }
    b1b_breadcrumb(190, 0, out->layer, out->kind, -1, tensor);
    if (trace_post_meta) {
        b1b_ring_record_event(b1b_ring_event::post_meta_result_end, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::post_meta_before_return, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
        b1b_ring_record_event(b1b_ring_event::lookup_return, tensor, tensor->buffer, tensor->data,
            out->layer, out->kind, -1, 0, tensor_name);
    }
    // C1a: the loader context is freed before decode, so the identity map misses and
    // this name path is the hot one. Cache it once the runtime map already points at
    // this tensor, which makes any later call for it a pure read.
    if (runtime_map_unchanged) {
        identity_cache_publish(tensor, it_name->second);
    }
    return 1;
}
static bool is_on() { const char * p = std::getenv("LLAMA_EXPERT_INDIRECTION"); return p && p[0] == '1'; }
static bool verify_only() { const char * p = std::getenv("CACHE_LOAD_VERIFY_ONLY"); return p && p[0] == '1'; }

static bool parse_metadata(const char * name, int & layer, const char * & kind) {
    if (!name || std::strncmp(name, "blk.", 4) != 0) return false;
    char * end = nullptr; layer = (int) std::strtol(name + 4, &end, 10);
    if (end == name + 4 || layer < 0 || layer >= 40) return false;
    if (std::strstr(name, ".ffn_gate_exps.")) kind = "gate";
    else if (std::strstr(name, ".ffn_up_exps.")) kind = "up";
    else if (std::strstr(name, ".ffn_down_exps.")) kind = "down";
    else return false;
    return true;
}

extern "C" int ggml_expert_indirection_enabled(void) {
    if (is_on()) return 1;
    const char * external = std::getenv("B1B_EXTERNAL_EXPERT_STORAGE");
    return external && external[0] == '1';
}
extern "C" int ggml_expert_verify_only_enabled(void) { return verify_only() ? 1 : 0; }
extern "C" int ggml_expert_cached_bounds_enabled(void) { const char * p=std::getenv("CACHE_VALIDATE_CACHED_BOUNDS_ONLY"); return p && p[0]=='1'; }
extern "C" int ggml_expert_cached_compute_enabled(void) {
    const char * p = std::getenv("CACHE_USE_CACHED_POINTER");
    if (p && p[0] == '1') return 1;
    const char * external = std::getenv("B1B_EXTERNAL_EXPERT_STORAGE");
    return external && external[0] == '1';
}
extern "C" int ggml_expert_b1b_detached_enabled(void) {
    const char * p = std::getenv("B1B_DETACH_CPU_EXPERTS");
    if (p && p[0] == '1') return 1;
    const char * external = std::getenv("B1B_EXTERNAL_EXPERT_STORAGE");
    return external && external[0] == '1';
}
extern "C" int ggml_expert_external_storage_enabled(void) { const char * p=std::getenv("B1B_EXTERNAL_EXPERT_STORAGE"); return p && p[0]=='1'; }
extern "C" int ggml_expert_placement_diagnostics_enabled(void) { const char * p=std::getenv("B1B_PLACEMENT_DIAGNOSTICS"); return p && p[0]=='1'; }
extern "C" int ggml_expert_external_is_tensor(const ggml_tensor * tensor) {
    if (!ggml_expert_external_storage_enabled() || !tensor) return 0;
    ggml_expert_storage_info info = {};
    return ggml_expert_storage_lookup(tensor, &info) && info.cpu_override ? 1 : 0;
}
extern "C" void ggml_expert_external_note_tensor(size_t logical, size_t reserved, size_t committed) {
    ++g_external_tensors; g_external_logical_bytes += logical; g_external_reserved_bytes += reserved; g_external_committed_bytes += committed;
    ++g_external_created;
}
extern "C" void ggml_expert_external_note_placement(const ggml_expert_storage_info * info,
        int cpu_candidate, int mmap_candidate, int external_assigned,
        const char * buft_name, const char * device_name) {
    if (!info) return;
    ++g_placement_registered;
    if (cpu_candidate) ++g_placement_cpu_candidates; else ++g_placement_gpu_candidates;
    if (info->kind >= 0 && info->kind < 3) ++g_placement_kind[info->kind];
    if (cpu_candidate) ++g_external_expected;
    if (mmap_candidate) {
        if (cpu_candidate) ++g_cpu_mmap_bound; else ++g_gpu_mmap_bound;
    }
    if (cpu_candidate && mmap_candidate && !external_assigned) ++g_unexpected_cpu_mmap;
    if (!cpu_candidate && external_assigned) ++g_unexpected_external_gpu;
    std::fprintf(stderr, "B1B_PLACEMENT name=%s layer=%d kind=%d logical=%llu cpu=%d mmap_candidate=%d external_assigned=%d buft=%s device=%s\n",
        info->name ? info->name : "<none>", info->layer, info->kind,
        (unsigned long long) info->tensor_size, cpu_candidate, mmap_candidate, external_assigned,
        buft_name ? buft_name : "<none>", device_name ? device_name : "<none>");
}
extern "C" void ggml_expert_external_note_get(const ggml_tensor * tensor) { GGML_UNUSED(tensor); ++g_external_get_attempts; }
extern "C" void ggml_expert_external_note_set(const ggml_tensor * tensor) { GGML_UNUSED(tensor); ++g_external_set_attempts; }
extern "C" void ggml_expert_external_note_copy(const ggml_tensor * tensor) { GGML_UNUSED(tensor); ++g_external_copy_attempts; }
extern "C" void ggml_expert_external_note_clear(const ggml_tensor * tensor) { GGML_UNUSED(tensor); ++g_external_clear_attempts; }
extern "C" void ggml_expert_external_note_view(const ggml_tensor * tensor) { GGML_UNUSED(tensor); ++g_external_view_attempts; }
extern "C" void ggml_expert_external_note_validation(const ggml_tensor * tensor) {
    std::fprintf(stderr, "NOTE_ENTER tensor=%p\n", (const void *) tensor);
    std::fflush(stderr);
    std::fprintf(stderr, "NOTE_ARG_CHECK_BEGIN\n");
    std::fflush(stderr);
    GGML_UNUSED(tensor);
    std::fprintf(stderr, "NOTE_ARG_CHECK_END\n");
    std::fflush(stderr);
    std::fprintf(stderr, "NOTE_COUNTER_BEGIN\n");
    std::fflush(stderr);
    ++g_external_validation_attempts;
    std::fprintf(stderr, "NOTE_COUNTER_END\n");
    std::fflush(stderr);
    std::fprintf(stderr, "NOTE_RETURN\n");
    std::fflush(stderr);
}
extern "C" void ggml_expert_external_note_failure(const char * reason, const ggml_tensor * tensor) {
    ++g_external_buffer_failures;
    std::fprintf(stderr, "B1B_EXTERNAL_BUFFER_VIOLATION op=%s tensor=%s\n", reason ? reason : "unknown", tensor ? ggml_get_name(tensor) : "<none>");
}
extern "C" void ggml_expert_external_report(void) {
    std::fprintf(stderr, "B1B_PLACEMENT_STATS registered_experts=%llu cpu_candidates=%llu gpu_candidates=%llu external_expected=%llu external_created=%llu cpu_mmap_bound=%llu gpu_mmap_bound=%llu unexpected_cpu_mmap=%llu unexpected_external_gpu=%llu kind_gate=%llu kind_up=%llu kind_down=%llu\n",
        (unsigned long long) g_placement_registered.load(), (unsigned long long) g_placement_cpu_candidates.load(),
        (unsigned long long) g_placement_gpu_candidates.load(), (unsigned long long) g_external_expected.load(),
        (unsigned long long) g_external_created.load(), (unsigned long long) g_cpu_mmap_bound.load(),
        (unsigned long long) g_gpu_mmap_bound.load(), (unsigned long long) g_unexpected_cpu_mmap.load(),
        (unsigned long long) g_unexpected_external_gpu.load(), (unsigned long long) g_placement_kind[0].load(),
        (unsigned long long) g_placement_kind[1].load(), (unsigned long long) g_placement_kind[2].load());
    std::fprintf(stderr, "B1B_EXTERNAL_STATS external_expert_tensors=%llu external_logical_bytes=%llu external_reserved_bytes=%llu external_committed_bytes=%llu external_get_attempts=%llu external_set_attempts=%llu external_copy_attempts=%llu external_clear_attempts=%llu external_view_attempts=%llu external_validation_content_attempts=%llu external_buffer_failures=%llu unexpected_mmap_expert_bindings=%llu\n",
        (unsigned long long) g_external_tensors.load(), (unsigned long long) g_external_logical_bytes.load(),
        (unsigned long long) g_external_reserved_bytes.load(), (unsigned long long) g_external_committed_bytes.load(),
        (unsigned long long) g_external_get_attempts.load(), (unsigned long long) g_external_set_attempts.load(),
        (unsigned long long) g_external_copy_attempts.load(), (unsigned long long) g_external_clear_attempts.load(),
        (unsigned long long) g_external_view_attempts.load(), (unsigned long long) g_external_validation_attempts.load(),
        (unsigned long long) g_external_buffer_failures.load(), (unsigned long long) g_unexpected_mmap_expert_bindings.load());
}
extern "C" void ggml_expert_external_memory_checkpoint(const char * stage) {
    uint32_t breadcrumb_stage = 9999, breadcrumb_substage = 0;
    if (stage && std::strcmp(stage, "before_model_load") == 0) { breadcrumb_stage = 1000; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "before_done_getting_tensors") == 0) { breadcrumb_stage = 1100; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_done_getting_tensors") == 0) { breadcrumb_stage = 1100; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "before_init_mappings") == 0) { breadcrumb_stage = 1200; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_init_mappings") == 0) { breadcrumb_stage = 1200; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "after_loader_validation") == 0) { breadcrumb_stage = 1900; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "before_backend_buffer_planning") == 0) { breadcrumb_stage = 1300; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_buft_selection") == 0) { breadcrumb_stage = 1300; breadcrumb_substage = 2; }
    else if (stage && std::strcmp(stage, "before_external_preallocation") == 0) { breadcrumb_stage = 1500; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_external_preallocation_context") == 0) { breadcrumb_stage = 1500; breadcrumb_substage = 2; }
    else if (stage && std::strcmp(stage, "before_mmap_buffer_allocation") == 0) { breadcrumb_stage = 1400; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_mmap_buffer_allocation") == 0) { breadcrumb_stage = 1400; breadcrumb_substage = 2; }
    else if (stage && std::strcmp(stage, "before_backend_buffer_allocation") == 0) { breadcrumb_stage = 1400; breadcrumb_substage = 3; }
    else if (stage && std::strcmp(stage, "after_backend_buffer_allocation") == 0) { breadcrumb_stage = 1400; breadcrumb_substage = 4; }
    else if (stage && std::strcmp(stage, "after_context_finalization") == 0) { breadcrumb_stage = 2300; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "after_external_preallocation") == 0) { breadcrumb_stage = 1500; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "after_ml_no_alloc_setup") == 0) { breadcrumb_stage = 1600; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "before_memory_breakdown") == 0) { breadcrumb_stage = 1700; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_memory_breakdown") == 0) { breadcrumb_stage = 1700; breadcrumb_substage = 99; }
    else if (stage && std::strcmp(stage, "before_runtime_map_insert") == 0) { breadcrumb_stage = 1800; breadcrumb_substage = 1; }
    else if (stage && std::strcmp(stage, "after_runtime_map_insert") == 0) { breadcrumb_stage = 1800; breadcrumb_substage = 2; }
    else if (stage && std::strcmp(stage, "after_validation_metadata_copy") == 0) { breadcrumb_stage = 1800; breadcrumb_substage = 3; }
    else if (stage && std::strcmp(stage, "before_validation_begin") == 0) { breadcrumb_stage = 1800; breadcrumb_substage = 4; }
    b1b_breadcrumb(breadcrumb_stage, breadcrumb_substage);
#ifdef _WIN32
    static std::atomic<uint64_t> last_checkpoint_ms{0};
    const uint64_t now_ms = GetTickCount64();
    uint64_t previous_ms = last_checkpoint_ms.load(std::memory_order_relaxed);
    if (previous_ms != 0 && now_ms - previous_ms < 5000) return;
    if (!last_checkpoint_ms.compare_exchange_strong(previous_ms, now_ms, std::memory_order_relaxed)) return;
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    pmc.cb = sizeof(pmc);
    const BOOL ok = GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc));
    std::fprintf(stderr, "B1B_MEMORY_CHECKPOINT stage=%s ok=%d working_set=%llu private_bytes=%llu peak_working_set=%llu external_logical=%llu external_reserved=%llu external_committed=%llu\n",
        stage ? stage : "<none>", ok ? 1 : 0,
        (unsigned long long) (ok ? pmc.WorkingSetSize : 0),
        (unsigned long long) (ok ? pmc.PrivateUsage : 0),
        (unsigned long long) (ok ? pmc.PeakWorkingSetSize : 0),
        (unsigned long long) g_external_logical_bytes.load(),
        (unsigned long long) g_external_reserved_bytes.load(),
        (unsigned long long) g_external_committed_bytes.load());
#else
    std::fprintf(stderr, "B1B_MEMORY_CHECKPOINT stage=%s ok=0 working_set=0 private_bytes=0 peak_working_set=0 external_logical=%llu external_reserved=%llu external_committed=%llu\n",
        stage ? stage : "<none>",
        (unsigned long long) g_external_logical_bytes.load(),
        (unsigned long long) g_external_reserved_bytes.load(),
        (unsigned long long) g_external_committed_bytes.load());
#endif
    std::fflush(stderr);
}
extern "C" void ggml_expert_note_original_pointer_use(void) { ++g_original_pointer_uses; }
extern "C" void ggml_expert_note_cached_pointer_use(void) { ++g_cached_pointer_uses; }
#if B1S_PROFILE
extern "C" int ggml_b1s_profile_enabled(void) { return 1; }
extern "C" uint64_t ggml_b1s_mmid_enter(int worker) {
    const uint64_t now = b1s_now_ns();
    if (g_b1s_active_mmid.fetch_add(1, std::memory_order_acq_rel) == 0) {
        g_b1s_mmid_union_start.store(now, std::memory_order_release);
    }
    (void) worker;
    return now;
}
extern "C" void ggml_b1s_mmid_exit(int worker, uint64_t start_ns) {
    const uint64_t elapsed = b1s_now_ns() - start_ns;
    g_b1s_mmid_aggregate_ns.fetch_add(elapsed, std::memory_order_relaxed);
    if (worker >= 0 && worker < B1S_MAX_WORKERS) g_b1s_worker_mmid_ns[worker].fetch_add(elapsed, std::memory_order_relaxed);
    if (g_b1s_active_mmid.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        const uint64_t begin = g_b1s_mmid_union_start.load(std::memory_order_acquire);
        g_b1s_mmid_wall_ns.fetch_add(b1s_now_ns() - begin, std::memory_order_relaxed);
    }
}
extern "C" uint64_t ggml_b1s_region_begin(void) { return b1s_now_ns(); }
extern "C" void ggml_b1s_region_end(int worker, int region, uint64_t start_ns) {
    if (region < 0 || region >= 3) return;
    const uint64_t elapsed = b1s_now_ns() - start_ns;
    g_b1s_region_ns[region].fetch_add(elapsed, std::memory_order_relaxed);
    if (worker >= 0 && worker < B1S_MAX_WORKERS) g_b1s_worker_region_ns[worker][region].fetch_add(elapsed, std::memory_order_relaxed);
}
#else
extern "C" int ggml_b1s_profile_enabled(void) { return 0; }
extern "C" uint64_t ggml_b1s_mmid_enter(int worker) { (void) worker; return 0; }
extern "C" void ggml_b1s_mmid_exit(int worker, uint64_t start_ns) { (void) worker; (void) start_ns; }
extern "C" uint64_t ggml_b1s_region_begin(void) { return 0; }
extern "C" void ggml_b1s_region_end(int worker, int region, uint64_t start_ns) { (void) worker; (void) region; (void) start_ns; }
#endif
extern "C" void ggml_expert_note_external_vec_dot(const void * vx, const void * reserved_begin, size_t reserved_size) {
#if B1R4_COMPILE_OUT_DIAGNOSTICS
    (void) vx; (void) reserved_begin; (void) reserved_size;
    return;
#else
    const bool b1k_sample = ggml_expert_b1k_profile_enabled() && ((++g_b1k_local_seq & 1023u) == 0);
    const uint64_t b1k_start = b1k_sample ? ggml_expert_mmid_now_ns() : 0;
    ++g_external_vec_dot_calls;
    const uintptr_t begin = (uintptr_t) reserved_begin;
    const uintptr_t value = (uintptr_t) vx;
    const bool in_reserved = begin && reserved_size <= UINTPTR_MAX - begin && value >= begin && value < begin + reserved_size;
    if (in_reserved) {
        ++g_reserved_pointer_violations;
        std::fprintf(stderr, "B1B_RESERVED_POINTER_VIOLATION vx=%p reserved_begin=%p reserved_size=%zu\n", vx, reserved_begin, reserved_size);
        std::fflush(stderr);
    }
    if (b1k_sample) ggml_expert_b1k_add(4, ggml_expert_mmid_now_ns() - b1k_start);
#endif
}
extern "C" void ggml_expert_note_unexpected_mmap_fallback(void) { ++g_unexpected_mmap_fallbacks; }
extern "C" int ggml_expert_validate_cached_row(const ggml_tensor * tensor, int expert_id, const char * original_plane, const char * cached_plane, int64_t ir0, size_t nb01, size_t nb02, size_t required_bytes, const char * vec_dot_name) {
    ++g_tl_counters.row_bounds_checks;
    const bool b1k_sample = ggml_expert_b1k_profile_enabled() && ((++g_b1k_local_seq & 1023u) == 0);
    const uint64_t b1k_start = b1k_sample ? ggml_expert_mmid_now_ns() : 0;
    const uintptr_t cb=(uintptr_t)cached_plane, ob=(uintptr_t)original_plane; size_t row_offset=0, row_end=0; uintptr_t plane_end=0, dot_ptr=0, dot_end=0, original_dot=0;
    bool bad = ir0 < 0 || (nb01 && (uint64_t)ir0 > SIZE_MAX/nb01) || !safe_mul((size_t)ir0, nb01, row_offset);
    if (bad) { ++g_row_offset_overflows; }
    if (!bad && !safe_add(row_offset, required_bytes, row_end)) { ++g_row_end_overflows; bad=true; }
    if (!bad && (!safe_addr_add(cb, nb02, plane_end) || !safe_addr_add(cb, row_offset, dot_ptr) || !safe_addr_add(dot_ptr, required_bytes, dot_end) ||
                 dot_ptr < cb || dot_ptr > plane_end || required_bytes > nb02 || dot_end > plane_end || dot_ptr > dot_end)) { bad=true; }
    if (!bad && !safe_addr_add(ob, row_offset, original_dot)) { ++g_row_end_overflows; bad=true; }
    if (!bad && (original_dot < ob || original_dot - ob != row_offset)) bad=true;
    if (bad) {
        ++g_row_bounds_failures;
        int layer=-1; const char * kind=nullptr; parse_metadata(ggml_get_name(tensor), layer, kind); (void) kind;
        if (!g_row_failure_recorded) { g_row_failure_recorded=true; g_row_failure={layer,expert_id,-1,-1,ir0,nb01,nb02, (size_t) tensor->ne[0],(size_t) tensor->ne[1],(size_t) tensor->ne[2],row_offset,required_bytes,cb,plane_end,(uintptr_t)cached_plane,dot_ptr,dot_end,vec_dot_name}; }
        if (b1k_sample) ggml_expert_b1k_add(1, ggml_expert_mmid_now_ns() - b1k_start);
        return 0;
    }
    if (b1k_sample) ggml_expert_b1k_add(1, ggml_expert_mmid_now_ns() - b1k_start);
    return 1;
}

extern "C" const char * ggml_expert_resolve_ptr(const struct ggml_tensor * src0, int expert_id, size_t expert_stride) {
#if B1R4_COMPILE_OUT_DIAGNOSTICS
    const bool pointer_trace_enabled = false;
#else
    const bool pointer_trace_enabled = pointer_trace_on();
#endif
    const uint64_t call_number = g_calls.fetch_add(1) + 1;
    const bool cache_enabled = cache_on();
    const bool b2d_sample = ggml_expert_b2d_enabled() && ((call_number & 15u) == 0);
    const uint64_t b2d_resolve = b2d_sample ? b2d_now() : 0;
#if B1R4_COMPILE_OUT_DIAGNOSTICS && !B1U_PROFILE
    const bool b1p_sample = false;
    const bool mmid_resolver_sample = false;
#else
    const bool b1p_sample = (B1U_PROFILE ? ((call_number & 65535u) == 0) : (ggml_expert_b1p_profile_env() && ((call_number & B1P_SAMPLE_MASK) == 0)));
    const bool mmid_resolver_sample = ggml_expert_mmid_profile_enabled() && ((call_number & 63u) == 0);
#endif
    const uint64_t b1p_resolver_start = b1p_sample ? lock_now_ns() : 0;
    const uint64_t mmid_resolver_start_ns = mmid_resolver_sample ? ggml_expert_mmid_now_ns() : 0;
    if (b1p_sample) ++g_b1p_resolver_samples;
    if (pointer_trace_enabled && pointer_trace_take_enabled()) {
        std::fprintf(stderr, "RESOLVE_BEGIN src0=%p src0_data=%p expert=%d stride=%zu name=%s indirection=%d cache_on=%d cache_compute=%d\n",
            (const void *) src0, src0 ? src0->data : nullptr, expert_id, expert_stride,
            src0 ? ggml_get_name(src0) : "<null>",
            ggml_expert_indirection_enabled(), cache_on(), ggml_expert_cached_compute_enabled());
        std::fflush(stderr);
    }
    const uint64_t b1p_registry_start = b1p_sample ? lock_now_ns() : 0;
    ggml_expert_storage_info storage = {};
    const bool found = ggml_expert_storage_lookup(src0, &storage) != 0;
    if (b1p_sample) g_b1p_registry_ns += lock_now_ns() - b1p_registry_start;
    if (!found) {
#if !B1R4_COMPILE_OUT_DIAGNOSTICS
        g_registry_misses.fetch_add(1); g_metadata_failures.fetch_add(1);
#endif
        return nullptr;
    }
#if !B1R4_COMPILE_OUT_DIAGNOSTICS
    g_registry_hits.fetch_add(1);
#endif
    const int layer = storage.layer;
    const char * kind = storage.kind == 0 ? "gate" : storage.kind == 1 ? "up" : storage.kind == 2 ? "down" : nullptr;
    if (!kind) {
#if !B1R4_COMPILE_OUT_DIAGNOSTICS
        g_metadata_failures.fetch_add(1);
#endif
        return nullptr;
    }
    if (expert_id < 0 || expert_id >= 256) {
#if !B1R4_COMPILE_OUT_DIAGNOSTICS
        g_invalid.fetch_add(1);
#endif
        return nullptr;
    }
    const char * expected = (const char *) src0->data + (size_t) expert_id * expert_stride;
    g_profile_current_sample = ggml_expert_profile_env() && ((call_number & 4095u) == 0);
    g_profile_current_start_us = g_profile_current_sample ? ggml_time_us() : 0;
#if !B1R4_COMPILE_OUT_DIAGNOSTICS
    { std::lock_guard<std::mutex> lock(g_mutex); g_layers.insert(layer); g_kinds.insert(kind); }
#endif
    const char * resolved = cache_enabled ? cache_resolve(&storage, expert_id, expert_stride, b1p_sample, b2d_sample, b2d_resolve) : expected;
    if (b1p_sample) g_b1p_resolver_ns += lock_now_ns() - b1p_resolver_start;
    if (g_profile_current_sample) { ++g_profile_resolver_samples; g_profile_resolver_us += std::max<int64_t>(0, ggml_time_us() - g_profile_current_start_us); g_profile_current_sample = false; }
    if (!cache_enabled) ++g_normal_returns;
    if (!resolved) ++g_resolver_failures; else ++g_resolver_success;
#if !B1R4_COMPILE_OUT_DIAGNOSTICS
    if (!cache_on() && resolved != expected) g_mismatches.fetch_add(1);
#endif
    if (pointer_trace_enabled && pointer_trace_take_enabled()) {
        std::fprintf(stderr, "RESOLVE_END src0=%p expert=%d expected=%p resolved=%p cache_on=%d\n",
            (const void *) src0, expert_id, (const void *) expected, (const void *) resolved, cache_on());
        std::fflush(stderr);
    }
    if (mmid_resolver_sample) ggml_expert_mmid_add_region(1, (ggml_expert_mmid_now_ns() - mmid_resolver_start_ns) * 64);
    return resolved;
}
extern "C" int ggml_expert_profile_enabled(void) { return ggml_expert_profile_env() ? 1 : 0; }
extern "C" void ggml_expert_note_mul_mat_id_time(uint64_t elapsed_us) { if (ggml_expert_profile_env()) { ++g_profile_mul_mat_calls; g_profile_mul_mat_us += elapsed_us; } }
extern "C" int ggml_expert_mmid_profile_enabled(void) { static const bool enabled = []() { const char * p = std::getenv("LLAMA_EXPERT_DECOMP"); return p && p[0] == '1'; }(); return enabled ? 1 : 0; }
extern "C" uint64_t ggml_expert_mmid_now_ns(void) { return lock_now_ns(); }
extern "C" void ggml_expert_mmid_note_call(uint64_t elapsed_ns) { ++g_mmid_external_calls; g_mmid_external_total_ns += elapsed_ns; }
extern "C" void ggml_expert_mmid_add_region(int region, uint64_t elapsed_ns) { if (region >= 0 && region < 7) g_mmid_region_ns[region] += elapsed_ns; }
extern "C" int ggml_expert_mmid_vecdot_sample(void) { return (++g_mmid_vecdot_local_seq & 1023u) == 0; }
extern "C" int ggml_expert_mmid_sample_ratio(uint32_t ratio) {
    if (ratio == 0) return 0;
    const uint32_t n = ++g_mmid_region_local_seq;
    return (ratio & (ratio - 1)) == 0 ? ((n & (ratio - 1)) == 0) : (n % ratio) == 0;
}
extern "C" void ggml_expert_mmid_note_vecdot(uint64_t elapsed_ns) { ++g_mmid_vecdot_samples; g_mmid_vecdot_sample_ns += elapsed_ns; }
extern "C" int ggml_expert_b1k_profile_enabled(void) { return ggml_expert_mmid_profile_enabled(); }
extern "C" int ggml_expert_b1k_sample(void) { return (++g_b1k_local_seq & 1023u) == 0; }
extern "C" void ggml_expert_b1k_add(int region, uint64_t elapsed_ns) { if (region >= 0 && region < 8) { g_b1k_ns[region] += elapsed_ns * 1024; ++g_b1k_calls[region]; } }
extern "C" void ggml_expert_b1k_count(int region) { if (region >= 0 && region < 8) ++g_b1k_calls[region]; }
extern "C" void ggml_expert_b1l_add(int region, uint64_t elapsed_ns) { if (region >= 0 && region < 4) { g_b1l_ns[region] += elapsed_ns * 1024; ++g_b1l_calls[region]; } }
extern "C" void ggml_expert_b1l_branch(int branch, int taken) { if (branch >= 0 && branch < 4) { if (taken) ++g_b1l_branch_taken[branch]; else ++g_b1l_branch_not_taken[branch]; } }
extern "C" void ggml_expert_b1m_note_external_storage_eval(void) { ++g_b1m_external_storage_evals; }
extern "C" int ggml_expert_b1n_sample(void) { return (++g_b1n_local_seq & 16383u) == 0; }
extern "C" int ggml_expert_b1n_profile_enabled(void) {
    b1n_calibrate_timers();
    static const bool enabled = []() { const char * p = std::getenv("LLAMA_EXPERT_B1N"); return p && p[0] == '1'; }();
    return enabled ? 1 : 0;
}
extern "C" void ggml_expert_b1n_add(int region, uint64_t elapsed_ns) {
    if (region >= 0 && region < 8) {
        g_b1n_ns[region] += elapsed_ns * 16384;
        ++g_b1n_sample_calls[region];
    }
}
extern "C" void ggml_expert_b1n_note_sampled_call(int region) {
    if (region >= 0 && region < 8) ++g_b1n_sample_calls[region];
}

extern "C" void ggml_expert_indirection_dump(const char * path) {
    if (!path) return; FILE * f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path, "w");
#else
    f = std::fopen(path, "w");
#endif
    if (!f) return;
    std::fprintf(f, "registered=%zu\nvalidation_pass=%llu\nvalidation_fail=%llu\nvalidation_bytes=%llu\ncalls=%llu\nmismatches=%llu\ninvalid_ids=%llu\nmetadata_failures=%llu\nregistry_hits=%llu\nregistry_misses=%llu\n",
        g_storage_by_name.size(),
        (unsigned long long) g_validation_pass, (unsigned long long) g_validation_fail, (unsigned long long) g_validation_bytes,
        (unsigned long long)g_calls.load(), (unsigned long long)g_mismatches.load(),
        (unsigned long long)g_invalid.load(), (unsigned long long)g_metadata_failures.load(),
        (unsigned long long)g_registry_hits.load(), (unsigned long long)g_registry_misses.load());
    std::fprintf(f, "EXPERT_CACHE_STATS_BEGIN\nresolver_calls=%llu\ncached_pointer_returns=%llu\nnormal_pointer_returns=%llu\nhits=%llu\nmisses=%llu\nloads=%llu\nevictions=%llu\npins=%llu\nunpins=%llu\ncurrent_pins=%llu\npeak_simultaneous_pins=%llu\ninvalid_unpins=%llu\neviction_attempt_while_pinned=%llu\nevicted_while_pinned=%llu\ndirect_read_failures=%llu\nresolver_failures=%llu\ngguf_bytes_read=%llu\npeak_occupied_slots=%llu\nEXPERT_CACHE_STATS_END\n",
        (unsigned long long)g_calls.load(), (unsigned long long)g_cached_returns, (unsigned long long)g_normal_returns,
        (unsigned long long)g_cache_hits, (unsigned long long)g_cache_misses, (unsigned long long)g_cache_loads,
        (unsigned long long)g_cache_evictions, (unsigned long long)g_cache_pins, (unsigned long long)g_cache_unpins,
        (unsigned long long)(g_cache_pins-g_cache_unpins), (unsigned long long)g_cache_pins, (unsigned long long)g_cache_bad_unpins,
        (unsigned long long)g_cache_pinned_evictions, (unsigned long long)g_cache_pinned_evictions,
        (unsigned long long)g_direct_read_failures, (unsigned long long)g_resolver_failures,
        (unsigned long long)g_cache_bytes, (unsigned long long)g_cache_peak);
    for (const auto & item : g_storage_by_name) {
        const auto & x = item.second;
        if (x.layer == 0 || x.layer == 15 || x.layer == 29) {
            std::fprintf(f, "storage=%s,file_index=%u,file_id=%d,absolute_offset=%llu,plane_stride=%llu,plane_size=%llu,tensor_size=%llu,type=%d,experts=%u\n",
                x.name, x.file_index, (int)x.os_handle, (unsigned long long)x.absolute_offset,
                (unsigned long long)x.plane_stride, (unsigned long long)x.plane_size,
                (unsigned long long)x.tensor_size, x.ggml_type, x.n_experts);
        }
    }
    std::fprintf(f, "layers=");
    for (int x : g_layers) std::fprintf(f, "%d,", x);
    std::fprintf(f, "\nkinds="); for (const auto & x : g_kinds) std::fprintf(f, "%s,", x.c_str());
    std::fprintf(f, "\n"); std::fclose(f);
}

struct indirection_kind_append {
    ~indirection_kind_append() {
        ggml_expert_b1w_shutdown();
        if (g_perf_path.empty()) return;
        FILE * f = nullptr;
#ifdef _WIN32
        fopen_s(&f, g_perf_path.c_str(), "a");
#else
        f = std::fopen(g_perf_path.c_str(), "a");
#endif
        if (!f) return;
        std::fprintf(f, "short_read_count=%llu\ngate_read_ops=%llu\ngate_bytes_read=%llu\ngate_read_latency_total_us=%llu\ngate_read_latency_min_us=%llu\ngate_read_latency_max_us=%llu\nup_read_ops=%llu\nup_bytes_read=%llu\nup_read_latency_total_us=%llu\nup_read_latency_min_us=%llu\nup_read_latency_max_us=%llu\ndown_read_ops=%llu\ndown_bytes_read=%llu\ndown_read_latency_total_us=%llu\ndown_read_latency_min_us=%llu\ndown_read_latency_max_us=%llu\n",
            (unsigned long long) g_short_reads,
            (unsigned long long) g_kind_ops[0], (unsigned long long) g_kind_bytes[0], (unsigned long long) g_kind_latency_total_us[0], (unsigned long long) (g_kind_ops[0] ? g_kind_latency_min_us[0] : 0), (unsigned long long) g_kind_latency_max_us[0],
            (unsigned long long) g_kind_ops[1], (unsigned long long) g_kind_bytes[1], (unsigned long long) g_kind_latency_total_us[1], (unsigned long long) (g_kind_ops[1] ? g_kind_latency_min_us[1] : 0), (unsigned long long) g_kind_latency_max_us[1],
            (unsigned long long) g_kind_ops[2], (unsigned long long) g_kind_bytes[2], (unsigned long long) g_kind_latency_total_us[2], (unsigned long long) (g_kind_ops[2] ? g_kind_latency_min_us[2] : 0), (unsigned long long) g_kind_latency_max_us[2]);
        std::fprintf(f, "geometry_invariant_failures=%llu\nresolved_pointer_range_failures=%llu\n",
            (unsigned long long) g_geometry_failures, (unsigned long long) g_pointer_range_failures);
#if B1W_PROFILE
        b1w_write_stats(f);
#endif
        std::fprintf(f, "row_bounds_checks=%llu\nrow_bounds_failures=%llu\nrow_offset_overflows=%llu\nrow_end_overflows=%llu\n",
            (unsigned long long) (g_row_bounds_checks + hot_counter_totals().row_bounds_checks), (unsigned long long) g_row_bounds_failures,
            (unsigned long long) g_row_offset_overflows, (unsigned long long) g_row_end_overflows);
        std::fprintf(f, "original_expert_pointer_uses=%llu\ncached_pointer_uses=%llu\nunexpected_mmap_fallbacks=%llu\n",
            (unsigned long long) g_original_pointer_uses, (unsigned long long) g_cached_pointer_uses,
            (unsigned long long) g_unexpected_mmap_fallbacks);
        std::fprintf(f, "parallel_bundle_loads=%llu\n", (unsigned long long) g_parallel_bundle_loads);
        {
            const hot_counters hot = hot_counter_totals();
            std::fprintf(f, "identity_fast_hits=%llu\nidentity_locked_lookups=%llu\nidentity_cache_flushes=%llu\nregistry_epoch=%llu\nregistry_pool_size=%zu\n",
                (unsigned long long) hot.identity_fast_hits, (unsigned long long) hot.identity_locked_lookups,
                (unsigned long long) hot.identity_cache_flushes,
                (unsigned long long) g_registry_epoch.load(std::memory_order_relaxed), g_registry_pool.size());
        }
        if (g_row_failure_recorded) {
            std::fprintf(f, "row_failure layer=%d expert=%d slot=%d pin=%d ir0=%lld nb01=%zu nb02=%zu ne0=%zu ne1=%zu ne2=%zu row_offset=%zu required_bytes=%zu plane_begin=0x%llx plane_end=0x%llx cached_ptr=0x%llx dot_ptr=0x%llx dot_end=0x%llx vec_dot=%s\n",
                g_row_failure.layer, g_row_failure.expert, g_row_failure.slot, g_row_failure.pin, (long long)g_row_failure.ir0,
                g_row_failure.nb01, g_row_failure.nb02, g_row_failure.ne0, g_row_failure.ne1, g_row_failure.ne2,
                g_row_failure.row_offset, g_row_failure.required, (unsigned long long)g_row_failure.plane_begin,
                (unsigned long long)g_row_failure.plane_end, (unsigned long long)g_row_failure.cached_ptr,
                (unsigned long long)g_row_failure.dot_ptr, (unsigned long long)g_row_failure.dot_end,
                g_row_failure.vec_dot ? g_row_failure.vec_dot : "unknown");
        }
        std::fclose(f);
    }
} g_kind_append;

#if B1T_PROFILE
struct b1t_profile_exit_dump {
    ~b1t_profile_exit_dump() {
        const char * path = std::getenv("LLAMA_EXPERT_B1T_OUT");
        if (!path || !path[0]) return;
        FILE * f = nullptr;
        fopen_s(&f, path, "w");
        if (!f) return;
        for (int worker = 0; worker < 64; ++worker) {
            bool any = false;
            for (int phase = 0; phase < B1T_MAX_PHASES; ++phase) any = any || g_b1t_worker_phase_ns[worker][phase].load() != 0;
            if (!any) continue;
            std::fprintf(f, "worker=%d,total_ns=%llu", worker, (unsigned long long) g_b1t_worker_total_ns[worker].load());
            for (int phase = 0; phase < B1T_MAX_PHASES; ++phase) std::fprintf(f, ",phase%d_ns=%llu", phase, (unsigned long long) g_b1t_worker_phase_ns[worker][phase].load());
            std::fprintf(f, "\n");
        }
        std::fclose(f);
    }
} g_b1t_profile_exit_dump;
#endif

#if B1S_PROFILE
struct b1s_profile_exit_dump {
    ~b1s_profile_exit_dump() {
        const char * path = std::getenv("LLAMA_EXPERT_B1S_OUT");
        if (!path || !path[0]) return;
        FILE * f = nullptr;
        fopen_s(&f, path, "w");
        if (!f) return;
        std::fprintf(f, "mmid_wall_ns=%llu\nmmid_aggregate_ns=%llu\nresolver_ns=%llu\ncompute_ns=%llu\nunpin_ns=%llu\n",
            (unsigned long long) g_b1s_mmid_wall_ns.load(),
            (unsigned long long) g_b1s_mmid_aggregate_ns.load(),
            (unsigned long long) g_b1s_region_ns[B1S_REGION_RESOLVER].load(),
            (unsigned long long) g_b1s_region_ns[B1S_REGION_COMPUTE].load(),
            (unsigned long long) g_b1s_region_ns[B1S_REGION_UNPIN].load());
        for (int w = 0; w < B1S_MAX_WORKERS; ++w) {
            const uint64_t mmid = g_b1s_worker_mmid_ns[w].load();
            if (!mmid) continue;
            std::fprintf(f, "worker=%d,mmid_ns=%llu,resolver_ns=%llu,compute_ns=%llu,unpin_ns=%llu\n", w,
                (unsigned long long) mmid,
                (unsigned long long) g_b1s_worker_region_ns[w][B1S_REGION_RESOLVER].load(),
                (unsigned long long) g_b1s_worker_region_ns[w][B1S_REGION_COMPUTE].load(),
                (unsigned long long) g_b1s_worker_region_ns[w][B1S_REGION_UNPIN].load());
        }
        std::fclose(f);
    }
} g_b1s_profile_exit_dump;
#endif

struct profile_exit_dump_late {
    ~profile_exit_dump_late() {
        if ((!ggml_expert_profile_env() && !ggml_expert_mmid_profile_enabled() && !ggml_expert_b1n_profile_enabled() && !ggml_expert_b1p_profile_env() && !B1U_PROFILE) || g_perf_path.empty()) return;
        FILE * f = nullptr; fopen_s(&f, g_perf_path.c_str(), "a"); if (!f) return;
        std::fprintf(f, "profile_resolver_samples=%llu\nprofile_resolver_us=%llu\nprofile_hit_samples=%llu\nprofile_hit_us=%llu\nprofile_miss_samples=%llu\nprofile_miss_us=%llu\nprofile_mul_mat_id_calls=%llu\nprofile_mul_mat_id_us=%llu\ndirect_lookup_hits=%llu\ndirect_lookup_misses=%llu\nstale_mapping_detections=%llu\nmapping_invalidations=%llu\nmapping_publications=%llu\nfallback_scans=%llu\nlockless_hit_attempts=%llu\nlockless_hit_successes=%llu\nlockless_validation_failures=%llu\nlockless_generation_failures=%llu\nlockless_state_failures=%llu\nlockless_post_pin_revalidation_failures=%llu\nlockless_hit_retries=%llu\nlockless_to_locked_fallbacks=%llu\nlock_contention_threshold_ns=%llu\nmutex_acquisition_count=%llu\nmutex_total_wait_ns=%llu\nmutex_max_wait_ns=%llu\ncritical_section_count=%llu\ncritical_section_total_ns=%llu\ncritical_section_max_ns=%llu\nmutex_contended_acquisitions=%llu\nresolver_hit_lock_acquisitions=%llu\nresolver_hit_lock_wait_ns=%llu\nresolver_hit_critical_ns=%llu\nresolver_miss_lock_acquisitions=%llu\nresolver_miss_lock_wait_ns=%llu\nresolver_miss_critical_ns=%llu\nunpin_lock_acquisitions=%llu\nunpin_lock_wait_ns=%llu\nunpin_critical_ns=%llu\n",
            (unsigned long long) g_profile_resolver_samples.load(), (unsigned long long) g_profile_resolver_us.load(),
            (unsigned long long) g_profile_hit_samples.load(), (unsigned long long) g_profile_hit_us.load(),
            (unsigned long long) g_profile_miss_samples.load(), (unsigned long long) g_profile_miss_us.load(),
            (unsigned long long) g_profile_mul_mat_calls.load(), (unsigned long long) g_profile_mul_mat_us.load(),
            (unsigned long long) g_direct_lookup_hits, (unsigned long long) g_direct_lookup_misses,
            (unsigned long long) g_stale_mapping_detections, (unsigned long long) g_mapping_invalidations,
            (unsigned long long) g_mapping_publications, (unsigned long long) g_fallback_scans,
            (unsigned long long) g_lockless_hit_attempts.load(), (unsigned long long) g_lockless_hit_successes.load(),
            (unsigned long long) g_lockless_validation_failures.load(), (unsigned long long) g_lockless_generation_failures.load(),
            (unsigned long long) g_lockless_state_failures.load(), (unsigned long long) g_lockless_post_pin_revalidation_failures.load(),
            (unsigned long long) g_lockless_hit_retries.load(), (unsigned long long) g_lockless_to_locked_fallbacks.load(),
            (unsigned long long) lock_contention_threshold_ns(), (unsigned long long) g_lock_acquisitions.load(),
            (unsigned long long) g_lock_wait_ns.load(), (unsigned long long) g_lock_max_wait_ns.load(),
            (unsigned long long) g_lock_critical_count.load(), (unsigned long long) g_lock_critical_ns.load(),
            (unsigned long long) g_lock_max_critical_ns.load(), (unsigned long long) g_lock_contended.load(),
            (unsigned long long) g_hit_lock_acquisitions.load(), (unsigned long long) g_hit_lock_wait_ns.load(), (unsigned long long) g_hit_lock_critical_ns.load(),
             (unsigned long long) g_miss_lock_acquisitions.load(), (unsigned long long) g_miss_lock_wait_ns.load(), (unsigned long long) g_miss_lock_critical_ns.load(),
             (unsigned long long) g_unpin_lock_acquisitions.load(), (unsigned long long) g_unpin_lock_wait_ns.load(), (unsigned long long) g_unpin_lock_critical_ns.load());
        std::fprintf(f, "mmid_external_calls=%llu\nmmid_external_total_ns=%llu\nmmid_setup_ns=%llu\nmmid_resolver_ns=%llu\nmmid_pointer_setup_ns=%llu\nmmid_dispatch_ns=%llu\nmmid_compute_ns=%llu\nmmid_accumulate_ns=%llu\nmmid_unpin_ns=%llu\nmmid_vecdot_samples=%llu\nmmid_vecdot_sample_ns=%llu\nmmid_vecdot_sampling_ratio=1/1024\nmmid_resolver_sampling_ratio=1/64\nmmid_unpin_sampling_ratio=1/64\n",
            (unsigned long long) g_mmid_external_calls.load(), (unsigned long long) g_mmid_external_total_ns.load(),
            (unsigned long long) g_mmid_region_ns[0].load(), (unsigned long long) g_mmid_region_ns[1].load(),
            (unsigned long long) g_mmid_region_ns[2].load(), (unsigned long long) g_mmid_region_ns[3].load(),
            (unsigned long long) g_mmid_region_ns[4].load(), (unsigned long long) g_mmid_region_ns[5].load(),
            (unsigned long long) g_mmid_region_ns[6].load(), (unsigned long long) g_mmid_vecdot_samples.load(),
            (unsigned long long) g_mmid_vecdot_sample_ns.load());
        std::fprintf(f, "b1k_row_prepare_calls=%llu\nb1k_row_prepare_ns=%llu\nb1k_validate_cached_row_calls=%llu\nb1k_validate_cached_row_ns=%llu\nb1k_external_pointer_calls=%llu\nb1k_external_pointer_ns=%llu\nb1k_loop_overhead_calls=%llu\nb1k_loop_overhead_ns=%llu\nb1k_note_external_vec_dot_calls=%llu\nb1k_note_external_vec_dot_ns=%llu\nb1k_sampling_ratio=1/1024\n",
            (unsigned long long) g_b1k_calls[0].load(), (unsigned long long) g_b1k_ns[0].load(),
            (unsigned long long) g_b1k_calls[1].load(), (unsigned long long) g_b1k_ns[1].load(),
            (unsigned long long) g_b1k_calls[2].load(), (unsigned long long) g_b1k_ns[2].load(),
            (unsigned long long) g_b1k_calls[3].load(), (unsigned long long) g_b1k_ns[3].load(),
            (unsigned long long) g_b1k_calls[4].load(), (unsigned long long) g_b1k_ns[4].load());
        std::fprintf(f, "b1l_inner_row_calls=%llu\nb1l_inner_row_ns=%llu\nb1l_final_vx_calls=%llu\nb1l_final_vx_ns=%llu\nb1l_external_check_calls=%llu\nb1l_external_check_ns=%llu\nb1l_external_storage_enabled_calls=%llu\nb1l_external_storage_enabled_ns=%llu\nb1l_sampling_ratio=1/1024\nb1l_branch_external_taken=%llu\nb1l_branch_external_not_taken=%llu\nb1l_branch_cached_plane_taken=%llu\nb1l_branch_cached_plane_not_taken=%llu\n",
            (unsigned long long) g_b1l_calls[0].load(), (unsigned long long) g_b1l_ns[0].load(),
            (unsigned long long) g_b1l_calls[1].load(), (unsigned long long) g_b1l_ns[1].load(),
            (unsigned long long) g_b1l_calls[2].load(), (unsigned long long) g_b1l_ns[2].load(),
            (unsigned long long) g_b1l_calls[3].load(), (unsigned long long) g_b1l_ns[3].load(),
            (unsigned long long) g_b1l_branch_taken[0].load(), (unsigned long long) g_b1l_branch_not_taken[0].load(),
            (unsigned long long) g_b1l_branch_taken[1].load(), (unsigned long long) g_b1l_branch_not_taken[1].load());
        std::fprintf(f, "b1m_external_storage_enabled_calls=%llu\n", (unsigned long long) g_b1m_external_storage_evals.load());
        std::fprintf(f, "b1n_sampling_ratio=1/16384\nb1n_timer_read_ns=%llu\nb1n_empty_scope_ns=%llu\nb1n_ir0_calls=%llu\nb1n_ir0_ns=%llu\nb1n_validation_calls_sampled=%llu\nb1n_validation_sampled_ns=%llu\nb1n_pointer_trace_calls_sampled=%llu\nb1n_pointer_trace_sampled_ns=%llu\nb1n_sample_decision_calls_sampled=%llu\nb1n_sample_decision_sampled_ns=%llu\nb1n_pointer_path_calls_sampled=%llu\nb1n_pointer_path_sampled_ns=%llu\nb1n_vecdot_sampling_calls_sampled=%llu\nb1n_vecdot_sampling_sampled_ns=%llu\nb1n_vecdot_calls_sampled=%llu\nb1n_vecdot_sampled_ns=%llu\nb1n_bookkeeping_calls_sampled=%llu\nb1n_bookkeeping_sampled_ns=%llu\n",
            (unsigned long long) g_b1n_timer_read_ns, (unsigned long long) g_b1n_empty_scope_ns,
            (unsigned long long) g_b1n_sample_calls[0].load(), (unsigned long long) g_b1n_ns[0].load(),
            (unsigned long long) g_b1n_sample_calls[1].load(), (unsigned long long) g_b1n_ns[1].load(),
            (unsigned long long) g_b1n_sample_calls[2].load(), (unsigned long long) g_b1n_ns[2].load(),
            (unsigned long long) g_b1n_sample_calls[3].load(), (unsigned long long) g_b1n_ns[3].load(),
            (unsigned long long) g_b1n_sample_calls[4].load(), (unsigned long long) g_b1n_ns[4].load(),
            (unsigned long long) g_b1n_sample_calls[5].load(), (unsigned long long) g_b1n_ns[5].load(),
            (unsigned long long) g_b1n_sample_calls[6].load(), (unsigned long long) g_b1n_ns[6].load(),
            (unsigned long long) g_b1n_sample_calls[7].load(), (unsigned long long) g_b1n_ns[7].load());
        std::fprintf(f, "b1p_sample_mask=%llu\nb1p_resolver_samples=%llu\nb1p_hit_samples=%llu\nb1p_miss_samples=%llu\nb1p_resolver_ns=%llu\nb1p_registry_ns=%llu\nb1p_metadata_ns=%llu\nb1p_cache_hit_ns=%llu\nb1p_shared_gate_ns=%llu\nb1p_direct_map_ns=%llu\nb1p_slot_validation_ns=%llu\nb1p_pin_ns=%llu\nb1p_hit_return_ns=%llu\nb1p_miss_lock_ns=%llu\nb1p_lru_ns=%llu\nb1p_backing_read_ns=%llu\nb1p_publication_ns=%llu\n",
            (unsigned long long) B1P_SAMPLE_MASK,
            (unsigned long long) g_b1p_resolver_samples.load(), (unsigned long long) g_b1p_hit_samples.load(), (unsigned long long) g_b1p_miss_samples.load(),
            (unsigned long long) g_b1p_resolver_ns.load(), (unsigned long long) g_b1p_registry_ns.load(), (unsigned long long) g_b1p_metadata_ns.load(),
            (unsigned long long) g_b1p_cache_hit_ns.load(), (unsigned long long) g_b1p_shared_gate_ns.load(), (unsigned long long) g_b1p_direct_map_ns.load(),
            (unsigned long long) g_b1p_slot_validation_ns.load(), (unsigned long long) g_b1p_pin_ns.load(), (unsigned long long) g_b1p_hit_return_ns.load(),
            (unsigned long long) g_b1p_miss_lock_ns.load(), (unsigned long long) g_b1p_lru_ns.load(), (unsigned long long) g_b1p_backing_read_ns.load(), (unsigned long long) g_b1p_publication_ns.load());
        std::fclose(f);
    }
} g_profile_exit_dump_late;

struct indirection_exit_dump {
    indirection_exit_dump() { if (const char * p = std::getenv("LLAMA_EXPERT_INDIRECTION_STATS")) g_path = p; if (const char * p = std::getenv("LLAMA_EXPERT_PERF_OUT")) g_perf_path = p; }
    ~indirection_exit_dump() { if (!g_path.empty()) ggml_expert_indirection_dump(g_path.c_str()); if (!g_perf_path.empty()) { FILE * f=nullptr; fopen_s(&f,g_perf_path.c_str(),"w"); if(f){auto occ=std::count_if(g_cache_slots.begin(),g_cache_slots.end(),[](const cache_slot&s){return s.valid;});std::fprintf(f,"resolver_calls=%llu\ncached_pointer_returns=%llu\nnormal_pointer_returns=%llu\ncache_hits=%llu\ncache_misses=%llu\ncache_loads=%llu\ncache_evictions=%llu\npins=%llu\nunpins=%llu\ncurrent_pins=%llu\ninvalid_unpins=%llu\nevicted_while_pinned=%llu\ndirect_read_failures=%llu\nresolver_failures=%llu\ninvalid_ids=%llu\ngguf_bytes_read=%llu\ngguf_read_operations=%llu\nread_latency_total_us=%llu\nread_latency_min_us=%llu\nread_latency_max_us=%llu\nread_latency_count=%llu\ncache_capacity_bytes=%llu\ncache_slot_count=%zu\nslot_size=%zu\npeak_occupied_slots=%llu\nfinal_occupied_slots=%zu\n",(unsigned long long)g_calls.load(),(unsigned long long)g_cached_returns,(unsigned long long)g_normal_returns,(unsigned long long)g_cache_hits,(unsigned long long)g_cache_misses,(unsigned long long)g_cache_loads,(unsigned long long)g_cache_evictions,(unsigned long long)g_cache_pins,(unsigned long long)g_cache_unpins,(unsigned long long)(g_cache_pins-g_cache_unpins),(unsigned long long)g_cache_bad_unpins,(unsigned long long)g_cache_pinned_evictions,(unsigned long long)g_direct_read_failures,(unsigned long long)g_resolver_failures,(unsigned long long)g_invalid.load(),(unsigned long long)g_read_bytes,(unsigned long long)g_read_ops,(unsigned long long)g_read_latency_total_us,(unsigned long long)(g_read_latency_count?g_read_latency_min_us:0),(unsigned long long)g_read_latency_max_us,(unsigned long long)g_read_ops,(unsigned long long)g_cache_slab.size(),g_cache_slots.size(),g_cache_bundle,(unsigned long long)g_cache_peak,occ);std::fprintf(f,"resolver_success=%llu\nexternal_vec_dot_calls=%llu\nvec_dot_reserved_pointer_violations=%llu\nexternal_logical_bytes=%llu\nexternal_reserved_bytes=%llu\nexternal_committed_bytes=%llu\nexternal_buffer_failures=%llu\nunexpected_mmap_expert_bindings=%llu\nvalidation_started=%d\nvalidation_completed=%d\n",(unsigned long long)g_resolver_success.load(),(unsigned long long)g_external_vec_dot_calls.load(),(unsigned long long)g_reserved_pointer_violations.load(),(unsigned long long)g_external_logical_bytes.load(),(unsigned long long)g_external_reserved_bytes.load(),(unsigned long long)g_external_committed_bytes.load(),(unsigned long long)g_external_buffer_failures.load(),(unsigned long long)g_unexpected_mmap_expert_bindings.load(),g_validation_started.load()?1:0,g_validation_completed.load()?1:0);std::fclose(f);}} }
} g_exit_dump;

struct profile_exit_dump {
    ~profile_exit_dump() {
        if (!ggml_expert_profile_env() || g_perf_path.empty()) return;
        FILE * f = nullptr; fopen_s(&f, g_perf_path.c_str(), "a"); if (!f) return;
        std::fprintf(f, "profile_resolver_samples=%llu\nprofile_resolver_us=%llu\nprofile_hit_samples=%llu\nprofile_hit_us=%llu\nprofile_miss_samples=%llu\nprofile_miss_us=%llu\nprofile_mul_mat_id_calls=%llu\nprofile_mul_mat_id_us=%llu\n",
            (unsigned long long) g_profile_resolver_samples.load(), (unsigned long long) g_profile_resolver_us.load(),
            (unsigned long long) g_profile_hit_samples.load(), (unsigned long long) g_profile_hit_us.load(),
            (unsigned long long) g_profile_miss_samples.load(), (unsigned long long) g_profile_miss_us.load(),
            (unsigned long long) g_profile_mul_mat_calls.load(), (unsigned long long) g_profile_mul_mat_us.load());
        std::fclose(f);
    }
} g_profile_exit_dump;

#ifdef _WIN32
static LONG WINAPI b1b_unhandled_exception_filter(EXCEPTION_POINTERS * ep) {
    if (ep && ep->ExceptionRecord) {
        const auto * er = ep->ExceptionRecord;
        std::fprintf(stderr, "B1B_SEH_EXCEPTION code=0x%08lx address=%p thread=%lu access_type=%ld fault_target=%p\n",
            (unsigned long) er->ExceptionCode, er->ExceptionAddress, (unsigned long) GetCurrentThreadId(),
            er->NumberParameters > 0 ? (long) er->ExceptionInformation[0] : -1L,
            er->NumberParameters > 1 ? (void *) (uintptr_t) er->ExceptionInformation[1] : nullptr);
        std::fflush(stderr);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
struct b1b_seh_install {
    b1b_seh_install() { SetUnhandledExceptionFilter(b1b_unhandled_exception_filter); }
} g_b1b_seh_install;
#endif
