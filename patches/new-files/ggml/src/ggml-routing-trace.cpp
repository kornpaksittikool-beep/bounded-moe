// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 kornpaksittikool-beep
#include "ggml-routing-trace.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct routing_record { uint64_t seq; int layer; int ids[8]; };
static std::mutex g_mutex;
static std::vector<routing_record> g_records;
static std::atomic<bool> g_enabled{false};
static std::atomic<uint64_t> g_failures{0};
static std::atomic<bool> g_atexit_registered{false};
static std::string g_path;

static void dump_at_exit() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_path.empty()) return;
    FILE * f = nullptr;
#ifdef _WIN32
    fopen_s(&f, g_path.c_str(), "w");
#else
    f = fopen(g_path.c_str(), "w");
#endif
    if (!f) return;
    fprintf(f, "sequence,layer,expert0,expert1,expert2,expert3,expert4,expert5,expert6,expert7\n");
    for (const auto & r : g_records) {
        fprintf(f, "%llu,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", (unsigned long long) r.seq, r.layer,
                r.ids[0],r.ids[1],r.ids[2],r.ids[3],r.ids[4],r.ids[5],r.ids[6],r.ids[7]);
    }
    fclose(f);
}

void ggml_routing_trace_set_enabled(int enabled) {
    if (!g_atexit_registered.exchange(true)) {
        const char * p = std::getenv("LLAMA_ROUTING_TRACE_OUT");
        if (p) { g_path = p; std::atexit(dump_at_exit); }
    }
    g_enabled.store(enabled != 0, std::memory_order_release);
}
void ggml_routing_trace_reset(void) { std::lock_guard<std::mutex> lock(g_mutex); g_records.clear(); g_failures.store(0); }
int ggml_routing_trace_is_enabled(void) {
    return g_enabled.load(std::memory_order_acquire) ? 1 : 0;
}
uint64_t ggml_routing_trace_parse_failures(void) { return g_failures.load(); }
void ggml_routing_trace_dump(const char * path) { if (path) { g_path = path; dump_at_exit(); } }

void ggml_routing_trace_record(const char * name, const int32_t * ids, int n_ids) {
    if (!ggml_routing_trace_is_enabled()) return;
    if (!name || !strstr(name, "ffn_gate_exps")) return;
    if (n_ids != 8) { g_failures.fetch_add(1); return; }
    const char * p = strstr(name, "blk.");
    if (!p) { g_failures.fetch_add(1); return; }
    p += 4; char * end = nullptr; long layer = strtol(p, &end, 10);
    if (end == p || (strncmp(end, ".ffn_gate_exps.weight", 21) != 0 && strncmp(end, ".ffn_gate_exps", 14) != 0) || layer < 0 || layer >= 40) { g_failures.fetch_add(1); return; }
    routing_record r{}; r.layer = (int) layer;
    for (int i = 0; i < 8; ++i) { if (ids[i] < 0 || ids[i] > 255) { g_failures.fetch_add(1); return; } r.ids[i] = ids[i]; }
    std::lock_guard<std::mutex> lock(g_mutex); r.seq = g_records.size(); g_records.push_back(r);
}
