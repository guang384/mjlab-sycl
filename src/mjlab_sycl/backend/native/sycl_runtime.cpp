/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Implementation of the SYCL micro-driver (see sycl_runtime.h).
// Built once into warpsycl.dll; kernel module DLLs link against warpsycl.lib.

#include "sycl_runtime.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr unsigned int kIntelVendorId = 0x8086;

sycl::queue make_intel_queue() {
    for (auto const& platform : sycl::platform::get_platforms()) {
        for (auto const& dev : platform.get_devices(sycl::info::device_type::gpu)) {
            if (dev.get_info<sycl::info::device::vendor_id>() == kIntelVendorId) {
                // in_order: submission order == execution order, matching Warp's
                // sequential launch semantics (also a prerequisite for the
                // async-submission optimization).
                return sycl::queue{dev, sycl::property::queue::in_order()};
            }
        }
    }
    throw std::runtime_error("wp_sycl: no Intel GPU found");
}

// ---------------------------------------------------------------------------
// Watchdog: a hung device kernel (e.g. a deadlocked work-group barrier) never
// completes, so the host would spin in queue wait forever -- and because this
// is an iGPU, the display driver hangs with it. Windows TDR can reset the
// GPU, but only if the process stops re-submitting into the dead queue.
//
// wp_sycl_synchronize() therefore arms a deadline before waiting; a watcher
// thread that fires if the wait exceeds it hard-exits the process with the
// name of the last submitted kernel. The process dies (with a diagnostic),
// TDR resets the iGPU, the desktop survives. Timeout via
// WARP_SYCL_SYNC_TIMEOUT_S (seconds, default 180, 0 disables).
// ---------------------------------------------------------------------------
static std::atomic<const char*> g_last_kernel{nullptr};
static std::atomic<bool> g_in_sync{false};
static std::atomic<long long> g_sync_deadline_ms{0};
static std::atomic<long long> g_timeout_ms{180 * 1000};

static long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

static void watchdog_loop() {
    long long fired = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (!g_in_sync.load(std::memory_order_acquire)) continue;
        const long long deadline = g_sync_deadline_ms.load(std::memory_order_relaxed);
        if (now_ms() < deadline || fired == deadline) continue;
        fired = deadline;
        const char* k = g_last_kernel.load(std::memory_order_relaxed);
        std::fprintf(stderr,
            "\nwp_sycl: WATCHDOG -- the device queue did not drain within the sync timeout.\n"
            "  last kernel submitted: %s\n"
            "  This is a device-side hang (divergent work-group barrier or GPU fault).\n"
            "  Aborting so the display driver (TDR) can reset the GPU; the desktop\n"
            "  should recover in seconds. Tune with WARP_SYCL_SYNC_TIMEOUT_S.\n",
            k ? k : "<unknown>");
        std::fflush(stderr);
        std::_Exit(2);  // no destructors, no further submissions
    }
}

static void start_watchdog() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* t = std::getenv("WARP_SYCL_SYNC_TIMEOUT_S");
        if (t) {
            const long long v = std::atoll(t);
            g_timeout_ms.store(v > 0 ? v * 1000 : 0);
        }
        if (g_timeout_ms.load() > 0)
            std::thread(watchdog_loop).detach();
    });
}

sycl::queue& the_queue() {
    static sycl::queue q = make_intel_queue();
    return q;
}

// ---------------------------------------------------------------------------
// USM pool: sycl::malloc_shared is a driver round-trip (tens of us) and the
// naive free must first drain the whole queue, because an in-flight kernel
// may still hold the pointer. Training workloads allocate and free the same
// array shapes every step, so recycling blocks is a large win.
//
// Safety rule: a freed block is only handed out again after a full
// wp_sycl_synchronize() happened *after* the free. Kernels submitted before
// that drain have completed by then, so nothing can still be referencing the
// block; kernels submitted after the alloc see the new owner in program
// order. Blocks freed since the last drain sit in `pending` and are moved
// into the per-size free lists by the drain itself.
// ---------------------------------------------------------------------------
struct UsmPool {
    std::mutex mutex;
    std::unordered_map<void*, size_t> sizes;      // live + pending blocks
    std::vector<std::pair<void*, size_t>> pending;  // freed, not yet drained
    std::unordered_map<size_t, std::vector<void*>> free_lists;
};

UsmPool& the_pool() {
    static UsmPool pool;
    return pool;
}

}  // namespace

namespace {

// Drain the queue and recycle every block freed before this point: kernels
// submitted earlier have completed, so nothing can still reference them.
void drain_and_recycle() {
    the_queue().wait();
    UsmPool& pool = the_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (auto& [ptr, size] : pool.pending) {
        pool.free_lists[size].push_back(ptr);
        pool.sizes[ptr] = size;
    }
    pool.pending.clear();
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------
// Command-graph capture/replay (see sycl_runtime.h). While recording, every
// submission to the shared queue becomes a graph node whose arguments are
// baked as POINTERS -- a replayed node reads the same USM slots again. The
// stage_args / memtile rings recycle their slots after kRing submissions, so
// a later replay would eventually observe another launch's arguments.
// Recording therefore draws from a dedicated per-graph arena that lives and
// dies with the graph.
// ---------------------------------------------------------------------------
namespace exptl = sycl::ext::oneapi::experimental;
using mod_graph_t = exptl::command_graph<exptl::graph_state::modifiable>;
using exec_graph_t = exptl::command_graph<exptl::graph_state::executable>;

struct GraphState {
    std::optional<mod_graph_t> mod;
    std::optional<exec_graph_t> exec;
    std::vector<void*> chunks;  // argument arena blocks, never recycled
    size_t chunk_off = 0;
    size_t chunk_cap = 0;
};

thread_local GraphState* g_recording = nullptr;

void* graph_arena_alloc(GraphState* g, size_t size) {
    const size_t need = (size + 15) & ~size_t(15);
    if (g->chunk_off + need > g->chunk_cap) {
        const size_t cap = need > (size_t(1) << 20) ? need : (size_t(1) << 20);
        void* p = sycl::malloc_shared(cap, the_queue());
        if (!p) throw std::runtime_error("wp_sycl: graph argument arena alloc failed");
        g->chunks.push_back(p);
        g->chunk_off = 0;
        g->chunk_cap = cap;
    }
    void* out = static_cast<unsigned char*>(g->chunks.back()) + g->chunk_off;
    g->chunk_off += need;
    return out;
}

}  // namespace

namespace {

template <int N>
static void chol_solve_submit(sycl::queue& q, const float* h,
                              const float* grad, const unsigned char* done,
                              const int* changed,
                              const unsigned char* lvalid_in, float* L,
                              unsigned char* lvalid_out, float* Mgrad,
                              long long stride, long long batch) {
    q.parallel_for(sycl::range<1>(static_cast<size_t>(batch)),
                   [=](sycl::id<1> idx) {
        const int w = static_cast<int>(idx.get(0));
        if (done[w]) return;
        float* Lw = L + static_cast<size_t>(w) * stride * stride;
        const float* hw = h + static_cast<size_t>(w) * stride * stride;
        const float* gw = grad + static_cast<size_t>(w) * stride;
        float* mw = Mgrad + static_cast<size_t>(w) * stride;
        if (changed[w] != 0 || !lvalid_in[w]) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j <= i; ++j) {
                    float s = hw[i * stride + j];
                    for (int k = 0; k < j; ++k)
                        s -= Lw[i * stride + k] * Lw[j * stride + k];
                    if (i == j) Lw[i * stride + i] = ::sqrtf(s);
                    else Lw[i * stride + j] = s / Lw[j * stride + j];
                }
            }
            lvalid_out[w] = 1;
        }
        for (int i = 0; i < N; ++i) {
            float s = gw[i];
            for (int k = 0; k < i; ++k) s -= Lw[i * stride + k] * mw[k];
            mw[i] = s / Lw[i * stride + i];
        }
        for (int i = 0; i < N; ++i) {
            const int ii = N - 1 - i;
            float s = mw[ii];
            for (int k = 0; k < N - 1 - ii; ++k) {
                const int kk = N - 1 - k;
                s -= Lw[kk * stride + ii] * mw[kk];
            }
            mw[ii] = s / Lw[ii * stride + ii];
        }
    });
}

}  // namespace

namespace {

template <int N>
static void chol_fs_submit(sycl::queue& q, const float* M, const float* y,
                           float* x, float* L, const int* adr,
                           long long stride, long long batch) {
    q.parallel_for(sycl::range<1>(static_cast<size_t>(batch)),
                   [=](sycl::id<1> idx) {
        const int w = static_cast<int>(idx.get(0));
        // tile anchor read device-side: a host read here would sync the
        // queue mid-step, which is illegal inside command-graph recording
        const int off = adr[0];
        float* Lw = L + static_cast<size_t>(w) * stride * stride;
        const float* Mw = M + static_cast<size_t>(w) * stride * stride;
        const float* yw = y + static_cast<size_t>(w) * stride;
        float* xw = x + static_cast<size_t>(w) * stride;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                float s = Mw[(off + i) * stride + (off + j)];
                for (int k = 0; k < j; ++k)
                    s -= Lw[(off + i) * stride + (off + k)] *
                         Lw[(off + j) * stride + (off + k)];
                if (i == j) Lw[(off + i) * stride + (off + i)] = ::sqrtf(s);
                else Lw[(off + i) * stride + (off + j)] = s / Lw[(off + j) * stride + (off + j)];
            }
        }
        for (int i = 0; i < N; ++i) {
            float s = yw[off + i];
            for (int k = 0; k < i; ++k)
                s -= Lw[(off + i) * stride + (off + k)] * xw[off + k];
            xw[off + i] = s / Lw[(off + i) * stride + (off + i)];
        }
        for (int i = 0; i < N; ++i) {
            const int ii = N - 1 - i;
            float s = xw[off + ii];
            for (int k = 0; k < N - 1 - ii; ++k) {
                const int kk = N - 1 - k;
                s -= Lw[(off + kk) * stride + (off + ii)] * xw[off + kk];
            }
            xw[off + ii] = s / Lw[(off + ii) * stride + (off + ii)];
        }
    });
}

}  // namespace

extern "C" {

void* wp_sycl_queue_ptr() { return &the_queue(); }

void* wp_sycl_alloc_shared(size_t size) {
    if (size == 0) {
        size = 1;
    }

    UsmPool& pool = the_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);

    // only blocks released before the last drain are safe to reuse
    auto it = pool.free_lists.find(size);
    if (it != pool.free_lists.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    void* ptr = sycl::malloc_shared(size, the_queue());
    if (ptr != nullptr) {
        pool.sizes[ptr] = size;
    }
    return ptr;
}

void wp_sycl_free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    UsmPool& pool = the_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);

    auto it = pool.sizes.find(ptr);
    if (it == pool.sizes.end()) {
        // not ours (e.g. staging ring owns its slots separately): fall back
        // to the conservative wait-and-free
        the_queue().wait();
        sycl::free(ptr, the_queue());
        return;
    }

    // hold the block until a drain proves no in-flight kernel references it
    pool.pending.emplace_back(ptr, it->second);
    pool.sizes.erase(it);
}

// Return every pooled block to the OS: free lists retain the construction
// high-water mark forever otherwise (measured: ~0.6 MB/env of churn kept
// resident). Drains first so pending frees become safe to release. Only
// touches pooled blocks -- live arrays are never in the free lists.

void wp_sycl_pool_trim() {
    drain_and_recycle();
    UsmPool& pool = the_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (auto& [size, blocks] : pool.free_lists) {
        for (void* p : blocks) {
            sycl::free(p, the_queue());
        }
    }
    pool.free_lists.clear();
}

// Pool census: (live_bytes, free_bytes, pending_bytes) across the pool.
void wp_sycl_pool_stats(long long* live, long long* free_b, long long* pending) {
    UsmPool& pool = the_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    long long lv = 0, frb = 0, pd = 0;
    for (auto& [ptr, size] : pool.sizes) lv += size;
    for (auto& [size, blocks] : pool.free_lists) frb += size * blocks.size();
    for (auto& [ptr, size] : pool.pending) pd += size;
    if (live) *live = lv;
    if (free_b) *free_b = frb;
    if (pending) *pending = pd;
}

// Pool size histogram: fills (size_bytes, count) pairs sorted by bytes
// descending, up to max_n entries; returns the number written. The size
// fingerprints the allocating call site.
int wp_sycl_pool_hist(long long* out_pairs, int max_n) {
    UsmPool& pool = the_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    std::unordered_map<size_t, long long> counts;
    for (auto& [ptr, size] : pool.sizes) counts[size] += 1;
    for (auto& [size, blocks] : pool.free_lists) counts[size] += (long long)blocks.size();
    for (auto& [ptr, size] : pool.pending) counts[size] += 1;
    std::vector<std::pair<long long, size_t>> rows;
    for (auto& [size, n] : counts) rows.push_back({(long long)size * n, size});
    std::sort(rows.rbegin(), rows.rend());
    int k = 0;
    for (auto& [bytes, size] : rows) {
        if (k >= max_n) break;
        out_pairs[k * 2] = (long long)size;
        out_pairs[k * 2 + 1] = counts[size];
        ++k;
    }
    return k;
}

void wp_sycl_memset(void* ptr, int value, size_t size) {
    g_last_kernel.store("wp_sycl_memset");
    if (ptr != nullptr && size > 0) {
        try {
            the_queue().memset(ptr, value, size);
        } catch (sycl::exception const& e) {
            std::fprintf(stderr, "wp_sycl_memset failed: %s\n", e.what());
            std::abort();
        }
    }
}

void wp_sycl_memcpy(void* dst, const void* src, size_t size) {
    g_last_kernel.store("wp_sycl_memcpy");
    if (dst != nullptr && src != nullptr && size > 0) {
        try {
            the_queue().memcpy(dst, src, size);
        } catch (sycl::exception const& e) {
            std::fprintf(stderr, "wp_sycl_memcpy failed: %s\n", e.what());
            std::abort();
        }
    }
}

void wp_sycl_memtile(void* dst, const void* src, size_t src_size, size_t reps) {
    g_last_kernel.store("wp_sycl_memtile");
    constexpr size_t kPatternCap = 256;  // covers scalar/vector/matrix fills
    if (dst == nullptr || src == nullptr || src_size == 0 || reps == 0) {
        return;
    }
    if (src_size > kPatternCap) {
        // struct-sized patterns are rare: conservative host path
        drain_and_recycle();
        unsigned char* d = static_cast<unsigned char*>(dst);
        unsigned char const* s = static_cast<unsigned char const*>(src);
        for (size_t r = 0; r < reps; ++r) {
            std::memcpy(d + r * src_size, s, src_size);
        }
        return;
    }

    // Stage the pattern in USM shared memory so the device kernel can read it;
    // a plain host pointer (e.g. thread_local storage) is NOT device-accessible
    // under Level-Zero, and the kernel would silently read nothing. The slot
    // must stay alive while the fill is in flight, so reuse the same ring
    // discipline as wp_sycl_stage_args (drain every kPatternRing-th call).
    constexpr size_t kPatternRing = 256;
    struct PatternSlots {
        // one USM block per slot; allocated lazily, never freed
        unsigned char* ptrs[kPatternRing] = {};
    };
    thread_local PatternSlots* slots = nullptr;
    thread_local unsigned long counter = 0;

    unsigned char* pat = nullptr;
    if (g_recording != nullptr) {
        // graph recording: same lifetime rule as stage_args -- the pattern
        // slot must never be recycled under a future replay
        pat = static_cast<unsigned char*>(graph_arena_alloc(g_recording, kPatternCap));
    } else {
        if (slots == nullptr) {
            slots = new PatternSlots();
        }
        if (counter != 0 && counter % kPatternRing == 0) {
            drain_and_recycle();
        }
        pat = slots->ptrs[counter % kPatternRing];
        if (pat == nullptr) {
            pat = static_cast<unsigned char*>(sycl::malloc_shared(kPatternCap, the_queue()));
            if (pat == nullptr) {
                std::fprintf(stderr, "wp_sycl_memtile: pattern USM alloc failed\n");
                std::abort();
            }
            slots->ptrs[counter % kPatternRing] = pat;
        }
        ++counter;
    }
    std::memcpy(pat, src, src_size);

    unsigned char* d = static_cast<unsigned char*>(dst);
    the_queue().parallel_for(
        sycl::range<1>(reps),
        [=](sycl::id<1> r) {
            unsigned char* row = d + r * src_size;
            for (size_t b = 0; b < src_size; ++b) {
                row[b] = pat[b];
            }
        });
}

void* wp_sycl_stage_args(size_t size) {
    // Recording a command graph: the slot must outlive every future replay,
    // so it comes from the graph's private arena instead of the ring below.
    if (g_recording != nullptr) {
        return graph_arena_alloc(g_recording, size);
    }
    // Ring of per-thread staging slots. Kernels are submitted asynchronously,
    // so the host can run ahead of the device: a single buffer would be
    // overwritten while an in-flight kernel still reads it. Slots are reused
    // in ring order, and every kRing-th staging call drains the in-order
    // queue, so by the time a slot comes around again every kernel that was
    // handed its address has completed.
    constexpr unsigned long kRing = 8192;
    struct Slot {
        void* data = nullptr;
        size_t cap = 0;
    };
    thread_local Slot slots[kRing];
    thread_local unsigned long counter = 0;

    if (counter != 0 && counter % kRing == 0) {
        drain_and_recycle();
    }
    Slot& s = slots[counter % kRing];
    ++counter;

    if (size > s.cap) {
        if (s.data != nullptr) {
            the_queue().wait();  // the old buffer may still be referenced
            sycl::free(s.data, the_queue());
        }
        s.cap = size * 2;
        s.data = sycl::malloc_shared(s.cap, the_queue());
        if (s.data == nullptr) {
            s.cap = 0;
            std::fprintf(stderr, "wp_sycl: failed to allocate %zu-byte args staging buffer\n", size);
            std::abort();
        }
    }
    return s.data;
}

void wp_sycl_note_kernel(const char* name) { g_last_kernel.store(name); }

void wp_sycl_synchronize() {
    start_watchdog();
    const long long timeout_ms = g_timeout_ms.load(std::memory_order_relaxed);
    if (timeout_ms > 0) {
        g_sync_deadline_ms.store(now_ms() + timeout_ms, std::memory_order_relaxed);
        g_in_sync.store(true, std::memory_order_release);
        try {
            the_queue().wait_and_throw();
        } catch (const sycl::exception& e) {
            g_in_sync.store(false, std::memory_order_release);
            const char* k = g_last_kernel.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                "\nwp_sycl: device error during synchronize: %s\n"
                "  last kernel submitted: %s\n  Aborting (see WATCHDOG notes).\n",
                e.what(), k ? k : "<unknown>");
            std::fflush(stderr);
            std::_Exit(2);
        }
        g_in_sync.store(false, std::memory_order_release);
    } else {
        the_queue().wait();
    }
    drain_and_recycle();
}


const char* wp_sycl_device_name() {
    static std::string name = the_queue().get_device().get_info<sycl::info::device::name>();
    return name.c_str();
}

// ---- command-graph capture/replay (see sycl_runtime.h) --------------------

void* wp_sycl_graph_begin() {
    try {
        auto* g = new GraphState();
        g->mod.emplace(the_queue());
        g->mod->begin_recording({the_queue()});
        g_recording = g;
        return g;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_graph_begin failed: %s\n", e.what());
        return nullptr;
    }
}

int wp_sycl_graph_end(void* handle) {
    auto* g = static_cast<GraphState*>(handle);
    if (!g) return -1;
    g_recording = nullptr;
    try {
        g->mod->end_recording();
        g->exec.emplace(g->mod->finalize());
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_graph_end failed: %s\n", e.what());
        return -1;
    }
}

int wp_sycl_graph_submit(void* handle) {
    auto* g = static_cast<GraphState*>(handle);
    if (!g || !g->exec) return -1;
    try {
        // watchdog attribution: replay runs the whole batch as one command,
        // so a hang inside it names the batch instead of one kernel
        g_last_kernel.store("wp_sycl_graph_batch");
        the_queue().ext_oneapi_graph(*g->exec);
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_graph_submit failed: %s\n", e.what());
        return -1;
    }
}

void wp_sycl_graph_free(void* handle) {
    auto* g = static_cast<GraphState*>(handle);
    if (!g) return;
    if (g_recording == g) g_recording = nullptr;
    try {
        the_queue().wait();  // in-flight replays must finish first
    } catch (...) {
    }
    for (void* p : g->chunks) sycl::free(p, the_queue());
    delete g;
}

// ---- fused mv + jv for the solver linesearch (bit-exact native rewrite) ----
//
// Replaces the warp-language one-work-item-per-world fused kernel: mv =
// qM @ search (nv rows) and jv = efc_J @ search (min(njmax, nefc) rows).
// Row-block schedule: a 32-lane group per world, each item owning rows
// l, l+32, ... -- every output element stays ONE item's k-ascending dot
// (identical arithmetic to the warp kernel, bit-exact by construction),
// while the 32x item fan-out and DPC++ codegen (measured ~70 GB/s on
// shared USM vs warp's ~59) attack the kernel's latency bound. Rows at
// and beyond nefc are skipped, so no bytes are read for idle rows.
// qM/efc_J/search are read-only; mv/jv are written only below.

int wp_sycl_mv_jv(const void* qM, const void* J, const void* search,
                  const int* nefc, const unsigned char* done, void* mv,
                  void* jv, long long nv, long long njmax, long long nv_pad,
                  long long njmax_pad, long long batch) {
    if (batch <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_mv_jv");
        q.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch) * 32),
                sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const int w = static_cast<int>(it.get_group_linear_id());
                // done worlds skip entirely -- the warp kernel's early return
                // is what makes converged ghost iterations nearly free, and
                // their mv/jv are dead stores (every consumer done-guards)
                if (done[w]) return;
                const int lane = static_cast<int>(it.get_local_linear_id());
                const int rows_total = static_cast<int>(nv + njmax);
                const auto* qMw = static_cast<const float*>(qM) +
                    static_cast<size_t>(w) * nv_pad * nv_pad;
                const auto* Jw = static_cast<const float*>(J) +
                    static_cast<size_t>(w) * njmax_pad * nv_pad;
                const auto* sw = static_cast<const float*>(search) +
                    static_cast<size_t>(w) * nv_pad;
                const int nef = static_cast<int>(nefc[w]);
                auto* mvw = static_cast<float*>(mv) + static_cast<size_t>(w) * nv_pad;
                auto* jvw = static_cast<float*>(jv) + static_cast<size_t>(w) * njmax;

                for (int r = lane; r < rows_total; r += 32) {
                    if (r < nv) {
                        // mv row: qM[r, :] @ search, k-ascending in this item
                        const auto* row = qMw + static_cast<size_t>(r) * nv_pad;
                        float s = 0.0f;
                        for (int i = 0; i < nv; ++i) s += row[i] * sw[i];
                        mvw[r] = s;
                    } else {
                        const int e = r - nv;
                        if (e >= nef || e >= njmax) continue;
                        const auto* row = Jw + static_cast<size_t>(e) * nv_pad;
                        float s = 0.0f;
                        for (int i = 0; i < nv; ++i) s += row[i] * sw[i];
                        jvw[e] = s;
                    }
                }
            });
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_mv_jv failed: %s\n", e.what());
        return -1;
    }
}

// ---- JTDAJ: h = qM + J^T D' J (bit-exact native rewrite) -------------------
//
// One output element (i, j) of h per work-item, dot over constraints
// k-ascending with the same zeroing rules as the warp kernel (Dk forced to
// zero for non-QUADRATIC states, zero Dk skipped) -- identical arithmetic,
// bit-exact by construction. 32-lane row-block schedule: adjacent items
// cover adjacent columns of the same h row, so their per-k J loads share
// cache lines, and Dk/state reads broadcast through L1. Rows at and beyond
// nefc are skipped; done worlds return entirely (dead stores downstream).

int wp_sycl_jtdaj(const void* qM, const void* J, const void* D,
                  const int* state, const int* nefc,
                  const unsigned char* done, void* h,
                  long long nv_pad, long long njmax_pad, long long batch) {
    if (batch <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_jtdaj");
        q.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch) * 32),
                sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const int w = static_cast<int>(it.get_group_linear_id());
                if (done[w]) return;
                const int lane = static_cast<int>(it.get_local_linear_id());
                const int elems = static_cast<int>(nv_pad * nv_pad);
                const auto* qMw = static_cast<const float*>(qM) +
                    static_cast<size_t>(w) * nv_pad * nv_pad;
                const auto* Jw = static_cast<const float*>(J) +
                    static_cast<size_t>(w) * njmax_pad * nv_pad;
                const auto* Dw = static_cast<const float*>(D) +
                    static_cast<size_t>(w) * njmax_pad;
                const auto* stw = state + static_cast<size_t>(w) * njmax_pad;
                const int nef = nefc[w];
                auto* hw = static_cast<float*>(h) +
                    static_cast<size_t>(w) * nv_pad * nv_pad;
                constexpr int kQuadratic = 1;  // mjCNSTRSTATE_QUADRATIC

                for (int e = lane; e < elems; e += 32) {
                    const int i = e / static_cast<int>(nv_pad);
                    const int j = e - i * static_cast<int>(nv_pad);
                    float s = qMw[e];
                    for (int k = 0; k < nef; ++k) {
                        float Dk = Dw[k];
                        if (stw[k] != kQuadratic) Dk = 0.0f;
                        if (Dk == 0.0f) continue;
                        s += (Jw[static_cast<size_t>(k) * nv_pad + i] * Dk) *
                             Jw[static_cast<size_t>(k) * nv_pad + j];
                    }
                    hw[e] = s;
                }
            });
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_jtdaj failed: %s\n", e.what());
        return -1;
    }
}

// ---- solver cholesky factor+solve (bit-exact native rewrite) ---------------
//
// One work-item per world, exactly the warp flat kernel's structure: LLT of
// h into L (skipped per world when `changed` == 0 and L is still valid --
// the incremental path's skip contract), then forward/back substitution
// grad -> Mgrad. Buffers are nv_pad-strided; loops run over the real n.
// Instantiated per size so the N-loops fully unroll (the warp kernel's
// max_unroll is load-bearing -- measured 94x slower unrolled-off); exotic
// sizes report -5 and the caller stays on the warp kernel.


WP_SYCL_API int wp_sycl_chol_solve(const void* h, const void* grad,
                       const unsigned char* done, const int* changed,
                       const unsigned char* lvalid_in, void* L,
                       unsigned char* lvalid_out, void* Mgrad,
                       long long n, long long stride, long long batch) {
    if (batch <= 0 || n <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_chol_solve");
        switch (n) {
        case 4:  chol_solve_submit<4>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 8:  chol_solve_submit<8>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 12: chol_solve_submit<12>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 16: chol_solve_submit<16>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 20: chol_solve_submit<20>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 24: chol_solve_submit<24>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 28: chol_solve_submit<28>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        case 32: chol_solve_submit<32>(q, static_cast<const float*>(h), static_cast<const float*>(grad), done, changed, lvalid_in, static_cast<float*>(L), lvalid_out, static_cast<float*>(Mgrad), stride, batch); return 0;
        default: return -5;
        }
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_chol_solve failed: %s\n", e.what());
        return -1;
    }
}

// ---- set-const cholesky factorize+solve (bit-exact native rewrite) ---------
//
// Same math as the warp _tile_cholesky_factorize_solve for the single-tile
// case (tile anchored at the origin): LLT of M into L, then y -> x through
// L. Always factorizes; no done guard (reset path). Per-size templates.


WP_SYCL_API int wp_sycl_chol_fs(const void* M, const void* y, void* x,
                                void* L, const int* adr, long long n,
                                long long stride, long long batch) {
    if (batch <= 0 || n <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_chol_fs");
        switch (n) {
        case 4:  chol_fs_submit<4>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 8:  chol_fs_submit<8>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 12: chol_fs_submit<12>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 16: chol_fs_submit<16>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 20: chol_fs_submit<20>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 24: chol_fs_submit<24>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 28: chol_fs_submit<28>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        case 32: chol_fs_submit<32>(q, static_cast<const float*>(M), static_cast<const float*>(y), static_cast<float*>(x), static_cast<float*>(L), adr, stride, batch); return 0;
        default: return -5;
        }
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_chol_fs failed: %s\n", e.what());
        return -1;
    }
}

// ---- incremental Hessian update h += +-D J^T J over changed constraints ----
//
// Bit-exact native rewrite of update_gradient_h_incremental: one item per
// (world, lower-triangle element); the element -> (i, j) mapping and the
// per-changed-constraint accumulation order are the warp kernel's.

WP_SYCL_API int wp_sycl_hinc(const void* J, const void* D, const int* state,
                             const int* changed_ids, const int* changed_count,
                             void* h, long long nv_pad, long long efc_stride,
                             long long ids_stride, long long batch) {
    if (batch <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_hinc");
        q.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch) * 32),
                sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const int w = static_cast<int>(it.get_group_linear_id());
                const int lane = static_cast<int>(it.get_local_linear_id());
                const int elems = static_cast<int>(nv_pad * (nv_pad + 1) / 2);
                const auto* Jw = static_cast<const float*>(J) +
                    static_cast<size_t>(w) * efc_stride * nv_pad;
                const auto* Dw = static_cast<const float*>(D) +
                    static_cast<size_t>(w) * efc_stride;
                const auto* stw = state + static_cast<size_t>(w) * efc_stride;
                const int n_changes = changed_count[w];
                auto* hw = static_cast<float*>(h) +
                    static_cast<size_t>(w) * nv_pad * nv_pad;
                constexpr int kQuadratic = 1;

                for (int e = lane; e < elems; e += 32) {
                    const int i = (static_cast<int>(
                                       ::sqrtf(static_cast<float>(1 + 8 * e))) -
                                   1) / 2;
                    const int j = e - (i * (i + 1)) / 2;
                    float delta = 0.0f;
                    for (int ci = 0; ci < n_changes; ++ci) {
                        const int efcid = changed_ids[w * ids_stride + ci];
                        const float Ji = Jw[static_cast<size_t>(efcid) * nv_pad + i];
                        if (Ji == 0.0f) continue;
                        const float Jj = Jw[static_cast<size_t>(efcid) * nv_pad + j];
                        if (Jj == 0.0f) continue;
                        const float Dk = Dw[efcid];
                        if (stw[efcid] == kQuadratic) delta += Dk * Ji * Jj;
                        else delta -= Dk * Ji * Jj;
                    }
                    if (delta != 0.0f) hw[i * nv_pad + j] += delta;
                }
            });
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_hinc failed: %s\n", e.what());
        return -1;
    }
}

// ---- qfrc_constraint = efc_J^T @ force (bit-exact native rewrite) ----------
//
// One work-item per (world, dof) row-block: the per-dof dot over efc rows is
// k-ascending inside one item (identical arithmetic), done worlds return.

WP_SYCL_API int wp_sycl_qfrc_constraint(const void* J, const void* force,
                                        const int* nefc,
                                        const unsigned char* done, void* out,
                                        long long nv, long long nv_pad,
                                        long long njmax_pad, long long batch) {
    if (batch <= 0 || nv <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_qfrc_constraint");
        q.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch) * 32),
                sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const int w = static_cast<int>(it.get_group_linear_id());
                if (done[w]) return;
                const int lane = static_cast<int>(it.get_local_linear_id());
                const auto* Jw = static_cast<const float*>(J) +
                    static_cast<size_t>(w) * njmax_pad * nv_pad;
                const auto* fw = static_cast<const float*>(force) +
                    static_cast<size_t>(w) * njmax_pad;
                const int nef = nefc[w] < njmax_pad ? nefc[w] : (int)njmax_pad;
                auto* ow = static_cast<float*>(out) +
                    static_cast<size_t>(w) * nv_pad;
                for (int dof = lane; dof < nv; dof += 32) {
                    float s = 0.0f;
                    for (int e = 0; e < nef; ++e) {
                        s += Jw[static_cast<size_t>(e) * nv_pad + dof] * fw[e];
                    }
                    ow[dof] = s;
                }
            });
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_qfrc_constraint failed: %s\n", e.what());
        return -1;
    }
}

// ---- fused linesearch_jaref + zero-ahead (bit-exact native rewrite) --------
//
// The fused_solver._jaref_zeroahead contract, verbatim semantics: one
// (world, row) work-item family. Item lane 0 per world does the zero-ahead
// bookkeeping (changed_count zero for EVERY world including done ones --
// memset semantics; the done-guarded rotate/zeros for live worlds), then
// rows >= nefc return and live rows do Jaref += alpha * jv.

WP_SYCL_API int wp_sycl_jaref(const void* jv, const void* alpha, const int* nefc,
                              const unsigned char* done, const void* cost,
                              void* Jaref, void* gauss, void* cost_out,
                              void* prev_cost, void* grad_dot, void* search_dot,
                              void* changed_count, long long njmax,
                              long long batch) {
    if (batch <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_jaref");
        q.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch) * 32),
                sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const int w = static_cast<int>(it.get_group_linear_id());
                const int lane = static_cast<int>(it.get_local_linear_id());
                const auto* jvw = static_cast<const float*>(jv) +
                    static_cast<size_t>(w) * njmax;
                const auto* Jw = static_cast<const float*>(Jaref) +
                    static_cast<size_t>(w) * njmax;
                auto* Jout = static_cast<float*>(Jaref) +
                    static_cast<size_t>(w) * njmax;
                const int nef = nefc[w];
                const float alpha_w = static_cast<const float*>(alpha)[w];

                if (lane == 0) {
                    // changed_efc_count: memset semantics for EVERY world
                    static_cast<int*>(changed_count)[w] = 0;
                }
                if (done[w]) return;

                if (lane == 0) {
                    static_cast<float*>(gauss)[w] = 0.0f;
                    static_cast<float*>(prev_cost)[w] = static_cast<const float*>(cost)[w];
                    static_cast<float*>(cost_out)[w] = 0.0f;
                    static_cast<float*>(grad_dot)[w] = 0.0f;
                    static_cast<float*>(search_dot)[w] = 0.0f;
                }

                for (int e = lane; e < njmax; e += 32) {
                    if (e >= nef) return;
                    Jout[e] = Jw[e] + alpha_w * jvw[e];
                }
            });
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_jaref failed: %s\n", e.what());
        return -1;
    }
}

// ---- fused prepare_quad + prepare_gauss (bit-exact native rewrite) --------
//
// The fused_linesearch._quad_gauss_fused contract verbatim: item (world, 0)
// runs the per-world gauss reduction (dofs_per_thread >= nv case: single
// writer, no atomics), rows >= nefc return, per-row quad = (0.5*Jaref^2*D,
// jv*Jaref*D, 0.5*jv^2*D) with the elliptic-cone branch writing quad1/quad2
// to the cone's rows 1/2 from its row-0 item (single writer per row, same
// early-return structure as the warp kernel).

WP_SYCL_API int wp_sycl_quad_gauss(
    const void* impratio_invsqrt,
    const int* nefc,
    const void* contact_friction, const int* contact_dim,
    const int* contact_efc_address,
    const int* efc_type, const int* efc_id,
    const void* efc_D, const int* nacon,
    const void* Jaref, const void* jv,
    const unsigned char* done,
    const void* qfrc_smooth, const void* efc_Ma,
    const void* search, const void* gauss, const void* mv,
    void* quad_out, void* quad_gauss_out,
    long long impratio_n, long long nv,
    long long efc_stride, long long ctx_stride,
    long long nv_stride, long long adr_stride,
    long long batch) {
    if (batch <= 0) return 0;
    try {
        sycl::queue& q = the_queue();
        g_last_kernel.store("wp_sycl_quad_gauss");
        q.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch) * 32),
                sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const int w = static_cast<int>(it.get_group_linear_id());
                const int lane = static_cast<int>(it.get_local_linear_id());
                constexpr int kElliptic = 1;  // ConstraintType.CONTACT_ELLIPTIC

                // ---- prepare_gauss fold: work-item (world, 0) only --------
                if (lane == 0) {
                    const auto* s = static_cast<const float*>(search) +
                        static_cast<size_t>(w) * nv_stride;
                    const auto* ma = static_cast<const float*>(efc_Ma) +
                        static_cast<size_t>(w) * nv_stride;
                    const auto* qs = static_cast<const float*>(qfrc_smooth) +
                        static_cast<size_t>(w) * nv_stride;
                    const auto* mvw = static_cast<const float*>(mv) +
                        static_cast<size_t>(w) * nv_stride;
                    float g1 = 0.0f, g2 = 0.0f;
                    for (int i = 0; i < nv; ++i) {
                        const float si = s[i];
                        g1 += si * (ma[i] - qs[i]);
                        g2 += 0.5f * si * mvw[i];
                    }
                    float* qg = static_cast<float*>(quad_gauss_out) +
                        static_cast<size_t>(w) * 3;
                    const float* g0 = static_cast<const float*>(gauss);
                    qg[0] = g0[w];
                    qg[1] = g1;
                    qg[2] = g2;
                }

                if (done[w]) return;

                // ---- prepare_quad per row --------------------------------
                const auto* Jw = static_cast<const float*>(Jaref) +
                    static_cast<size_t>(w) * ctx_stride;
                const auto* jvw = static_cast<const float*>(jv) +
                    static_cast<size_t>(w) * ctx_stride;
                const auto* Dw = static_cast<const float*>(efc_D) +
                    static_cast<size_t>(w) * efc_stride;
                const auto* typ = efc_type + static_cast<size_t>(w) * efc_stride;
                const auto* ids = efc_id + static_cast<size_t>(w) * efc_stride;
                float* quw = static_cast<float*>(quad_out) +
                    static_cast<size_t>(w) * ctx_stride * 3;
                const int nef = nefc[w];

                for (int e = lane; e < ctx_stride; e += 32) {
                    if (e >= nef) return;
                    const float Jaref_e = Jw[e];
                    const float jv_e = jvw[e];
                    const float D_e = Dw[e];
                    float q0 = 0.5f * Jaref_e * Jaref_e * D_e;
                    float q1 = jv_e * Jaref_e * D_e;
                    float q2 = 0.5f * jv_e * jv_e * D_e;

                    if (typ[e] == kElliptic) {
                        const int conid = ids[e];
                        const int* adr = contact_efc_address +
                            static_cast<size_t>(conid) * adr_stride;
                        if (conid >= nacon[0]) return;
                        const int efcid0 = adr[0];
                        if (e != efcid0) return;
                        const int dim = contact_dim[conid];
                        const float* fr = static_cast<const float*>(contact_friction) +
                            static_cast<size_t>(conid) * 5;
                        const float* iis = static_cast<const float*>(impratio_invsqrt);
                        const float mu = fr[0] * iis[w % impratio_n];
                        const float u0 = Jaref_e * mu;
                        const float v0 = jv_e * mu;
                        float uu = 0.0f, uv = 0.0f, vv = 0.0f;
                        for (int j = 1; j < dim; ++j) {
                            const int efcidj = adr[j];
                            if (efcidj < 0) return;
                            const float jvj = jvw[efcidj];
                            const float jarefj = Jw[efcidj];
                            const float dj = Dw[efcidj];
                            const float DJj = dj * jarefj;
                            q0 += 0.5f * jarefj * DJj;
                            q1 += jvj * DJj;
                            q2 += 0.5f * jvj * dj * jvj;
                            const float fj = fr[j - 1];
                            const float uj = jarefj * fj;
                            const float vj = jvj * fj;
                            uu += uj * uj;
                            uv += uj * vj;
                            vv += vj * vj;
                        }
                        const int efcid1 = adr[1];
                        float* q1w = quw + static_cast<size_t>(efcid1) * 3;
                        q1w[0] = u0;
                        q1w[1] = v0;
                        q1w[2] = uu;
                        const float mu2 = mu * mu;
                        const int efcid2 = adr[2];
                        float* q2w = quw + static_cast<size_t>(efcid2) * 3;
                        q2w[0] = uv;
                        q2w[1] = vv;
                        q2w[2] = D_e / (mu2 * (1.0f + mu2));
                    }

                    float* qe = quw + static_cast<size_t>(e) * 3;
                    qe[0] = q0;
                    qe[1] = q1;
                    qe[2] = q2;
                }
            });
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "wp_sycl_quad_gauss failed: %s\n", e.what());
        return -1;
    }
}

// CRT helpers expected by generated kernel modules (declared in crt.h).
// Host-side implementations; kernels that call these from device code are
// not supported yet.
void _wp_assert(const char* message, const char* file, unsigned int line) {
    std::fprintf(stderr, "%s:%u: assert: %s\n", file ? file : "?", line, message ? message : "?");
    std::abort();
}

int _wp_isfinite(double x) { return std::isfinite(x); }

int _wp_isnan(double x) { return std::isnan(x); }

int _wp_isinf(double x) { return std::isinf(x); }

}
