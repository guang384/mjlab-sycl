/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    if (slots == nullptr) {
        slots = new PatternSlots();
    }
    if (counter != 0 && counter % kPatternRing == 0) {
        drain_and_recycle();
    }
    unsigned char*& pat = slots->ptrs[counter % kPatternRing];
    if (pat == nullptr) {
        pat = static_cast<unsigned char*>(sycl::malloc_shared(kPatternCap, the_queue()));
        if (pat == nullptr) {
            std::fprintf(stderr, "wp_sycl_memtile: pattern USM alloc failed\n");
            std::abort();
        }
    }
    ++counter;
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
