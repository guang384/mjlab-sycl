/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal SYCL micro-driver shared by all SYCL-compiled Warp kernel modules.
//
// The implementation lives in warpsycl.dll (built on demand into the Warp
// kernel cache) and each module DLL links against warpsycl.lib so that the
// whole process shares a single sycl::queue and USM allocator.
//
// Device code cannot call through function pointers (SYCL/SPIR-V
// restriction), so the generated module entry points embed the
// parallel_for directly and only obtain the shared queue from here.

#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>

// Exported symbols: dllexport while building warpsycl.dll (which defines
// WP_SYCL_BUILDING_RUNTIME), dllimport in the kernel modules that include
// this header and link against warpsycl.lib.
#if defined(_WIN32)
#if defined(WP_SYCL_BUILDING_RUNTIME)
#define WP_SYCL_API __declspec(dllexport)
#else
#define WP_SYCL_API __declspec(dllimport)
#endif
#else
#define WP_SYCL_API __attribute__((visibility("default")))
#endif

extern "C" {
// Address of the shared in-order sycl::queue bound to the first Intel GPU.
WP_SYCL_API void* wp_sycl_queue_ptr();
// USM shared-memory allocation served from a drain-gated pool (see the .cpp):
// blocks freed since the last synchronize are recycled without touching the
// driver allocator, and frees never wait on the queue.
WP_SYCL_API void* wp_sycl_alloc_shared(size_t size);
WP_SYCL_API void wp_sycl_free(void* ptr);
// Enqueue a byte fill / copy on the shared in-order queue. These are ordered
// against kernels already in flight (unlike a raw host write) and complete
// before any kernel submitted afterwards, so no queue drain is needed as long
// as the host does not read the target before the next synchronize.
WP_SYCL_API void wp_sycl_memset(void* ptr, int value, size_t size);
WP_SYCL_API void wp_sycl_memcpy(void* dst, const void* src, size_t size);
// Enqueue a tiled fill (repeat the src_size-byte pattern reps times) on the
// shared in-order queue; the pattern is staged through a USM ring so the
// device kernel can read it. Same ordering guarantees as memset/memcpy.
WP_SYCL_API void wp_sycl_memtile(void* dst, const void* src, size_t src_size, size_t reps);
// Per-thread ring of USM staging buffers for kernel argument structs. The
// struct produced by the Python launcher lives in plain host memory, which
// the device cannot dereference; entry points copy it here before
// submitting to the queue. Slots are reused in ring order; every kRing-th
// staging call drains the in-order queue so a slot is never overwritten
// while an in-flight launch still reads it.
WP_SYCL_API void* wp_sycl_stage_args(size_t size);
// Block until every submitted kernel on the shared queue has completed.
WP_SYCL_API void wp_sycl_synchronize();
// Record the name of the kernel about to be submitted (called at the top of
// every generated module entry). The watchdog prints it before aborting so a
// hang names its culprit instead of wedging the machine silently.
WP_SYCL_API void wp_sycl_note_kernel(const char* name);
WP_SYCL_API const char* wp_sycl_device_name();
// SLM high-water tracking: tile.h alloc() reports bytes used per task;
// the device side atomically maxes it into USM so the host can size the
// arena to the real workload instead of guessing WP_MAX_SYCL_SHARED.
WP_SYCL_API void wp_sycl_note_slm_usage(unsigned int bytes);
WP_SYCL_API unsigned int wp_sycl_get_slm_highwater();
WP_SYCL_API void wp_sycl_reset_slm_highwater();
}

inline sycl::queue& wp_sycl_queue() {
    return *static_cast<sycl::queue*>(wp_sycl_queue_ptr());
}
