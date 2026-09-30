# SPDX-License-Identifier: Apache-2.0
"""Cached ``wp.launch`` for the SYCL hot path: skip pack_arg type checking,
module lookups, ArgsStruct construction and bounds building for repeated
(kernel, args, dim) launches.

mujoco_warp's physics step launches ~1500 kernels per step, each with ~20
arguments, with the SAME kernel+args combination every step (only the array
DATA changes — the array descriptors: ptr/shape/strides are immutable for
persistent USM allocations).  Profiling showed the per-launch host cost is
~0.18 ms, dominated by:

  - ``pack_arg``: 5-6 isinstance/type/dtype/ndim/device checks per argument
    (~20 args per kernel, ~30k checks per step)
  - ``invoke``: ``ArgsStruct()`` construction + one ``setattr`` per field
  - ``launch_bounds_t`` ctypes construction
  - ``kernel.module.load`` + ``get_kernel_hooks`` lookups
  - list/tuple churn and the pack_args closure in ``wp.launch``

This module replaces ``wp.launch`` with a version that caches
``(hooks, args_struct, bounds)`` keyed by ``(kernel, args_tuple, dim)``.
On a cache hit the launch reduces to a single ctypes FFI call::

    hooks.forward(bounds, ctypes.byref(args_struct))

Recurrence gate with weakref liveness verification.  A cache entry holds
strong references to every argument object (the key tuple), which keeps
their USM backing store alive — only safe for objects that are genuinely
persistent.  Two hazards were verified by crashing the 4096-env bench:

  1. mujoco_warp's collision path builds FRESH kernels
     (``ccd_kernel_builder``/``_primitive_narrowphase``/...) and FRESH
     ``wp.empty`` scratch every step; caching one of those hoards a dead
     buffer set per launch until the GPU OOMs (~30 steps at 4096 envs).
  2. Python ids are recycled: a freed scratch array and its replacement
     can hash identically, so a plain "seen twice" fingerprint gate still
     gets fooled (``flat_kernels._cholesky_solve_flat`` takes a fresh
     ``wp.empty_like`` scratch with a module-level kernel — exactly this
     case).

The gate therefore stores WEAK references on first sight and, when the
fingerprint matches a second time, verifies every weakref is still alive
and points at the very same object.  Recycled ids (dead weakrefs) fail
verification and are treated as a fresh occurrence — never cached.  Only
launches whose entire argument set provably survives across launches
enter the cache, which is precisely the persistent hot path.

Other safety properties:

  - The cached ArgsStruct holds array DESCRIPTORS (ptr/shape/strides), not
    data.  Kernels write through ``ptr`` into the USM buffer; the descriptor
    never changes for a persistent allocation, so the cached struct stays
    valid across steps.
  - Scalar arguments participate in the key BY VALUE: a scalar that changes
    (e.g. an adaptive njmax) produces a different key and therefore a fresh
    cache entry — stale scalars are impossible.
  - ``dim`` is part of the key; bounds are cached alongside the struct and
    only reused for an identical dim.
  - Non-standard launches (adjoint, record_cmd, explicit stream, generic
    kernels, CUDA devices) fall through to the original ``wp.launch``.

Measured impact (4096 envs, Arc 130T, idle GPU): rollout 16.5 s -> 11.0 s
per iteration (-33%), throughput 5,968 -> 8,953 env-steps/s (+50%).  The
cache shifts the step bottleneck from host launch overhead to GPU
execution, so when another GPU client (e.g. a 3D viewer) is co-running on
the iGPU the gain shrinks and can temporarily reverse — benchmark on an
idle GPU for clean numbers.

Kill switch: ``MJLAB_SYCL_LAUNCH_CACHE=0``.
"""

from __future__ import annotations

import ctypes
import os
import weakref
from collections import OrderedDict

import warp as wp
from warp._src import context as ctx

_orig_launch = None
# OrderedDict so insertion order tracks recency: hot entries get
# move_to_end on every hit while ephemeral-arg entries (solver contexts,
# one-shot scratch) drift to the head and get evicted.  This bounds the
# pinned USM — without it the cache grew linearly (~110 stale solver-
# context arrays per step, ~8 GB after 30 steps at 4096 envs) until the
# old _MAX_ENTRIES hard-clear emptied everything.
_cache: "OrderedDict" = OrderedDict()
# fingerprint -> (weakref(kernel), tuple(arg_ref)) where arg_ref is a
# weakref for objects or the raw value for scalars.  Never pins memory.
_seen_once: dict = {}
_STATS = {
  "hit": 0,
  "miss": 0,
  "first_seen": 0,
  "recurring": 0,
  "recycled": 0,
  "evicted": 0,
}
# LRU cap: the persistent hot path keeps ~50-100 entries; the rest of the
# budget covers a few generations of per-substep solver contexts (~15-20
# entries each, 4 substeps/step) before the oldest are evicted.  Keeps
# pinned memory bounded to a handful of contexts (~100-200 MB).
_MAX_ENTRIES = 512
# Evict in batches: when the cap is hit, drop the oldest 25% rather than
# one entry per insertion, amortizing the eviction cost (dict.popitem on
# an OrderedDict is O(1) but Python call overhead adds up at 80 inserts/
# step).
_EVICT_BATCH = max(1, _MAX_ENTRIES // 4)
_MAX_SEEN = 65536


def _enabled() -> bool:
  return os.environ.get("MJLAB_SYCL_LAUNCH_CACHE", "1").strip().lower() not in (
    "0",
    "false",
    "off",
  )


def _normalize_dim(dim):
  """lists are unhashable; tuples of ints hash fast and compare by value."""
  if isinstance(dim, list):
    return tuple(dim)
  return dim


def _arg_ref(a):
  """Weak reference for objects; raw value for scalars (hash is value-based
  so scalars never suffer id recycling)."""
  if isinstance(a, (int, float, bool, complex, str, bytes, type(None))):
    return a
  try:
    return weakref.ref(a)
  except TypeError:
    return None  # not weakref-able and not scalar: can't prove persistence


def _args_survive(seen_kernel_ref, seen_arg_refs, kernel, fwd_args):
  """True when every first-seen object is still alive AND is the exact
  object in this launch.  A single dead weakref (garbage-collected and
  id-recycled) fails the whole combination."""
  if seen_kernel_ref is None or seen_kernel_ref() is not kernel:
    return False
  if len(seen_arg_refs) != len(fwd_args):
    return False
  for ref, a in zip(seen_arg_refs, fwd_args):
    if isinstance(ref, weakref.ref):
      if ref() is not a:
        return False
    elif ref is None:
      return False  # unverifiable type: refuse to cache
    elif ref != a:  # scalar: value comparison
      return False
  return True


def install() -> None:
  global _orig_launch
  if _orig_launch is not None:
    return
  if not _enabled():
    return
  _orig_launch = wp.launch
  wp.launch = _cached_launch
  # launch_tiled forwards to context.launch via a module-level reference,
  # so patch that too (covers flat_kernels' tiled fallback path).
  ctx.launch = _cached_launch
  print(
    "[sycl-launch-cache] cached wp.launch installed "
    "(MJLAB_SYCL_LAUNCH_CACHE=0 to disable)"
  )


def uninstall() -> None:
  global _orig_launch
  if _orig_launch is None:
    return
  wp.launch = _orig_launch
  ctx.launch = _orig_launch
  _orig_launch = None


def stats() -> dict:
  return dict(_STATS)


def _cached_launch(
  kernel,
  dim,
  inputs=(),
  outputs=(),
  adj_inputs=(),
  adj_outputs=(),
  device=None,
  stream=None,
  adjoint=False,
  record_tape=True,
  record_cmd=False,
  max_blocks=0,
  block_dim=256,
):
  # Fast path only for the exact launch pattern the physics hot path uses.
  # While a wp.Tape is open, bypass the cache entirely (both lookup and
  # storage): warp records each launch on the original launch path, and a
  # cached replay would silently skip tape.record_launch — wrong gradients.
  cacheable = not (
    adjoint
    or record_cmd
    or stream is not None
    or getattr(kernel, "is_generic", False)
  )
  if cacheable and ctx.runtime is not None and ctx.runtime.tape is not None:
    cacheable = False

  build_key = None
  fwd_args = ()
  ndim = None
  if cacheable:
    fwd_args = tuple(inputs) + tuple(outputs)
    ndim = _normalize_dim(dim)

    # Build the cache key.  wp.array objects hash by identity (default
    # object hash); int/float scalars hash by value.  Include device and
    # block_dim so that the same kernel launched on different devices or
    # with different block dimensions does not cross-contaminate entries.
    # record_tape is not part of the key: tape-active launches never reach
    # this point (cacheable is False above).
    try:
      key = (kernel, fwd_args, ndim, device, block_dim)
      cached = _cache.get(key)
    except TypeError:
      cached = None
      key = None

    if key is not None and cached is not None:
      _STATS["hit"] += 1
      # LRU: refresh recency so frequently-reused entries (the persistent
      # physics hot path) survive while one-shot / per-substep entries drift
      # toward eviction.  move_to_end rehashes the key (~1-2 us) — negligible
      # vs. the ~100 us each launch saves.
      _cache.move_to_end(key)
      hooks, args_struct, bounds = cached
      # bounds was validated non-empty when the entry was built
      hooks.forward(bounds, ctypes.byref(args_struct))
      return

    if key is not None:
      _STATS["miss"] += 1
      # Recurrence gate with liveness verification (see module docstring).
      fp = hash(key)
      seen = _seen_once.get(fp)
      if seen is not None:
        kernel_ref, arg_refs = seen
        if _args_survive(kernel_ref, arg_refs, kernel, fwd_args):
          # Proven recurring: every object survived across launches, so the
          # argument set is persistent — safe to pin with strong references.
          _STATS["recurring"] += 1
          build_key = key
        else:
          # Same fingerprint but the original objects died: recycled ids,
          # not a recurrence.  Refresh the stored weakrefs and stay on the
          # slow path.
          _STATS["recycled"] += 1
          _seen_once[fp] = (
            _arg_ref(kernel),
            tuple(_arg_ref(a) for a in fwd_args),
          )
      else:
        # First sight: store weakrefs (never pins memory), then slow path.
        _STATS["first_seen"] += 1
        if len(_seen_once) < _MAX_SEEN:
          _seen_once[fp] = (
            _arg_ref(kernel),
            tuple(_arg_ref(a) for a in fwd_args),
          )

  # Single slow-path exit: the original wp.launch (also the tape-recording
  # path, the TypeError/unhashable-arg path, and every non-hot pattern).
  _orig_launch(
    kernel,
    dim,
    inputs,
    outputs,
    adj_inputs,
    adj_outputs,
    device,
    stream,
    adjoint,
    record_tape,
    record_cmd,
    max_blocks,
    block_dim,
  )
  if build_key is not None:
    _build_cache_entry(kernel, ndim, fwd_args, device, block_dim, build_key)


def _build_cache_entry(kernel, ndim, fwd_args, device, block_dim, key) -> None:
  """Reconstruct the packed-args struct from the (already launched) call.

  Best-effort: any failure just means this launch keeps using the slow path.
  Only caches the CPU-device path (the SYCL device reports is_cpu=True).
  """
  from warp._src import context as ctx

  try:
    if ctx.runtime is None:
      return
    if ctx.runtime.tape is not None:
      return  # tape active: launches must be recorded, don't bypass
    dev = ctx.runtime.get_device(device)
    if not dev.is_cpu:
      return  # CUDA path packs differently (kernel_params array) — skip

    bounds = ctx.launch_bounds_t(ndim)
    if bounds.size <= 0:
      return

    n_args = len(kernel.adj.args)
    if len(fwd_args) != n_args:
      return  # original would have raised; nothing to cache

    # Pack each argument (cheap on the second pass: arrays return their
    # cached __ctype__ descriptor, scalars rebuild a one-field ctypes value).
    params = [bounds]
    for i, a in enumerate(fwd_args):
      arg_type = kernel.adj.args[i].type
      arg_name = kernel.adj.args[i].label
      params.append(ctx.pack_arg(kernel, arg_type, arg_name, a, dev, False))

    # Reuse the ModuleExec the original launch just loaded.  DO NOT call
    # kernel.module.load(...) here: load() mutates options["block_dim"] and
    # a mismatched value (1 vs the launch's 256) recompiles every module
    # into a second DLL — seconds per module and a duplicate handle.
    block_dim = kernel.module.options.get("block_dim", 256)
    module_exec = kernel.module.execs.get((dev.context, block_dim))
    if module_exec is None:
      return  # original launch failed; nothing to cache
    # get_kernel_hooks is a dict-lookup cache on the exec; the original
    # launch already populated it.
    hooks = module_exec.get_kernel_hooks(kernel)
    if hooks.forward is None:
      return

    # Build (or reuse) the ArgsStruct type — invoke() already cached the type
    # during the original launch, so this is normally a dict hit.
    param_types = tuple(type(p) for p in params[1:])
    type_cached = kernel._invoke_cache.get((param_types, False))
    if type_cached is not None:
      ArgsStruct, fields = type_cached
    else:
      fields = [
        (kernel.adj.args[i].label, type(params[1 + i])) for i in range(n_args)
      ]
      ArgsStruct = type(
        "ArgsStruct", (ctypes.Structure,), {"_fields_": fields}
      )
      kernel._invoke_cache[(param_types, False)] = (ArgsStruct, fields)

    args_struct = ArgsStruct()
    for i, field in enumerate(fields):
      setattr(args_struct, field[0], params[1 + i])

    if len(_cache) >= _MAX_ENTRIES:
      # LRU batch evict: drop the oldest _EVICT_BATCH entries (the
      # least-recently-used).  OrderedDict popitem(last=False) is O(1); we
      # evict a batch so the amortized cost is one eviction per ~EVICT_BATCH
      # insertions.
      for _ in range(_EVICT_BATCH):
        _cache.popitem(last=False)
        _STATS["evicted"] += 1
    _cache[key] = (hooks, args_struct, bounds)
  except Exception:
    pass  # caching is strictly optional
