# SPDX-License-Identifier: Apache-2.0
"""Batch-level command-graph capture/replay for the solver iteration loop.

``loop_poll`` runs the solver body in fixed batches (``POLL_EVERY``
iterations).  The batch is a static sequence of ~33 kernel submissions per
iteration over stable argument buffers (solver scratch is reused across
solves -- see solver_ctx.py), which makes it a textbook command-graph
candidate: capture the batch once, then replay it as ONE queue submission
instead of ~264 individual ones (~20 us of host+driver cost each).

State per (model, data):

  UNSEEN  -> run the batch plain and count top-level launch calls;
  ARMED   -> record the batch as a graph (submissions are captured, NOT
             executed -- the graph is submitted right after finalize so the
             batch's work still happens), verify the launch count matched;
             a mismatch disables the graph path (the sequence is not static);
  REPLAY  -> submit the graph.  If any argument object changed (e.g. solver
             scratch reuse is off and a fresh ctx appears), the baked
             pointers are stale: re-record a few times, then give up.

Safety:

  - argument staging during recording goes to a per-graph arena inside
    warpsycl.dll, never the shared ring, so replays cannot observe an
    overwritten slot;
  - no capture while a wp.Tape is recording (a replayed launch would skip
    tape.record_launch -- wrong gradients);
  - every failure falls back to the plain per-iteration loop.

Kill switch: ``MJLAB_SYCL_GRAPH=0``.
"""

from __future__ import annotations

import ctypes
import os

import warp as wp

_api_cache = None
_broken = False  # hard failure (missing DLL exports, runtime error)
_MAX_RECAPTURES = 4
_STATS = {"plain": 0, "recorded": 0, "replayed": 0, "fallback": 0, "disabled": 0}

# Depth of the outer sequence currently establishing itself (UNSEEN count or
# ARMED record). While > 0, every nested run_batch/run_sequence must run
# PLAIN so its kernels become nodes of the outer graph: the outer count and
# record runs then see the same launch sequence (a nested graph REPLAY would
# submit zero launches and break the outer count-verification).
_OUTER = 0

# True while a graph record's fn is being captured (submissions are NOT
# executing). Queue waits are illegal during capture, so code inside the
# recorded region (loop_poll's convergence drain) checks this and skips
# them; the captured sequence keeps its launch order either way.
_RECORDING = False


def recording_active() -> bool:
    return _RECORDING


def stats() -> dict:
    return dict(_STATS)

# (id(m), id(d)) -> entry; entries hold strong refs so ids stay valid.
_CACHE: dict = {}
_SEQ: dict = {}  # tag -> entry, for fixed sequences like the lite forward

_PLAIN = -2  # entry marker: graphs disabled for this (m, d)


class _Entry:
    __slots__ = ("m", "d", "arg_ids", "n", "launches", "graph", "recaptures")

    def __init__(self, m, d):
        self.m = m
        self.d = d
        self.arg_ids = None
        self.n = 0
        self.launches = -1  # -1: unseen; >=0: armed with this count; -2: plain
        self.graph = None
        self.recaptures = 0


def _enabled() -> bool:
    return os.environ.get("MJLAB_SYCL_GRAPH", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _api():
    global _api_cache, _broken
    if _api_cache is None:
        try:
            from mjlab_sycl.backend._src import build as b

            dll = b.ensure_sycl_runtime()
            dll.wp_sycl_graph_begin.restype = ctypes.c_void_p
            dll.wp_sycl_graph_begin.argtypes = []
            dll.wp_sycl_graph_end.restype = ctypes.c_int
            dll.wp_sycl_graph_end.argtypes = [ctypes.c_void_p]
            dll.wp_sycl_graph_submit.restype = ctypes.c_int
            dll.wp_sycl_graph_submit.argtypes = [ctypes.c_void_p]
            dll.wp_sycl_graph_free.restype = None
            dll.wp_sycl_graph_free.argtypes = [ctypes.c_void_p]
            _api_cache = dll
        except Exception:
            _broken = True
            return None
    return _api_cache


def _tape_active() -> bool:
    from warp._src import context as wctx

    return wctx.runtime is not None and wctx.runtime.tape is not None


def _counted_call(fn) -> int:
    """Run fn() plain, return the number of top-level wp.launch calls."""
    counter = [0]
    inner = wp.launch

    def counting(*args, **kws):
        counter[0] += 1
        return inner(*args, **kws)

    wp.launch = counting
    try:
        fn()
    finally:
        wp.launch = inner
    return counter[0]


def _counted_batch(while_body, n, kwargs) -> int:
    def run_all():
        for _ in range(n):
            while_body(**kwargs)

    return _counted_call(run_all)


def _record_call(api, fn):
    """Record fn() as a graph and run it (post-finalize submit).

    Returns (handle, launch_count) on success.  handle is None means fn has
    NOT executed and the caller must fall back to a plain run.
    """
    global _RECORDING
    h = api.wp_sycl_graph_begin()
    if not h:
        return None, -1
    _RECORDING = True
    try:
        count = _counted_call(fn)
    except Exception:
        try:
            if api.wp_sycl_graph_end(h) == 0:
                api.wp_sycl_graph_free(h)
        except Exception:
            pass
        raise
    finally:
        _RECORDING = False
    if api.wp_sycl_graph_end(h) != 0:
        # captured but never executed: discard and let the caller re-run
        api.wp_sycl_graph_free(h)
        return None, -1
    if api.wp_sycl_graph_submit(h) != 0:
        api.wp_sycl_graph_free(h)
        return None, -1
    return h, count


def _record(api, while_body, n, kwargs):
    def run_all():
        for _ in range(n):
            while_body(**kwargs)

    return _record_call(api, run_all)


def run_sequence(tag: str, fn, ident=None, scoped: bool = False) -> bool:
    """Capture a static kernel sequence (e.g. lite forward) as one graph.

    Same three-state arming as run_batch: run once plain to count launches,
    record+verify once, then replay as a single submission.  The sequence
    must be static (same launches every call) and its array arguments must
    stay identical objects -- per-call scratch allocated *inside* fn is
    fine, the graph pins the recorded buffers and reuses them.  ``ident``
    (typically the model) keys liveness: a different object re-records.
    Returns True when the sequence already ran; False when the caller
    should just call fn() itself.

    ``scoped=True`` marks a WHOLE-SUBSTEP sequence (e.g. mjwarp.step): while
    it counts or records, nested graph users are forced plain (see _OUTER)
    so the outer graph absorbs them; on replay the nested users are never
    called at all.
    """
    global _OUTER
    if _broken or not _enabled() or _tape_active() or (_OUTER > 0 and not scoped):
        return False
    if scoped:
        _OUTER += 1
        try:
            return _run_sequence(tag, fn, ident)
        finally:
            _OUTER -= 1
    return _run_sequence(tag, fn, ident)


def _run_sequence(tag: str, fn, ident) -> bool:
    global _broken
    if _broken or not _enabled() or _tape_active():
        return False
    api = _api()
    if api is None:
        return False

    e = _SEQ.get(tag)
    if e is None or e.m is not ident:
        e = _Entry(ident, None)
        if len(_SEQ) < 16:
            _SEQ[tag] = e

    if e.launches == _PLAIN:
        return False

    if e.graph is not None:
        if api.wp_sycl_graph_submit(e.graph) == 0:
            _STATS["replayed"] += 1
            return True
        api.wp_sycl_graph_free(e.graph)
        e.graph = None
        _STATS["fallback"] += 1

    if e.launches < 0:
        e.launches = _counted_call(fn)
        _STATS["plain"] += 1
        return True

    h, count = _record_call(api, fn)
    if h is None:
        e.launches = _PLAIN
        _STATS["fallback"] += 1
        return False
    if count != e.launches:
        api.wp_sycl_graph_free(h)
        e.launches = _PLAIN
        _STATS["fallback"] += 1
        return True
    e.graph = h
    _STATS["recorded"] += 1
    return True


def run_batch(while_body, n, kwargs) -> bool:
    """Run ``n`` body iterations as one graph submission when possible.

    Returns True when the batch already ran (graph replay or recorded-run);
    False when the caller must run the plain per-iteration loop.
    """
    global _broken
    if _broken or not _enabled() or n < 2 or _tape_active():
        return False
    if _OUTER > 0:
        # an outer (substep-level) sequence is counting or recording: run
        # plain so its kernels become nodes of the outer graph
        return False

    m = kwargs.get("m")
    d = kwargs.get("d")
    if m is None or d is None:
        return False
    api = _api()
    if api is None:
        return False

    key = (id(m), id(d))
    arg_ids = tuple(id(v) for v in kwargs.values())
    e = _CACHE.get(key)
    if e is None or e.m is not m or e.d is not d:
        e = _Entry(m, d)
        if len(_CACHE) < 8:
            _CACHE[key] = e

    if e.launches == _PLAIN:
        _STATS["disabled"] += 1
        return False

    # argument objects changed -> the baked pointers are stale
    if e.graph is not None and (e.arg_ids != arg_ids or e.n != n):
        api.wp_sycl_graph_free(e.graph)
        e.graph = None
        e.recaptures += 1
        if e.recaptures > _MAX_RECAPTURES:
            # scratch churn (solver_ctx disabled): graphs cannot amortize
            e.launches = _PLAIN
            return False

    if e.graph is not None:
        if api.wp_sycl_graph_submit(e.graph) == 0:
            _STATS["replayed"] += 1
            return True
        api.wp_sycl_graph_free(e.graph)  # submit failed: fall back below
        e.graph = None
        _STATS["fallback"] += 1

    if e.launches < 0:
        # UNSEEN: plain run to learn the expected launch count
        e.n = n
        e.arg_ids = arg_ids
        e.launches = _counted_batch(while_body, n, kwargs)
        _STATS["plain"] += 1
        return True

    # ARMED: record; a successful record executes the batch via the
    # post-finalize submit, so only a failed record leaves work undone.
    h, count = _record(api, while_body, n, kwargs)
    if h is None:
        e.launches = _PLAIN  # recording unsupported here: stay plain
        _STATS["fallback"] += 1
        return False
    if count != e.launches:
        # sequence is not static: never replay this entry again
        api.wp_sycl_graph_free(h)
        e.launches = _PLAIN
        _STATS["fallback"] += 1
        return True  # the recorded batch already ran (post-finalize submit)
    e.graph = h
    e.arg_ids = arg_ids
    e.n = n
    _STATS["recorded"] += 1
    return True
