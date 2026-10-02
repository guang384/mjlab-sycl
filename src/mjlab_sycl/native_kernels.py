# SPDX-License-Identifier: Apache-2.0
"""Native (warpsycl.dll) replacements for hot warp-language kernels.

The DPC++ codegen measures ~70 GB/s on shared USM where warp's generated
kernels cap at ~59 (docs/optimization_ideas.md) -- a ~1.2x structural gap
that hand-written native kernels recover for the hottest launches, with
identical per-element arithmetic (bit-exact by construction).

Each helper returns True when the native kernel ran; False means the
caller must fall back to the warp path (unavailable export, kill switch,
or runtime failure). Kill switch: MJLAB_SYCL_NATIVE_MVJV=0.
"""

from __future__ import annotations

import ctypes
import os

_fn_mv_jv = None
_probed = False


def _enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in (
        "0", "false", "off",
    )


def _api():
    """Resolve the DLL exports once; None means unavailable (stale DLL)."""
    global _fn_mv_jv, _probed
    if _probed:
        return _fn_mv_jv
    _probed = True
    try:
        from mjlab_sycl.backend._src import build as _build

        dll = _build.ensure_sycl_runtime()
        fn = dll.wp_sycl_mv_jv
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_longlong] * 5
        _fn_mv_jv = fn
    except Exception:
        _fn_mv_jv = None
    return _fn_mv_jv


def mv_jv(qM, J, search, nefc, done, mv, jv, nv: int, njmax: int,
          nv_pad: int, njmax_pad: int) -> bool:
    """Native fused mv+jv. All arguments are warp sycl arrays; ``done`` is
    the solver's per-world bool array. Returns True when the batch ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_MVJV"):
        return False
    fn = _api()
    if fn is None:
        return False
    return fn(qM.ptr, J.ptr, search.ptr, nefc.ptr, done.ptr, mv.ptr, jv.ptr,
              nv, njmax, nv_pad, njmax_pad, qM.shape[0]) == 0
