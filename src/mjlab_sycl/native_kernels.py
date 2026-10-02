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
_fn_jtdaj = None
_fn_chol = None
_fn_chol_fs = None
_fn_hinc = None
_probed = False


def _enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in (
        "0", "false", "off",
    )


def _api():
    """Resolve the DLL exports once; None means unavailable (stale DLL)."""
    global _fn_mv_jv, _fn_jtdaj, _fn_chol, _fn_chol_fs, _fn_hinc, _probed
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
        try:
            fj = dll.wp_sycl_jtdaj
            fj.restype = ctypes.c_int
            fj.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_longlong] * 3
            _fn_jtdaj = fj
        except AttributeError:
            _fn_jtdaj = None
        try:
            fc = dll.wp_sycl_chol_solve
            fc.restype = ctypes.c_int
            fc.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_longlong] * 3
            _fn_chol = fc
        except AttributeError:
            _fn_chol = None
        fs = getattr(dll, "wp_sycl_chol_fs")
        fs.restype = ctypes.c_int
        fs.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 3
        _fn_chol_fs = fs
        try:
            fx = dll.wp_sycl_hinc
            fx.restype = ctypes.c_int
            fx.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_longlong] * 4
            _fn_hinc = fx
        except AttributeError:
            _fn_hinc = None
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


def jtdaj(qM, J, D, state, nefc, done, h, nv_pad: int, njmax_pad: int) -> bool:
    """Native JTDAJ (h = qM + J^T D' J). Returns True when the batch ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_JTDAJ"):
        return False
    _api()
    fn = _fn_jtdaj
    if fn is None:
        return False
    return fn(qM.ptr, J.ptr, D.ptr, state.ptr, nefc.ptr, done.ptr, h.ptr,
              nv_pad, njmax_pad, qM.shape[0]) == 0


def chol_solve(h, grad, done, changed, lvalid_in, L, lvalid_out, Mgrad,
               n: int, nv_pad: int) -> bool:
    """Native solver cholesky factor+solve (one item per world, per-size
    template). Returns True when the batch ran; False for exotic n or a
    disabled/unavailable route (caller stays on the warp kernel)."""
    if not _enabled("MJLAB_SYCL_NATIVE_CHOL"):
        return False
    _api()
    fn = _fn_chol
    if fn is None:
        return False
    return fn(h.ptr, grad.ptr, done.ptr, changed.ptr, lvalid_in.ptr, L.ptr,
              lvalid_out.ptr, Mgrad.ptr, n, nv_pad, h.shape[0]) == 0


def chol_fs(M, y, x, L, adr, n: int, nv_pad: int) -> bool:
    """Native set-const cholesky factorize+solve (single-tile case).
    The tile anchor is read device-side from ``adr`` -- the route must be
    free of host syncs, because it runs inside the recorded substep.
    Returns True when the batch ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_CHOL"):
        return False
    _api()
    fn = _fn_chol_fs
    if fn is None:
        return False
    return fn(M.ptr, y.ptr, x.ptr, L.ptr, adr.ptr, n, nv_pad, M.shape[0]) == 0


def hinc(J, D, state, changed_ids, changed_count, h, nv_pad: int,
         efc_stride: int, ids_stride: int) -> bool:
    """Native incremental Hessian update over changed constraints.
    Returns True when the batch ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_HINC"):
        return False
    _api()
    fn = _fn_hinc
    if fn is None:
        return False
    return fn(J.ptr, D.ptr, state.ptr, changed_ids.ptr, changed_count.ptr,
              h.ptr, nv_pad, efc_stride, ids_stride, J.shape[0]) == 0
