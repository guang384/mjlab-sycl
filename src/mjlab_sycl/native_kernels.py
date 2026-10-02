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
_fn_qfrc = None
_fn_jaref = None
_fn_quad = None
_fn_efc = None
_fn_fold = None
_fn_gauss = None
_fn_lstd = None
_probed = False


def _enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in (
        "0", "false", "off",
    )


def _is_sycl(arr) -> bool:
    """Device guard: the native kernels read raw pointers on the sycl
    queue, so a non-sycl launch must never be routed to them."""
    try:
        return bool(arr.device.is_sycl)
    except Exception:
        return False


def _api():
    """Resolve the DLL exports once; None means unavailable (stale DLL)."""
    global _fn_mv_jv, _fn_jtdaj, _fn_chol, _fn_chol_fs, _fn_hinc, _fn_qfrc, _fn_jaref, _fn_quad, _fn_efc, _fn_fold, _fn_gauss, _fn_lstd, _probed
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
        try:
            fq = dll.wp_sycl_qfrc_constraint
            fq.restype = ctypes.c_int
            fq.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 4
            _fn_qfrc = fq
        except AttributeError:
            _fn_qfrc = None
        try:
            fj = dll.wp_sycl_jaref
            fj.restype = ctypes.c_int
            fj.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_longlong] * 2
            _fn_jaref = fj
        except AttributeError:
            _fn_jaref = None
        try:
            fe = dll.wp_sycl_efc_force
            fe.restype = ctypes.c_int
            fe.argtypes = [ctypes.c_void_p] * 19 + [ctypes.c_longlong] * 6
            _fn_efc = fe
            ff = dll.wp_sycl_cost_fold
            ff.restype = ctypes.c_int
            ff.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_longlong] * 2
            _fn_fold = ff
        except AttributeError:
            _fn_efc = None
            _fn_fold = None
        try:
            fg = dll.wp_sycl_gauss_cost
            fg.restype = ctypes.c_int
            fg.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_longlong] * 3
            _fn_gauss = fg
            fl = dll.wp_sycl_ls_teardown
            fl.restype = ctypes.c_int
            fl.argtypes = [ctypes.c_longlong, ctypes.c_float] + [ctypes.c_void_p] * 4 + [ctypes.c_longlong] + [ctypes.c_void_p] * 3 + [ctypes.c_longlong] * 3
            _fn_lstd = fl
        except AttributeError:
            _fn_gauss = None
            _fn_lstd = None
        try:
            fq2 = dll.wp_sycl_quad_gauss
            fq2.restype = ctypes.c_int
            fq2.argtypes = [ctypes.c_void_p] * 19 + [ctypes.c_longlong] * 7  # ptrs then scalars
            _fn_quad = fq2
        except AttributeError:
            _fn_quad = None
    except Exception:
        _fn_mv_jv = None
    return _fn_mv_jv


def mv_jv(qM, J, search, nefc, done, mv, jv, nv: int, njmax: int,
          nv_pad: int, njmax_pad: int) -> bool:
    """Native fused mv+jv. All arguments are warp sycl arrays; ``done`` is
    the solver's per-world bool array. Returns True when the batch ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_MVJV"):
        return False
    if not _is_sycl(qM):
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
    if not _is_sycl(qM):
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
    if not _is_sycl(h):
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
    if not _is_sycl(M):
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
    if not _is_sycl(J):
        return False
    _api()
    fn = _fn_hinc
    if fn is None:
        return False
    return fn(J.ptr, D.ptr, state.ptr, changed_ids.ptr, changed_count.ptr,
              h.ptr, nv_pad, efc_stride, ids_stride, J.shape[0]) == 0


def efc_force(impratio, ne, nf, nefc, friction, cdim, adr, type_, ids, D,
              fricloss, nacon, Jaref, done, force, state, partial,
              changed_ids, changed_count, efc_stride, ctx_stride, adr_stride,
              track_changes: bool) -> bool:
    """Native update_constraint_efc with per-row cost partials. Returns True
    when the kernel ran; the caller must then run cost_fold."""
    if not _enabled("MJLAB_SYCL_NATIVE_EFC"):
        return False
    if not _is_sycl(Jaref):
        return False
    _api()
    fn = _fn_efc
    if fn is None:
        return False
    return fn(impratio.ptr, impratio.shape[0], ne.ptr, nf.ptr, nefc.ptr,
              friction.ptr, cdim.ptr, adr.ptr, type_.ptr, ids.ptr, D.ptr,
              fricloss.ptr, nacon.ptr, Jaref.ptr, done.ptr, force.ptr,
              state.ptr, partial.ptr, changed_ids.ptr, changed_count.ptr,
              efc_stride, ctx_stride, adr_stride, 1 if track_changes else 0,
              Jaref.shape[0]) == 0


def cost_fold(partial, nefc, done, cost, ctx_stride: int) -> bool:
    """Deterministic sum of the efc cost partials into ctx cost."""
    if not _enabled("MJLAB_SYCL_NATIVE_EFC"):
        return False
    _api()
    fn = _fn_fold
    if fn is None:
        return False
    return fn(partial.ptr, nefc.ptr, done.ptr, cost.ptr, ctx_stride,
              partial.shape[0]) == 0


def gauss_cost(qacc, qfrc_smooth, qacc_smooth, Ma, done, gauss, cost,
               nv: int) -> bool:
    """Native update_constraint_gauss_cost (single-writer per world)."""
    if not _enabled("MJLAB_SYCL_NATIVE_GAUSS"):
        return False
    if not _is_sycl(qacc):
        return False
    _api()
    fn = _fn_gauss
    if fn is None:
        return False
    return fn(qacc.ptr, qfrc_smooth.ptr, qacc_smooth.ptr, Ma.ptr, done.ptr,
              gauss.ptr, cost.ptr, nv, qacc.shape[1], qacc.shape[0]) == 0


def ls_teardown(ls_iterations: int, min_step: float, cost, done, search, mv,
                nv: int, alpha, qacc, Ma) -> bool:
    """Native fused best_alpha + qacc_ma teardown."""
    if not _enabled("MJLAB_SYCL_NATIVE_LSTD"):
        return False
    if not _is_sycl(qacc):
        return False
    _api()
    fn = _fn_lstd
    if fn is None:
        return False
    return fn(ls_iterations, ctypes.c_float(min_step), cost.ptr, done.ptr,
              search.ptr, mv.ptr, nv, alpha.ptr, qacc.ptr, Ma.ptr,
              cost.shape[1], qacc.shape[1], qacc.shape[0]) == 0


def pool_stats() -> tuple:
    """(live_bytes, free_list_bytes, pending_bytes) of the USM pool."""
    try:
        import ctypes

        from mjlab_sycl.backend._src import build as _build

        dll = _build.ensure_sycl_runtime()
        fn = dll.wp_sycl_pool_stats
        fn.argtypes = [ctypes.POINTER(ctypes.c_longlong)] * 3
        a = ctypes.c_longlong()
        b = ctypes.c_longlong()
        c = ctypes.c_longlong()
        fn(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
        return a.value, b.value, c.value
    except Exception:
        return -1, -1, -1


def pool_trim(with_graphs: bool = True) -> None:
    """Return all pooled USM blocks to the OS (free-list trim).

    With ``with_graphs`` (default) every recorded command graph is freed
    first: replayed graphs bake raw pointers and cannot survive the pool
    releasing blocks (measured access violation). The graphs re-arm within
    a few calls. The trim drains the queue first; live arrays never sit in
    the free lists."""
    try:
        if with_graphs:
            from mjlab_sycl import graph_batch

            graph_batch.free_all_graphs()
        from mjlab_sycl.backend._src import build as _build

        dll = _build.ensure_sycl_runtime()
        dll.wp_sycl_pool_trim()
    except Exception:
        pass  # best-effort


def qfrc_constraint(J, force, nefc, done, out, nv: int, nv_pad: int,
                    njmax_pad: int) -> bool:
    """Native qfrc_constraint = J^T @ force. Returns True when it ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_QFRC"):
        return False
    if not _is_sycl(J):
        return False
    _api()
    fn = _fn_qfrc
    if fn is None:
        return False
    return fn(J.ptr, force.ptr, nefc.ptr, done.ptr, out.ptr, nv, nv_pad,
              njmax_pad, J.shape[0]) == 0


def jaref(jv, alpha, nefc, done, cost, Jaref, gauss, cost_out, prev_cost,
          grad_dot, search_dot, changed_count) -> bool:
    """Native fused linesearch_jaref + zero-ahead. Returns True when it ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_JAREF"):
        return False
    if not _is_sycl(jv):
        return False
    _api()
    fn = _fn_jaref
    if fn is None:
        return False
    return fn(jv.ptr, alpha.ptr, nefc.ptr, done.ptr, cost.ptr, Jaref.ptr,
              gauss.ptr, cost_out.ptr, prev_cost.ptr, grad_dot.ptr,
              search_dot.ptr, changed_count.ptr, Jaref.shape[1],
              Jaref.shape[0]) == 0


def quad_gauss(impratio, nefc, friction, dim_, adr, efc_type, efc_id, efc_D,
               nacon, Jaref, jv, done, nv, qfrc_smooth, efc_Ma, search,
               gauss, mv, quad_out, quad_gauss_out) -> bool:
    """Native fused prepare_quad + prepare_gauss. Returns True when it ran."""
    if not _enabled("MJLAB_SYCL_NATIVE_QUAD"):
        return False
    if not _is_sycl(Jaref):
        return False
    _api()
    fn = _fn_quad
    if fn is None:
        return False
    return fn(impratio.ptr, nefc.ptr, friction.ptr, dim_.ptr, adr.ptr,
              efc_type.ptr, efc_id.ptr, efc_D.ptr, nacon.ptr, Jaref.ptr,
              jv.ptr, done.ptr, qfrc_smooth.ptr, efc_Ma.ptr, search.ptr,
              gauss.ptr, mv.ptr, quad_out.ptr, quad_gauss_out.ptr,
              impratio.shape[0], nv, efc_D.shape[1], Jaref.shape[1],
              search.shape[1], adr.shape[1], Jaref.shape[0]) == 0
