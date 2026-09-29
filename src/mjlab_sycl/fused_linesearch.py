# SPDX-License-Identifier: Apache-2.0
"""Fuse the linesearch parallel teardown and jv+mv computation kernels.

Two fusions in the parallel linesearch path (the path microduck uses,
``m.opt.ls_parallel=True``):

1. **Teardown fusion**: ``linesearch_parallel_best_alpha`` +
   ``linesearch_qacc_ma`` → one kernel.  Both depend on ``alpha`` from
   best_alpha.  A single work-item per world scans the cost array, then
   updates qacc/Ma (nv dofs).  Saves 1 launch × ~10 iterations × 4
   substeps = ~40 launches/step.

   NOTE: ``linesearch_jaref`` is NOT included here — it is handled by
   ``fused_solver.py``'s ``_jaref_zeroahead`` which also folds in the
   per-iteration zero/rotate kernels.  Including jaref+zeroahead here
   would double-execute them (both this kernel and the jaref interceptor
   would fire), corrupting cost/prev_cost/gauss.

2. **mv+jv fusion**: ``mul_m_dense`` + ``linesearch_jv_fused`` → one
   kernel.  Both read ``search`` independently.  A single work-item per
   world computes mv = qM @ search (nv×nv dense dot) then jv = J @ search
   (nefc×nv dense dot).  Saves 1 launch × ~10 iterations × 4 substeps =
   ~40 launches/step.

Both fusions are installed as replacements inside the
``_linesearch_parallel`` and ``_linesearch`` functions.  The interceptor
layer is the same pattern as ``fused_solver.py``: wrap wp.launch to
replace specific kernel keys, and suppress the originals.

Installed AFTER launch_cache and fused_solver (outermost layer) so the
cache sees the fused kernels.

Kill switch: ``MJLAB_SYCL_FUSED_LINESEARCH=0``.
"""

from __future__ import annotations

import os

import warp as wp

_prev_launch = None
_ORIG = {}


def _enabled() -> bool:
  return os.environ.get("MJLAB_SYCL_FUSED_LINESEARCH", "1").strip().lower() not in (
    "0",
    "false",
    "off",
  )


# ---------------------------------------------------------------------------
# Fused teardown: best_alpha + qacc_ma (NOT jaref — that's fused_solver's job)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _ls_teardown_fused(
  # best_alpha args:
  opt_ls_iterations: int,
  opt_ls_parallel_min_step: float,
  cost_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  # qacc_ma args:
  ctx_search_in: wp.array2d[float],
  ctx_mv_in: wp.array2d[float],
  # model:
  nv: int,
  # outputs:
  ctx_alpha_out: wp.array[float],
  qacc_out: wp.array2d[float],
  efc_Ma_out: wp.array2d[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  # ── best_alpha: scan cost array ────────────────────────────────────
  bestid = int(0)
  best_cost = float(1e30)  # MJ_MAXVAL
  for i in range(opt_ls_iterations):
    c = cost_in[worldid, i]
    if c < best_cost:
      best_cost = c
      bestid = i

  # _log_scale inline
  log_min = wp.log(opt_ls_parallel_min_step)
  log_max = wp.log(1.0)
  step = (log_max - log_min) / wp.max(1.0, float(opt_ls_iterations - 1))
  alpha = wp.exp(log_min + float(bestid) * step)
  ctx_alpha_out[worldid] = alpha

  # ── qacc_ma: update qacc and Ma ────────────────────────────────────
  for d in range(nv):
    qacc_out[worldid, d] += alpha * ctx_search_in[worldid, d]
    efc_Ma_out[worldid, d] += alpha * ctx_mv_in[worldid, d]


# ---------------------------------------------------------------------------
# Fused mv+jv: mul_m_dense + linesearch_jv_fused (dense, threads_per_efc=1)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _mv_jv_fused(
  # model:
  nv: int,
  # mv args (mul_m_dense):
  qM_in: wp.array3d[float],
  ctx_search_in: wp.array2d[float],
  # jv args:
  nefc_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  njmax_in: int,
  # skip:
  ctx_done_in: wp.array[bool],
  # outputs:
  ctx_mv_out: wp.array2d[float],
  ctx_jv_out: wp.array2d[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  # ── mv = qM @ search (dense, nv×nv) ─────────────────────────────────
  for d in range(nv):
    s = float(0.0)
    for i in range(nv):
      s += qM_in[worldid, d, i] * ctx_search_in[worldid, i]
    ctx_mv_out[worldid, d] = s

  # ── jv = J @ search (dense, nefc×nv) ───────────────────────────────
  n = wp.min(njmax_in, nefc_in[worldid])
  for e in range(n):
    s = float(0.0)
    for i in range(nv):
      s += efc_J_in[worldid, e, i] * ctx_search_in[worldid, i]
    ctx_jv_out[worldid, e] = s


# ---------------------------------------------------------------------------
# Interceptor: replace specific kernel launches in the linesearch path
# ---------------------------------------------------------------------------

_CTX = None
_SUPPRESS_TEARDOWN = frozenset(
  {
    "linesearch_qacc_ma",
  }
)


def _intercept_launch(kernel, dim, inputs=(), outputs=(), *args, **kwargs):
  global _CTX
  if _CTX is not None:
    key = getattr(kernel, "key", None)

    # mv+jv fusion: replace mul_m_dense + linesearch_jv_fused with one launch
    if key == "mul_m_dense__locals___mul_m_dense":
      # Defer: stash inputs/outputs and suppress; the jv_fused interceptor
      # will merge both.
      _CTX._pending_mv_inputs = list(inputs)
      _CTX._pending_mv_outputs = list(outputs)
      return None

    if key == "linesearch_jv_fused__locals__kernel":
      mv_in = getattr(_CTX, "_pending_mv_inputs", None)
      mv_out = getattr(_CTX, "_pending_mv_outputs", None)
      if mv_in is not None:
        _CTX._pending_mv_inputs = None
        _CTX._pending_mv_outputs = None
        m = _CTX._model
        d = _CTX._data
        # mv_in = [qM, search]  mv_out = [mv]
        # jv inputs = [nefc, J_rownnz, J_rowadr, J_colind, J, search, done]
        # jv outputs = [jv]
        return _prev_launch(
          _mv_jv_fused,
          dim=d.nworld,
          inputs=[
            m.nv,
            mv_in[0],  # qM
            mv_in[1],  # search
            d.nefc,
            inputs[4],  # efc_J (dense path: index 4)
            d.njmax,
            _CTX.done,
          ],
          outputs=[
            mv_out[0],  # mv
            outputs[0] if outputs else _CTX.jv,  # jv
          ],
        )

    # Teardown fusion: replace best_alpha + qacc_ma with one kernel.
    # jaref is NOT included — fused_solver handles it via _jaref_zeroahead.
    if key == "linesearch_parallel_best_alpha":
      ctx = _CTX
      m = ctx._model
      d = ctx._data
      return _prev_launch(
        _ls_teardown_fused,
        dim=d.nworld,
        inputs=[
          m.opt.ls_iterations,
          m.opt.ls_parallel_min_step,
          inputs[3] if len(inputs) > 3 else ctx._cost,
          ctx.done,
          ctx.search,
          ctx.mv,
          m.nv,
        ],
        outputs=[
          ctx.alpha,
          d.qacc,
          d.efc.Ma,
        ],
      )

    if key in _SUPPRESS_TEARDOWN:
      return None  # already done by the fused teardown

  return _prev_launch(kernel, dim, inputs, outputs, *args, **kwargs)


def install() -> None:
  global _prev_launch
  if _prev_launch is not None:
    return
  if not _enabled():
    return
  _prev_launch = wp.launch
  wp.launch = _intercept_launch
  from warp._src import context as _ctx

  _ctx.launch = _intercept_launch

  from mujoco_warp._src import solver

  _ORIG["solve"] = solver.solve

  def solve_fused(m, d):
    if not _enabled() or not m.opt.ls_parallel or d.njmax == 0 or m.nv == 0:
      return _ORIG["solve"](m, d)

    # Wrap _linesearch (not _linesearch_parallel) so _CTX is active during
    # the mul_m + jv_fused launches that happen BEFORE _linesearch_parallel.
    _ORIG["_linesearch"] = solver._linesearch

    def _linesearch_fused(m, d, ctx, cost):
      global _CTX
      ctx._model = m
      ctx._data = d
      ctx._cost = cost
      _CTX = ctx
      try:
        _ORIG["_linesearch"](m, d, ctx, cost)
      finally:
        _CTX = None

    solver._linesearch = _linesearch_fused
    try:
      return _ORIG["solve"](m, d)
    finally:
      solver._linesearch = _ORIG["_linesearch"]

  solver.solve = solve_fused
  print(
    "[sycl-fused-linesearch] linesearch teardown (2→1) + mv+jv (2→1) "
    "fused (MJLAB_SYCL_FUSED_LINESEARCH=0 to disable)"
  )


def uninstall() -> None:
  global _prev_launch
  if _prev_launch is None:
    return
  wp.launch = _prev_launch
  from warp._src import context as _ctx

  _ctx.launch = _prev_launch
  _prev_launch = None
  if "solve" in _ORIG:
    from mujoco_warp._src import solver

    solver.solve = _ORIG["solve"]
    if "_linesearch" in _ORIG:
      solver._linesearch = _ORIG["_linesearch"]
  _ORIG.clear()
