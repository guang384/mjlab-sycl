# SPDX-License-Identifier: Apache-2.0
"""Fuse the solver loop's zero/rotate launches into the tail of
``linesearch_jaref`` — the last kernel of every solver iteration.

Each Newton solver iteration launches four pure zero/rotate kernels whose
only job is to prepare accumulators for the phase that follows:

  - ``update_constraint_init_cost``  (rotate prev_cost=cost; zero cost+gauss)
  - ``update_gradient_zero_grad_dot`` (zero grad_dot)
  - ``solve_zero_search_dot``         (zero search_dot)
  - ``ctx.changed_efc_count.zero_()`` (SYCL memset)

All four sit between ``linesearch_jaref`` (end of iteration k's linesearch)
and their first consumer in iteration k's update phase — nothing reads or
writes those buffers in between.  This module appends the zeroing to the
jaref kernel itself (work-item ``efcid==0`` per world), so one launch does
the work of five.  Per iteration that is 4 fewer launches; at ~10
iterations x 4 physics substeps per env step it removes ~160
launches/memsets per step.

The jaref replacement must be layered OUTSIDE ``launch_cache`` (installed
after it) so the cache sees the *fused* kernel and its args — otherwise a
cache hit would replay the unfused jaref and skip the zeroing while the
suppressed kernels stay suppressed.

Safety properties (verified against the mujoco_warp sources):
  - ``changed_efc_count`` is zeroed for EVERY world (including done ones):
    the original is a whole-array memset and ``update_gradient_h_incremental``
    reads it without a done-guard.
  - The rotate/zeros skip done worlds exactly like
    ``update_constraint_init_cost``.
  - The zeroahead happens at the same logical point as the original
    kernels — after the last linesearch kernel, before the first update
    kernel — with a kernel boundary (implicit device-wide sync) on both
    sides.
  - Only active when ``m.opt.ls_parallel`` (the parallel linesearch is the
    path that ends with ``linesearch_jaref``); other configurations fall
    back to the original unmodified iteration.

Kill switch: ``MJLAB_SYCL_FUSED_SOLVER=0``.
"""

from __future__ import annotations

import os

import warp as wp

_CTX = None  # active SolverContext while a fused iteration runs
_SUPPRESS = frozenset(
  {
    "update_constraint_init_cost",
    "update_gradient_zero_grad_dot",
    "solve_zero_search_dot",
  }
)
_prev_launch = None
_orig_solver_iteration = None


def _enabled() -> bool:
  return os.environ.get("MJLAB_SYCL_FUSED_SOLVER", "1").strip().lower() not in (
    "0",
    "false",
    "off",
  )


@wp.kernel(enable_backward=False)
def _jaref_zeroahead(
  # original linesearch_jaref args:
  nefc_in: wp.array[int],
  ctx_jv_in: wp.array2d[float],
  ctx_alpha_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  ctx_cost_in: wp.array[float],
  # original output:
  ctx_Jaref_out: wp.array2d[float],
  # zero-ahead outputs:
  ctx_gauss_out: wp.array[float],
  ctx_cost_out: wp.array[float],
  ctx_prev_cost_out: wp.array[float],
  ctx_grad_dot_out: wp.array[float],
  ctx_search_dot_out: wp.array[float],
  changed_count_out: wp.array[int],
):
  worldid, efcid = wp.tid()

  # changed_efc_count: zero for EVERY world (memset semantics; the
  # incremental-H kernel reads it without a done-guard).
  if efcid == 0:
    changed_count_out[worldid] = 0

  if ctx_done_in[worldid]:
    return

  # rotate + zeros, done-guarded exactly like update_constraint_init_cost.
  if efcid == 0:
    ctx_gauss_out[worldid] = 0.0
    ctx_prev_cost_out[worldid] = ctx_cost_in[worldid]
    ctx_cost_out[worldid] = 0.0
    ctx_grad_dot_out[worldid] = 0.0
    ctx_search_dot_out[worldid] = 0.0

  if efcid >= nefc_in[worldid]:
    return

  ctx_Jaref_out[worldid, efcid] += ctx_alpha_in[worldid] * ctx_jv_in[worldid, efcid]


def _intercept_launch(kernel, dim, inputs=(), outputs=(), *args, **kwargs):
  global _CTX
  if _CTX is not None:
    key = getattr(kernel, "key", None)
    if key == "linesearch_jaref":
      ctx = _CTX
      return _prev_launch(
        _jaref_zeroahead,
        dim,
        list(inputs) + [ctx.cost],
        list(outputs)
        + [
          ctx.gauss,
          ctx.cost,
          ctx.prev_cost,
          ctx.grad_dot,
          ctx.search_dot,
          ctx.changed_efc_count,
        ],
        *args,
        **kwargs,
      )
    if key in _SUPPRESS:
      return None  # already zeroed by the jaref tail
  return _prev_launch(kernel, dim, inputs, outputs, *args, **kwargs)


def install() -> None:
  global _prev_launch, _orig_solver_iteration
  if _prev_launch is not None:
    return
  if not _enabled():
    return
  from mujoco_warp._src import solver
  from mujoco_warp._src.types import ConeType
  from mujoco_warp._src.types import SolverType

  _prev_launch = wp.launch
  wp.launch = _intercept_launch
  from warp._src import context as _ctx_mod

  _ctx_mod.launch = _intercept_launch

  _orig_solver_iteration = solver._solver_iteration

  def _solver_iteration_fused(m, d, ctx, step_size_cost, nsolving):
    global _CTX
    fused = _enabled() and bool(m.opt.ls_parallel)
    _CTX = ctx if fused else None
    try:
      solver._linesearch(m, d, ctx, step_size_cost)

      if m.opt.solver == SolverType.CG:
        wp.launch(
          solver.solve_prev_grad_Mgrad,
          dim=(d.nworld, m.nv),
          inputs=[ctx.grad, ctx.Mgrad, ctx.done],
          outputs=[ctx.prev_grad, ctx.prev_Mgrad],
        )

      incremental = (
        m.opt.solver == SolverType.NEWTON and m.opt.cone != ConeType.ELLIPTIC
      )

      # changed_efc_count: zeroed by the jaref tail when fused; otherwise
      # keep the original memset (must complete before update_constraint_efc
      # atomically increments it).
      if incremental and not fused:
        ctx.changed_efc_count.zero_()

      solver._update_constraint(m, d, ctx, track_changes=incremental)

      if incremental:
        solver._update_gradient_incremental(m, d, ctx)
      else:
        solver._update_gradient(m, d, ctx)

      if m.opt.solver == SolverType.CG:
        wp.launch(
          solver.solve_beta,
          dim=d.nworld,
          inputs=[m.nv, ctx.grad, ctx.Mgrad, ctx.prev_grad, ctx.prev_Mgrad, ctx.done],
          outputs=[ctx.beta],
        )

      # When fused, the interceptor suppresses this launch (the jaref tail
      # already zeroed search_dot); when falling back it passes through.
      wp.launch(
        solver.solve_zero_search_dot,
        dim=d.nworld,
        inputs=[ctx.done],
        outputs=[ctx.search_dot],
      )

      wp.launch(
        solver.solve_search_update,
        dim=(d.nworld, m.nv),
        inputs=[m.opt.solver, ctx.Mgrad, ctx.search, ctx.beta, ctx.done],
        outputs=[ctx.search, ctx.search_dot],
      )

      wp.launch(
        solver.solve_done,
        dim=d.nworld,
        inputs=[
          m.nv,
          m.opt.tolerance,
          m.opt.iterations,
          m.stat.meaninertia,
          ctx.grad_dot,
          ctx.cost,
          ctx.prev_cost,
          ctx.done,
        ],
        outputs=[d.solver_niter, nsolving, ctx.done],
      )
    finally:
      _CTX = None

  solver._solver_iteration = _solver_iteration_fused
  print(
    "[sycl-fused-solver] solver zero/rotate launches fused into "
    "linesearch_jaref tail (MJLAB_SYCL_FUSED_SOLVER=0 to disable)"
  )


def uninstall() -> None:
  global _prev_launch, _orig_solver_iteration
  if _prev_launch is None:
    return
  wp.launch = _prev_launch
  from warp._src import context as _ctx_mod

  _ctx_mod.launch = _prev_launch
  _prev_launch = None
  if _orig_solver_iteration is not None:
    from mujoco_warp._src import solver

    solver._solver_iteration = _orig_solver_iteration
    _orig_solver_iteration = None
