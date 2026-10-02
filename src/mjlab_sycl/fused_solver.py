# SPDX-License-Identifier: Apache-2.0
"""Fuse the solver loop's small per-iteration launches into their neighbors.

Two merges live here:

1. ``linesearch_jaref`` zero-ahead (below): the four pure zero/rotate
   kernels fold into the tail of ``linesearch_jaref`` — the last kernel of
   every solver iteration.

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

2. ``solve_search_update`` + ``solve_done``: the per-world convergence
   bookkeeping runs in the per-dof search-update kernel's work-item
   ``(world, 0)``.  The two launches are adjacent in the iteration and
   ``solve_done`` only reads cost/prev_cost/grad_dot (final long before
   both), so values are identical.

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

# The two mujoco_warp references used inside the _search_done_fused kernel
# BODY. Warp evaluates kernel signature annotations at decoration time (see
# fused_tree.py's NOTE) but resolves body-only global references later, so
# these just need to be populated before the kernel can ever launch -- which
# install(), below, guarantees. That keeps this module importable without
# mujoco_warp (fused_tree/fused_linesearch cannot do the same: their kernel
# signatures themselves use mujoco_warp types).
_mw_types = None
_mw_rescale = None

_CTX = None  # active SolverContext while a fused iteration runs
_SUPPRESS = frozenset(
  {
    "update_constraint_init_cost",
    "update_gradient_zero_grad_dot",
    "solve_zero_search_dot",
  }
)
_prev_launch = None
_EFC_PARTIAL_CACHE = {}  # (nworld, njmax) -> scratch wp.array


def _efc_partial(nworld: int, njmax: int):
    key = (nworld, njmax)
    arr = _EFC_PARTIAL_CACHE.get(key)
    if arr is None:
        arr = wp.zeros((nworld, njmax), dtype=float)
        if len(_EFC_PARTIAL_CACHE) < 4:
            _EFC_PARTIAL_CACHE[key] = arr
    return arr
_orig_solver_iteration = None
_pending_search_update = None  # deferred launch call awaiting the solve_done merge


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


@wp.kernel(enable_backward=False)
def _search_done_fused(
  # solve_search_update args:
  opt_solver: int,
  ctx_Mgrad_in: wp.array2d[float],
  ctx_search_in: wp.array2d[float],
  ctx_beta_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # solve_done args:
  nv: int,
  opt_tolerance: wp.array[float],
  opt_iterations: int,
  stat_meaninertia: wp.array[float],
  ctx_grad_dot_in: wp.array[float],
  ctx_cost_in: wp.array[float],
  ctx_prev_cost_in: wp.array[float],
  # outputs (search_update first, then done):
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
  solver_niter_out: wp.array[int],
  nsolving_out: wp.array[int],
  ctx_done_out: wp.array[bool],
):
  """solve_search_update + solve_done in one launch.

  solve_done launches immediately after solve_search_update in the
  iteration and only reads cost/prev_cost/grad_dot (final long before both),
  so its per-world bookkeeping can run in work-item (world, 0) alongside
  the per-dof search update: same values, one fewer launch.
  """
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  # ── solve_search_update (verbatim) ─────────────────────────────────
  search = -1.0 * ctx_Mgrad_in[worldid, dofid]
  if opt_solver == _mw_types.SolverType.CG:
    search += ctx_beta_in[worldid] * ctx_search_in[worldid, dofid]
  ctx_search_out[worldid, dofid] = search
  wp.atomic_add(ctx_search_dot_out, worldid, search * search)

  # ── solve_done (verbatim, single writer at dof 0) ──────────────────
  if dofid == 0:
    solver_niter_out[worldid] += 1
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
    improvement = _mw_rescale(nv, meaninertia, ctx_prev_cost_in[worldid] - ctx_cost_in[worldid])
    gradient = _mw_rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))
    done = (improvement < tolerance) or (gradient < tolerance)
    if done or solver_niter_out[worldid] == opt_iterations:
      ctx_done_out[worldid] = True
      wp.atomic_add(nsolving_out, 0, -1)


def _intercept_launch(kernel, dim, inputs=(), outputs=(), *args, **kwargs):
  global _CTX, _pending_search_update
  if _CTX is not None:
    key = getattr(kernel, "key", None)
    if key == "linesearch_jaref":
      ctx = _CTX
      from mjlab_sycl import native_kernels

      if native_kernels.jaref(
          inputs[1], inputs[2], inputs[0], inputs[3], ctx.cost,
          outputs[0], ctx.gauss, ctx.cost, ctx.prev_cost, ctx.grad_dot,
          ctx.search_dot, ctx.changed_efc_count,
      ):
        return None  # the native kernel ran; suppress the warp launch
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
    if key == "update_constraint_ecf" or key == "update_constraint_efc__locals__kernel":
      from mjlab_sycl import native_kernels

      # inputs = [impratio, ne, nf, nefc, friction, dim, adr, type, id, D,
      #           fricloss, nacon, Jaref, done]
      # outputs = [force, state, cost, changed_ids, changed_count]
      # per-row cost partials replace the same-address atomic storm; the
      # deterministic fold lands in ctx.cost, then gauss_cost adds its term
      partial = _efc_partial(inputs[12].shape[0], inputs[12].shape[1])
      if native_kernels.efc_force(
          inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
          inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
          inputs[12], inputs[13], outputs[0], outputs[1], partial,
          outputs[3], outputs[4],
          inputs[9].shape[1], inputs[12].shape[1], inputs[6].shape[1],
          getattr(kernel, "_mjlab_track", False),
      ) and native_kernels.cost_fold(
          partial, inputs[3], inputs[13], outputs[2], inputs[12].shape[1],
      ):
        return None  # the native pair ran; suppress the warp launch
      return _prev_launch(kernel, dim, inputs, outputs, *args, **kwargs)

    if key == "update_constraint_init_qfrc_constraint_dense":
      from mjlab_sycl import native_kernels

      # inputs = [nefc, efc_J, efc_force, njmax, done]; outputs = [qfrc]
      if native_kernels.qfrc_constraint(
          inputs[1], inputs[2], inputs[0], inputs[4], outputs[0],
          outputs[0].shape[1], inputs[1].shape[2], inputs[1].shape[1],
      ):
        return None  # the native kernel ran; suppress the warp launch
      return _prev_launch(kernel, dim, inputs, outputs, *args, **kwargs)

    if key in _SUPPRESS:
      return None  # already zeroed by the jaref tail

    # search_update + solve_done merge: defer the per-dof search update,
    # merge it with the per-world done bookkeeping at solve_done's launch
    # (the two launches are adjacent in the iteration).
    if key == "solve_search_update":
      if _pending_search_update is not None:
        # unexpected kernel between the pair: keep the original order
        k, d_, i_, o_, a_, kw_ = _pending_search_update
        _pending_search_update = None
        _prev_launch(k, d_, i_, o_, *a_, **kw_)
      _pending_search_update = (kernel, dim, inputs, outputs, args, kwargs)
      return None
    if key == "solve_done" and _pending_search_update is not None:
      sk, sdim, sin, sout, sargs, skw = _pending_search_update
      _pending_search_update = None
      # merged params: search_update's 5 inputs + solve_done's first 7
      # inputs (its 8th is ctx.done, already in sin[4]), then search_update's
      # 2 outputs + solve_done's 3 outputs
      return _prev_launch(
        _search_done_fused,
        sdim,
        list(sin) + list(inputs[:7]),
        list(sout) + list(outputs),
        *sargs,
        **skw,
      )
    if _pending_search_update is not None:
      # any intervening launch must see the original order
      k, d_, i_, o_, a_, kw_ = _pending_search_update
      _pending_search_update = None
      _prev_launch(k, d_, i_, o_, *a_, **kw_)
  return _prev_launch(kernel, dim, inputs, outputs, *args, **kwargs)


def install() -> None:
  global _prev_launch, _orig_solver_iteration, _mw_types, _mw_rescale
  if _prev_launch is not None:
    return
  if not _enabled():
    return
  from mujoco_warp._src import solver
  from mujoco_warp._src import types as _types
  from mujoco_warp._src.solver import _rescale as _rescale
  from mujoco_warp._src.types import ConeType
  from mujoco_warp._src.types import SolverType
  # kernel-body globals (see the comment at their declaration)
  _mw_types = _types
  _mw_rescale = _rescale

  _prev_launch = wp.launch
  wp.launch = _intercept_launch
  from warp._src import context as _ctx_mod

  _ctx_mod.launch = _intercept_launch

  # tag each update_constraint_efc kernel with its track_changes variant so
  # the seam can reproduce it exactly (same pattern as the chol factory wrap)
  _orig_efc_factory = solver.update_constraint_efc

  def _efc_factory_tracked(track_changes):
    k = _orig_efc_factory(track_changes)
    try:
      k._mjlab_track = bool(track_changes)
    except Exception:
      pass
    return k

  solver.update_constraint_efc = _efc_factory_tracked

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
