# SPDX-License-Identifier: Apache-2.0
"""Reuse mujoco_warp's per-solve solver scratch allocations across solves.

``solver.solve()`` builds a fresh SolverContext (~18 ``wp.empty`` arrays)
plus ``step_size_cost`` and ``nsolving`` on EVERY call -- 4 solves per
env.step.  The launch cache keys on argument identity, so every solver
kernel whose args touch a fresh array gets a fresh key each solve: the
cache re-runs its recurrence gate and the full pack/ArgsStruct build
before hits resume.  Measured at 4096 envs (microduck): ~230 first-seen
launch keys per step, ~28% of all launches on the slow rebuild path,
~65 ms/step of host submit overhead.

The context is scratch state: ``init_context`` re-initializes it at the
top of every solve and the SYCL queue orders solves against each other,
so reusing the allocations is safe.  Entries are cached per (model,
data) with a shape signature check and hold strong references, so Python
id recycling can never surface a stale entry.

This copies the pinned mujoco-warp 3.8.1 bodies of ``solve``/``_solve``
to hoist the two allocations that live inside ``_solve`` (there is no
seam to hook them otherwise).  Any environment the copy does not
recognize falls through to the originals unchanged.

Kill switch: ``MJLAB_SYCL_SOLVER_CTX=0``.
"""

from __future__ import annotations

import os

import warp as wp

_orig_solve = None
# (id(m), id(d)) -> (m, d, ctx, step_size_cost, nsolving, sig).  The strong
# refs make the id key safe: entries cannot recycle underneath a lookup.
_CACHE: dict = {}


def _enabled() -> bool:
  return os.environ.get("MJLAB_SYCL_SOLVER_CTX", "1").strip().lower() not in (
    "0",
    "false",
    "off",
  )


def _sig(m, d) -> tuple:
  # Everything create_solver_context and _solve allocate or branch on.
  return (
    d.nworld,
    m.nv,
    m.nv_pad,
    d.njmax,
    int(m.opt.solver),
    bool(m.opt.graph_conditional),
    int(m.opt.iterations),
    bool(m.opt.ls_parallel),
    int(m.opt.ls_iterations),
  )


def _scratch(m, d):
  from mujoco_warp._src import solver as _solver

  sig = _sig(m, d)
  entry = _CACHE.get((id(m), id(d)))
  if entry is not None and entry[5] == sig:
    return entry[2], entry[3], entry[4]

  ctx = _solver.create_solver_context(m, d)
  step_size_cost = wp.empty(
    (d.nworld, m.opt.ls_iterations if m.opt.ls_parallel else 0), dtype=float
  )
  nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)
  _CACHE[(id(m), id(d))] = (m, d, ctx, step_size_cost, nsolving, sig)
  return ctx, step_size_cost, nsolving


def install() -> None:
  global _orig_solve
  if _orig_solve is not None:
    return
  if not _enabled():
    return

  from mujoco_warp._src import solver as _solver
  from mujoco_warp._src import types as _types
  from mujoco_warp._src.warp_util import event_scope

  _orig_solve = _solver.solve

  def solve_patched(m, d):
    if d.njmax == 0 or m.nv == 0:
      wp.copy(d.qacc, d.qacc_smooth)
      d.solver_niter.fill_(0)
      return

    ctx, step_size_cost, nsolving = _scratch(m, d)
    # wp.full every call in the original; the cached counter just refills.
    nsolving.fill_(d.nworld)

    # body of solver._solve (mujoco-warp 3.8.1) with the two per-solve
    # allocations replaced by the cached pair
    if not (m.opt.disableflags & _types.DisableBit.WARMSTART):
      wp.copy(d.qacc, d.qacc_warmstart)
    else:
      wp.copy(d.qacc, d.qacc_smooth)

    _solver.init_context(m, d, ctx, grad=True)

    wp.launch(
      _solver.solve_init_search,
      dim=(d.nworld, m.nv),
      inputs=[ctx.Mgrad],
      outputs=[ctx.search, ctx.search_dot],
    )

    if m.opt.iterations != 0 and m.opt.graph_conditional:
      wp.capture_while(
        nsolving,
        while_body=_solver._solver_iteration,
        m=m,
        d=d,
        ctx=ctx,
        step_size_cost=step_size_cost,
        nsolving=nsolving,
      )
    else:
      for _ in range(m.opt.iterations):
        _solver._solver_iteration(m, d, ctx, step_size_cost, nsolving)

  _solver.solve = event_scope(solve_patched, name="solve")
  print(
    "[sycl-solver-ctx] per-solve scratch reuse installed "
    "(MJLAB_SYCL_SOLVER_CTX=0 to disable)"
  )


def uninstall() -> None:
  global _orig_solve
  if _orig_solve is None:
    return
  from mujoco_warp._src import solver as _solver

  _solver.solve = _orig_solve
  _orig_solve = None
