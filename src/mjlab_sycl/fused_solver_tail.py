# SPDX-License-Identifier: Apache-2.0
"""Fuse the 4-5 small per-iteration solver kernels into 1 launch.

Each solver iteration (m.opt.iterations, default 11) ends with:
  1. solve_prev_grad_Mgrad  (nworld×nv)  — copy grad/Mgrad to prev
  2. solve_beta             (nworld)     — Polak-Ribiere beta (CG only)
  3. solve_zero_search_dot  (nworld)     — search_dot = 0
  4. solve_search_update    (nworld×nv)  — new search + atomic search_dot
  5. solve_done             (nworld)     — convergence check

That is 5 launches × 11 iterations = 55 launches/step. For nv=20 all four
fit in a single per-world work-item: the reduction over nv is a trivial loop.

This module replaces the 5 launches with 1, reducing solver-iteration
overhead by ~4/5.  Enabled with MJLAB_SYCL_FUSED_SOLVER_TAIL=1 (default on
when patch_simulation_for_sycl runs); set to 0 to fall back.
"""

from __future__ import annotations

import os

import warp as wp

_ORIG = None


def _enabled() -> bool:
    return os.environ.get("MJLAB_SYCL_FUSED_SOLVER_TAIL", "1").strip().lower() not in (
        "0", "false", "off",
    )


# ---------------------------------------------------------------------------
# Fused kernel: per-world, inlines prev_grad + beta + search + done
# ---------------------------------------------------------------------------

# Solver type constants (avoid importing mujoco_warp.types at module level —
# the kernel is compiled before the patch runs)
_SOLVER_CG = 1     # SolverType.CG
_SOLVER_NEWTON = 0  # SolverType.NEWTON


@wp.kernel(enable_backward=False)
def _solver_tail_fused(
    nv: int,
    opt_solver: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    stat_meaninertia: wp.array[float],
    # In:
    ctx_grad_in: wp.array2d[float],
    ctx_Mgrad_in: wp.array2d[float],
    ctx_search_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    ctx_cost_in: wp.array[float],
    ctx_prev_cost_in: wp.array[float],
    ctx_grad_dot_in: wp.array[float],
    # In/out:
    ctx_prev_grad_out: wp.array2d[float],
    ctx_prev_Mgrad_out: wp.array2d[float],
    ctx_search_out: wp.array2d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_beta_out: wp.array[float],
    # Out:
    solver_niter_out: wp.array[int],
    nsolving_out: wp.array[int],
    ctx_done_out: wp.array[bool],
):
    worldid = wp.tid()

    # If already done, skip entirely (all kernels check this)
    if ctx_done_in[worldid]:
        return

    # ── 1. beta (CG only) — MUST read prev_grad/Mgrad BEFORE overwriting ──
    # The previous iteration stored its grad/Mgrad into prev_grad_out/
    # prev_Mgrad_out.  We read those OLD values here, then overwrite below.
    beta = 0.0
    if opt_solver == _SOLVER_CG:
        beta_num = float(0.0)
        beta_den = float(0.0)
        for dofid in range(nv):
            prev_g = ctx_prev_grad_out[worldid, dofid]
            prev_Mg = ctx_prev_Mgrad_out[worldid, dofid]
            beta_num += ctx_grad_in[worldid, dofid] * (
                ctx_Mgrad_in[worldid, dofid] - prev_Mg
            )
            beta_den += prev_g * prev_Mg
        beta = wp.max(0.0, beta_num / wp.max(1e-14, beta_den))
    ctx_beta_out[worldid] = beta

    # ── 2. Save current grad/Mgrad as prev for next iteration ────────────
    for dofid in range(nv):
        ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]
        ctx_prev_Mgrad_out[worldid, dofid] = ctx_Mgrad_in[worldid, dofid]

    # ── 3. zero_search_dot ────────────────────────────────────────────────
    ctx_search_dot_out[worldid] = 0.0

    # ── 4. search_update + search_dot ─────────────────────────────────────
    search_dot = float(0.0)
    for dofid in range(nv):
        search = -1.0 * ctx_Mgrad_in[worldid, dofid]
        if opt_solver == _SOLVER_CG:
            search += beta * ctx_search_in[worldid, dofid]
        ctx_search_out[worldid, dofid] = search
        search_dot += search * search
    ctx_search_dot_out[worldid] = search_dot

    # ── 5. done ───────────────────────────────────────────────────────────
    solver_niter_out[worldid] += 1
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]

    # _rescale: scale by 1/(meaninertia * nv)
    scale = 1.0 / (meaninertia * float(nv))
    improvement = (ctx_prev_cost_in[worldid] - ctx_cost_in[worldid]) * scale
    gradient = wp.sqrt(ctx_grad_dot_in[worldid]) * scale

    done = (improvement < tolerance) or (gradient < tolerance)
    if done or solver_niter_out[worldid] == opt_iterations:
        ctx_done_out[worldid] = True
        wp.atomic_add(nsolving_out, 0, -1)


# ---------------------------------------------------------------------------
# Install: wrap _solver_iteration to replace the 5 tail kernels with 1
# ---------------------------------------------------------------------------

def install() -> None:
    global _ORIG
    if _ORIG is not None:
        return
    from mujoco_warp._src import solver as _solver

    _ORIG = _solver._solver_iteration

    def patched_iteration(m, d, ctx, step_size_cost, nsolving):
        if not _enabled():
            return _ORIG(m, d, ctx, step_size_cost, nsolving)

        # Run linesearch + constraint + gradient (unchanged — heavy kernels
        # with tile-based parallelism that we don't touch).
        _solver._linesearch(m, d, ctx, step_size_cost)

        if m.opt.solver == _solver.types.SolverType.CG:
            # prev_grad_Mgrad was a separate launch in the original; now fused.
            pass  # handled inside _solver_tail_fused

        incremental = (
            m.opt.solver == _solver.types.SolverType.NEWTON
            and m.opt.cone != _solver.types.ConeType.ELLIPTIC
        )
        if incremental:
            ctx.changed_efc_count.zero_()

        _solver._update_constraint(m, d, ctx, track_changes=incremental)

        if incremental:
            _solver._update_gradient_incremental(m, d, ctx)
        else:
            _solver._update_gradient(m, d, ctx)

        # ── FUSED TAIL: 1 launch replaces 5 ───────────────────────────────
        wp.launch(
            _solver_tail_fused,
            dim=d.nworld,
            inputs=[
                m.nv,
                int(m.opt.solver),
                m.opt.tolerance,
                m.opt.iterations,
                m.stat.meaninertia,
                ctx.grad,
                ctx.Mgrad,
                ctx.search,
                ctx.done,
                ctx.cost,
                ctx.prev_cost,
                ctx.grad_dot,
            ],
            outputs=[
                ctx.prev_grad,
                ctx.prev_Mgrad,
                ctx.search,
                ctx.search_dot,
                ctx.beta,
                d.solver_niter,
                nsolving,
                ctx.done,
            ],
            device=d.qLD.device,
        )

    _solver._solver_iteration = patched_iteration
    print("[sycl-fused] fused solver tail installed "
          "(MJLAB_SYCL_FUSED_SOLVER_TAIL=0 to disable)")


def uninstall() -> None:
    global _ORIG
    if _ORIG is None:
        return
    from mujoco_warp._src import solver as _solver
    _solver._solver_iteration = _ORIG
    _ORIG = None
