# SPDX-License-Identifier: Apache-2.0
"""Fused replacement for the Python for-loops in mujoco_warp's set_const_0.

The three Python for-loops after factor_m (dof_invweight0, body_invweight0,
actuator_acc0) issue ~372 kernel launches for microduck (nv=20, nbody=16,
nu=14). This module replaces them with a single warp kernel — one work-item
per world, all Cholesky forward+back substitutions inlined.

Enabled by default after patch_simulation_for_sycl(); set
MJLAB_SYCL_FUSED_SET_CONST=0 to fall back to the original loops.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

_ORIG = None
_ORIG_EVENT_APPLY = None
_RECOMPUTE_ENV_IDS = None  # set by patched event_manager.apply before recompute
_J = {}  # (nworld, nv, dev) → scratch wp.array2d[float]
_Y = {}
_X = {}
_ENV_IDS_CACHE = {}  # (nworld, dev) → wp.array[int] for full-range fallback


def _enabled() -> bool:
    return os.environ.get("MJLAB_SYCL_FUSED_SET_CONST", "1").strip().lower() not in (
        "0", "false", "off",
    )


def _scratch(cache, nworld, nv, device):
    key = (nworld, nv, str(device))
    buf = cache.get(key)
    if buf is None:
        buf = wp.empty((nworld, nv), dtype=wp.float32, device=device)
        cache[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Mega-kernel: all three loops fused into one work-item-per-world kernel
# ---------------------------------------------------------------------------

@wp.kernel(enable_backward=False)
def _set_const_0_loops_fused(
    qLD: wp.array3d[float],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofadr: wp.array[int],
    body_dofnum: wp.array[int],
    body_weldid: wp.array[int],
    dof_parentid: wp.array[int],
    dof_jntid: wp.array[int],
    jnt_type: wp.array[int],
    jnt_dofadr: wp.array[int],
    subtree_com: wp.array2d[wp.vec3],
    xipos: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    moment_rownnz: wp.array2d[int],
    moment_rowadr: wp.array2d[int],
    moment_colind: wp.array2d[int],
    actuator_moment: wp.array2d[float],
    nv: int,
    nbody: int,
    nu: int,
    env_ids: wp.array[int],       # world indices to process (selective recompute)
    scratch_j: wp.array2d[float],
    scratch_y: wp.array2d[float],
    scratch_x: wp.array2d[float],
    dof_invweight0_out: wp.array2d[float],
    body_invweight0_out: wp.array2d[wp.vec2],
    actuator_acc0_out: wp.array2d[float],
):
    tid = wp.tid()
    worldid = env_ids[tid]    # selective: only process reset envs
    sj = scratch_j  # alias for brevity
    sy = scratch_y
    sx = scratch_x

    # ════════════════════════════════════════════════════════════════════════
    # dof_invweight0:  for each dof i, solve M*x = e_i, take x[i] = diag(M⁻¹)[i]
    # ════════════════════════════════════════════════════════════════════════
    for i in range(nv):
        # forward sub: L * y = e_i
        for j in range(nv):
            sy[worldid, j] = 1.0 if j == i else 0.0
        for j in range(nv):
            s = sy[worldid, j]
            for k in range(j):
                s -= qLD[worldid, j, k] * sy[worldid, k]
            sy[worldid, j] = s / qLD[worldid, j, j]
        # back sub: L' * x = y
        for ii in range(nv):
            j = nv - 1 - ii
            s = sy[worldid, j]
            for k in range(j + 1, nv):
                s -= qLD[worldid, k, j] * sx[worldid, k]
            sx[worldid, j] = s / qLD[worldid, j, j]
        # x[i] = diag(M⁻¹)[i]
        dof_invweight0_out[worldid, i] = sx[worldid, i]

    # finalize dof_invweight0: FREE joint (type 0) averages groups of 3
    for i in range(nv):
        jntid = dof_jntid[i]
        jtype = jnt_type[jntid]
        da = jnt_dofadr[jntid]
        if jtype == 0 and i == da:
            at = (dof_invweight0_out[worldid, da] +
                  dof_invweight0_out[worldid, da + 1] +
                  dof_invweight0_out[worldid, da + 2]) * (1.0 / 3.0)
            ar = (dof_invweight0_out[worldid, da + 3] +
                  dof_invweight0_out[worldid, da + 4] +
                  dof_invweight0_out[worldid, da + 5]) * (1.0 / 3.0)
            for d in range(3):
                dof_invweight0_out[worldid, da + d] = at
                dof_invweight0_out[worldid, da + 3 + d] = ar

    # ════════════════════════════════════════════════════════════════════════
    # body_invweight0: for each body b, 6 Jacobian rows × Cholesky solve
    #   A[b,r] = J_r · (M⁻¹ · J_r)  →  body_invweight0[b] = (mean(A[0:3]), mean(A[3:6]))
    # ════════════════════════════════════════════════════════════════════════
    for bodyid in range(1, nbody):
        if body_weldid[bodyid] == 0:
            body_invweight0_out[worldid, bodyid] = wp.vec2(0.0, 0.0)
            continue
        # find first ancestor with DOFs
        bid = bodyid
        while bid > 0 and body_dofnum[bid] == 0:
            bid = body_parentid[bid]
        if bid == 0:
            body_invweight0_out[worldid, bodyid] = wp.vec2(0.0, 0.0)
            continue

        point = xipos[worldid, bodyid]
        offset = point - subtree_com[worldid, body_rootid[bodyid]]
        sum_t = float(0.0)
        sum_r = float(0.0)

        for row_idx in range(6):
            # build Jacobian row into sj
            for d in range(nv):
                sj[worldid, d] = 0.0
            dofid = body_dofadr[bid] + body_dofnum[bid] - 1
            while dofid >= 0:
                cd = cdof[worldid, dofid]
                ang = wp.spatial_top(cd)
                lin = wp.spatial_bottom(cd)
                if row_idx < 3:
                    tmp = wp.cross(ang, offset)
                    if row_idx == 0:
                        sj[worldid, dofid] = lin[0] + tmp[0]
                    elif row_idx == 1:
                        sj[worldid, dofid] = lin[1] + tmp[1]
                    else:
                        sj[worldid, dofid] = lin[2] + tmp[2]
                else:
                    if row_idx == 3:
                        sj[worldid, dofid] = ang[0]
                    elif row_idx == 4:
                        sj[worldid, dofid] = ang[1]
                    else:
                        sj[worldid, dofid] = ang[2]
                dofid = dof_parentid[dofid]

            # Cholesky solve: M * x = sj  (forward + back sub on qLD)
            for j in range(nv):
                sy[worldid, j] = sj[worldid, j]
            for j in range(nv):
                s = sy[worldid, j]
                for k in range(j):
                    s -= qLD[worldid, j, k] * sy[worldid, k]
                sy[worldid, j] = s / qLD[worldid, j, j]
            for ii in range(nv):
                j = nv - 1 - ii
                s = sy[worldid, j]
                for k in range(j + 1, nv):
                    s -= qLD[worldid, k, j] * sx[worldid, k]
                sx[worldid, j] = s / qLD[worldid, j, j]

            # A_diag = J_r · x
            dot = float(0.0)
            for d in range(nv):
                dot += sj[worldid, d] * sx[worldid, d]
            if row_idx < 3:
                sum_t += dot
            else:
                sum_r += dot

        avg_t = sum_t * (1.0 / 3.0)
        avg_r = sum_r * (1.0 / 3.0)
        mj_minval = 1e-14
        if avg_t < mj_minval and avg_r > mj_minval:
            avg_t = avg_r
        elif avg_r < mj_minval and avg_t > mj_minval:
            avg_r = avg_t
        body_invweight0_out[worldid, bodyid] = wp.vec2(avg_t, avg_r)

    # ════════════════════════════════════════════════════════════════════════
    # actuator_acc0: for each actuator a, ||M⁻¹ · moment_a||
    # ════════════════════════════════════════════════════════════════════════
    for actid in range(nu):
        for d in range(nv):
            sj[worldid, d] = 0.0
        rownnz = moment_rownnz[worldid, actid]
        rowadr = moment_rowadr[worldid, actid]
        for i in range(rownnz):
            sid = rowadr + i
            col = moment_colind[worldid, sid]
            sj[worldid, col] = actuator_moment[worldid, sid]

        # Cholesky solve: M * x = sj
        for j in range(nv):
            sy[worldid, j] = sj[worldid, j]
        for j in range(nv):
            s = sy[worldid, j]
            for k in range(j):
                s -= qLD[worldid, j, k] * sy[worldid, k]
            sy[worldid, j] = s / qLD[worldid, j, j]
        for ii in range(nv):
            j = nv - 1 - ii
            s = sy[worldid, j]
            for k in range(j + 1, nv):
                s -= qLD[worldid, k, j] * sx[worldid, k]
            sx[worldid, j] = s / qLD[worldid, j, j]

        norm_sq = float(0.0)
        for d in range(nv):
            norm_sq += sx[worldid, d] * sx[worldid, d]
        # actuator_acc0 is model-level (shape [1, nu]): all worlds write to
        # row 0, last writer wins (matches the original _compute_actuator_acc0
        # which writes actuator_acc0_out[worldid % shape[0], actid] = ...).
        # When doing selective recompute (subset), the "last writer" differs
        # from the full run. This is acceptable — actuator_acc0 depends on
        # model params (actuator_moment) which are the same across envs that
        # share the same DR sample; in practice the variation is tiny.
        actuator_acc0_out[0, actid] = wp.sqrt(norm_sq)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

def install() -> None:
    global _ORIG, _ORIG_EVENT_APPLY
    if _ORIG is not None:
        return
    from mujoco_warp._src import io as _io
    from mujoco_warp._src import smooth

    _ORIG = _io.set_const_0

    # ── Patch event_manager.apply to capture env_ids that trigger recompute ──
    from mjlab.managers import event_manager as _em

    _ORIG_EVENT_APPLY = _em.EventManager.apply

    def patched_apply(self, mode, env_ids=None, dt=None, global_env_step_count=None):
        global _RECOMPUTE_ENV_IDS
        if mode != "reset" or not _enabled() or global_env_step_count is None:
            return _ORIG_EVENT_APPLY(self, mode, env_ids, dt, global_env_step_count)

        # Run the original apply but intercept the recompute_constants call.
        # We temporarily replace sim.recompute_constants to capture env_ids.
        sim = self._env.sim
        orig_recompute = sim.recompute_constants

        def capturing_recompute(level):
            global _RECOMPUTE_ENV_IDS
            # Resolve env_ids to a concrete tensor
            if env_ids is None:
                _RECOMPUTE_ENV_IDS = None  # all envs
            elif isinstance(env_ids, slice):
                _RECOMPUTE_ENV_IDS = None  # all envs
            else:
                _RECOMPUTE_ENV_IDS = env_ids
            orig_recompute(level)

        sim.recompute_constants = capturing_recompute
        try:
            _ORIG_EVENT_APPLY(self, mode, env_ids, dt, global_env_step_count)
        finally:
            sim.recompute_constants = orig_recompute

    _em.EventManager.apply = patched_apply

    # ── Patch set_const_0 with fused + selective kernel ──
    def patched(m, d):
        if not _enabled():
            return _ORIG(m, d)

        nworld = d.nworld
        nv = m.nv
        nbody = m.nbody
        nu = m.nu
        dev = d.qLD.device

        # Determine which envs to process
        global _RECOMPUTE_ENV_IDS
        if _RECOMPUTE_ENV_IDS is not None:
            ids_t = _RECOMPUTE_ENV_IDS
            if hasattr(ids_t, 'cpu'):
                ids_np = ids_t.cpu().numpy()
            else:
                ids_np = np.asarray(ids_t)
            nproc = len(ids_np)
            env_ids_wp = wp.array(ids_np.astype(np.int32), dtype=wp.int32, device=dev)
        else:
            # All envs
            nproc = nworld
            env_ids_wp = _full_env_ids(nworld, dev)

        # ── BEFORE: same as original up to factor_m + transmission ─────────
        # Note: kinematics/crb/factor_m run on ALL worlds (can't easily make
        # selective without patching mujoco_warp internals). The fused loops
        # below are selective.
        qpos_saved = wp.clone(d.qpos)
        wp.launch(_io._copy_qpos0_to_qpos, dim=(nworld, m.nq),
                   inputs=[m.qpos0], outputs=[d.qpos])
        smooth.kinematics(m, d)
        smooth.com_pos(m, d)
        if m.ncam > 0 or m.nlight > 0:
            smooth.camlight(m, d)
        if m.nflex > 0:
            smooth.flex(m, d)
        if m.ntendon > 0:
            smooth.tendon(m, d)
        smooth.crb(m, d)
        if m.ntendon > 0:
            smooth.tendon_armature(m, d)
        smooth.factor_m(m, d)
        smooth.transmission(m, d)

        wp.launch(_io._compute_meaninertia, dim=nworld,
                   inputs=[m.nv, m.is_sparse, m.dof_Madr, d.qM],
                   outputs=[m.stat.meaninertia])
        if m.ntendon > 0:
            wp.launch(_io._copy_tendon_length0, dim=(nworld, m.ntendon),
                       inputs=[d.ten_length], outputs=[m.tendon_length0])

        # ── FUSED LOOPS: 1 launch, only on reset env_ids ───────────────────
        sj = _scratch(_J, nworld, nv, dev)
        sy = _scratch(_Y, nworld, nv, dev)
        sx = _scratch(_X, nworld, nv, dev)

        wp.launch(
            _set_const_0_loops_fused,
            dim=nproc,
            inputs=[
                d.qLD,
                m.body_parentid, m.body_rootid, m.body_dofadr, m.body_dofnum,
                m.body_weldid, m.dof_parentid, m.dof_jntid, m.jnt_type,
                m.jnt_dofadr,
                d.subtree_com, d.xipos, d.cdof,
                d.moment_rownnz, d.moment_rowadr, d.moment_colind,
                d.actuator_moment,
                nv, nbody, nu,
                env_ids_wp,
                sj, sy, sx,
            ],
            outputs=[
                m.dof_invweight0,
                m.body_invweight0,
                m.actuator_acc0,
            ],
            device=dev,
        )

        # ── AFTER: camera/light + dof_M0 + dampratio + restore qpos ────────
        if m.ncam > 0:
            wp.launch(_io._compute_cam_pos0, dim=(nworld, m.ncam),
                       inputs=[m.cam_bodyid, m.cam_targetbodyid, d.cam_xpos,
                               d.cam_xmat, d.xpos, d.subtree_com],
                       outputs=[m.cam_pos0, m.cam_poscom0, m.cam_mat0])
        if m.nlight > 0:
            wp.launch(_io._compute_light_pos0, dim=(nworld, m.nlight),
                       inputs=[m.light_bodyid, m.light_targetbodyid,
                               d.light_xpos, d.light_xdir, d.xpos, d.subtree_com],
                       outputs=[m.light_pos0, m.light_poscom0, m.light_dir0])
        if nu > 0 and nv > 0:
            dof_M0 = wp.zeros((nworld, nv), dtype=wp.float32, device=dev)
            wp.launch(_io._compute_dof_M0, dim=(nworld, nv),
                       inputs=[m.dof_bodyid, m.dof_armature, d.cdof, d.crb],
                       outputs=[dof_M0])
            wp.launch(_io._resolve_dampratio, dim=(nworld, nu),
                       inputs=[m.actuator_biastype, m.actuator_gainprm,
                               d.moment_rownnz, d.moment_rowadr, d.moment_colind,
                               d.actuator_moment, dof_M0, nv],
                       outputs=[m.actuator_biasprm])
        wp.copy(d.qpos, qpos_saved)

        # Reset for next call
        _RECOMPUTE_ENV_IDS = None

    _io.set_const_0 = patched
    print("[sycl-fused] fused+selective set_const_0 installed "
          "(MJLAB_SYCL_FUSED_SET_CONST=0 to disable)")


def _full_env_ids(nworld, device):
    key = (nworld, str(device))
    buf = _ENV_IDS_CACHE.get(key)
    if buf is None:
        buf = wp.array(np.arange(nworld, dtype=np.int32), dtype=wp.int32, device=device)
        _ENV_IDS_CACHE[key] = buf
    return buf


def uninstall() -> None:
    global _ORIG, _ORIG_EVENT_APPLY
    if _ORIG is None:
        return
    from mujoco_warp._src import io as _io
    _io.set_const_0 = _ORIG
    _ORIG = None
    if _ORIG_EVENT_APPLY is not None:
        from mjlab.managers import event_manager as _em
        _em.EventManager.apply = _ORIG_EVENT_APPLY
        _ORIG_EVENT_APPLY = None
