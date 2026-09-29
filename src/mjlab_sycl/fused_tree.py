# SPDX-License-Identifier: Apache-2.0
"""Fuse mujoco_warp's per-level tree-chain kernel launches into one launch
per chain.

mujoco_warp computes tree reductions (subtree CoM, composite rigid body,
cfrc backward, linear/angular momentum) with a *level-synchronous* sweep:
``for body_tree in reversed(m.body_tree): wp.launch(kernel, ...)`` — one
launch per kinematic-tree depth level.  For microduck that is 7 launches
per chain invocation, and the five chains run 30 times per env step:
210 launches/step where each launch carries ~0.10 ms of pure submission +
tiny-kernel execution overhead while doing almost no work (the deepest
levels have 1-3 bodies).

The fused replacement runs ONE work-item per world which walks all bodies
in deepest->shallowest order (exactly the order the level loop used) and
does the same read-modify-write per body.  Because a single work-item owns
the whole world, plain ``+=`` replaces ``wp.atomic_add`` — same values,
deterministic, no atomic traffic.  Kernel-boundary semantics are preserved:
every consumer of these buffers runs after the fused kernel completes.

Order data (all bodies concatenated deepest-first + cumulative level
boundaries) is precomputed once per model and cached on the module —
``m.body_tree`` is fixed at model build time.

Semantics notes (verified against the originals):
  - ``_subtree_com_acc`` skips body 0 (parent is itself).
  - ``_crb_accumulate`` skips bodies whose parent is the world body (pid==0),
    which also covers body 0.
  - ``_cfrc_backward`` skips body 0.
  - ``_linear_momentum`` accumulates into the parent with the PRE-division
    value, then divides the body's own slot (division runs for body 0 too).
  - ``_angular_momentum`` adds the body's own momentum term, pushes the
    updated own-value plus the parent-frame term into the parent slot.

Kill switch: ``MJLAB_SYCL_FUSED_TREE=0``.
"""

from __future__ import annotations

import os
import weakref

import numpy as np
import warp as wp

from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import vec10

# Weak-keyed: entries vanish with the model, so a recycled id can never
# surface a stale body order.  (Same id-recycling hazard class the launch
# cache guards against with weakref liveness checks.)
_ORDER_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_ORDER_FALLBACK: dict = {}  # id(model) -> (model_ref, entry) if not weakref-able


def _enabled() -> bool:
  return os.environ.get("MJLAB_SYCL_FUSED_TREE", "1").strip().lower() not in (
    "0",
    "false",
    "off",
  )


def _tree_order(m):
  """Concatenated deepest->shallowest body ids + cumulative level ends."""
  try:
    cached = _ORDER_CACHE.get(m)
    if cached is not None:
      return cached
  except TypeError:
    # unhashable/unweakref-able model: id-keyed fallback.  The entry holds
    # a STRONG model reference, so the id cannot be recycled while cached.
    entry = _ORDER_FALLBACK.get(id(m))
    if entry is not None and entry[0] is m:
      return entry[1]
  levels = list(reversed(m.body_tree))  # deepest first, matching the loops
  device = m.body_parentid.device
  flat = []
  ends = []
  total = 0
  for lvl in levels:
    ids = lvl.numpy().astype(np.int32)
    flat.append(ids)
    total += len(ids)
    ends.append(total)
  order = wp.array(np.concatenate(flat), dtype=wp.int32, device=device)
  level_ends = wp.array(np.array(ends, dtype=np.int32), dtype=wp.int32, device=device)
  entry = (order, level_ends, len(levels), device)
  try:
    _ORDER_CACHE[m] = entry
  except TypeError:
    _ORDER_FALLBACK[id(m)] = (m, entry)
  return entry


# ---------------------------------------------------------------------------
# Fused kernels: one work-item per world, sequential deepest->shallowest walk
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _tree_subtree_com_acc(
  body_parentid: wp.array[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  # Order data:
  body_order: wp.array[int],
  level_ends: wp.array[int],
  nlevels: int,
  # Data out:
  subtree_com_out: wp.array2d[wp.vec3],
):
  worldid = wp.tid()
  prev = int(0)
  for lvl in range(nlevels):
    end = level_ends[lvl]
    for k in range(prev, end):
      b = body_order[k]
      if b != 0:
        p = body_parentid[b]
        subtree_com_out[worldid, p] += subtree_com_in[worldid, b]
    prev = end


@wp.kernel(enable_backward=False)
def _tree_crb_accumulate(
  body_parentid: wp.array[int],
  # Data in:
  crb_in: wp.array2d[vec10],
  # Order data:
  body_order: wp.array[int],
  level_ends: wp.array[int],
  nlevels: int,
  # Data out:
  crb_out: wp.array2d[vec10],
):
  worldid = wp.tid()
  prev = int(0)
  for lvl in range(nlevels):
    end = level_ends[lvl]
    for k in range(prev, end):
      b = body_order[k]
      p = body_parentid[b]
      if p == 0:
        continue
      crb_out[worldid, p] += crb_in[worldid, b]
    prev = end


@wp.kernel(enable_backward=False)
def _tree_cfrc_backward(
  body_parentid: wp.array[int],
  # Data in:
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # Order data:
  body_order: wp.array[int],
  level_ends: wp.array[int],
  nlevels: int,
  # Data out:
  cfrc_int_out: wp.array2d[wp.spatial_vector],
):
  worldid = wp.tid()
  prev = int(0)
  for lvl in range(nlevels):
    end = level_ends[lvl]
    for k in range(prev, end):
      b = body_order[k]
      if b != 0:
        p = body_parentid[b]
        cfrc_int_out[worldid, p] += cfrc_int_in[worldid, b]
    prev = end


@wp.kernel(enable_backward=False)
def _tree_linear_momentum(
  body_parentid: wp.array[int],
  body_subtreemass: wp.array2d[float],
  # Data in:
  subtree_linvel_in: wp.array2d[wp.vec3],
  # Order data:
  body_order: wp.array[int],
  level_ends: wp.array[int],
  nlevels: int,
  # Data out:
  subtree_linvel_out: wp.array2d[wp.vec3],
):
  worldid = wp.tid()
  prev = int(0)
  for lvl in range(nlevels):
    end = level_ends[lvl]
    for k in range(prev, end):
      b = body_order[k]
      if b != 0:
        p = body_parentid[b]
        subtree_linvel_out[worldid, p] += subtree_linvel_in[worldid, b]
      # divide the body's own (fully accumulated) value; runs for body 0 too
      subtree_linvel_out[worldid, b] /= wp.max(
        MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], b]
      )
    prev = end


@wp.kernel(enable_backward=False)
def _tree_angular_momentum(
  body_parentid: wp.array[int],
  body_mass: wp.array2d[float],
  body_subtreemass: wp.array2d[float],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  subtree_linvel_in: wp.array2d[wp.vec3],
  subtree_bodyvel_in: wp.array2d[wp.spatial_vector],
  # Order data:
  body_order: wp.array[int],
  level_ends: wp.array[int],
  nlevels: int,
  # Data out:
  subtree_angmom_out: wp.array2d[wp.vec3],
):
  worldid = wp.tid()
  prev = int(0)
  for lvl in range(nlevels):
    end = level_ends[lvl]
    for k in range(prev, end):
      b = body_order[k]
      if b == 0:
        continue
      p = body_parentid[b]

      xipos = xipos_in[worldid, b]
      com = subtree_com_in[worldid, b]
      com_parent = subtree_com_in[worldid, p]
      vel = subtree_bodyvel_in[worldid, b]
      linvel = subtree_linvel_in[worldid, b]
      linvel_parent = subtree_linvel_in[worldid, p]
      mass = body_mass[worldid % body_mass.shape[0], b]
      subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], b]

      # momentum wrt body b
      dx = xipos - com
      dv = wp.spatial_bottom(vel) - linvel
      dp = dv * mass
      dL = wp.cross(dx, dp)

      # add to own subtree
      subtree_angmom_out[worldid, b] += dL

      # push own (updated) value to parent
      subtree_angmom_out[worldid, p] += subtree_angmom_out[worldid, b]

      # momentum wrt parent
      dx = com - com_parent
      dv = linvel - linvel_parent
      dv *= subtreemass
      dL = wp.cross(dx, dv)
      subtree_angmom_out[worldid, p] += dL
    prev = end


# ---------------------------------------------------------------------------
# Patched module functions (replace the per-level loops with one launch)
# ---------------------------------------------------------------------------

_ORIG = {}


def install() -> None:
  if _ORIG:
    return
  if not _enabled():
    return
  from mujoco_warp._src import smooth

  _ORIG["com_pos"] = smooth.com_pos
  _ORIG["crb"] = smooth.crb
  _ORIG["rne_cfrc_backward"] = smooth._rne_cfrc_backward
  _ORIG["subtree_vel"] = smooth.subtree_vel

  def com_pos(m, d):
    if not _enabled():
      return _ORIG["com_pos"](m, d)
    order, ends, nlevels, _dev = _tree_order(m)
    # _subtree_com_init: subtree_com = xipos * mass
    wp.launch(
      smooth._subtree_com_init,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_mass, d.xipos],
      outputs=[d.subtree_com],
    )
    # fused accumulation (replaces len(body_tree) launches)
    wp.launch(
      _tree_subtree_com_acc,
      dim=d.nworld,
      inputs=[m.body_parentid, d.subtree_com, order, ends, nlevels],
      outputs=[d.subtree_com],
    )
    wp.launch(
      smooth._subtree_div,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_subtreemass, d.subtree_com],
      outputs=[d.subtree_com],
    )
    wp.launch(
      smooth._cinert,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
      outputs=[d.cinert],
    )
    wp.launch(
      smooth._cdof,
      dim=(d.nworld, m.njnt),
      inputs=[m.body_rootid, m.jnt_type, m.jnt_dofadr, m.jnt_bodyid, d.xmat, d.xanchor, d.xaxis, d.subtree_com],
      outputs=[d.cdof],
    )

  def crb(m, d):
    if not _enabled():
      return _ORIG["crb"](m, d)
    order, ends, nlevels, _dev = _tree_order(m)
    wp.copy(d.crb, d.cinert)
    wp.launch(
      _tree_crb_accumulate,
      dim=d.nworld,
      inputs=[m.body_parentid, d.crb, order, ends, nlevels],
      outputs=[d.crb],
    )
    d.qM.zero_()
    if m.is_sparse:
      wp.launch(
        smooth._qM_sparse,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_bodyid, m.dof_parentid, m.dof_Madr, m.dof_armature, d.cdof, d.crb],
        outputs=[d.qM],
      )
    else:
      wp.launch(
        smooth._qM_dense,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, d.cdof, d.crb],
        outputs=[d.qM],
      )

  def _rne_cfrc_backward(m, d):
    if not _enabled():
      return _ORIG["rne_cfrc_backward"](m, d)
    order, ends, nlevels, _dev = _tree_order(m)
    wp.launch(
      _tree_cfrc_backward,
      dim=d.nworld,
      inputs=[m.body_parentid, d.cfrc_int, order, ends, nlevels],
      outputs=[d.cfrc_int],
    )

  def subtree_vel(m, d):
    if not _enabled():
      return _ORIG["subtree_vel"](m, d)
    order, ends, nlevels, _dev = _tree_order(m)
    subtree_bodyvel = wp.empty((d.nworld, m.nbody), dtype=wp.spatial_vector)
    wp.launch(
      smooth._subtree_vel_forward,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com, d.cvel],
      outputs=[d.subtree_linvel, d.subtree_angmom, subtree_bodyvel],
    )
    # fused linear momentum (replaces len(body_tree) launches)
    wp.launch(
      _tree_linear_momentum,
      dim=d.nworld,
      inputs=[m.body_parentid, m.body_subtreemass, d.subtree_linvel, order, ends, nlevels],
      outputs=[d.subtree_linvel],
    )
    # fused angular momentum (replaces len(body_tree) launches)
    wp.launch(
      _tree_angular_momentum,
      dim=d.nworld,
      inputs=[
        m.body_parentid,
        m.body_mass,
        m.body_subtreemass,
        d.xipos,
        d.subtree_com,
        d.subtree_linvel,
        subtree_bodyvel,
        order,
        ends,
        nlevels,
      ],
      outputs=[d.subtree_angmom],
    )

  smooth.com_pos = com_pos
  smooth.crb = crb
  smooth._rne_cfrc_backward = _rne_cfrc_backward
  smooth.subtree_vel = subtree_vel
  print(
    "[sycl-fused-tree] tree chains fused: subtree_com/crb/cfrc_backward/"
    "linvel+angmom (MJLAB_SYCL_FUSED_TREE=0 to disable)"
  )


def uninstall() -> None:
  if not _ORIG:
    return
  from mujoco_warp._src import smooth

  smooth.com_pos = _ORIG["com_pos"]
  smooth.crb = _ORIG["crb"]
  smooth._rne_cfrc_backward = _ORIG["rne_cfrc_backward"]
  smooth.subtree_vel = _ORIG["subtree_vel"]
  _ORIG.clear()
