# SPDX-License-Identifier: Apache-2.0
"""Barrier-free (flat) SYCL replacements for mujoco_warp's hottest tiled kernels.

See the module docstring sections below for each kernel. Installed together
with the SYCL simulation patch (sycl_patch.py); MJLAB_SYCL_FLAT_JTDAJ=0
disables (historical name -- it kills all four rewrites).
"""

from __future__ import annotations

import os


import warp as wp

_TILE = 16  # must divide nv_pad (mujoco_warp pads nv to TILE_SIZE_JTDAJ_DENSE)
_QUADRATIC = 1  # mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC

_KERNEL_CACHE = {}
_orig_launch_tiled = None


def _make_flat_kernel(nv_pad: int):
  nt = (nv_pad + _TILE - 1) // _TILE  # sub-tiles per axis; edge tiles are clipped

  @wp.kernel(enable_backward=False)
  def kernel(
    nefc_in: wp.array[int],
    qM_in: wp.array3d[float],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    ctx_done_in: wp.array[bool],
    ctx_h_out: wp.array3d[float],
  ):
    worldid, tile_id = wp.tid()

    if ctx_done_in[worldid]:
      return

    # (ti, tj) selects this work-item's (clipped) 16x16 sub-tile of the output
    # matrix; the full (nt x nt) grid partitions h exactly, so no element is
    # written by two work-items and no synchronization is needed. nv_pad is
    # only padded to a multiple of 4 by mujoco_warp (dense path), so edge
    # sub-tiles may be smaller than 16.
    ti = tile_id // nt
    tj = tile_id - ti * nt
    r0 = ti * _TILE
    c0 = tj * _TILE
    rows = wp.min(_TILE, nv_pad - r0)
    cols = wp.min(_TILE, nv_pad - c0)

    for i in range(rows):
      for j in range(cols):
        h_out_val = qM_in[worldid, r0 + i, c0 + j]
        ctx_h_out[worldid, r0 + i, c0 + j] = h_out_val

    n = nefc_in[worldid]
    for k in range(n):
      Dk = efc_D_in[worldid, k]
      if efc_state_in[worldid, k] != wp.static(_QUADRATIC):
        Dk = 0.0
      if Dk == 0.0:
        continue

      for i in range(rows):
        Jik = efc_J_in[worldid, k, r0 + i] * Dk
        for j in range(cols):
          h_out_val = ctx_h_out[worldid, r0 + i, c0 + j] + Jik * efc_J_in[worldid, k, c0 + j]
          ctx_h_out[worldid, r0 + i, c0 + j] = h_out_val

  return kernel


def _get_kernel(nv_pad: int):
  k = _KERNEL_CACHE.get(nv_pad)
  if k is None:
    k = _make_flat_kernel(nv_pad)
    _KERNEL_CACHE[nv_pad] = k
  return k


def nworld_from(kwargs):
  dim = kwargs["dim"]
  return dim[0] if isinstance(dim, (tuple, list)) else dim


def _enabled() -> bool:
  # install() is only ever called from the sycl patch path, so reaching here
  # means the sim runs on the sycl device; the env var is the kill switch.
  return os.environ.get("MJLAB_SYCL_FLAT_JTDAJ", "1").strip().lower() not in (
    "0", "false", "off"
  )



_ADR_SIZE_CACHE = {}  # id(adr array) -> tile size (TileSet: uniform across tiles)


def _adr_tile_size(adr, unpadded_dim):
  """TileSet tiles are uniform-size; size = gap between the first two
  addresses. A single-tile set spans the UNPADDED matrix dimension (callers
  must pass nv -- the qM/qLD buffers are padded and their garbage rows would
  poison a Cholesky factorization sized to the padded end; that was the
  cartpole NaN: qM padded 4x4, tile actually 2x2)."""
  key = adr.ptr
  size = _ADR_SIZE_CACHE.get(key)
  if size is None:
    a = adr.numpy()
    size = int(a[1] - a[0]) if len(a) > 1 else int(unpadded_dim)
    _ADR_SIZE_CACHE[key] = size
  return size


def install() -> None:
  """Wrap wp.launch_tiled to re-route the flat-rewritten kernels on sycl."""
  global _orig_launch_tiled
  if _orig_launch_tiled is not None:
    return

  # pre-instantiate the tiled contact-jac kernels for both cones so the
  # interception can tell them apart by object identity (their keys are equal;
  # only the closure differs). Falls back to the tiled original (safe) if
  # mujoco runs with a non-default tile size and the id misses.
  try:
    from mujoco_warp._src import constraint as _constraint
    from mujoco_warp._src import types as _mw_types
    for cone in (_mw_types.ConeType.PYRAMIDAL, _mw_types.ConeType.ELLIPTIC):
      k = _constraint._efc_contact_jac_dense(32, cone)
      _CONTACT_CONE_IDS[id(k)] = cone == _mw_types.ConeType.ELLIPTIC
  except Exception as e:
    print(f"[sycl-flat] contact-jac cone probe failed ({e!r}); JTDAJ only")

  def patched_launch_tiled(*args, **kwargs):
    kernel = args[0] if args else kwargs.get("kernel")
    key = getattr(kernel, "key", "")
    if key == "_tile_cholesky_factorize__locals__cholesky_factorize" and _enabled():
      adr = kwargs["inputs"][1]
      # L_out is unpadded: single-tile size = its row dim
      n = _adr_tile_size(adr, kwargs["outputs"][0].shape[1])
      nworld = nworld_from(kwargs)
      return wp.launch(
        _cholesky_factorize_tiles_flat,
        dim=(nworld, adr.shape[0]),
        inputs=kwargs["inputs"] + [n],
        outputs=kwargs["outputs"],
        device=kwargs.get("device"),
      )
    if key == "_tile_cholesky_solve__locals__cholesky_solve" and _enabled():
      adr = kwargs["inputs"][2]
      # L (inputs[0]) is unpadded: its row dim IS the single-tile size
      n = _adr_tile_size(adr, kwargs["inputs"][0].shape[1])
      nworld = nworld_from(kwargs)
      return wp.launch(
        _cholesky_solve_tiles_flat,
        dim=(nworld, adr.shape[0]),
        inputs=kwargs["inputs"] + [n],
        outputs=kwargs["outputs"],
        device=kwargs.get("device"),
      )
    if key.startswith("_efc_contact_jac_dense") and _enabled():
      flat = _contact_flat_for(kernel)
      if flat is not None:
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        nworld = kwargs["dim"][0]
        njmax_pad = outputs[0].shape[1]
        return wp.launch(
          flat,
          dim=(nworld, njmax_pad),
          inputs=inputs,
          outputs=outputs,
          device=kwargs.get("device"),
        )
    if key == "update_gradient_cholesky__locals__kernel" and _enabled():
      inputs = kwargs["inputs"]
      outputs = kwargs["outputs"]
      h = inputs[1]
      n = h.shape[1]
      scratch = wp.empty_like(h)
      return wp.launch(
        _cholesky_solve_flat,
        dim=nworld_from(kwargs),
        inputs=[inputs[0], h, inputs[2], n],
        outputs=[scratch, outputs[0]],
        device=kwargs.get("device"),
      )
    if key == "_tile_cholesky_factorize_solve__locals__cholesky_factorize_solve" and _enabled():
      adr = kwargs["inputs"][2]
      n = _adr_tile_size(adr, kwargs["outputs"][0].shape[1])
      return wp.launch(
        _cholesky_factorize_solve_flat,
        dim=nworld_from(kwargs),
        inputs=kwargs["inputs"] + [n],
        outputs=kwargs["outputs"],
        device=kwargs.get("device"),
      )
    if key.startswith("update_gradient_JTDAJ_dense_tiled") and _enabled():
      inputs = kwargs["inputs"]
      outputs = kwargs["outputs"]
      qM = inputs[1]
      nv_pad = qM.shape[1]
      nworld = kwargs["dim"]
      if isinstance(nworld, (tuple, list)):
        nworld = nworld[0]
      nt = (nv_pad + _TILE - 1) // _TILE
      return wp.launch(
        _get_kernel(nv_pad),
        dim=(nworld, nt * nt),
        inputs=inputs,
        outputs=outputs,
        device=kwargs.get("device"),
      )
    return _orig_launch_tiled(*args, **kwargs)

  _orig_launch_tiled = wp.launch_tiled
  wp.launch_tiled = patched_launch_tiled
  print("[sycl-flat] flat replacements installed: JTDAJ + contact_jac (MJLAB_SYCL_FLAT_JTDAJ=0 to disable)")


# ---------------------------------------------------------------------------
# _efc_contact_jac_dense: per-(world, efcid) constraint Jacobian rows.
# The tiled original loads 32-dof cdof tiles and reduces Jqvel per block with
# an atomic; every (world, efcid) pair is independent, so the flat version
# recomputes the per-contact data per work-item (cheap, L2-resident) and
# writes its J row and Jqvel directly -- no reduction, no atomics.
# ---------------------------------------------------------------------------

def _make_contact_jac_flat(is_elliptic: bool):
  from mujoco_warp._src import support as _support

  @wp.kernel(enable_backward=False)
  def kernel(
    body_rootid: wp.array[int],
    geom_bodyid: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nl_in: wp.array[int],
    nefc_in: wp.array[int],
    qvel_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    contact_efc_address_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    njmax_in: int,
    nv_pad: int,
    condim_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    flex_in: wp.array[wp.vec2i],
    vert_in: wp.array[wp.vec2i],
    pos_in: wp.array[wp.vec3],
    frame_in: wp.array2d[wp.vec3],
    friction_in: wp.array2d[float],
    efc_J_out: wp.array3d[float],
    efc_Jqvel_out: wp.array2d[float],
  ):
    worldid, efcid = wp.tid()

    efc_start = ne_in[worldid] + nf_in[worldid] + nl_in[worldid]
    efc_end = wp.min(nefc_in[worldid], njmax_in)
    if efcid < efc_start or efcid >= efc_end:
      return

    conid = efc_id_in[worldid, efcid]
    condim = condim_in[conid]

    geom = geom_in[conid]
    if geom[0] >= 0:
      body1 = geom_bodyid[geom[0]]
    else:
      flex = flex_in[conid]
      vert = vert_in[conid]
      body1 = flex_vertbodyid[flex_vertadr[flex[0]] + vert[0]]
    if geom[1] >= 0:
      body2 = geom_bodyid[geom[1]]
    else:
      flex = flex_in[conid]
      vert = vert_in[conid]
      body2 = flex_vertbodyid[flex_vertadr[flex[1]] + vert[1]]

    con_pos = pos_in[conid]
    offset1 = con_pos - subtree_com_in[worldid, body_rootid[body1]]
    offset2 = con_pos - subtree_com_in[worldid, body_rootid[body2]]

    dimid = efcid - contact_efc_address_in[conid, 0]

    Jqvel = float(0.0)
    for d in range(nv_pad):
      cdof_clip = cdof_in[worldid, d]
      jpd = _support._compute_jacp(cdof_clip, offset2, body_isdofancestor[body2, d])             - _support._compute_jacp(cdof_clip, offset1, body_isdofancestor[body1, d])
      jrd = _support._compute_jacr(cdof_clip, body_isdofancestor[body2, d])             - _support._compute_jacr(cdof_clip, body_isdofancestor[body1, d])

      if wp.static(is_elliptic):
        frame_idx = dimid if dimid < 3 else dimid - 3
        frame_row = frame_in[conid, frame_idx]
        if dimid < 3:
          J0 = wp.dot(jpd, frame_row)
        else:
          J0 = wp.dot(jrd, frame_row)
      else:
        J0 = wp.dot(jpd, frame_in[conid, 0])
        if condim > 1:
          dimid2 = dimid / 2 + 1
          frii = friction_in[conid, dimid2 - 1]
          frii_sign = frii * (1.0 - 2.0 * float(dimid & 1))
          if dimid2 == 1:
            J0 = J0 + wp.dot(jpd, frame_in[conid, 1]) * frii_sign
          elif dimid2 == 2:
            J0 = J0 + wp.dot(jpd, frame_in[conid, 2]) * frii_sign
          elif dimid2 == 3:
            J0 = J0 + wp.dot(jrd, frame_in[conid, 0]) * frii_sign
          elif dimid2 == 4:
            J0 = J0 + wp.dot(jrd, frame_in[conid, 1]) * frii_sign
          else:
            J0 = J0 + wp.dot(jrd, frame_in[conid, 2]) * frii_sign

      efc_J_out[worldid, efcid, d] = J0
      Jqvel = Jqvel + J0 * qvel_in[worldid, d]

    efc_Jqvel_out[worldid, efcid] = Jqvel

  return kernel


@wp.kernel(enable_backward=False)
def _cholesky_solve_tiles_flat(
  L_in: wp.array3d[float],
  y_in: wp.array2d[float],
  adr_in: wp.array[int],
  n: int,
  x_out: wp.array2d[float],
):
  """Dense backsubstitution x = inv(L'L) y over the qM tile set.

  One work-item per (world, tile); n (the tile size) is computed on the host
  from the TileSet's address gaps. Replaces smooth._tile_cholesky_solve,
  which mujoco_warp's set-const path launches once per tile per world --
  260 launches per recompute at 4096 envs, ~93% of recompute time.
  """
  worldid, nodeid = wp.tid()
  off = adr_in[nodeid]

  # forward substitution in-place on the output (y = L^-1 b)
  for i in range(n):
    s = y_in[worldid, off + i]
    for k in range(i):
      s = s - L_in[worldid, off + i, off + k] * x_out[worldid, off + k]
    x_out[worldid, off + i] = s / L_in[worldid, off + i, off + i]

  # back substitution (x = L^-T y)
  for i in range(n):
    ii = n - 1 - i
    s = x_out[worldid, off + ii]
    for k in range(n - 1 - ii):
      kk = n - 1 - k
      s = s - L_in[worldid, off + kk, off + ii] * x_out[worldid, off + kk]
    x_out[worldid, off + ii] = s / L_in[worldid, off + ii, off + ii]


@wp.kernel(enable_backward=False)
def _cholesky_factorize_tiles_flat(
  qM_in: wp.array3d[float],
  adr_in: wp.array[int],
  tile_n: int,
  L_out: wp.array3d[float],
):
  """Dense LLT factorization over the qM tile set, one work-item per tile.

  Replaces smooth._tile_cholesky_factorize (the set-const path's per-tile
  LLT; the last non-flat tiled kernel in the reset path). Writes the lower
  triangle; the upper triangle mirrors what warp's tile_cholesky leaves in
  the destination (upper rows beyond the factor are zero-filled by the
  caller's zero-init of L, so we zero them explicitly to stay bit-identical
  with the tiled original's store of a full tile).

  NOTE on exactness: warp's tile_cholesky accumulation order per element is
  the standard k-ascending dot product, same as ours, so values match the
  tiled original to float rounding.
  """
  worldid, nodeid = wp.tid()
  off = adr_in[nodeid]
  n = tile_n

  for i in range(n):
    for j in range(i + 1):
      s = qM_in[worldid, off + i, off + j]
      for k in range(j):
        s = s - L_out[worldid, off + i, off + k] * L_out[worldid, off + j, off + k]
      if i == j:
        L_out[worldid, off + i, off + j] = wp.sqrt(s)
      else:
        L_out[worldid, off + i, off + j] = s / L_out[worldid, off + j, off + j]
    # zero the strictly-upper part of row i within the tile
    for j in range(i + 1, n):
      L_out[worldid, off + i, off + j] = 0.0


_CONTACT_CONE_IDS = {}  # id(kernel object) -> is_elliptic
_CONTACT_FLAT = {}  # is_elliptic -> flat kernel


def _contact_flat_for(kernel_obj):
  is_elliptic = _CONTACT_CONE_IDS.get(id(kernel_obj))
  if is_elliptic is None:
    return None
  k = _CONTACT_FLAT.get(is_elliptic)
  if k is None:
    k = _make_contact_jac_flat(is_elliptic)
    _CONTACT_FLAT[is_elliptic] = k
  return k


# ---------------------------------------------------------------------------
# Cholesky pair: update_gradient_cholesky (solver, nv<=32) and
# _tile_cholesky_factorize_solve (smooth, per qM tile). For nv=20 both are
# tiny dense factorizations whose cost is pure tile machinery; the flat
# versions run one work-item per (world[, tile]) with scalar row-by-row
# LLT + in-place forward/back substitution on the output buffer.
# ---------------------------------------------------------------------------

@wp.kernel(enable_backward=False)
def _cholesky_solve_flat(
  ctx_grad_in: wp.array2d[float],
  h_in: wp.array3d[float],
  ctx_done_in: wp.array[bool],
  n: int,
  L_out: wp.array3d[float],
  ctx_Mgrad_out: wp.array2d[float],
):
  worldid = wp.tid()
  if ctx_done_in[worldid]:
    return

  # row-by-row LLT into the scratch buffer
  for i in range(n):
    for j in range(i + 1):
      s = h_in[worldid, i, j]
      for k in range(j):
        s = s - L_out[worldid, i, k] * L_out[worldid, j, k]
      if i == j:
        L_out[worldid, i, j] = wp.sqrt(s)
      else:
        L_out[worldid, i, j] = s / L_out[worldid, j, j]

  # forward substitution in-place on the output (y = L^-1 g)
  for i in range(n):
    s = ctx_grad_in[worldid, i]
    for k in range(i):
      s = s - L_out[worldid, i, k] * ctx_Mgrad_out[worldid, k]
    ctx_Mgrad_out[worldid, i] = s / L_out[worldid, i, i]

  # back substitution (x = L^-T y)
  for i in range(n):
    ii = n - 1 - i
    s = ctx_Mgrad_out[worldid, ii]
    for k in range(n - 1 - ii):
      kk = n - 1 - k
      s = s - L_out[worldid, kk, ii] * ctx_Mgrad_out[worldid, kk]
    ctx_Mgrad_out[worldid, ii] = s / L_out[worldid, ii, ii]


@wp.kernel(enable_backward=False)
def _cholesky_factorize_solve_flat(
  M_in: wp.array3d[float],
  y_in: wp.array2d[float],
  adr_in: wp.array[int],
  n: int,
  x_out: wp.array2d[float],
  L_out: wp.array3d[float],
):
  worldid, nodeid = wp.tid()
  off = adr_in[nodeid]

  for i in range(n):
    for j in range(i + 1):
      s = M_in[worldid, off + i, off + j]
      for k in range(j):
        s = s - L_out[worldid, off + i, off + k] * L_out[worldid, off + j, off + k]
      if i == j:
        L_out[worldid, off + i, off + j] = wp.sqrt(s)
      else:
        L_out[worldid, off + i, off + j] = s / L_out[worldid, off + j, off + j]

  for i in range(n):
    s = y_in[worldid, off + i]
    for k in range(i):
      s = s - L_out[worldid, off + i, off + k] * x_out[worldid, off + k]
    x_out[worldid, off + i] = s / L_out[worldid, off + i, off + i]

  for i in range(n):
    ii = n - 1 - i
    s = x_out[worldid, off + ii]
    for k in range(n - 1 - ii):
      kk = n - 1 - k
      s = s - L_out[worldid, off + kk, off + ii] * x_out[worldid, off + kk]
    x_out[worldid, off + ii] = s / L_out[worldid, off + ii, off + ii]

