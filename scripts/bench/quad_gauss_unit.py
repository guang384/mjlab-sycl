"""Unit: _quad_gauss_fused (warp) vs wp_sycl_quad_gauss (native).

Covers the pyramidal path + the per-world gauss fold -- the elliptic-cone
branch is ported verbatim but not synthesized here (neither microduck nor
the gate model uses elliptic cones).
"""
import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, NJMAX, NV = 6, 176, 20
rng = np.random.default_rng(37)

NEFC = np.array([5, 100, 20, 176, 46, 1], dtype=np.int32)
DONE = np.array([False, False, True, False, False, True])
# efc types: mix incl. true ELLIPTIC=7 rows (exercises the cone branch)
TYPE = rng.integers(0, 8, (NW, NJMAX)).astype(np.int32)
ID = rng.integers(0, 16, (NW, NJMAX)).astype(np.int32)
D = rng.random((NW, NJMAX)).astype(np.float32) + 0.5
JAREF = rng.standard_normal((NW, NJMAX)).astype(np.float32)
JV = rng.standard_normal((NW, NJMAX)).astype(np.float32)
ADR = rng.integers(0, NJMAX, (16, 4)).astype(np.int32)
FRIC = rng.random((16, 5)).astype(np.float32)
CDIM = rng.integers(1, 4, (16,)).astype(np.int32)
NACON = np.array([8], dtype=np.int32)
IMPR = rng.random(1).astype(np.float32)
QSM = rng.standard_normal((NW, NV)).astype(np.float32)
MA = rng.standard_normal((NW, NV)).astype(np.float32)
SEARCH = rng.standard_normal((NW, NV)).astype(np.float32)
GAUSS = rng.random(NW).astype(np.float32)
MV = rng.standard_normal((NW, NV)).astype(np.float32)

from mujoco_warp._src import types as _mw_types

def arr(a, dtype=float):
    return wp.array(a, dtype=dtype, device="sycl")

w = {
    "impr": arr(IMPR), "nefc": arr(NEFC, int), "fric": wp.array(FRIC, dtype=_mw_types.vec5, device="sycl"), "cdim": arr(CDIM, int),
    "adr": arr(ADR, int), "typ": arr(TYPE, int), "id": arr(ID, int), "D": arr(D),
    "nacon": arr(NACON, int), "Jaref": arr(JAREF), "jv": arr(JV), "done": arr(DONE, bool),
    "qsm": arr(QSM), "ma": arr(MA), "search": arr(SEARCH), "gauss": arr(GAUSS),
    "mv": arr(MV),
}

from mjlab_sycl.fused_linesearch import _quad_gauss_fused

def run(native):
    quad = wp.zeros((NW, NJMAX), dtype=wp.vec3, device="sycl")
    qg = wp.zeros((NW,), dtype=wp.vec3, device="sycl")
    if native:
        from mjlab_sycl import native_kernels
        ok = native_kernels.quad_gauss(
            w["impr"], w["nefc"], w["fric"], w["cdim"], w["adr"], w["typ"],
            w["id"], w["D"], w["nacon"], w["Jaref"], w["jv"], w["done"], NV,
            w["qsm"], w["ma"], w["search"], w["gauss"], w["mv"], quad, qg)
    else:
        wp.launch(_quad_gauss_fused, dim=(NW, NJMAX),
                  inputs=[w["impr"], w["nefc"], w["fric"], w["cdim"], w["adr"],
                          w["typ"], w["id"], w["D"], w["nacon"], w["Jaref"],
                          w["jv"], w["done"], NV, w["qsm"], w["ma"],
                          w["search"], w["gauss"], w["mv"]],
                  outputs=[quad, qg], device="sycl")
        ok = "warp"
    wp.synchronize_device("sycl")
    return ok, quad.numpy(), qg.numpy()

ok1, q1, g1 = run(False)
ok2, q2, g2 = run(True)
print("rc:", ok1, ok2)
# done worlds leave outputs untouched in both paths; compare active worlds
act = ~DONE
eq_q = np.array_equal(q1[act], q2[act], equal_nan=True)
eq_g = np.array_equal(g1[act], g2[act], equal_nan=True)
print("quad equal (active):", eq_q, " max diff:", np.abs(q1[act] - q2[act]).max())
print("gauss equal (active):", eq_g, " max diff:", np.abs(g1[act] - g2[act]).max())
