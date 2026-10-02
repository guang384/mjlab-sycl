"""Unit: gauss_cost + ls_teardown natives vs their warp kernels (bit-exact)."""
import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, NV, LSIT = 8, 20, 10
rng = np.random.default_rng(91)


def arr(x, dt=float):
    return wp.array(x, dtype=dt, device="sycl")


QACC = rng.standard_normal((NW, NV)).astype(np.float32)
QFRC = rng.standard_normal((NW, NV)).astype(np.float32)
QSMT = rng.standard_normal((NW, NV)).astype(np.float32)
MA = rng.standard_normal((NW, NV)).astype(np.float32)
DONE = rng.random(NW) < 0.25
GAUSS0 = rng.random(NW).astype(np.float32)
COST0 = rng.random(NW).astype(np.float32)

wq = arr(QACC); wf = arr(QFRC); w0 = arr(QSMT); wm = arr(MA)
wd = arr(DONE, bool)

# ---- gauss_cost: warp factory (single-writer variant) vs native ----
from mujoco_warp._src.solver import update_constraint_gauss_cost
K = update_constraint_gauss_cost(NV, 32)  # dofs_per_thread >= nv
g1 = arr(GAUSS0.copy()); c1 = arr(COST0.copy())
g2 = arr(GAUSS0.copy()); c2 = arr(COST0.copy())
wp.launch(K, dim=(NW, 1), inputs=[wq, wf, w0, wm, wd], outputs=[g1, c1],
          device="sycl")
from mjlab_sycl import native_kernels
ok = native_kernels.gauss_cost(wq, wf, w0, wm, wd, g2, c2, NV)
wp.synchronize_device("sycl")
print("gauss rc:", ok,
      " gauss bit-equal:", np.array_equal(g1.numpy(), g2.numpy()),
      " cost bit-equal:", np.array_equal(c1.numpy(), c2.numpy()))

# ---- ls_teardown: warp fused vs native ----
COST = rng.standard_normal((NW, LSIT)).astype(np.float32)
SEARCH = rng.standard_normal((NW, NV)).astype(np.float32)
MV = rng.standard_normal((NW, NV)).astype(np.float32)
A0 = rng.random(NW).astype(np.float32)
Q0 = rng.standard_normal((NW, NV)).astype(np.float32)
M0 = rng.standard_normal((NW, NV)).astype(np.float32)
wc = arr(COST); ws = arr(SEARCH); wmv = arr(MV)

from mjlab_sycl import fused_linesearch as fl
fl._mw_types = fl._mw_types  # module globals already set at import
K2 = fl._ls_teardown_fused
a1 = arr(A0); q1 = arr(Q0); m1 = arr(M0)
a2 = arr(A0); q2 = arr(Q0); m2 = arr(M0)
wp.launch(K2, dim=NW,
          inputs=[LSIT, 0.01, wc, wd, ws, wmv, NV],
          outputs=[a1, q1, m1], device="sycl")
ok2 = native_kernels.ls_teardown(LSIT, 0.01, wc, wd, ws, wmv, NV, a2, q2, m2)
wp.synchronize_device("sycl")
print("lstd rc:", ok2,
      " alpha bit-equal:", np.array_equal(a1.numpy(), a2.numpy()),
      " qacc bit-equal:", np.array_equal(q1.numpy(), q2.numpy()),
      " Ma bit-equal:", np.array_equal(m1.numpy(), m2.numpy()))
