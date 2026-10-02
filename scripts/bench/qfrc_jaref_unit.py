import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, NV, NJMAX = 6, 20, 176
rng = np.random.default_rng(31)
J = rng.standard_normal((NW, NJMAX, NV)).astype(np.float32) * 0.1
FORCE = rng.standard_normal((NW, NJMAX)).astype(np.float32)
NEFC = np.array([5, 100, 20, 176, 46, 1], dtype=np.int32)
DONE = np.array([False, False, True, False, False, True])

Jw = wp.array(J, dtype=float, device="sycl")
Fw = wp.array(FORCE, dtype=float, device="sycl")
Nw = wp.array(NEFC, dtype=int, device="sycl")
Dw = wp.array(DONE, dtype=bool, device="sycl")

# ---- qfrc_constraint: warp reference vs native ----
from mujoco_warp._src.solver import update_constraint_init_qfrc_constraint_dense as KQ
o1 = wp.zeros((NW, NV), dtype=float, device="sycl")
o2 = wp.zeros((NW, NV), dtype=float, device="sycl")
wp.launch(KQ, dim=(NW, NV), inputs=[Nw, Jw, Fw, NJMAX, Dw], outputs=[o1], device="sycl")
from mjlab_sycl import native_kernels
rc = native_kernels.qfrc_constraint(Jw, Fw, Nw, Dw, o2, NV, NV, NJMAX)
wp.synchronize_device("sycl")
print("qfrc rc:", rc, " equal:", np.array_equal(o1.numpy(), o2.numpy(), equal_nan=True),
      " max diff:", np.abs(o1.numpy() - o2.numpy()).max())

# ---- jaref zeroahead: warp fused reference vs native ----
from mjlab_sycl import fused_solver as _fs
from mujoco_warp._src import types as _t, solver as _s
_fs._mw_types = _t                      # install() would populate these
_fs._mw_rescale = _s._rescale
from mjlab_sycl.fused_solver import _jaref_zeroahead
JV = rng.standard_normal((NW, NJMAX)).astype(np.float32)
ALPHA = rng.random(NW).astype(np.float32)
COST = rng.random(NW).astype(np.float32)
JVw = wp.array(JV, dtype=float, device="sycl")
Aw = wp.array(ALPHA, dtype=float, device="sycl")
Cw = wp.array(COST, dtype=float, device="sycl")

def run_jaref(native):
    Jaref0 = rng.random((NW, NJMAX)).astype(np.float32)  # fixed below
    return Jaref0

Jaref_init = rng.random((NW, NJMAX)).astype(np.float32)
outs = {}
for label, native in (("warp", False), ("natv", True)):
    Ja = wp.array(Jaref_init.copy(), dtype=float, device="sycl")
    gauss = wp.zeros(NW, dtype=float, device="sycl")
    cost = wp.array(COST.copy(), dtype=float, device="sycl")
    prev = wp.zeros(NW, dtype=float, device="sycl")
    gdot = wp.zeros(NW, dtype=float, device="sycl")
    sdot = wp.zeros(NW, dtype=float, device="sycl")
    chg = wp.ones(NW, dtype=int, device="sycl")
    if native:
        ok = native_kernels.jaref(JVw, Aw, Nw, Dw, Cw, Ja, gauss, cost, prev,
                                  gdot, sdot, chg)
    else:
        wp.launch(_jaref_zeroahead, dim=(NW, NJMAX),
                  inputs=[Nw, JVw, Aw, Dw, Cw], outputs=[Ja],
                  # zero-ahead outputs appended like the interceptor does
                  outputs2=None, device="sycl") if False else wp.launch(
            _jaref_zeroahead, dim=(NW, NJMAX),
            inputs=[Nw, JVw, Aw, Dw, Cw],
            outputs=[Ja, gauss, cost, prev, gdot, sdot, chg], device="sycl")
        ok = "warp"
    wp.synchronize_device("sycl")
    outs[label] = (Ja.numpy().copy(), gauss.numpy().copy(), cost.numpy().copy(),
                   prev.numpy().copy(), gdot.numpy().copy(), sdot.numpy().copy(),
                   chg.numpy().copy(), ok)

w, n = outs["warp"], outs["natv"]
names = ["Jaref", "gauss", "cost", "prev_cost", "grad_dot", "search_dot", "changed"]
bad = 0
for i, nm in enumerate(names):
    eq = np.array_equal(w[i], n[i], equal_nan=True)
    if not eq:
        bad += 1
        print(f"  MISMATCH {nm}: max diff {np.abs(w[i].astype(float) - n[i].astype(float)).max()}")
print("jaref rc:", n[7], " all outputs equal:", bad == 0)
