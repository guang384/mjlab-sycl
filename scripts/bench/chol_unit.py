import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, N, PAD = 6, 20, 20
rng = np.random.default_rng(13)
R = rng.standard_normal((NW, N, N)).astype(np.float32)
H = R @ R.transpose(0, 2, 1) + N * np.eye(N, dtype=np.float32)  # SPD
GRAD = rng.standard_normal((NW, N)).astype(np.float32)
DONE = np.array([False, True, False, False, False, True])
CHANGED = np.array([1, 1, 0, 0, 1, 0], dtype=np.int32)
LVALID = np.array([True, False, True, False, True, False])  # w2: skip path

def run(native):
    h = wp.array(H, dtype=float, device="sycl")
    g = wp.array(GRAD, dtype=float, device="sycl")
    d = wp.array(DONE, dtype=bool, device="sycl")
    c = wp.array(CHANGED, dtype=int, device="sycl")
    lv = wp.array(LVALID, dtype=bool, device="sycl")
    L = wp.zeros((NW, PAD, PAD), dtype=float, device="sycl")
    lv_out = wp.zeros(NW, dtype=bool, device="sycl")
    mg = wp.zeros((NW, PAD), dtype=float, device="sycl")
    if native:
        from mjlab_sycl import native_kernels
        rc = native_kernels.chol_solve(h, g, d, c, lv, L, lv_out, mg, N, PAD)
    else:
        from mjlab_sycl.flat_kernels import _get_chol_solve_kernel
        wp.launch(_get_chol_solve_kernel(N), dim=NW,
                  inputs=[g, h, d, c, lv], outputs=[L, lv_out, mg], device="sycl")
        rc = "warp"
    wp.synchronize_device("sycl")
    return rc, L.numpy(), lv_out.numpy(), mg.numpy()

rc1, L1, lv1, m1 = run(False)
rc2, L2, lv2, m2 = run(True)
print("rc:", rc1, rc2)
print("L equal:", np.array_equal(L1, L2, equal_nan=True),
      " lvalid equal:", np.array_equal(lv1, lv2),
      " Mgrad equal:", np.array_equal(m1, m2, equal_nan=True))
if not np.array_equal(m1, m2, equal_nan=True):
    print("max diff:", np.nanmax(np.abs(np.nan_to_num(m1 - m2))))
    bad = np.argwhere(~((m1 == m2) | (np.isnan(m1) & np.isnan(m2))))
    print("first mismatch:", bad[:3])
