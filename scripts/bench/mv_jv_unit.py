import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, NV, NJMAX, NJMAX_PAD = 4, 20, 168, 176
rng = np.random.default_rng(7)
qM = rng.standard_normal((NW, NV, NV)).astype(np.float32)
J = rng.standard_normal((NW, NJMAX_PAD, NV)).astype(np.float32) * 0.1
S = rng.standard_normal((NW, NV)).astype(np.float32)
NEFC = np.array([5, 100, 20, 168], dtype=np.int32)
DONE = np.array([False, False, True, False])

wqM = wp.array(qM, dtype=float, device="sycl")
wJ = wp.array(J, dtype=float, device="sycl")
wS = wp.array(S, dtype=float, device="sycl")
wN = wp.array(NEFC, dtype=int, device="sycl")
wD = wp.array(DONE, dtype=bool, device="sycl")

mv1 = wp.zeros((NW, NV), dtype=float, device="sycl")
jv1 = wp.zeros((NW, NJMAX), dtype=float, device="sycl")
mv2 = wp.zeros((NW, NV), dtype=float, device="sycl")
jv2 = wp.zeros((NW, NJMAX), dtype=float, device="sycl")

from mjlab_sycl.fused_linesearch import _mv_jv_fused
wp.launch(_mv_jv_fused, dim=NW,
          inputs=[NV, wqM, wS, wN, wJ, NJMAX, wD], outputs=[mv1, jv1],
          device="sycl")

from mjlab_sycl import native_kernels
rc = native_kernels.mv_jv(wqM, wJ, wS, wN, wD, mv2, jv2, NV, NJMAX, NV, NJMAX_PAD)
wp.synchronize_device("sycl")
print("native rc:", rc)

m1, j1 = mv1.numpy(), jv1.numpy()
m2, j2 = mv2.numpy(), jv2.numpy()
print("mv equal:", np.array_equal(m1, m2), " max diff:", np.abs(m1 - m2).max())
mask_done = DONE
print("jv equal (active worlds):", np.array_equal(j1[~mask_done], j2[~mask_done]),
      " max diff:", np.abs(j1[~mask_done] - j2[~mask_done]).max())
if not np.array_equal(m1, m2):
    d = np.abs(m1 - m2)
    w, r = np.unravel_index(np.argmax(d), d.shape)
    print("worst mv at world", w, "row", r, " warp:", m1[w, r], " native:", m2[w, r])
