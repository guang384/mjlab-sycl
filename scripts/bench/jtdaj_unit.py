import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, NV_PAD, NJMAX_PAD = 6, 20, 176
rng = np.random.default_rng(11)
qM = rng.standard_normal((NW, NV_PAD, NV_PAD)).astype(np.float32)
J = rng.standard_normal((NW, NJMAX_PAD, NV_PAD)).astype(np.float32) * 0.1
D = rng.random((NW, NJMAX_PAD)).astype(np.float32)
STATE = rng.integers(0, 4, (NW, NJMAX_PAD)).astype(np.int32)  # mixes non-QUADRATIC
NEFC = np.array([5, 100, 20, 176, 46, 1], dtype=np.int32)
DONE = np.array([False, False, True, False, False, True])

wqM = wp.array(qM, dtype=float, device="sycl")
wJ = wp.array(J, dtype=float, device="sycl")
wD = wp.array(D, dtype=float, device="sycl")
wS = wp.array(STATE, dtype=int, device="sycl")
wN = wp.array(NEFC, dtype=int, device="sycl")
wDn = wp.array(DONE, dtype=bool, device="sycl")

h1 = wp.zeros((NW, NV_PAD, NV_PAD), dtype=float, device="sycl")
h2 = wp.zeros((NW, NV_PAD, NV_PAD), dtype=float, device="sycl")

from mjlab_sycl.flat_kernels import _get_kernel
wp.launch(_get_kernel(NV_PAD), dim=(NW, NV_PAD * NV_PAD),
          inputs=[wN, wqM, wJ, wD, wS, wDn], outputs=[h1], device="sycl")

from mjlab_sycl import native_kernels
rc = native_kernels.jtdaj(wqM, wJ, wD, wS, wN, wDn, h2, NV_PAD, NJMAX_PAD)
wp.synchronize_device("sycl")
print("native rc:", rc)
a, b = h1.numpy(), h2.numpy()
done_mask = DONE
print("h equal (active worlds):", np.array_equal(a[~done_mask], b[~done_mask]),
      " max diff:", np.abs(a[~done_mask] - b[~done_mask]).max())
