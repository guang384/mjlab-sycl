import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
wp.init()

NW, N, PAD, NJMAX_PAD = 5, 20, 20, 176
rng = np.random.default_rng(23)

# ---- chol_fs: single-tile factorize+solve ----
R = rng.standard_normal((NW, N, N)).astype(np.float32)
M = R @ R.transpose(0, 2, 1) + N * np.eye(N, dtype=np.float32)
Y = rng.standard_normal((NW, N)).astype(np.float32)
Mw = wp.array(M, dtype=float, device="sycl")
Yw = wp.array(Y, dtype=float, device="sycl")
X1 = wp.zeros((NW, N), dtype=float, device="sycl"); L1 = wp.zeros((NW, N, N), dtype=float, device="sycl")
X2 = wp.zeros((NW, N), dtype=float, device="sycl"); L2 = wp.zeros((NW, N, N), dtype=float, device="sycl")
from mjlab_sycl.flat_kernels import _get_chol_fs_kernel
ADR = wp.array(np.array([0], dtype=np.int32), dtype=int, device="sycl")
wp.launch(_get_chol_fs_kernel(N), dim=(NW, 1), inputs=[Mw, Yw, ADR], outputs=[X1, L1], device="sycl")
from mjlab_sycl import native_kernels
rc = native_kernels.chol_fs(Mw, Yw, X2, L2, N, PAD)
wp.synchronize_device("sycl")
print("chol_fs rc:", rc, " X equal:", np.array_equal(X1.numpy(), X2.numpy(), equal_nan=True),
      " L equal:", np.array_equal(L1.numpy(), L2.numpy(), equal_nan=True))

# ---- hinc: incremental Hessian ----
J = rng.standard_normal((NW, NJMAX_PAD, N)).astype(np.float32) * 0.1
D = rng.random((NW, NJMAX_PAD)).astype(np.float32)
STATE = rng.integers(0, 4, (NW, NJMAX_PAD)).astype(np.int32)
NC = np.array([0, 3, 176, 1, 8], dtype=np.int32)
CID = rng.integers(0, NJMAX_PAD, (NW, NJMAX_PAD)).astype(np.int32)
H0 = rng.standard_normal((NW, N, N)).astype(np.float32)

Jw = wp.array(J, dtype=float, device="sycl"); Dw = wp.array(D, dtype=float, device="sycl")
Sw = wp.array(STATE, dtype=int, device="sycl"); Cw = wp.array(CID, dtype=int, device="sycl")
Nw = wp.array(NC, dtype=int, device="sycl")
H1 = wp.array(H0.copy(), dtype=float, device="sycl"); H2 = wp.array(H0.copy(), dtype=float, device="sycl")

# warp reference: replicate via mujoco_warp kernel
from mujoco_warp._src.solver import update_gradient_h_incremental
wp.launch(update_gradient_h_incremental, dim=(NW, N * (N + 1) // 2),
          inputs=[Jw, Dw, Sw, Cw, Nw], outputs=[H1], device="sycl")
rc = native_kernels.hinc(Jw, Dw, Sw, Cw, Nw, H2, N, NJMAX_PAD, NJMAX_PAD)
wp.synchronize_device("sycl")
print("hinc rc:", rc, " equal:", np.array_equal(H1.numpy(), H2.numpy(), equal_nan=True),
      " max diff:", np.abs(H1.numpy() - H2.numpy()).max())
