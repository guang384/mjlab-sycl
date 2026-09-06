# mjlab-sycl

Intel GPU (SYCL) training support for [mjlab](https://github.com/mujocolab/mjlab):
a runtime patch that routes mjlab/mujoco_warp physics onto a warp SYCL backend
([warp-sycl](https://github.com/<owner>/warp), branch `sycl`), barrier-free
flat rewrites of mujoco_warp's hottest tiled kernels, and a train entry that
bypasses mjlab's CUDA-only `select_gpus`.

See README-SYCL-TRAINING.md for setup, usage, footguns and performance data.
