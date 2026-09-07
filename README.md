# mjlab-sycl

Intel GPU (SYCL) training support for [mjlab](https://github.com/mujocolab/mjlab):
a runtime patch that routes mjlab/mujoco_warp physics onto a vendored warp
SYCL backend (7 files patched on top of
[NVIDIA/warp 1.12.0](https://github.com/NVIDIA/warp); development history
archived in warp_backend/history.bundle), barrier-free flat rewrites of mujoco_warp's hottest tiled
kernels, and a train entry that bypasses mjlab's CUDA-only `select_gpus`.

See README-SYCL-TRAINING.md for setup, usage, footguns and performance data.
