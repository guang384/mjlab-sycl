"""warp-sycl: Intel GPU (SYCL) training stack for mjlab/mujoco_warp.

Layers bundled in this one package (the goal is training, not full warp
parity):
  - backend/: a warp 1.12.0 SYCL backend (9 patched files + prebuilt
    warpsycl.dll), applied to the environment's warp package by
    `python -m mjlab_sycl install`
  - runtime_patch: routes mjlab/mujoco_warp onto the sycl device, drains at
    sim boundaries, skips viewer-only sites
  - flat kernels: barrier-free rewrites of mujoco_warp's hot tiled kernels
  - train/bench: entries that bypass mjlab's CUDA-only select_gpus

Requires MJLAB_SYCL=1 (or the train entry) at runtime; does nothing when
imported.
"""

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

__all__ = ["patch_simulation_for_sycl"]
