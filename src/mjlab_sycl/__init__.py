"""mjlab-sycl: Intel GPU (SYCL) training stack for mjlab/mujoco_warp.

Layers bundled in this one package (the goal is training, not full warp
parity):
  - backend/: a warp 1.12.0 SYCL backend (7 patched files + 2 new SYCL
    runtime sources + prebuilt warpsycl.dll), applied to the environment's
    warp package by `python -m mjlab_sycl install`
  - runtime_patch: routes mjlab/mujoco_warp onto the sycl device, drains at
    sim boundaries, skips viewer-only sites
  - flat kernels: barrier-free rewrites of mujoco_warp's hot tiled kernels
  - train/bench/train_viewer: entries that bypass mjlab's CUDA-only
    select_gpus; doctor: read-only environment preflight (mjlab-sycl-check);
    test_overlay (host-only) + test_e2e/test_mujoco: verification gates
    (mjlab-sycl-test)

The bundled entries (train, bench, train_viewer, etc.) call
``prepare_sycl_runtime_path()`` and ``patch_simulation_for_sycl()`` themselves;
users should call those functions explicitly rather than relying on import
side effects.  This module re-exports ``patch_simulation_for_sycl`` for
convenience.
"""

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

__all__ = ["patch_simulation_for_sycl"]
