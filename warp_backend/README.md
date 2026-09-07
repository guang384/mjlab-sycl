# Vendored warp SYCL backend

This directory holds the warp 1.12.0 SYCL backend that
`python -m mjlab_sycl install` overlays onto the environment's warp package.
It is **vendored from our fork of NVIDIA/warp** (branch `sycl`) so that this
package is self-contained; the fork remains the development home for the
backend itself.

## Contents

- `files/_src/` — the 5 patched Python modules (`build.py`, `codegen.py`,
  `context.py`, `types.py`, `builtins.py`), copied over
  `<venv>/Lib/site-packages/warp/_src/` by the installer.
- `files/native/` (shown as `native/`) — the 4 patched C++/header files
  (`builtin.h`, `tile.h`, `sycl_runtime.h`, `sycl_runtime.cpp`) that kernels
  are compiled against.
- `native/warpsycl.dll` — the prebuilt micro-driver (SYCL queue + USM pool +
  watchdog). Placed into warp's kernel cache at install time.

## When to rebuild

Only when changing the backend itself (new kernels support, watchdog tuning,
a warp upstream upgrade). Then: apply the overlay to a warp checkout, rebuild
`warpsycl.dll` per `REBUILD.md`, re-run the e2e + mujoco_warp gates, and
refresh the copies here.

## Upstream

Based on NVIDIA/warp 1.12.0. The fork carries ~9 modified files and no
deletions; the CUDA and CPU backends are untouched, so the overlay is
strictly additive (a machine without an Intel GPU is unaffected).
