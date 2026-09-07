# Vendored warp SYCL backend — provenance & rebuild docs

The warp 1.12.0 SYCL backend this package ships is vendored from our fork of
NVIDIA/warp ([guang384/warp](https://github.com/guang384/warp), branch `sycl`)
— the development home for the backend. The files themselves live in
`src/mjlab_sycl/backend/`, the single copy: it is what goes into the wheel and
what `python -m mjlab_sycl install` overlays onto the environment's warp
package. This directory holds only the documentation.

- `README.md` — this file: what is vendored, from where, and the modification
  inventory.
- `REBUILD.md` — how to rebuild `warpsycl.dll` (oneAPI DPC++ / icx).

## What's vendored (src/mjlab_sycl/backend/)

- `_src/` — the 5 patched Python modules (`build.py`, `codegen.py`,
  `context.py`, `types.py`, `builtins.py`), copied over
  `<venv>/Lib/site-packages/warp/_src/` by the installer.
- `native/` — the 2 patched upstream C++ headers (`builtin.h`, `tile.h`) plus
  the 2 new SYCL runtime sources (`sycl_runtime.h`, `sycl_runtime.cpp`) that
  kernels are compiled against.
- `warpsycl.dll` — the prebuilt micro-driver (SYCL queue + USM pool +
  watchdog). Placed into warp's kernel cache at install time.
- `LICENSE.md` + `third_party_licenses/` — warp 1.12.0's Apache-2.0 license
  and its bundled third-party notices. The 7 files derived from upstream each
  carry a MODIFIED notice (Apache-2.0 §4(b)); the 2 SYCL runtime sources are
  original to this project.

## When to rebuild

Only when changing the backend itself (new kernel support, watchdog tuning, a
warp upstream upgrade). Then: rebuild `warpsycl.dll` per `REBUILD.md`, re-run the
verification gates (`mjlab-sycl-test`), and commit the refreshed
sources together.

## Upstream

Based on NVIDIA/warp 1.12.0 (Apache-2.0 — see `src/mjlab_sycl/backend/LICENSE.md`).
The fork carries 7 modified files and 2 new ones, no deletions; the CUDA and
CPU backends are untouched, so the overlay is strictly additive (a machine
without an Intel GPU is unaffected).