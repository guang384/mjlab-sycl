# Vendored warp SYCL backend — provenance & rebuild docs

The warp 1.12.0 SYCL backend this package ships was developed in a local
clone of NVIDIA/warp; the full commit history is preserved in
`history.bundle` in this directory (see "Development history" below).
The files themselves live in `src/mjlab_sycl/backend/`, the single copy:
it is what goes into the wheel and what `python -m mjlab_sycl install`
overlays onto the environment's warp package. This directory holds the
documentation and the history archive.

- `README.md` — this file: what is vendored, from where, and the modification
  inventory.
- `REBUILD.md` — how to rebuild `warpsycl.dll` (oneAPI DPC++ / icx).
- `history.bundle` — the backend's full development history (see below).

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

## Development history (history.bundle)

A thin git bundle carrying the backend's complete development history on top
of the official NVIDIA/warp v1.12.0 release — 15 commits across two branches:

- `sycl` — the delivered arc: toolchain spike, micro-driver, SYCL codegen,
  device integration, tile SLM, the performance passes (async submission,
  8-task work-groups, args ring buffer), the watchdog, and the measured
  end-to-end results (4.7x on microduck RL training).
- `wip-tile-cooperative` — unreleased experiments (~780 lines): cooperative
  tile execution for block>1 (freezes the machine — kept as a recorded dead
  end), chunked `capture_while`, opt-in `/fp:fast`, incremental args packing.

Restore anywhere; the only prerequisite is upstream's v1.12.0 tag:

    git init warp-sycl && cd warp-sycl
    git remote add origin https://github.com/NVIDIA/warp.git
    git fetch --depth 1 origin tag v1.12.0
    git fetch /path/to/history.bundle sycl:sycl wip-tile-cooperative:wip-tile-cooperative

Why keep it: commit-message archaeology (why each change was made),
`git bisect` on backend regressions, and a rebase base for a future warp
upgrade. Restore verified: tip hashes match the original clone bit-for-bit.

## Upstream

Based on NVIDIA/warp 1.12.0 (Apache-2.0 — see `src/mjlab_sycl/backend/LICENSE.md`).
The backend carries 7 modified files and 2 new ones, no deletions; the CUDA and
CPU backends are untouched, so the overlay is strictly additive (a machine
without an Intel GPU is unaffected).