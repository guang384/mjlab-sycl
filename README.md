# mjlab-sycl

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/guang384/mjlab-sycl)](https://github.com/guang384/mjlab-sycl/releases)
[![CI](https://github.com/guang384/mjlab-sycl/actions/workflows/ci.yml/badge.svg)](https://github.com/guang384/mjlab-sycl/actions/workflows/ci.yml)

**Run mjlab (MuJoCo Warp) PPO training on Intel GPUs — no NVIDIA required.**

[mjlab](https://github.com/mujocolab/mjlab) (with its CUDA-only `select_gpus`)
currently dies before iteration 0 on Intel-only machines. `mjlab-sycl` is a
companion package that makes the *existing* mjlab stack train on an Intel
iGPU/Arc, **without touching your project's source**: install it into your
mjlab project's venv, run one overlay command, and every registered task
(e.g. from [microduck_rl](https://github.com/pollen-robotics/microduck_rl))
trains as-is.

What it bundles:

- a **vendored warp 1.12.0 SYCL backend** (patched files + `warpsycl.dll`,
  strictly additive — CUDA/CPU paths untouched) applied by `python -m mjlab_sycl install`
- a **runtime patch** routing mjlab/mujoco_warp physics onto the `sycl` device
  (torch stays on CPU/XPU, queue drained at every sim boundary)
- **barrier-free flat kernels** replacing mujoco_warp's hottest tiled kernels
  (one-work-item-per-world tiled kernels are extremely slow on an iGPU)
- **train/bench/viewer/check entries** that bypass mjlab's CUDA-only GPU
  selection, plus **verification gates** and a one-command environment preflight

## Quickstart (into your mjlab project venv)

```powershell
# 1. install this package into the venv (editable keeps your clone live;
#    on machines with a global pip `target=` redirect, clear PIP_CONFIG_FILE/
#    PIP_TARGET first or use --isolated)
<project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps -e <path-to-mjlab-sycl>

# 2. overlay the warp SYCL backend + self-check
<project>\.venv\Scripts\python.exe -m mjlab_sycl install

# 3. environment preflight (read-only): platform, overlay sync, oneAPI,
#    Intel GPU, a real device kernel vs cpu, torch XPU, task registry
<project>\.venv\Scripts\mjlab-sycl-check.exe
```

Then train any registered task:

```powershell
# smoke test first (64 envs, 5 iterations)
<project>\.venv\Scripts\mjlab-sycl-train.exe Mjlab-Velocity-Flat-MicroDuck --num-envs 64 --max-iterations 5

# full training (4096 envs)
<project>\.venv\Scripts\mjlab-sycl-train.exe Mjlab-Velocity-Flat-MicroDuck --num-envs 4096 --max-iterations 1000
```

One-command setup into a fresh project: `scripts/setup_microduck.ps1 -Repo <project>`.

See **[README-SYCL-TRAINING.md](README-SYCL-TRAINING.md)** for requirements
(Windows + Python 3.12 + Intel GPU + oneAPI), the full install/usage story,
verification gates, environment variables and the hard-won footguns.

## Measured performance

Intel Arc 130T (Lunar Lake iGPU), microduck velocity task, 4096 envs:
**~22k env-steps/s (~5.5 s/iteration) on a quiet desktop** end-to-end
(quiet matters: a background GPU app costs 10–45 % — the bench and train
entries now warn on startup), vs ~258 on the same stack's warp-cpu
device (0.2.0 archive: 5,485). Full
measured archive (per-step budget, kernel ranking, every optimization attempt
and its verdict, hardware comparisons): [`docs/performance.md`](docs/performance.md);
argued kernel-level candidates and the measured bounds that closed them:
[`docs/optimization_ideas.md`](docs/optimization_ideas.md).

## Viewer tooling

| tool | what it shows |
|---|---|
| `python -m mjlab_sycl.train_viewer` | real BAM-sim training with a native MuJoCo window (env 0 mirrored; slow-motion ~8 updates/s — physics, not rendering, is the cap) |
| `python -m mjlab_sycl.kview` | K ducks side by side from K live training envs |
| `python -m mjlab_sycl.cpu_replay` | smooth ~50 Hz approximate playback of a checkpoint on CPU MuJoCo; can watch a running training dir and hot-swap the newest checkpoint |
| `python -m mjlab_sycl.play` | real-sim checkpoint playback with a viewer |

## Verification gates

`mjlab-sycl-test` runs three gates in order: host-only overlay-sync check,
backend e2e (bit-exact vs cpu), and mujoco_warp physics vs cpu (~1e-5). GPU
gates are also available as a self-hosted GitHub Actions workflow
(`.github/workflows/gpu-gates.yml`).

## Repo layout

```
src/mjlab_sycl/       the package: backend/, runtime_patch, flat_kernels,
                      train/bench/viewer/play/kview/cpu_replay, doctor,
                      verification gates
scripts/              setup_microduck.ps1, probes, run_guarded.py watchdog
docs/performance.md   measured performance baseline archive
warp_backend/         provenance + rebuild docs for the vendored backend
```

## Relationship to NVIDIA/warp

`mjlab-sycl` is **not a fork of NVIDIA/warp** and never replaces it. Training
runs against the ordinary `warp-lang==1.12.0` package, with a strictly
additive **overlay** applied on top of it by `python -m mjlab_sycl install`:

- 7 modified files (5 Python modules under `warp/_src/`, 2 headers under
  `warp/native/`) + 2 new files (`sycl_runtime.{h,cpp}`) + a prebuilt
  `warpsycl.dll` micro-driver. File-by-file inventory:
  [`warp_backend/README.md`](warp_backend/README.md).
- `warp.dll` is never rebuilt and the CUDA/CPU code paths are untouched — on a
  machine without an Intel GPU the overlay is inert.
- Version-locked to 1.12.0. Every entry point verifies the overlay is
  byte-identical to `src/mjlab_sycl/backend/` before physics starts, because
  `uv sync` / `pip install` silently reinstall warp and wipe it.

Licensing: [NVIDIA/warp](https://github.com/NVIDIA/warp) is Apache-2.0; the
derived files carry MODIFIED notices (§4(b)) and the licenses ship in
`src/mjlab_sycl/backend/`.

Provenance: the SYCL backend was developed on a `sycl` branch in a local
NVIDIA/warp clone checked out at the `v1.12.0` tag. That clone has been
deleted; its complete history (15 commits, including the recorded dead-end
`wip-tile-cooperative` experiments) is archived in
[`warp_backend/history.bundle`](warp_backend/history.bundle) and restores on
top of the `v1.12.0` tag (recipe in `warp_backend/README.md`). Since that
2026-09-06 snapshot the backend evolves **inside this repo**:
`src/mjlab_sycl/backend/` is the single live copy, and later changes are
ordinary commits in this project's history.

## Limitations

- Windows + Intel GPU only (battle-tested on an Arc 130T iGPU); no Linux.
- This is a *training path*, not full warp parity (no official warp test suite).
- Version-locked to warp 1.12.0 / mjlab 1.3.0 / mujoco-warp 3.8.1.

## Feedback

Bugs, benchmarks from other Intel GPUs and feature ideas: [Issues](https://github.com/guang384/mjlab-sycl/issues)
/ [Discussions](https://github.com/guang384/mjlab-sycl/discussions). See
[CHANGELOG.md](CHANGELOG.md) for history.
