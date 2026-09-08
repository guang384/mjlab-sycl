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
**~5.5k env-steps/s (~19 s/iteration)** — about **14x the same stack on its
warp-cpu device**. Full measured archive (per-step budget, kernel ranking,
every optimization attempt and its verdict, hardware comparisons):
[`docs/performance.md`](docs/performance.md).

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

## Limitations

- Windows + Intel GPU only (battle-tested on an Arc 130T iGPU); no Linux.
- This is a *training path*, not full warp parity (no official warp test suite).
- Version-locked to warp 1.12.0 / mjlab 1.3.0 / mujoco-warp 3.8.1.
- The vendored backend derives from [NVIDIA/warp 1.12.0](https://github.com/NVIDIA/warp)
  (Apache-2.0; modified files carry MODIFIED notices; licenses ship in
  `src/mjlab_sycl/backend/`).

## Feedback

Bugs, benchmarks from other Intel GPUs and feature ideas: [Issues](https://github.com/guang384/mjlab-sycl/issues)
/ [Discussions](https://github.com/guang384/mjlab-sycl/discussions). See
[CHANGELOG.md](CHANGELOG.md) for history.
