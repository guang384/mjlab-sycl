# mjlab-sycl

English | [简体中文](README.zh-CN.md)

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
  strictly additive — CUDA/CPU paths untouched) applied by `mjlab-sycl-install`
- a **runtime patch** routing mjlab/mujoco_warp physics onto the `sycl` device
  (torch stays on CPU/XPU, queue drained at every sim boundary)
- **barrier-free flat kernels + native SYCL kernels** replacing mujoco_warp's
  hottest kernels (the tiled originals are extremely slow on an iGPU)
- **train/bench/viewer/check entries** that bypass mjlab's CUDA-only GPU
  selection, plus **verification gates** and a one-command environment preflight

## Quickstart (into your mjlab project venv)

```powershell
# 1. install this package into the venv (editable keeps your clone live;
#    clear PIP_CONFIG_FILE/PIP_TARGET first if pip is redirected globally,
#    or just use --isolated as below)
<project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps -e <path-to-mjlab-sycl>

# 2. overlay the warp SYCL backend (add --warmup <TASK> to pre-compile kernels)
<project>\.venv\Scripts\mjlab-sycl-install.exe --warmup Mjlab-Velocity-Flat-MicroDuck

# 3. environment preflight (read-only, 8 checks, each with its fix)
<project>\.venv\Scripts\mjlab-sycl-check.exe

# torch must be the +xpu wheel -- a plain `pip install torch` installs the
# CPU build and PPO silently runs ~3x slower. Only if the check FAILs it:
<project>\.venv\Scripts\python.exe -m pip install "torch==2.9.1+xpu" --index-url https://download.pytorch.org/whl/xpu
```

Then train any registered task:

```powershell
# smoke test first (64 envs, 5 iterations)
<project>\.venv\Scripts\mjlab-sycl-train.exe Mjlab-Velocity-Flat-MicroDuck --num-envs 64 --max-iterations 5

# full training (4096 envs)
<project>\.venv\Scripts\mjlab-sycl-train.exe Mjlab-Velocity-Flat-MicroDuck --num-envs 4096 --max-iterations 1000
```

One-command setup into a fresh project: `scripts/setup_microduck.ps1 -Repo <project>`.

**First run (one-time):** kernel modules are JIT-compiled on first use
(measured 3.5 min on this box, cached afterwards) — the entries say so up
front, and step 2's `--warmup` moves that cost into the install. Full path
from a fresh clone to the first training step: ≈10 minutes, mostly `uv sync`
plus the one-time JIT. The command family is
`mjlab-sycl-{install,check,train,bench,test}`.

Requirements, usage, environment variables, verification gates, footguns
and known limitations: **[README-SYCL-TRAINING.md](README-SYCL-TRAINING.md)**
(the single source for those facts).

## Measured performance

Intel Arc 130T (Lunar Lake iGPU), microduck velocity task, 4096 envs:
**~22k env-steps/s (~5.5 s/iteration) on a quiet desktop** end-to-end, vs
~258 on the same stack's warp-cpu device. All measurements, their error
bars and every optimization's verdict live in
[`docs/performance.md`](docs/performance.md) — the single source for
numbers. A background GPU app costs 10–45 % of throughput; the train/bench
entries warn at startup.

## Viewer tooling

| tool | what it shows |
|---|---|
| `python -m mjlab_sycl.train_viewer` | real BAM-sim training with a native MuJoCo window (env 0 mirrored; slow-motion ~8 updates/s — physics, not rendering, is the cap) |
| `python -m mjlab_sycl.kview` | K ducks side by side from K live training envs |
| `python -m mjlab_sycl.cpu_replay` | smooth ~50 Hz approximate playback of a checkpoint on CPU MuJoCo; can watch a running training dir and hot-swap the newest checkpoint |
| `python -m mjlab_sycl.play` | real-sim checkpoint playback with a viewer |

## Relationship to NVIDIA/warp

`mjlab-sycl` is **not a fork of NVIDIA/warp** and never replaces it. Training
runs against the ordinary `warp-lang==1.12.0` package, with a strictly
additive **overlay** applied on top of it by `mjlab-sycl-install`:

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

## Feedback

Bugs, benchmarks from other Intel GPUs and feature ideas: [Issues](https://github.com/guang384/mjlab-sycl/issues)
/ [Discussions](https://github.com/guang384/mjlab-sycl/discussions). See
[CHANGELOG.md](CHANGELOG.md) for history and [RELEASING.md](RELEASING.md)
for the maintainer release checklist.
