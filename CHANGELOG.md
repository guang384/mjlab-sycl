# Changelog

All notable changes to mjlab-sycl.

## [Unreleased]

<!-- Add new changes here as they land; fold into a dated release section when
     tagging. -->

## [0.2.0] - 2026-09-08 — first public release candidate

First release shape for the community: environment preflight, one-command
setup, verified viewer tooling and the measured performance archive.

### Added
- `mjlab-sycl-check` (doctor): read-only environment preflight — platform/Python,
  warp overlay sync, Intel oneAPI runtime, sycl device, a real device kernel vs
  cpu, torch XPU, and the mjlab task registry. `--no-kernel` to skip the compile.
- `scripts/setup_microduck.ps1`: one-command install of this package into any
  mjlab project venv (handles the global-pip-`target=` trap), with
  `-InstallTorchXpu` and `-PipIndex` options.
- Viewer tooling: `play` (checkpoint playback), `kview` (K-duck theater from K
  live envs), `cpu_replay` (smooth approximate playback on CPU MuJoCo, incl.
  watching a running training dir with hot checkpoint swaps), and a fixed
  `train_viewer` that actually opens the native MuJoCo window (mjlab 1.3.0 has
  no human render mode; env 0 is mirrored into a passive viewer).
- `docs/performance.md`: measured baseline archive (with reproduction commands
  and verdicts on every optimization attempted).
- `.github/`: CI (CPU-side unit + build) and a self-hosted GPU-gate workflow.

### Changed
- torch CPU threads default 2 (`MJLAB_TORCH_THREADS`); PPO/xpu defaults in the
  entries; bench defaults PPO to xpu to match training.
- Overlay sync guard enforced in `patch_simulation_for_sycl()`; `install`
  self-checks the overlay it applies.
- probe scripts bootstrap the sycl8.dll PATH ordering like the entries.

### Fixed
- `train_viewer` opened no window on mjlab 1.3.0 (relied on a non-existent
  `render_mode="human"`); now mirrors env 0 into `mujoco.viewer.launch_passive`
  with env-0 tracking camera and a watcher thread (decoupled from physics).
- CPU/viewer models render black/void without injected floor+light, and fallen
  ducks sank through the plane (only feet collide): real mesh shell contact via
  the visual class contype flip; visible checkered floor.
- `kview` duck spacing (free-joint qpos world positions), qpos intra-block
  mapping, and multi-duck model lights.

### Deprecated / removed
- Experimental batched convergence polling kept but default-off (measured
  slower); kernel-arg by-value codegen experiment reverted (measured ~noise).

## [0.1.1] - 2026-09-08

- Overlay-sync guard (`overlay_problems` / `ensure_overlay_synced`) with a
  byte-for-byte check + `mjlab-sycl-test` host-side overlay gate (`test_overlay`).
- `mujoco-warp==3.8.1` pinned (flat_kernels intercepts internals by key string).

## [0.1.0] - initial development snapshot (pre-release)

- Vendored warp 1.12.0 SYCL backend (7 patched files + sycl_runtime + warpsycl.dll),
  runtime patch, barrier-free flat kernels, train/bench entries, verification gates.
