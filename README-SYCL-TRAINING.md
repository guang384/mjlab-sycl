# mjlab-sycl — Intel GPU (SYCL) training for mjlab

Run mjlab PPO training with mujoco_warp physics on an Intel Arc iGPU. The
package installs into an existing mjlab project's venv: the project's code,
config, and lock file stay untouched, and tasks resolve through mjlab's plugin
registry — everything from an installed task package (e.g.
[microduck_rl](https://github.com/pollen-robotics/microduck_rl)) trains as-is.
Developed and battle-tested against microduck_rl's 14-servo biped at 4096 envs.

## How it works

| layer | role |
|---|---|
| `backend/` | Vendored warp 1.12.0 SYCL backend: 7 patched warp files + 2 new SYCL runtime sources + a prebuilt `warpsycl.dll`. `python -m mjlab_sycl install` overlays it onto the environment's warp package — strictly additive, CUDA and CPU paths untouched. |
| `runtime_patch` | Routes mjlab/mujoco_warp physics onto the `sycl` device: warp arrays live in USM shared memory (`wp.to_torch` wraps them zero-copy), torch tensors stay on CPU, and the async kernel queue is drained at every sim call boundary. |
| `flat_kernels` | Barrier-free rewrites of mujoco_warp's hottest tiled kernels (JTDAJ, contact_jacobian, both Choleskys, factorize), intercepted at `wp.launch_tiled`. Warp's tiled kernels are one work-item-per-world — dead slow on an iGPU. |
| `train` / `bench` / `train_viewer` | Entry points that bypass mjlab's CUDA-only `select_gpus` (it indexes torch's empty CUDA list on Intel-only machines and dies before iteration 0). |

No third-party package files are modified on disk except the documented warp
overlay — a fresh `uv sync` remains ground truth (and wipes the overlay; see
footguns).

## Requirements

- Windows, Python 3.12
- Intel Arc iGPU (developed on Arc 130T / Lunar Lake)
- mjlab 1.3.0, mujoco-warp, warp-lang 1.12.0, torch 2.9.1 (pinned in the package)
- Intel oneAPI 2025.x compiler runtime; default location
  `C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3/bin` — the major version
  must match the one `warpsycl.dll` was built with

## Install

Into your mjlab project's venv:

    cd <your mjlab project>          # e.g. microduck_rl
    uv sync                          # project stays untouched
    uv pip install git+https://github.com/guang384/mjlab-sycl
    python -m mjlab_sycl install

`install` overlays the backend onto the venv's warp package, drops
`warpsycl.dll` into warp's kernel cache, verifies the `sycl` device comes up,
and reports torch XPU availability.

torch XPU is a per-machine step (deliberately NOT routed in pyproject — an
XPU-index dependency would change the wheel for every CPU/CUDA user):

    .venv/Scripts/python.exe -m pip install "torch==2.9.1+xpu" --index-url https://download.pytorch.org/whl/xpu

pip gotchas seen on real machines: a global `pip config` `target=` merges
installs into a shared directory (`PIP_TARGET= pip install ...` to escape),
and disabled Windows long paths (`LongPathsEnabled=0`) silently truncate the
torch install into an unusable state.

**Every `uv sync` — and every `uv run`, which auto-syncs — wipes the warp
overlay.** Re-run the two install commands afterwards, or the sycl device
silently disappears. `uv sync --inexact` skips the uninstall.

## Usage

    mjlab-sycl-train <TASK_ID> --num-envs 4096 --max-iterations 1000
        [--save-interval N] [--run-name NAME] [--checkpoint path]
        [--seed N] [--ppo-device xpu|cpu]

Full training: checkpoints, tensorboard logs under `logs/<TASK_ID>-sycl/`,
resume via `--checkpoint`. PPO runs on `torch.xpu` when available (physics
stays on the sycl device, env managers on CPU); `--ppo-device cpu` to compare.
Run the console script directly (`.venv\Scripts\mjlab-sycl-train ...` or an
activated shell) — `uv run mjlab-sycl-train` would auto-sync first and wipe
the overlay.

Benchmark and live viewer:

    mjlab-sycl-bench --device sycl --task Mjlab-Velocity-Flat-MicroDuck --num-envs 4096 --iters 6
    python -m mjlab_sycl.train_viewer <TASK_ID> --num-envs 1024

Task IDs come from mjlab's registry — anything installed in the venv works
(in microduck_rl: `uv run list-envs`). The bundled entries wire the SYCL patch
themselves; for a custom entry point, call `patch_simulation_for_sycl()` right
after `wp.init()`.

## Environment variables

| var | default | role |
|---|---|---|
| `WARP_SYCL_ONEAPI_BIN` | `C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3/bin` | oneAPI bin dir prepended to PATH before torch/warp import |
| `WARP_SYCL_PIP_BIN` | auto-detected from site-packages `Library/bin` | torch's pip SYCL runtime dir, appended so its older `sycl8.dll` can never shadow oneAPI's |
| `MJLAB_SYCL_FLAT_JTDAJ` | `1` | kill switch for the flat JTDAJ kernel rewrite |
| `WARP_SYCL_SHARED_KB` | 16 (`WP_MAX_SYCL_SHARED` in tile.h) | tile SLM arena size baked in at kernel-build time |
| `WARP_SYCL_SYNC_TIMEOUT_S` | 180 (0 disables) | in-process GPU watchdog; a hung kernel aborts and names the culprit |
| `MJLAB_PPO_DEVICE` | `cpu` | torch device used by `bench` |

## Performance (Arc 130T, microduck velocity, 4096 envs)

- ~5000–6800 env-steps/s end-to-end vs ~258 on the CPU device; ~22× the first
  working SYCL build, before the flat kernels and async submission.
- Numerics: `max |sycl − cpu|` over 100-step rollouts on the same
  model/actions ≈ 2–3e-06.
- Microbenchmark: 16M-float saxpy ~7.5× CPU bandwidth.

## Footguns — each learned the hard way

1. **Never add work-group barriers to tile kernels on this path.** One
   divergent barrier deadlocks the iGPU engine and freezes the whole desktop
   (Windows TDR cannot recover while the host keeps re-submitting). The
   in-dll watchdog (`WARP_SYCL_SYNC_TIMEOUT_S`) aborts and names the culprit
   kernel; wrap risky experiments in `scripts/run_guarded.py --timeout N --
   <cmd>` as the outer backstop (kills the process tree, exit code 3).
2. **The tile SLM arena fails silently when exceeded** — physics corrupts with
   no error. For bigger `nv` models raise `WARP_SYCL_SHARED_KB` and verify
   numerics against the cpu device before trusting the run.
3. **`uv sync` / `uv run` wipes the warp overlay.** Re-install after (see
   Install).
4. **`sycl8.dll` collision.** torch's XPU wheel and oneAPI ship the same DLL
   name at incompatible versions; if the wrong one resolves first, warp's
   device registration dies with `WinError 127`. The train entry orders PATH
   (oneAPI prepended, pip runtime appended) — in custom scripts, do the same
   before importing torch/warp.
5. **Steady-state step probes do not see the reset path.** Batched episode
   resets run `recompute_constants` every step (84% of rollout time in one
   profile). Profile inside `runner.learn`, never on synthetic steps.
6. **Before any long run:** a 64-env smoke test plus a numerics check against
   the cpu device on the same model/actions.

## Repo layout

    src/mjlab_sycl/    runtime_patch, flat_kernels, train/bench/train_viewer,
                       install, backend/ (the vendored warp files + warpsycl.dll
                       that ship in the wheel)
    warp_backend/      provenance + rebuild docs for the vendored backend
                       (README.md, REBUILD.md); its files/ tree is the
                       authoritative copy — keep src/mjlab_sycl/backend/ in sync
    scripts/           run_guarded.py (watchdog wrapper) + attribution probes

## Known limitations

- Windows + Intel iGPUs only (developed on Arc 130T). No Linux support, and
  the official warp test suite has not been run — this is a training path,
  not warp parity.
- Tile dynamic shared memory is capped by SLM capacity (~60 KB on current
  iGPUs).
- Device-side printf is a no-op in the SYCL backend.

## License

Apache-2.0. The vendored backend derives from NVIDIA/warp 1.12.0: derived
files carry MODIFIED notices, and warp's license plus its bundled third-party
notices ship alongside them (`backend/LICENSE.md`,
`backend/third_party_licenses/`). To rebuild `warpsycl.dll`, see
`warp_backend/REBUILD.md`.