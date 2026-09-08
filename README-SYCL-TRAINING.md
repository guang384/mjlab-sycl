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
| `test_overlay` / `test_e2e` / `test_mujoco` | Verification gates: host-only overlay-sync check (no GPU), backend mechanics bit-exact vs the cpu device, and real mujoco_warp physics agreeing with the cpu device (see Verification gates below). One command: `mjlab-sycl-test`. |

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
reports torch XPU availability, and **self-checks the overlay it just applied**
(byte-identical to this package). The training entries re-run that check on
every start (`patch_simulation_for_sycl()` → `ensure_overlay_synced()`): a
stale or uv-sync-wiped overlay aborts with the remediation instead of a
cryptic missing-device error or silently corrupt physics.

**Console scripts.** `mjlab-sycl-train/-bench/-test` are only created when the
package is installed *into the venv* — a global `pip config` `target=` (as on
this machine) redirects the install into a shared directory that receives no
scripts. Install from the local clone with the redirect bypassed so the
scripts land in `.venv\Scripts\`:

    cd D:\mjlab-sycl                 # the local clone
    # This machine's pip config file sits behind a PIP_CONFIG_FILE env var, and
    # --isolated CANNOT bypass an env-pointed config file (verified the hard
    # way) -- clear it first (PowerShell):
    #   Remove-Item Env:PIP_CONFIG_FILE, Env:PIP_TARGET -ErrorAction SilentlyContinue
    <project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps -e .
    # or: uv pip install --python <project>\.venv\Scripts\python.exe --no-deps -e .

(drop `--no-deps` / `-e` as you prefer; the point is that the install lands in
the venv itself, so its `.venv\Scripts\` receives the console scripts).

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

## Using with microduck_rl (or any mjlab task package)

mjlab-sycl installs *into* an existing mjlab project and trains its registered
tasks as-is — nothing in the project needs to know about SYCL. On a fresh
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) clone the
whole setup is one command (the script below is exactly the manual recipe
that follows, with the pip-config-env-var trap already handled):

    cd D:\mjlab-sycl
    .\scripts\setup_microduck.ps1 -Repo C:\dev\microduck_rl        # + -InstallTorchXpu on a new machine

Manual equivalent:

    # 1. project venv (stays untouched by everything that follows)
    cd microduck_rl
    uv sync

    # 2. this package, installed INTO the venv so its console scripts land in
    #    .venv\Scripts\ (clear the pip-config env vars first on this machine)
    #    Remove-Item Env:PIP_CONFIG_FILE, Env:PIP_TARGET -ErrorAction SilentlyContinue
    cd D:\mjlab-sycl
    ..\microduck_rl\.venv\Scripts\python.exe -m pip install --isolated --no-deps -e .

    # 3. per-machine torch XPU (see above), then overlay the warp backend
    ..\microduck_rl\.venv\Scripts\python.exe -m mjlab_sycl install

    # 4. one command decides whether this machine can train here:
    ..\microduck_rl\.venv\Scripts\mjlab-sycl-check        # (mjlab-sycl-check)

    # 5. every task microduck_rl registers trains as-is:
    mjlab-sycl-train Mjlab-Velocity-Flat-MicroDuck --num-envs 4096 --max-iterations 1000

`mjlab-sycl-check` is read-only diagnosis: platform/Python, warp overlay sync,
Intel oneAPI runtime, sycl device, a real device kernel vs cpu, torch XPU, and
that the venv's mjlab task registry (microduck_rl) imports — each line prints
its fix when it fails, exit code 0/1. Remember the footguns: after any
`uv sync` / `uv run` re-run steps 3 (+ re-install of this package) — the train
entries and `mjlab-sycl-check` both tell you when the overlay is gone.

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

## Verification gates

> Environment preflight first: `mjlab-sycl-check` (read-only). These gates are
> the full numerical verification on top of a healthy environment.

    mjlab-sycl-test                     # all gates, in order
    python -m mjlab_sycl.test_overlay   # host-only overlay-sync check alone
    python -m mjlab_sycl.test_e2e       # backend mechanics alone
    python -m mjlab_sycl.test_mujoco    # mujoco_warp physics vs cpu alone

`test_overlay` needs no GPU: it asserts every shipped backend file is
byte-identical to what install() placed in the environment's warp package and
kernel cache (plus the warp-version guard), so a uv-sync-wiped or drifted
overlay aborts before the GPU gates run. Its file-comparison logic is also
pytest-discoverable (`pytest src/mjlab_sycl/test_overlay.py`) against
throwaway temp dirs.

`test_e2e` exercises device registration, USM-backed arrays with host
readback, kernel compilation through the icx chain, module-cache relaunch,
vec3/struct types, atomics under million-way races, 2-D launches, and the
tape adjoint path — bit-exact (or atol-tight) against the cpu device.

`test_mujoco` runs real mujoco_warp physics steps (the pendula model shipped
with mujoco_warp) on the sycl device: no NaNs, run-to-run determinism, and
agreement with the cpu device within 1e-5 (typically ~2-3e-6).

Run the gates after `install` on a new machine, after any backend change or
`warpsycl.dll` rebuild (see warp_backend/REBUILD.md), and before any long
training run. Some backend misconfigurations fail silently — e.g. an SLM
arena overflow corrupts physics with no error — so the cpu-agreement check
is the only reliable detector.

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
   Install). The sycl entries now fail fast with a clear message if you
   forget (`patch_simulation_for_sycl()` verifies the overlay is byte-identical
   to the package before physics starts) — don't work around it, re-run
   `python -m mjlab_sycl install`.
4. **`sycl8.dll` collision.** torch's XPU wheel and oneAPI ship the same DLL
   name at incompatible versions; if the wrong one resolves first, warp's
   device registration dies with `WinError 127`. The train entry orders PATH
   (oneAPI prepended, pip runtime appended) — in custom scripts, do the same
   before importing torch/warp.
5. **Steady-state step probes do not see the reset path.** Batched episode
   resets run `recompute_constants` every step (84% of rollout time in one
   profile). Profile inside `runner.learn`, never on synthetic steps.
6. **Before any long run:** `mjlab-sycl-test` (the gates) plus a 64-env
   training smoke test.

## Repo layout

    src/mjlab_sycl/    runtime_patch, flat_kernels, train/bench/train_viewer,
                       the verification gates (test_e2e/test_mujoco/test +
                       _bootstrap), install, backend/ (the vendored warp files
                       + warpsycl.dll that ship in the wheel)
    warp_backend/      provenance + rebuild docs for the vendored backend
                       (README.md, REBUILD.md); the backend files themselves
                       live only in src/mjlab_sycl/backend/
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