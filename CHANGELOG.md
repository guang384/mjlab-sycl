# Changelog

All notable changes to mjlab-sycl.

## [Unreleased]

<!-- Add new changes here as they land; fold into a dated release section when
     tagging. -->

### Added
- Root `LICENSE` (Apache-2.0 full text) and a landing-page `README.md`
  (what/why, quickstart, measured performance, tooling, links).
- Launch-overhead suite for the physics hot path (Arc 130T, 4096-env
  microduck; every knob has a `MJLAB_SYCL_*` kill switch):
  - `launch_cache` — reuse (hooks, args_struct, bounds) for repeated
    launches, recurrence-gated with weakref liveness checks and LRU
    eviction so one-shot scratch buffers are never pinned.
  - `fused_tree` — level-synchronous tree chains fused to one launch per
    chain (210 → 30 launches/step), one work-item per world, no atomics.
  - `fused_solver` — the four per-iteration zero/rotate kernels folded
    into the `linesearch_jaref` tail (125 → 5 launches/step).
  - `skip_empty` — global 0-dim launch filter (120 wasted dispatches/step
    for featureless models; per-call, so feature-rich models untouched).
  - `fused_linesearch` — parallel-linesearch teardown and mv+jv fusions.
  Net: 1438 → 938 launches/step (−35%).
- `_bootstrap` preloads oneAPI SYCL DLLs before torch-xpu can pin the pip
  copies, so every entry point sees one consistent SYCL runtime.
- CI installs numpy in the CPU-side job (warp needs it at import).
- Explicit-Gaussian rollout inference (`act_fuse.py`, `MJLAB_SYCL_ACT_FUSE`):
  `GaussianDistribution.sample/log_prob` go through `torch.distributions.Normal`,
  whose Python machinery measured 5.5 ms + 3.8 ms per act call at 4096 envs
  (the underlying tensor math is ~0.1 ms). Closed-form tensor expressions
  cut the act path 19.3 → 12.2 ms/call (−7 ms/step); both sides of the PPO
  ratio use the same patched `log_prob`, so the surrogate stays internally
  consistent. Distributionally identical sampling.
- Command-graph batch replay (`graph_batch.py`, `MJLAB_SYCL_GRAPH`): the
  solver's 8-iteration batch (~264 kernel submissions) is captured as a
  SYCL command graph — new warpsycl.dll API (`wp_sycl_graph_begin/end/
  submit/free`) — and replayed as ONE queue submission. Argument staging
  during capture draws from a per-graph arena (never the shared ring), so
  a replayed node cannot observe a recycled slot. Replay arms only after
  the launch count verifies the batch sequence is static; any argument
  churn falls back to per-kernel submits. Paired A/B (5 pairs, t = −20):
  −35.0 ± 3.9 ms/step (−12.7%).

### Changed
- **Batched convergence polling is now default-on** (`MJLAB_SYCL_POLL_EVERY=8`,
  superseding the 0.2.0 "default-off" note): each solve polls convergence
  once after the first full batch instead of every iteration — with the
  8-iteration cap the batch covers the whole solve, and extra iterations
  are guarded no-ops, so physics stays bit-identical. Combined with
  restoring the lite final forward, env.step went 370 → 306 ms (−17%) at
  4096 envs. (The 0.2.0 verdict measured a smaller cadence that launched
  extra guard iterations; `docs/performance.md` carries the updated table.)
- Poll loop also skips the provably-useless initial drain (nsolving is
  host-initialized > 0) and reads the convergence counter once per poll
  instead of twice.
- `sim.step` no longer carries an inter-substep drain: kernel-to-kernel
  order is the queue's job, and the solve-end convergence poll is already
  the sync before every host read mjlab makes (air-time tracking,
  termination/reward all observe pre-solve outputs).
- Flat-kernel v2 (same `MJLAB_SYCL_FLAT_JTDAJ` switch):
  - JTDAJ (`h = qM + J^T D J`) runs one work-item per output element with
    a dot over constraints — the formulation upstream's own incremental
    Hessian update uses — instead of a per-tile read-modify-write of h;
    measured ~2.2x faster per launch on the Arc 130T.
  - The dense cholesky pair builds from a static-n factory (row loops
    unroll, triangular dots stay in registers) and reuses one scratch L
    buffer instead of allocating per call (~1.8x faster).
  - The solver's `skip_unchanged` contract is now honored on the small-nv
    path too (nv ≤ 32): when no constraint state changed since the last
    factorization, the LLT is skipped and only the triangular solves run
    (upstream only cached the factorization for nv > 32).
    `MJLAB_SYCL_LLT_SKIP=0` force-disables the skip for A/B testing.
- Solver scratch reuse (`solver_ctx.py`, `MJLAB_SYCL_SOLVER_CTX`):
  `solver.solve()` allocated a fresh SolverContext plus step_size_cost and
  nsolving on every call (4/step), so every solver kernel got a fresh
  launch-cache key each solve — the cache re-ran its recurrence gate and
  rebuild path before hits resumed (~28% of launches on the slow path).
  The context is scratch state re-initialized per solve; reusing the
  allocations keeps keys stable (cache hit rate 60% → 81%, slow path
  halved). +2–3% throughput. Installed before `fused_linesearch` so that
  module's `solver.solve` shim wraps this replacement — wrapping the other
  way round silently disabled its mul_m/jv/teardown fusions.
- `launch_cache.stats()` now reports slow-path reasons and per-path µs
  totals, making cache behavior measurable instead of opaque.
- `sense()` drains only when there is no sensor context — with one,
  `SensorContext.finalize()` drains before its torch-only host reads and
  `sense()` launches nothing afterwards. −1 queue drain/env.step
  (measured 8 → 7).
- `MJLAB_SYCL_BENCH_SC_CPU` renamed to `MJLAB_SYCL_SC_CPU`.
- README environment-variable table rewritten: all `WARP_SYCL_*` /
  `MJLAB_SYCL_*` knobs with defaults and roles, including previously
  undocumented ones (device, lite forward, iteration caps, every fusion
  kill switch) — plus `MJLAB_PPO_DEVICE` documented as `xpu`.
- Comments reconciled with code across runtime_patch, loop_poll,
  flat_kernels, fused_tree, fused_linesearch, bench, README (drain sites,
  tile padding, install ordering, gate count, drain/launch arithmetic).
- Two more iteration-kernel merges (`MJLAB_SYCL_FUSED_LINESEARCH` /
  `MJLAB_SYCL_FUSED_SOLVER`): `linesearch_prepare_gauss` folded into
  `prepare_quad`'s work-item (world, 0) — single writer, no atomics — and
  `solve_done` folded into `solve_search_update`. With graph replay
  active these are wall-neutral (paired A/B ±4.5 ms) but further cut the
  per-iteration kernel count and help the graph-less fallback.
- **Selective model-constant recompute is now default-on for the GPU too**
  (`MJLAB_SYCL_FUSED_SET_CONST`, was CPU-sim only): the reset path (fall →
  domain-randomization event → `recompute_constants`) used to recompute
  model constants for ALL 4096 worlds when a handful reset. The selective
  kernel limits the work to the reset env_ids — measured 50 → 10 ms/step
  at 4096 envs with active policies, **+17% end-to-end (13,458 → 15,762
  env-steps/s)**. Even the all-worlds case is a wash on the GPU (66 ms
  both ways) — the old "CPU only" verdict measured just that case.

### Fixed
- SYCL runtime loading hardened (`_bootstrap` / `build._load_sycl_dll`):
  sycl8.dll's own imports (libmmd from oneAPI's top-level `<ver>/bin`, not
  the compiler bin) are now preloaded in dependency order before warpsycl
  loads, and the runtime DLL loader walks that chain itself — on some
  loader states the dependency search alone missed them (WinError 127).
  Also: `act_fuse` imports torch lazily — importing torch before warpsycl
  pins pip's sycl8.dll by name and warpsycl then fails to load.
- `lite_forward` was clobbered by a blanket `drained(orig_forward)`
  assignment — the solver-skipping final forward never ran, costing ~50
  ms/step at 4096 envs (5 solver calls per step instead of 4).
- `play` crashed unconditionally (missing `import threading`).
- `launch_cache` key now includes device and block_dim (entries could
  cross-contaminate across devices/block sizes) and the cache no longer
  bypasses tape capture (torch.compile path).
- `flat_kernels` `_ADR_SIZE_CACHE` guards against Python `id()` recycling
  (a recycled id returned a wrong tile size → silent Cholesky error).
- `fused_solver_tail` CG beta=0 bug (prev values copied before beta was
  computed — CG silently degraded to steepest descent).
- `_tile_cholesky_factorize_solve`'s flat replacement processed only the
  first qM tile per world (a bare `nworld` launch dim left `nodeid` at 0);
  silent on single-tile models, wrong for multi-tile ones.
- `kview` `dict()` misuse, `install` self-check order, `fused_set_const`
  missing physics term, epsilon constants, `bench` division by zero on
  zero completed iterations, unused imports and dead globals (doctor's
  stray pip-path finder, `_ORIG_RECOMPUTE`, …).
- Dev-harness files removed from the index (`.temp/` scripts,
  `probe_field_hunt`, leaked scratch) and `.gitignore` completed — a
  clone contains only shippable files.
- `test_e2e`'s 2-D atomic reduction was a tolerance lottery: random-order
  float32 atomics carry ~3e-5 relative rounding which sat under the
  rtol=1e-5 check, so it failed depending on the GPU's atomic schedule.
  Now uses integer-valued float32 (exact in any order) with an exact
  comparison — deterministic and a stronger check.
- The overlay self-check no longer trips on `__pycache__` bytecode
  artifacts of the bundled backend sources (each tree compiles its own
  `.pyc`; comparing them byte-wise false-failed).

### Performance
- −35% kernel launches/step, −17% env.step from the poll/forward fixes,
  7 instead of 8 queue drains/step after the sense merge, −29% serialized
  kernel time from flat-kernel v2, and −12.7% from graph batch replay.
  Current census: ~938 launches, 7 drains per env.step, and 4 graph
  submissions replacing ~1,056 per-kernel submits; median ≈ 210 ms/step
  at 4096 envs (Arc 130T). End-to-end bench: 13,390 env-steps/s vs 5,485
  archived. Every change above is A/B-verified against the cpu device
  (max |sycl − cpu| ≈ 3e-06) and gated by `mjlab-sycl-test`.

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
