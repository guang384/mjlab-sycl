# Changelog

All notable changes to mjlab-sycl.

## [Unreleased]

<!-- Add new changes here as they land; fold into a dated release section when
     tagging. -->

### Added
- **Collision workspace pool + USM pool trim** (`MJLAB_SYCL_WS_POOL`):
  mujoco_warp's convex_narrowphase allocates ~1.5 GB of GJK/EPA scratch
  PER CALL at 4096 envs ((naccdmax, 112/224/256)) and drops it at return
  -- ~18 GB of alloc/free churn over 12 steps, RSS pinned at the churn
  high-water. A shape-keyed checkout pool reuses the scratch (in-order
  queue + write-before-read semantics make this safe and keep the step
  graph's baked pointers stable). Plus `wp_sycl_pool_trim` (returns free
  lists to the OS -- a trim under live graphs crashes the replay, so it
  frees graphs first), `wp_sycl_pool_stats`/`wp_sycl_pool_hist` census
  exports and `scripts/probe_memory.py`. Measured: free lists 2,474 ->
  44 MB, live 5,029 -> 2,871 MB, RSS steady with no churn spikes;
  determinism bit-equal and physics 1.863e-08 unchanged; perf
  wall-neutral (paired A/B -1.3 % / +1.1 %).
- **Native fused quad_gauss kernel** (`wp_sycl_quad_gauss`,
  `MJLAB_SYCL_NATIVE_QUAD=0` to disable): the prepare_quad+prepare_gauss
  fusion re-implemented natively -- unit-verified BIT-EXACT (0.0) on the
  pyramidal path, elliptic-cone branch ported verbatim. Measured
  wall-neutral (swapped-order A/B +1.8 % / -0.3 %, mean inside noise):
  at (nworld, njmax) the grid already saturates the device so only the
  codegen delta remains (same lesson as the jv reschedule). Kept on the
  wall-neutral/bit-exact standard. Completes the native rewrite of every
  top-10 kernel family (8 native kernels total).
- **Native qfrc_constraint + fused jaref kernels** (`wp_sycl_qfrc_constraint`,
  `wp_sycl_jaref`; kill switches `MJLAB_SYCL_NATIVE_QFRC` /
  `MJLAB_SYCL_NATIVE_JAREF`): J^T @ force and the fused linesearch-jaref +
  zero-ahead contract, re-implemented natively with bit-exact per-element
  arithmetic (qfrc unit-verified at 0.0; jaref bit-exact except the
  Jaref += alpha*jv FMA contraction, ULP 4.8e-7). Swapped-order paired A/B
  at 4096 envs on a fully-quiet machine: 21,463/21,293 -> 22,060/21,800
  env-steps/s (+2.4-2.8 %). All gates pass (patched stack 1.863e-08). The
  seams live inside the existing interceptor launch branches (no new
  function-level contracts).
- **`test_patched` gate** (runs last in `mjlab-sycl-test`): physics check
  for the PATCHED stack on collision.xml (nv=12 -> small-nv chol, nefc=8,
  ncon=11, Newton + pyramidal -> incremental path) -- the other gates call
  mjw.step directly and exercise none of the runtime patch layers. Adds a
  **wrapper-contract check**: function-level seams must replicate the
  replaced function's full launch sequence (verified against the hinc
  seam). Trajectory checks alone are NOT sufficient: measured with
  collision.xml, a skipped gradient launch is trajectory-neutral (qpos
  bit-identical) while dropping physics on other states -- the contract
  check catches it and was demonstrated red with the bug reintroduced.
  Native routes gain a device guard (non-sycl arrays fall back).
- **Native set-const chol_fs + incremental Hessian hinc kernels**
  (`wp_sycl_chol_fs`, `wp_sycl_hinc`; kill switches `MJLAB_SYCL_NATIVE_CHOL`
  / `MJLAB_SYCL_NATIVE_HINC`): the set-const single-tile factorize+solve and
  the changed-constraint Hessian delta. chol_fs's tile anchor is read
  device-side from the adr array so the route is host-sync-free (a host read
  in the route aborted the step-graph recording); hinc's route re-issues the
  gradient launches verbatim and replaces ONLY the Hessian-delta kernel --
  the first version replaced the whole _update_gradient_incremental and
  silently skipped the gradient computation (a fake +23 % that was broken
  physics; caught by contract review before landing). Quiet-machine A/B:
  hinc +1.7-4.0 %; full 5-kernel native suite vs all-off: 16,875/16,856 ->
  18,796/18,505 env-steps/s (+9.8-11.4 %). Gates pass at 3.308e-06.
- **Native solver-cholesky kernel** (`wp_sycl_chol_solve`,
  `MJLAB_SYCL_NATIVE_CHOL=0` to disable): LLT + forward/back substitution
  per world with per-size template instantiation (4..32) keeping the
  N-loops fully unrolled, done-guarded, changed/lvalid skip preserved.
  NOT bit-identical to the warp kernel (sycl::sqrt/division rounding,
  ULP-level) -- gated by the 1e-5 contract: all gates pass, physics vs
  cpu 3.308e-06 unchanged. Swapped-order paired A/B at 4096 envs:
  18,420/18,201 -> 18,869/18,991 env-steps/s (+2.4-4.3 %). Third harvest
  of the DPC++-vs-warp codegen gap.
- **Native JTDAJ kernel** (`wp_sycl_jtdaj`, `MJLAB_SYCL_NATIVE_JTDAJ=0` to
  disable): the Hessian update h = qM + J^T D' J re-implemented as a native
  SYCL kernel -- 32-lane row-block schedule, bit-exact per-element
  k-ascending dots with the same Dk-zeroing rules (unit-verified
  bit-identical incl. non-QUADRATIC states, done worlds, nefc edges).
  Swapped-order paired A/B at 4096 envs on a quiet desktop:
  17,697/17,351 -> 18,491/18,279 env-steps/s (+4.5-5.4 %); gates pass with
  physics-vs-cpu at 3.308e-06 unchanged. Second harvest of the measured
  DPC++-vs-warp codegen gap.
- **Native mv+jv kernel** (`native_kernels.py` + `wp_sycl_mv_jv` in
  warpsycl.dll, `MJLAB_SYCL_NATIVE_MVJV=0` to disable): the linesearch's
  fused mv+jv launch re-implemented as a native SYCL kernel -- 32-lane
  row-block schedule, bit-exact per-element k-ascending dots (unit-verified
  bit-identical to the warp kernel), done-guarded. Kernel 2.5x faster
  (0.62 -> 0.25 ms/launch); swapped-order paired A/B at 4096 envs:
  14,777/14,719 -> 15,493/15,216 env-steps/s (+3.4-4.8 %); all gates pass
  with physics-vs-cpu at 3.308e-06, unchanged. First harvest of the
  DPC++-vs-warp codegen gap measured in docs/optimization_ideas.md.
- **Whole-substep command graph** (`MJLAB_SYCL_STEP_GRAPH`, default-on;
  `MJLAB_SYCL_GRAPH=0` still kills all graphs): `mjwarp.step` — collision,
  constraints, solve, integrate — replays as ONE queue submission after a
  plain count and a launch-count-verified record. Scoped mode runs the
  solver batch graph plain while the substep establishes itself, so the
  outer graph absorbs it (nested replays would break count verification);
  the solver-end convergence poll drain is skipped during capture (queue
  waits are illegal mid-recording; with poll_every >= the iteration cap the
  poll count is constant, so the recorded sequence is unchanged) and the
  step wrapper drains once at the substep boundary instead — still exactly
  one sync per substep. Swapped-order paired A/B at 4096 envs: 13,552/13,834
  -> 15,108/14,998 env-steps/s (+8-11 %); all verification gates pass with
  physics-vs-cpu at 3.3e-06, unchanged.
- `docs/optimization_ideas.md`: argued-but-unattempted kernel-level
  optimization candidates (oneMKL batched cholesky, sub_group LLT, jv
  row-block parallelism, whole-substep command graphs, efc-cost atomic
  elimination, ghost-iteration world compaction) with feasibility, safety
  and numerics analysis per item — plus two facts measured while arguing:
  the kernel-time probe attributes fused executions to pre-fusion names
  (identity must be checked via launch-cache keys), and a `max_unroll`
  sweep is a measured dead end (unroll 64→8 makes the cholesky kernels
  94x slower; full unrolling is load-bearing).
- pytest configuration: `[tool.pytest.ini_options]` `testpaths` points at
  the host-only overlay gate (a bare `pytest` no longer collects the
  in-package GPU test modules), and pytest is declared as a `test` extra.
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
- `scripts/probe_kernel_times.py` now annotates its report with what
  actually executed: the hook sits outside the fusion interceptors, so
  fused executions were timed under pre-fusion names (times real, names
  not). Active fusions are detected from the launch cache's key space and
  each affected row carries an `[actually ...]` / `[suppressed ...]` note;
  the docstring records that kernel-level attribution requires
  `MJLAB_SYCL_GRAPH=0` (with the substep graph active, steady-state steps
  replay without Python launch calls).
- The 12 interceptor layers now install from a single auditable
  `_INTERCEPTOR_LAYERS` table in `runtime_patch` — one row per layer, each
  note recording why the row sits exactly there; the install sequence is
  verified identical to the previous sequential calls.
- `import mjlab_sycl` is side-effect free (PEP 562 lazy re-export of
  `patch_simulation_for_sycl`): the package no longer pulls warp or
  mujoco_warp at import time.
- `fused_solver` defers its mujoco_warp imports to `install()` (kernel
  body globals resolve at first launch). `fused_tree` / `fused_linesearch`
  cannot do the same — warp evaluates kernel SIGNATURE annotations at
  decoration time — now recorded as a NOTE in both files.
- Viewer entries share `viewer_common.snapshot_env0/apply_state`; `play`
  no longer imports `train_viewer`'s private helpers.
- README gains a **"Relationship to NVIDIA/warp"** section (not a fork — a
  strictly additive overlay on `warp-lang==1.12.0`; `warp.dll` and the
  CUDA/CPU paths untouched; Apache-2.0 §4(b) notices ship with the files),
  and `warp_backend/README.md` records the provenance endgame: the local
  NVIDIA/warp `sycl` development clone is retired (its `history.bundle` was
  verified against the clone before deletion). The bundle is now the archive
  of record and restores on top of the `v1.12.0` tag; backend work continues
  in `src/mjlab_sycl/backend/`.
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
- Lite forward replays as one command graph (`graph_batch.run_sequence`,
  same `MJLAB_SYCL_GRAPH` switch): the fwd_position/sensor/fwd_velocity
  sequence is static, so after two plain runs it replays as a single queue
  submission. Wall-neutral on its own (the batch replay already owns the
  big win) but removes ~15 more per-kernel submits from every step.
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
- `docs/performance.md` correction: the efc-buffer-padding note ("46 real
  vs 1504 compiled rows, ~33x idle") sat under "known device-side costs
  (out of adapter reach)" but adaptive njmax has been default-on since
  2026-09-30 — it landed uncredited inside the launch-cache commit,
  shrinking the task cfg's njmax=1500 to `max(baseline_nefc*16, nq*8, 96)`
  = 168 (buffer 176 rows). Moved to the attempts table with a fresh
  back-to-back A/B (12.05 -> 7.01 s/iter, 1.72x, ppo unchanged; 8,765 ->
  15,918 env-steps/s) and the reason shrinking below ~128 is a dead end
  (compute is already nefc-bounded; undersized buffers drop constraints
  silently -> NaN). Warm-device 2026-10-02 headline row added (15,918
  env-steps/s, consistent with the thermal attribution).

### Fixed
- `install` no longer crashes with WinError 32 when a live process holds
  the kernel-cache `warpsycl.dll`: an already byte-identical DLL is
  skipped, and a genuinely different one reports the close-the-holder
  remediation instead of a traceback.
- `_bootstrap.py` had been pasted into itself once (~200 duplicated lines;
  the orphan `2.0` at the seam is the paste artifact). The LIVE copy —
  including the eager `warpsycl.dll` preload fix, which had landed only in
  the second, shadowing definition — is kept; the dead first copy is gone
  (−185 lines).
- README's performance headline was ~3x stale (0.2.0's ~5.5k env-steps/s;
  the stack measures ~15.8k — see `docs/performance.md` for the error bar).
- `sycl_runtime.h` declared three SLM high-water exports that were never
  defined or called (the real mechanism is tile.h's compile-time
  `WP_MAX_SYCL_SHARED` arena); the dead declarations are removed.
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
