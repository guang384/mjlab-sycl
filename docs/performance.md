# mjlab-sycl performance baseline (measured archive)

One place for every throughput/latency number measured on this machine, the
tooling to reproduce it, and the verdict on each optimization attempted. Use
it as the baseline when tuning on this box or benchmarking any other machine.

Measured on: Intel Arc 130T iGPU (16 GB, Lunar Lake) · Windows · Python 3.12
· warp 1.12.0 (sycl overlay) · mujoco_warp 3.8.1 · mjlab 1.3.0 · torch 2.9.1+xpu
Task: `Mjlab-Velocity-Flat-MicroDuck` (14 servos, nv=20), 50 Hz control.

## Reproduce

```powershell
# end-to-end training loop timing (rollout | ppo | total per iteration)
.venv\Scripts\mjlab-sycl-bench.exe --device sycl  --num-envs 4096 --iters 3
.venv\Scripts\mjlab-sycl-bench.exe --device cpu   --num-envs 1024 --iters 2   # pure warp-cpu, no sycl patch
# per-step API census (launches / drains / allocs per env step)
python scripts\probe_sycl_profile.py --num-envs 4096 --steps 20
# per-kernel device time attribution (serialized; relative ranking valid)
python scripts\probe_kernel_times.py --num-envs 4096 --steps 4
# real efc usage vs compiled buffer (njmax padding audit)
python scripts\probe_efc_audit.py --num-envs 512 --steps 30
```

## Headline numbers

| config (4096 envs unless noted) | per-iteration wall | env-steps/s |
|---|---|---|
| sycl, PPO on **xpu** (default), selective-recompute refresh (2026-09-30) | mean 7.1 s (rollout 6.2 / ppo 0.9) | **15,762** |
| sycl, PPO on **xpu** (default), graph-batch refresh (2026-09-30) | mean 8.3 s (rollout 7.3 / ppo 1.0) | 13,390 |
| sycl, PPO on **xpu** (default), flat-kernel v2 refresh (2026-09-30, 2 runs) | mean 9.1 – 9.4 s | 11,835 – 11,940 |
| sycl, PPO on **xpu** (default), post-fusion (2026-09-30, 3 runs) | mean 10.2 – 17.1 s (run-to-run clock variance) | 6,301 – 10,782 |
| sycl, PPO on **cpu** | 23.7 s (rollout 19.1 / ppo 4.5) | 5,140 |
| sycl, PPO on **xpu** (default) — 2026-09-08 archive | **19.2 s** (rollout 17.9 / ppo 1.2) | 5,485 |
| sycl, 8192 envs, PPO xpu | 37.3 s | 5,634 (+2.7 %) |
| **pure warp-CPU device** (1024 envs) | 100.2 s | 246 |
| sycl (1024 envs) | 7.6 s | 3,436 |

- PPO xpu vs cpu: update 4.5 s -> 1.2 s/iter (~19 % faster iterations).
- sycl vs same-stack pure CPU device: **~14x** (3436 vs 246 env-steps/s @1024).
- Env-count scaling is flat beyond ~4096 (device saturated; doubling envs doubles
  per-step wall -> same samples/s).

## Where env.step goes (census)

Current (launch-fusion suite + flat-kernel v2 + command-graph batch
replay, measured 2026-09-30, 4096 envs):

- median ≈ 210 ms/env.step (paired A/B: graph replay worth
  −35.0 ± 3.9 ms/step, 5 pairs, t = −20)
- ~938 kernel launches/step, of which the solver's ~264-iteration batch
  replays as ONE graph submission (4 graphs/step replace ~1,056
  per-kernel submits)
- 7 queue drains/step: 4 solve-end convergence polls + lite forward +
  Bvh refit + sensor finalize; ~7 % of wall waiting in drains

Budget map (measured 2026-09-30 with true per-kernel launch counts --
cache hits bypass launch hooks, so hook-based counts under-sample ~3x):

- The Newton loop runs the full 8-iteration cap on all 4 substep solves
  (32 iterations/step); mean useful iterations per solve is 4.3 and 22% of
  worlds reach the cap, so ~45% of iteration slots are ghost work that
  cannot be dropped without changing numerics.
- Device time (wall is device-bound): cholesky factorize+solve ~18%
  (45 calls/step, ~1 ms each -- the serial LLT chain already runs at
  ~2 cycles/inner-iteration), constraint linearization (efc/jaref/init/
  gauss) ~15%, linesearch family (jv/quad/parallel) ~20%, Hessian builds
  (JTDAJ + h_incremental) ~10%, collision/kinematics/sensors ~15%.
- ~20 us of queue-submit overhead per kernel x ~1070 launches/step ≈ 8%.
- Host-side launch cost is GPU-queue backpressure in disguise: the pure
  enqueue floor measured on an idle queue is 20-45 us (raw pack ~45 us),
  while in-workload "submit" times reach 300+ us.

Rollout-phase budget (active policy / random actions, 2026-09-30):

- env.step ~260 ms: sim.step ~170 (the GPU physics), lite forward ~13,
  selective recompute on falls ~10 (was 50 before selectivity), reward ~8,
  policy act path ~17 (XPU launch-overhead bound, shares the iGPU with
  physics), obs/sense/termination/scene ~15, wrapper+manager misc ~8.
- torch thread count is NOT a lever here (paired A/B 2/4/8: ns).

Historical (2026-09-08 baseline, before the fusion suite and poll
default — kept for attribution):

- ~670 ms/env.step: sim:step x4 ~454 ms + env-level forward ~111 ms
  (5 full Newton solves/step — the lite forward was clobbered then)
- ~1,600 kernel launches/step; host submit ~300 ms/step (overlaps device)
- ~62 queue drains/step (~55 are the capture-while convergence polls),
  ~230 ms/step waiting on real device work
- residual host (managers/obs/etc.) ~120 ms/step

## Optimization attempts + verdicts

| change | result | verdict |
|---|---|---|
| PPO on torch.xpu | 4.5 -> 1.2 s/iter | kept (default) |
| torch CPU threads cap (MJLAB_TORCH_THREADS=2) | CPU 4.6 -> 1.9 cores, wall unchanged (~670 ms/step) | kept (default) |
| batched convergence polling — small cadence (poll < iteration cap) | 716 vs 671 ms/step: the extra guard no-op iterations cost more than the polls saved | dead end |
| batched convergence polling — cadence = iteration cap (poll_every=8) | 1 poll/solve instead of every iteration; with the lite-forward restore, env.step 370 -> 306 ms (−17 %) | kept (default-on) |
| launch cache + tree/solver/linesearch fusion + 0-dim launch skip | 1,600 -> 938 launches/step (−35 %), part of 670 -> ~330 ms/step | kept (default-on, `MJLAB_SYCL_*` kill switches) |
| flat-kernel v2: per-element JTDAJ dot, unrolled dense cholesky, LLT skip on unchanged constraints | serialized kernel time 435 -> 308 ms/step; env.step median ~330 -> ~270 ms | kept (same `MJLAB_SYCL_FLAT_JTDAJ` switch) |
| solver scratch reuse (SolverContext/step_size_cost/nsolving across solves) | launch-cache hit rate 60 -> 81 %, slow rebuild path halved; +2-3 % throughput | kept (`MJLAB_SYCL_SOLVER_CTX`) |
| command-graph batch replay (solver 8-iteration batch as one submission) | paired A/B −35.0 ± 3.9 ms/step (−12.7 %) | kept (`MJLAB_SYCL_GRAPH`) |
| iteration-kernel merges (prepare_gauss→prepare_quad, solve_done→search_update) | wall-neutral with graphs on (±4.5 ms), fewer kernels/launches | kept (same fusion switches) |
| selective set_const recompute on resets (fall → randomize event) | recompute_constants 50 → 10 ms/step with active policies; +17 % end-to-end | kept, default-on (`MJLAB_SYCL_FUSED_SET_CONST`) |
| torch thread sweep re-run under the new pipeline (2/4/8) | paired A/B: no significant wall difference | default 2 unchanged |
| selective kinematics/com_pos/crb for set_const recompute (copied prep kernels with world-id indirection) | unit A/B equivalent to 2e-4, but only ~0.8 ms/call at 4096 envs (4 reset ids): the cut stages are launch-bound at small nproc and factor_m stays all-world | reverted -- 400 lines of copied kernels not justified below the noise floor |
| lite forward as one command graph (`graph_batch.run_sequence`) | ~15 submits/step -> 1; wall-neutral | kept (`MJLAB_SYCL_GRAPH`) |
| sense() drain merge (sensor context already drains in finalize) | 8 -> 7 drains/step, no measurable wall change | kept |
| kernel args by-value capture (codegen) | ~1 % (noise); DPC++ also requires const kernel lambdas | dead end, reverted |
| 8192 envs | +2.7 % | not worth wall 2x |
| solver micro-kernel fusion (Route C) | infeasible here: fusion points need intra-world sync (barriers) which must never be added on the iGPU; only tiny per-world scalar merges remain (~1-2 %) | not viable, documented |
| viewer rendering | GL runs on the same iGPU as physics (hardware GL confirmed); Windows KnownDLLs block Mesa software-GL override; real-sim viewers cap at ~8 updates/s (per-step fixed cost) | structural |

## Known device-side costs (out of adapter reach)

- **efc buffer padding**: peak real efc/world = 46 vs compiled buffer 1504 rows
  (~33x idle). Shrinking needs model-level `<size njmax>` (microduck MJCF) or a
  mujoco_warp compile change; see scripts/probe_efc_audit.py.
- Kernel time leaders (serialized ranking @4096): update_constraint_efc,
  cholesky solves, linesearch family (jv/prepare_quad/jaref), flat JTDAJ/contact.

## Hardware context (web, for wall-clock planning)

- Same task, 4096 envs: RTX 5080 ~1.13 s/iter (community course,
  https://forum.d-robotics.cc/t/topic/35668) -> ~17x faster than this iGPU.
- PassMark GPU compute: Arc 130V ~2.2k vs RTX 5090 ~24.4k ops/s (~11x);
  memory bandwidth GDDR7 ~15-18x LPDDR5X-class iGPU.
- Wall-clock projection (tricks ~1000 iters, gaits 4000-6000 iters):
  this iGPU ~5.5 h / 21-32 h; RTX 5090-class ~12 min / 50-75 min.
