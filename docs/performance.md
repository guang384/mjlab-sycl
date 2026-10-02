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
.venv\Scripts\mjlab-sycl-bench.exe --device sycl  --task Mjlab-Velocity-Flat-MicroDuck --num-envs 4096 --iters 3
.venv\Scripts\mjlab-sycl-bench.exe --device cpu   --task Mjlab-Velocity-Flat-MicroDuck --num-envs 1024 --iters 2   # pure warp-cpu, no sycl patch
# per-step API census (launches / drains / allocs per env step)
python scripts\probe_sycl_profile.py --num-envs 4096 --steps 20
# drain call-site attribution (which wrapper drains how often per step)
python scripts\probe_drain_sites.py --num-envs 4096 --steps 20
# per-kernel device time attribution (serialized; relative ranking valid)
python scripts\probe_kernel_times.py --num-envs 4096 --steps 4
# real efc usage vs compiled buffer (njmax padding audit)
python scripts\probe_efc_audit.py --num-envs 512 --steps 30
```

## Headline numbers

| config (4096 envs unless noted) | per-iteration wall | env-steps/s |
|---|---|---|
| sycl, PPO on **xpu** (default), selective-recompute refresh (2026-09-30) | mean 7.1 s (rollout 6.2 / ppo 0.9) | **15,762** |
| sycl, PPO on **xpu** (default), 2026-10-01 re-check (3 runs, cold start) | mean 8.6 s (rollout 7.4 / ppo 1.2) | 13,276 |
| sycl, PPO on **xpu** (default), 2026-10-02 re-check (warm device, 3 iters) | mean 7.0 s (rollout 6.2 / ppo 0.8) | 15,918 |
| sycl, PPO on **xpu** (default), 2026-10-02 quiet-desktop + full native suite (8 kernels) | ~5.5 s/iter | **21,500 – 22,100** (contended baseline for comparison: 16.9k) |
| sycl, same stack, 2026-10-02 later-session re-check | ~6.3 s/iter | 18,100 – 19,400 (same code, machine-state drift; natives-vs-off pairing held at +19.8 %) |
| sycl, same stack, remote-desktop closed (measure it: RDP costs ~5 %) | 6.05 s/iter (rollout 5.06 / ppo 0.98) | 19,425 – 19,840 |
| sycl, PPO on **xpu** (default), graph-batch refresh (2026-09-30) | mean 8.3 s (rollout 7.3 / ppo 1.0) | 13,390 |
| sycl, PPO on **xpu** (default), flat-kernel v2 refresh (2026-09-30, 2 runs) | mean 9.1 – 9.4 s | 11,835 – 11,940 |
| sycl, PPO on **xpu** (default), post-fusion (2026-09-30, 3 runs) | mean 10.2 – 17.1 s (run-to-run clock variance) | 6,301 – 10,782 |
| sycl, PPO on **cpu** | 23.7 s (rollout 19.1 / ppo 4.5) | 5,140 |
| sycl, PPO on **xpu** (default) — 2026-09-08 archive | **19.2 s** (rollout 17.9 / ppo 1.2) | 5,485 |
| sycl, 8192 envs, PPO xpu | 37.3 s | 5,634 (+2.7 %) |
| **pure warp-CPU device** (1024 envs) | 100.2 s | 246 |
| sycl (1024 envs) | 7.6 s | 3,436 |
| sycl (1024 envs), 2026-10-01 re-check | 3.9 s | 6,902 |

- PPO xpu vs cpu: update 4.5 s -> 1.2 s/iter (~19 % faster iterations).
- sycl vs same-stack pure CPU device: **~14x** (3436 vs 246 env-steps/s @1024).
- Env-count scaling is flat beyond ~4096 (device saturated; doubling envs doubles
  per-step wall -> same samples/s).
- 2026-10-01 re-check: cold-start runs land at 8.4-8.7 s/iter, and a 5 min
  cooldown recovers 8.65 s from a heat-soaked 9.9-10.5 s -- sustained load
  costs ~20 %.  Per-kernel serialized time is up ~17 % (359 vs 308 ms/step)
  with an IDENTICAL kernel mix and call counts (cholesky 44 calls/step,
  32 solver iterations/step, same share per family), so the gap vs the
  09-30 best is device clock/thermal state, not workload or code (a
  warm-device run on 2026-10-02 landed back at 7.0 s/iter).  The
  compute engine sits at ~97 % busy during the bench (GPU-bound); desktop
  compositing (dwm and whatever the desktop shows) holds ~20 % of the shared 3D engine in
  both sessions -- constant contention, not the variable.  Read every
  number in this file with a +-20 % session error bar.

  RE-ATTRIBUTION (2026-10-02): the slow historical sessions ran with
  co-occurring GPU apps (video playback et al.), making
  BUS CONTENTION -- not thermals -- the leading variance driver. Measured
  the same day: a background torch.xpu copy hog at ~50 % duty (~85 GB/s
  while active) drops the bench from 14,453 to 8,000 env-steps/s (-45 %);
  heavier overlap reaches -65 %. A video-grade background load plausibly
  costs 10-30 %. Consequence: bench/train on a QUIET desktop -- closing
  GPU-consuming apps is worth more than any remaining code candidate, and
  the quiet-desktop numbers (~15.1-15.9k eps) are the true baseline. The
  paired A/Bs in this file compare arms under similar contamination and
  stay valid.

## Where env.step goes (census)

Current (launch-fusion suite + flat-kernel v2 + command-graph batch
replay, measured 2026-09-30, 4096 envs):

- median ≈ 210 ms/env.step (paired A/B: graph replay worth
  −35.0 ± 3.9 ms/step, 5 pairs, t = −20)
- ~938 kernel launches/step; since 2026-10-02 the whole substep
  (collision -> constraints -> solve -> integrate) replays as ONE graph
  submission (4/step) and the solver batch is absorbed inside it, so
  per-kernel submits are down to a handful of non-graph launches per step
- 7 queue drains/step: 4 solve-end convergence polls + lite forward +
  Bvh refit + sensor finalize; ~7 % of wall waiting in drains.  Call-site
  attribution (scripts/probe_drain_sites.py) shows exactly these four
  sites; a redundant pre-drain in the runtime_patch Bvh wrapper (the
  overlay's Bvh.__init__/refit already drain for sycl instances) was found
  and removed on 2026-10-01 -- before that the census counted 8.

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
| adaptive njmax (the task cfg pinned njmax=1500 -> 1504 padded vs peak real nefc 46; the runtime patch measures baseline nefc via mj_forward and shrinks to max(nefc\*16, nq\*8, 96) = 168 -> pad 176, never raising a caller-set value; `MJLAB_SYCL_NJMAX=N`/`=0` knob; landed inside the launch-cache commit) | back-to-back A/B 2026-10-02 @4096 (3 iters): rollout 11.22 -> 6.18 s, ppo unchanged (0.83/0.84), total 12.05 -> 7.01 s/iter (1.72x; 8,765 -> 15,918 env-steps/s); also frees ~0.9 GB of `efc.J` @4096 | kept (default-on since 2026-09-30; supersedes the 2026-09-09 "33x idle" audit that sat in "known device-side costs". Shrinking below ~128 is a dead end: flat/fused kernels already iterate min(nefc, njmax), only full-buffer zeroing scales with njmax (<1 %), and an undersized buffer drops constraints silently -> NaN -- microduck rolls at nefc ~122) |
| oneMKL batched potrf/potrs for the dense cholesky family (`wp_sycl_mkl_chol_solve` export, staged a->L/b->x copies, strided USM batch; correctness vs numpy PASS at ULP level, 1.9e-07 L / 4.1e-07 x) | isolated 2026-10-02: 137.5 ms/call @batch 4096, 17.2 @512 (linear in batch -- per-matrix submission internally) vs the flat kernel's ~1 ms; end-to-end probe 273 -> 3,275 ms/step | dead end, fully reverted same day -- vendor batched LAPACK is built for large n x modest batch, the opposite of our 20x20 x 4096 shape; rescheduling must be hand-written (see docs/optimization_ideas.md T1c/T2) |
| sub_group-cooperative LLT (`wp_sycl_sg_chol_solve` in the DLL: 16-lane column-oriented bit-exact factorization; v1 USM+global-fence barriers, v2 SLM+local barriers; correctness vs numpy PASS at ULP level, 1.7e-07 L / 3.8e-07 x) | 2.28 / 2.16 ms/call @4096 vs the flat kernel's ~1.0 ms -- 2x slower in both schedules | dead end, reverted -- LLT at n=20 is sync-bound when parallelized intra-world (~10 MACs per lane between 2 group-wide syncs x 40 columns); the flat zero-sync fully-unrolled kernel is the schedule floor. Revisit only for n > 64 models. With T1d this closes the cholesky family |
| ghost-iteration slot cost (gates T4 world compaction): paired `MJLAB_SYCL_ITER` 8 vs 7, swapped-order A/B 2026-10-02 under the substep graph | 14,839/14,635 vs 14,932/14,682 env-steps/s -- both differences inside +-0.6 % run noise: one full solver slot (~30 kernel nodes, done-guarded, graph-replayed) costs ~0 wall | T4 is dead before being built -- compaction's entire win is ~3.7 ghost slots that now measure free; the graph replay closed the door the compaction design was meant to open |
| native mv+jv rewrite (`wp_sycl_mv_jv` in warpsycl.dll: 32-lane row-block native SYCL, bit-exact per-element k-ascending dots, done-guarded, nefc-clamped; unit-verified bit-identical to the warp kernel on random data incl. done worlds) | kernel 0.62 -> 0.25 ms/launch (2.5x); swapped-order paired A/B 2026-10-02 @4096: 14,777/14,719 -> 15,493/15,216 env-steps/s (**+3.4-4.8 %**); all gates pass, physics vs cpu 3.308e-06 unchanged | kept (default-on, `MJLAB_SYCL_NATIVE_MVJV=0` kill switch; the DPC++-vs-warp codegen gap from the stream diagnosis is real and harvestable kernel by kernel) |
| native JTDAJ rewrite (`wp_sycl_jtdaj`: h = qM + J^T D' J, 32-lane row-block native SYCL, bit-exact per-element k-ascending dots with the same Dk-zeroing rules; unit-verified bit-identical incl. non-QUADRATIC states, done worlds, nefc edge cases) | serialized 277 -> 246 ms/step; swapped-order paired A/B 2026-10-02 @4096: 17,697/17,351 -> 18,491/18,279 env-steps/s (**+4.5-5.4 %**) on a quiet desktop; all gates pass, physics vs cpu 3.308e-06 unchanged | kept (default-on, `MJLAB_SYCL_NATIVE_JTDAJ=0` kill switch; second harvest of the codegen gap -- the warp flat kernel's 1.6M tiny-item grid cost more in-pipeline than its serialized attribution suggested) |
| native solver-cholesky rewrite (`wp_sycl_chol_solve`: one item per world, LLT + forward/back substitution, per-size template instantiation (4..32) so the N-loops stay fully unrolled, done-guarded, changed/lvalid skip preserved; NOT bit-identical to the warp kernel -- sycl::sqrt/division rounding differs at ULP level, gated by the 1e-5 contract) | swapped-order paired A/B 2026-10-02 @4096: 18,420/18,201 -> 18,869/18,991 env-steps/s (**+2.4-4.3 %**); all gates pass, physics vs cpu 3.308e-06 unchanged | kept (default-on, `MJLAB_SYCL_NATIVE_CHOL=0` kill switch; third harvest of the codegen gap) |
| native set-const chol_fs + incremental Hessian hinc (`wp_sycl_chol_fs`, `wp_sycl_hinc`; chol_fs's tile anchor is read DEVICE-side so the route is host-sync-free inside graph recording -- a host read in the route aborted the step graph). hinc's route re-issues the gradient launches (zero_grad_dot, grad) verbatim and replaces ONLY the Hessian-delta kernel | swapped-order paired A/B 2026-10-02 @4096 quiet: hinc +1.7-4.0 %; chol family (solve+fs) +-noise ~+1 %; FULL native suite (5 kernels) vs all-off: 16,875/16,856 -> 18,796/18,505 env-steps/s (**+9.8-11.4 %**) | kept (default-on, per-kernel kill switches). LESSON logged: the first hinc route replaced the whole _update_gradient_incremental and silently SKIPPED the gradient computation -- a fake +23 % that was broken physics; function-level seams must replicate the function's full contract. Gates do not exercise the incremental path (they passed either way) -- kernel-unit + contract review caught it |
| native qfrc_constraint + fused jaref (`wp_sycl_qfrc_constraint`: J^T @ force, k-ascending per dof row; `wp_sycl_jaref`: the fused jaref+zero-ahead contract verbatim -- zero-ahead bookkeeping on (world, 0), rows >= nefc skip) | qfrc unit-verified bit-exact (0.0); jaref bit-exact except the Jaref += alpha*jv accumulation (ULP 4.8e-7, FMA contraction); swapped-order paired A/B 2026-10-02 @4096 fully-quiet: 21,463/21,293 -> 22,060/21,800 env-steps/s (**+2.4-2.8 %**); all gates pass (patched stack 1.863e-08) | kept (default-on, `MJLAB_SYCL_NATIVE_QFRC` / `MJLAB_SYCL_NATIVE_JAREF` kill switches; seams live inside the existing interceptor branches -- no new function-level contracts) |
| native fused quad_gauss (`wp_sycl_quad_gauss`: the _quad_gauss_fused contract verbatim incl. the elliptic-cone branch and the (world,0) gauss fold; unit-verified BIT-EXACT (0.0) on the pyramidal path) | swapped-order paired A/B 2026-10-02 @4096: 21,454/21,869 OFF vs 21,837/21,811 ON -- wall-neutral (+1.8 % / -0.3 %, mean +0.7 % inside noise) | kept (wall-neutral, bit-exact, own kill switch MJLAB_SYCL_NATIVE_QUAD -- same standard as the sense-drain merge). Lesson consistent with T2: at (nworld, njmax) the grid already saturates the device, so only the codegen delta (~1.2x) remains and it lands inside the noise floor |
| native update_constraint_efc + deterministic cost fold (`wp_sycl_efc_force` + `wp_sycl_cost_fold`: force/state/change-tracking bit-identical incl. the elliptic cone; per-row cost partials replace the same-address atomic storm, folded serially -- cost now deterministic where the atomic order was run-to-run nondeterministic; unit-verified incl. true elliptic rows) | swapped-order paired A/B 2026-10-02 @4096: 20,098/19,949 OFF vs 20,262/20,442 ON (**+0.8 / +2.5 %, mean +1.6 %**); all 6 gates green | kept (default-on, `MJLAB_SYCL_NATIVE_EFC=0`). NOTE: the 2026-10-02 T3 downgrade used a wrong launch count (9 vs the actual 36/step) -- with the corrected budget this kernel was ~8 % of step, and the measured win landed at the low end of the corrected 2-4 % estimate |
| native gauss_cost + linesearch teardown (`wp_sycl_gauss_cost`: bit-exact single-writer port; `wp_sycl_ls_teardown`: best_alpha+qacc_ma port, ULP-level on alpha via log/exp rounding) | swapped-order paired A/B 2026-10-02 @4096 (23k-class machine state): 23,402/23,086 OFF vs 23,458/23,418 ON -- wall-neutral (+0.2 / +1.4 %, mean +0.8 % inside noise) | kept (wall-neutral standard, `MJLAB_SYCL_NATIVE_GAUSS` / `_LSTD` kill switches) -- closes the visible-kernel native rewrite series (11 natives total) |
| collision workspace pool + USM pool trim (`MJLAB_SYCL_WS_POOL`; convex_narrowphase's GJK/EPA scratch -- (naccdmax=143360, 112/224/256) buffers -- was allocated and dropped PER CALL: ~18 GB of churn over 12 steps at 4096 envs. Now shape-keyed checkout/reuse; plus wp_sycl_pool_trim + pool_stats/ps_hist exports and scripts/probe_memory.py) | memory: pool free-lists 2,474 -> 44 MB, live 5,029 -> 2,871 MB, RSS steady ~4.3 GB with no churn spikes (was 2 GB swing between steps); deterministic bit-equal and physics 1.863e-08 UNCHANGED; perf wall-neutral (paired A/B -1.3 % / +1.1 %) | kept (default-on). NOTE: a pool trim under live command graphs crashed the replay (baked pointers) -- trim now frees graphs first (they re-arm in 2 calls). The remaining ~2.4 GB of collision workspaces is functional (sized nconmax x nworld); shrinking it needs the njmax-style overflow audit per task |
| `_mv_jv_fused` schedule change (the 1-item-per-world mv+jv kernel re-gridded two ways: row-per-item (world, nv+njmax) and 4-rows/item with independent accumulator chains; arithmetic per element unchanged -> bit-identical by construction) | kernel steady-state 0.56 / 0.59 ms/launch vs 0.47-0.77 for the original -- a wash across all three schedules | dead end, reverted -- the kernel moves ~21 MB/launch (nefc rows of efc_J + qM) at ~43 GB/s effective. NOTE (2026-10-02 microbench): the device's ACHIEVABLE bandwidth is ~80-88 GB/s (torch.xpu copy 87-88, read-only sum 79-81, vs ~136 GB/s theoretical) -- so ~2x pattern headroom exists in principle, but closing it needs lane-coalesced J reads, which requires cross-lane reductions (breaks bit-exactness) or the row-block mapping (measured wash above: short-dot latency dominates). Only shrinking or sharing the bytes can move it within the accuracy contract |
| flat-kernel v2: per-element JTDAJ dot, unrolled dense cholesky, LLT skip on unchanged constraints | serialized kernel time 435 -> 308 ms/step; env.step median ~330 -> ~270 ms | kept (same `MJLAB_SYCL_FLAT_JTDAJ` switch) |
| solver scratch reuse (SolverContext/step_size_cost/nsolving across solves) | launch-cache hit rate 60 -> 81 %, slow rebuild path halved; +2-3 % throughput | kept (`MJLAB_SYCL_SOLVER_CTX`) |
| command-graph batch replay (solver 8-iteration batch as one submission) | paired A/B −35.0 ± 3.9 ms/step (−12.7 %) | kept (`MJLAB_SYCL_GRAPH`) |
| whole-substep command graph (`mjwarp.step` — collision -> constraints -> solve -> integrate — replays as one queue submission after a plain count and a verified record; scoped mode runs the solver batch plain while the substep establishes so the outer graph absorbs it and launch counts match; the poll drain is skipped during capture — queue waits are illegal there and the poll count is constant with poll_every >= cap — and the step wrapper drains once at the substep boundary instead, keeping one sync/substep) | swapped-order paired A/B 2026-10-02 @4096: 13,552/13,834 -> 15,108/14,998 env-steps/s (**+8-11 %**, 8.2 -> 7.5 s/iter); all gates pass, physics vs cpu 3.3e-06 unchanged | kept (default-on, `MJLAB_SYCL_STEP_GRAPH=0` kill switch; `MJLAB_SYCL_GRAPH=0` still kills every graph) |
| iteration-kernel merges (prepare_gauss→prepare_quad, solve_done→search_update) | wall-neutral with graphs on (±4.5 ms), fewer kernels/launches | kept (same fusion switches) |
| selective set_const recompute on resets (fall → randomize event) | recompute_constants 50 → 10 ms/step with active policies; +17 % end-to-end | kept, default-on (`MJLAB_SYCL_FUSED_SET_CONST`) |
| torch thread sweep re-run under the new pipeline (2/4/8) | paired A/B: no significant wall difference | default 2 unchanged |
| selective kinematics/com_pos/crb for set_const recompute (copied prep kernels with world-id indirection) | unit A/B equivalent to 2e-4, but only ~0.8 ms/call at 4096 envs (4 reset ids): the cut stages are launch-bound at small nproc and factor_m stays all-world | reverted -- 400 lines of copied kernels not justified below the noise floor |
| lite forward as one command graph (`graph_batch.run_sequence`) | ~15 submits/step -> 1; wall-neutral | kept (`MJLAB_SYCL_GRAPH`) |
| sense() drain merge (sensor context already drains in finalize) | 8 -> 7 drains/step, no measurable wall change | kept |
| Bvh pre-drain removal (runtime_patch wrapper; the overlay's Bvh.__init__/refit already drain sycl instances) | 8 -> 7 drains/step as designed; paired A/B old-new +2.4 ± 14.4 ms/step (n=5, ns) | kept (wall-neutral, one less sync) |
| solver iteration cap 8 -> 6 (`MJLAB_SYCL_ITER=6`) | -3.7 % pre-step-graph (2026-09-30); re-measured 2026-10-02 under the substep graph: 15,077 vs 14,876 env-steps/s (+1.4 %, noise) -- the graph replay made empty solver slots ~free and the knob's value evaporated | kept as a knob, default remains 8: no throughput case, and the cap would only matter for the 22 % of worlds that hit it |
| poll batch matched to the smaller cap (`MJLAB_SYCL_POLL_EVERY=6` with ITER=6) | no win over ITER=6 alone (extra guard batches again), consistent with the small-cadence verdict above | dead end |
| kernel args by-value capture (codegen) | ~1 % (noise); DPC++ also requires const kernel lambdas | dead end, reverted |
| 8192 envs | +2.7 % | not worth wall 2x |
| solver micro-kernel fusion (Route C) | infeasible here: fusion points need intra-world sync (barriers) which must never be added on the iGPU; only tiny per-world scalar merges remain (~1-2 %) | not viable, documented |
| viewer rendering | GL runs on the same iGPU as physics (hardware GL confirmed); Windows KnownDLLs block Mesa software-GL override; real-sim viewers cap at ~8 updates/s (per-step fixed cost) | structural |

## Known device-side costs (out of adapter reach)

- Kernel time leaders (serialized ranking @4096): update_constraint_efc,
  cholesky solves, linesearch family (jv/prepare_quad/jaref), flat JTDAJ/contact.

## Hardware context (web, for wall-clock planning)

- Same task, 4096 envs: RTX 5080 ~1.13 s/iter (community course,
  https://forum.d-robotics.cc/t/topic/35668) -> ~17x faster than this iGPU.
- PassMark GPU compute: Arc 130V ~2.2k vs RTX 5090 ~24.4k ops/s (~11x);
  memory bandwidth GDDR7 ~15-18x LPDDR5X-class iGPU.
- Wall-clock projection (tricks ~1000 iters, gaits 4000-6000 iters):
  this iGPU ~5.5 h / 21-32 h; RTX 5090-class ~12 min / 50-75 min.

## Memory map (measured 2026-10-02, 4096 envs, microduck velocity)

Lunar Lake has unified memory (no discrete VRAM): "GPU memory" and system
RAM draw from the same LPDDR5X pool. Full training loop (probe_training_memory.py),
peak after 2 PPO iterations: **RSS ~7.8 GB** decomposed as:

| component | size | notes |
|---|---|---|
| process baseline (python/torch/warp + JIT) | ~1.6 GB | not reclaimable |
| warp USM pool (collide workspaces + Data + solver scratch) | ~2.9 GB | after the workspace pool fix; was 5.0 GB live + 2.5 GB free-list churn |
| torch.xpu pool (PPO rollout/policy/opt states) | 572 MB reserved, 77 MB live | `torch.xpu.empty_cache()` reclaims 472 MB |
| **torch CPU transients retained by the allocator** | **~3.4 GB** | Python heap delta is only ~5 MB (tracemalloc); the churn is C-level |

The dominant CPU transient is `bam.mjlab._dof_friction_fo` (BamActuator's
friction term): two `torch.zeros_like(efc_force)` (4096x168, 2.6 MB each)
per act call -- ~1.25 GB of alloc/free churn per 2 iterations. The two
zeros_like are `torch.where` fillers; `torch.where(cond, x, 0)` (scalar
fill) is bit-identical and removes both allocations -- recommend upstream
in the bam task package. Remaining unattributed transients are in the
same class (per-step reward/obs temporaries kept by the CPU allocator).

Levers by ROI: the collision workspace pool + USM trim (landed: churn
-18 GB/12 steps, free lists 2.47 GB -> 44 MB); the bam scalar-where fix
(upstream, ~1.25 GB churn); `torch.xpu.empty_cache()` between phases
(472 MB); shrinking nconmax-driven workspaces needs the njmax-style
overflow audit per task (silent contact drops -> NaN).
