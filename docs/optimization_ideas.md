# Kernel-level optimization candidates (argued, not yet attempted)

Companion to [`performance.md`](performance.md): that file archives what was
measured and kept; this one records **candidate directions** with the
feasibility / safety / numerics argument worked out before any code is
written, plus the facts measured while arguing (2026-10-02, Arc 130T,
microduck velocity @4096 envs). When a candidate is attempted, its verdict
migrates to `performance.md`'s attempts table and the entry here shrinks to
a pointer.

Numerical contract used throughout (the repo's existing gates):

- backend e2e gate: **bit-exact vs cpu**;
- mujoco_warp physics gate: **~1e-5 vs cpu** (historically max |sycl−cpu| ≈ 3e-06);
- every candidate ships behind an `MJLAB_SYCL_*` kill switch, default-off
  until gated, so a revert is an env var, not a rollback.

## Fresh measurements this analysis is built on (2026-10-02)

`scripts/probe_kernel_times.py --num-envs 4096 --steps 4` (graphs off),
272–290 ms/step serialized, 108 kernels:

| kernel (pre-fusion attribution, see caveat) | ms/launch | launches/step | grid | reading |
|---|---|---|---|---|
| `_make_cholesky_solve_kernel` (flat LLT+solve) | ~1.0 | ~9 | 4096 | 1 work-item/world, latency-chain bound |
| `_mv_jv_fused` (attributed to `linesearch_jv_fused`) | 0.47–0.77 | 8–16 | 4096 | 1 work-item/world serial loops |
| `_make_cholesky_factorize_solve_kernel` (set-const path) | 2.4–2.7 | ~2 | (4096,1) | worst perf/launch on the board |
| `_make_flat_kernel` (JTDAJ) | 3.9 | 1 | (4096,400) | ~40 GFLOP/s — the healthy baseline |
| `update_constraint_efc` | 0.43 | ~9 | (4096,168) | ~46 atomic_add/row onto one address per world |

**Attribution caveat (methodology):** the probe hooks `wp.launch` OUTSIDE the
interceptor chain, so fused executions are timed under the pre-fusion kernel
name (e.g. `_mv_jv_fused` shows up as `linesearch_jv_fused`). Times are real;
names are not. Identity was verified via the launch cache's inner keys: all
five fusions (`_mv_jv_fused`, `_jaref_zeroahead`, `_quad_gauss_fused`,
`_ls_teardown_fused`, `_search_done_fused`) are live. Check the cache keys
before optimizing "a kernel" by name.

**ALU-utilization argument for the dense-linear-algebra family:** the
cholesky pair moves ~20 GFLOP/s at ~1 ms/launch against a ~1.5 TFLOP/s-class
device — under ~2 %. The cause is structural: one work-item per world walks a
serial dependency chain through an L that (400 floats at nv=20) cannot stay
in registers, so the chain pays private-memory latency. The fix must
restructure the schedule, not the compiler flags (see the dead end below).

## Measured while arguing: `max_unroll` is load-bearing (T1a — dead end)

Hypothesis: `max_unroll=64` over-unrolls the 20×20 LLT and spills; smaller
unroll would help. Measured 2026-10-02 (2-step probe, 4096 envs):

| max_unroll | chol solve ms/launch | chol factorize+solve ms/launch | ms/step |
|---|---|---|---|
| 64 (current) | 0.97–1.03 | 2.4–2.7 | ~273 |
| 8 | **96.7** | **323** | ~8,935 |

94× worse. Full unrolling keeps the loop bounds constant-folded and the hot
values in registers; rolling the loops collapses it. Verdict: **dead end —
compiler knobs are exhausted; gains must come from rescheduling the
algorithm** (T1c/T1d below). (Reverted same day; kernel cache is
content-hashed so both variants coexist on disk.)

---

## T1 — restructure the dense cholesky family (top priority)

Scope: `_make_cholesky_solve_kernel` (~9 launches/step, ~1 ms), the set-const
`_make_cholesky_factorize_solve_kernel` (~2/step, ~2.5 ms), i.e. the ~18 %
device-time share. The vendor-library route (T1d) is measured dead; the
hand-written reschedule (T1c) is the live path.

### T1d — oneMKL batched `potrf_batch` + `potrs_batch` — **measured dead end (2026-10-02)**

Implemented, correctness-gated (ULP-level PASS vs numpy: 1.9e-07 on L,
4.1e-07 on x, residual 1.2e-06), and measured a **135x regression**: 137.5
ms/call at batch 4096 (17.2 at 512 — linear in batch) vs the flat kernel's
~1 ms; end-to-end probe 273 -> 3,275 ms/step. oneMKL's strided batched
LAPACK submits per-matrix internally — it is built for large-n x modest-batch,
the exact opposite of our 20x20 x 4096 shape. Fully reverted; the verdict
lives in `performance.md`'s attempts table. What the attempt validated and
survives for T1c: the DLL rebuild + MKL link recipe (REBUILD.md), the
ctypes export seam, the a->L / b->x staging-copy requirement (the
incremental Hessian accumulates onto `a`), and the row-major/upper-L
layout equivalence.

- **Mechanism.** One `sycl::sub_group` (16 lanes on Xe2) per world; L
  distributed across lanes (400/16 = 25 floats per lane — **fits in
  registers**, which is exactly what the 1-item/world version cannot do);
  per row i, lanes compute the (i, j) elements in parallel and the pivot row
  is broadcast with sub_group collectives (hardware shuffles, no SLM, no
  work-group barriers).
- **Feasibility.** We own `sycl_runtime.cpp` and its rebuild pipeline
  (`warp_backend/README.md`); entry-point seam identical to T1d. Critical
  distinction: the archived `wip-tile-cooperative` dead end used
  **work-group** barriers — the documented iGPU killer. sub_group collectives
  are convergent hardware ops with no such cost; this is a different
  mechanism and does not inherit that verdict.
- **Safety.** Same kill switch/fallback seam as T1d. sub_group size is
  queried at runtime; kernels specialize for 8/16 and fall back to the flat
  kernel on unexpected sizes.
- **Numerics.** Bit-identical by construction: each (i, j) element's dot
  over k stays k-ascending **within a single lane** — the same op sequence as
  the current kernel, only the schedule across lanes changes. Verify with the
  bit-exact backend gate + physics gate.
- **Expected gain (estimate).** Similar range to T1d (register-resident L
  removes the private-memory chain); keep whichever of T1d/T1c measures
  better, they share the seam.
- **Why T1d first:** it buys vendor tuning for ~a quarter of the effort;
  T1c is the fallback and the bit-identical option if gates ever complain.

### T1b — register-tiled blocked LLT in pure warp language — **deferred**

Possible in principle (4×4 register tiles), but any blocking that regroups
the k-sums changes FP accumulation grouping (no longer bit-identical), and
the effort rivals T1c with less certain gain. Moot: T1c is now measured
dead too.

### T1c — native sub_group-cooperative LLT in the backend — **measured dead end (2026-10-02)**

Implemented natively in `warpsycl.dll` (`wp_sycl_sg_chol_solve`), correctness
gated at ULP level vs numpy (1.7e-07 L / 3.8e-07 x), two schedule variants:

- v1: 16-lane sub-group per world, column-oriented bit-exact factorization
  against USM with global-fence barriers — 2.28 ms/call @4096.
- v2: same schedule staged through SLM with local-space barriers (cheap
  intra-thread sync on Xe2, where a 16-lane group is one hardware thread) —
  2.16 ms/call.

Both ~2x SLOWER than the flat scalar kernel's ~1.0 ms. The lesson is
structural: LLT's dependency chain forces ~n sequential column steps with a
group-wide sync between them; at n=20 each step leaves ~10 MACs per lane —
a ~10:1 sync:work ratio — while the flat kernel's zero-sync, fully-unrolled,
one-item-per-world schedule already hides latency across 4096 independent
items. Intra-world parallelism cannot beat it at this matrix size; the chol
family's ~1.0 ms is its floor. Both variants reverted; the column-oriented
bit-exact schedule and the SLM staging pattern are recorded here for any
future larger-n model (n > 64), where this schedule amortizes.

With T1d and T1c measured, the cholesky family is closed.

## T2 — un-serialize `_mv_jv_fused` (row-block parallelism) — **measured dead end (2026-10-02)**

Implemented both variants (row-per-item over `(nworld, nv + njmax)`, and
4-rows/item with independent accumulator chains for ILP; arithmetic per
element unchanged → bit-identical by construction) and measured a wash:
0.56 / 0.59 ms/launch vs the original's 0.47–0.77. The kernel is
**memory-bandwidth bound**, not schedule bound: ~21 MB actually read per
launch (the nefc rows of `efc_J` plus `qM`) over ~0.5 ms ≈ 43 GB/s on the
LPDDR5X bus, and 4096 worlds already provide more item parallelism than the
device can use. Fully reverted; the lever for this family is reading fewer
bytes (fusing more consumers per J read), not rescheduling. Verdict in
`performance.md`'s attempts table.

## T5 — extend command-graph capture to the whole substep

- **Mechanism.** `graph_batch.run_sequence` is general and proven (lite
  forward already replays as one submission). Capture the per-substep
  collision → kinematics → integrate sequence the same way; the solver batch
  graph remains nested inside. Target: the remaining per-kernel submissions
  behind the ~8 % submit-overhead / queue-backpressure budget.
- **Feasibility.** All physics grids are buffer-sized (static); the existing
  three-state arming (count → record+verify → replay) **auto-disables** on
  any non-static sequence, so attempting it is safe by construction.
  Hazard to design for, not discover: host-side `zero_()`/`wp.copy` inside a
  candidate sequence must be captured or hoisted (the jaref-tail fusion
  already replaced one such memset; audit the remaining ones before arming).
  Drains stay outside the graph — the 7 drain sites are sequence boundaries.
- **Safety.** Same machinery, same kill switch (`MJLAB_SYCL_GRAPH=0`);
  launch-count verification is the guard.
- **Numerics.** Bit-identical: a graph replays the exact recorded kernel
  sequence.
- **Expected gain (estimate).** submit share 8 % → ~2 % → **3–6 %
  end-to-end**, plus reduced host contention. Effort: 1–2 days.

## T3 — eliminate `update_constraint_efc` cost atomics — **downgraded below the noise floor (2026-10-02)**

Fresh arithmetic from the 2026-10-02 kernel ranking: the whole kernel costs
0.43 ms x ~9/step = 3.9 ms/step = **1.4 % of the 273 ms step** — and the
atomics are only part of it (each row also does the elliptic-cone zone
math, per-row state writes, change tracking). Even eliminating every atomic
stall outright bounds the win at <0.7 % end-to-end, under the session noise
bar; the original 1–3 % estimate over-weighted the atomic share. The
rewrite itself (per-row partial stores + folding the reduction into the
`(nworld, 1)` gauss_cost kernel, both kernel copies owned) stays sound —
mechanism, safety and numerics notes below stand — but it only becomes
worth attempting if a future profile shows atomic stalls dominating the
kernel, or as a rider on other work in the same files.

- **Mechanism (when attempted).** Replace each of the five
  `atomic_add(ctx_cost_out, worldid, ...)` sites with a per-row store into
  a `(nworld, njmax)` partial buffer; the immediately-following
  `(nworld, 1)` `gauss_cost` kernel computes
  `cost = 0.5*gauss + sum(partial[:nefc])` instead of `+=` (its
  single-work-item-per-world shape already reads/writes cost non-atomically).
  `changed_ids/changed_count` atomics stay (no same-address contention).
- **Safety.** Kill switch; interceptor seam identical to the other fusions;
  scratch rides SolverContext (stable cache/graph keys).
- **Numerics.** A fixed ascending-row reduction is strictly MORE
  reproducible than today's run-to-run atomic order (ULP-level differences
  either way; a ULP flip can still move an individual world's borderline
  `solve_done` comparison — physically equivalent, validate via the physics
  gate + paired training A/B, not unit equality alone).

## T4 — ghost-iteration world compaction

- **Mechanism.** Mean useful solver iterations are 4.3 of 8; done-guarded
  kernels make ghost iterations cheap but not free (~30 kernel schedules ×
  full grids × ~3.7 ghost iterations/step). The earlier poll-cadence dead
  end lacked the second half: **compaction**. Poll `nsolving` at iteration 4
  (loop_poll already has the cadence machinery); if below a threshold, one
  tiny kernel atomic-appends the active world ids into a dense list; the
  tail iterations launch over the compacted grid with `worldid =
  active_in[slot]` indirection (precedent: the reverted selective-kinematics
  experiment used world-id indirection). Pad the list with a sentinel world
  whose `done` is permanently true so every kernel's existing done-guard
  no-ops the padding — the batch stays a **static graph**.
- **Feasibility.** Requires own copies of the ~10 iteration kernels with one
  indirection line each — mechanical, and the repo has done this scale of
  copy before. The graph re-records once at fixed dim.
- **Safety.** Kill switch; fallback is the current full-grid batch. The
  bookkeeping hazards are enumerable and must be handled explicitly:
  `changed_efc_count` must still be zeroed for ALL worlds (memset semantics
  — the jaref tail note), `solver_niter`/`nsolving` must stay exact, and
  per-solve `init_context` re-initialization must not depend on the ghost
  iterations having run.
- **Numerics.** Bit-identical **by design**: active worlds run the identical
  kernel sequence; done worlds skip exactly the work their done-guards
  already skip. Verify with a per-step state-parity check vs the uncompacted
  path (assert qacc equality for N steps) before any perf claim.
- **Expected gain (estimate).** 5–10 % of the solver pipeline → **2–5 %
  end-to-end**. Effort: 3–5 days. Highest complexity on this page — schedule
  after T1/T2/T5 have landed or failed.

## Rejected / deferred on the accuracy policy

- **`fast_math` on physics modules** (backend knob exists, default off):
  enables reassociation and approximate div/sqrt — by definition not
  bit-stable and not bounded ULP. Even if the 1e-5 gate passes on one
  machine, drift is unquantified across states. Opt-in research knob at
  most, never default. Excluded by the repo's accuracy contract.
- **fp16/bf16 J or h**: same objection, stronger (3 decimal digits).
- **`max_unroll` sweep**: measured dead end (table above).

## Where this leaves the list (2026-10-02 status)

Every kernel-schedule candidate has now been measured except T4:

- **T1d / T1c / T2: measured dead** (vendor library, intra-world parallel
  LLT, jv rescheduling). The chol family's ~1.0 ms is its floor at n=20;
  the jv family is bandwidth-bound.
- **T3: parked** below the noise floor (<0.7 % bound by its kernel's own
  1.4 % share).
- **T5: landed** (+8-11 %, default-on).
- **T4: measured dead before being built (2026-10-02).** The gating
  measurement — a paired `MJLAB_SYCL_ITER` 8 vs 7 swapped-order A/B under
  the substep graph — puts one full solver slot (~30 done-guarded,
  graph-replayed kernel nodes) at ~0 % wall (both runs inside +-0.6 %
  noise). Compaction's entire win was ~3.7 ghost slots, which now measure
  free; and the ITER=6 knob's pre-graph -3.7 % re-measured at +1.4 %
  (noise) for the same reason. The graph replay closed the door the
  compaction design was meant to open.
- Outside kernel schedules, the measured levers that remain are the
  thermal/session error bar itself (+-20 %, larger than every candidate
  here) and the ITER=6 knob (-3.7 % pre-graph; +1.4 % = noise after).
- **The warp codegen bandwidth gap (found 2026-10-02, OPEN -- the biggest
  remaining software lever).** A warp-language streaming-sum kernel over
  256 MB caps at ~42 GB/s (block_dim sweep 32->1024: 30 -> 42, flat after)
  on the same device where torch.xpu's copy kernel streams 87-88 GB/s --
  so the wall is in the warp->SYCL generated code / launch path, not the
  memory system, and we OWN that layer (backend/_src). Item mapping is
  ruled out (row-per-item and linear-stream variants cap identically);
  block_dim helps 40 % then flat. Diagnosed 2026-10-02 with a standalone DPC++ matrix (scripts/bench/
  stream_matrix.cpp; read-only sum + memcpy, 256 MB):

  - device USM: 88-90 GB/s read, 85 GB/s memcpy
  - shared USM (DPC++): 67-70 GB/s read, 43 GB/s memcpy  <- the 2x write
    coherence tax on shared, and ~1.25x on reads
  - warp (shared, no-guard): ~59 GB/s; my guard test 42. Vectorization
    and ILP are NOT factors (scalar loads reach 88 on device USM).

  So the original 42-vs-88 gap decomposes into USM type (~1.25x reads,
  2x writes) plus warp-vs-DPC++ codegen (~1.2x). Vectorization: excluded.
  Revised estimate: warp arrays must stay shared USM (mjlab's WarpBridge
  zero-copies the pointers into torch CPU tensors; device USM breaks
  host visibility model-wide). The feasible slice is SELECTIVE device
  USM for solver-internal scratch never host-read inside a step (ctx.*,
  efc.J): ~1.25x on ~16 % of step time -> ~1.5-3 % e2e, 2-4 days, medium
  risk (bridge/field enumeration must be exact). The 2x memcpy win only
  benefits copy-heavy paths (set_const staging, ~1-2 ms/step). The
  10-20 % jackpot hypothesized earlier is OFF: the shared-USM constraint
  caps the recovery near 1.25x for the read-dominant mix.
- **Bandwidth ceiling, corrected 2026-10-02 by microbenchmark**: the
  device achieves ~80-88 GB/s (torch.xpu copy 87-88 r+w, read-only sum
  79-81; ~65 % of the ~136 GB/s LPDDR5X-8533 x 128-bit theoretical), so
  the jv family's ~43 GB/s effective is NOT the hardware wall -- ~2x
  pattern headroom exists. It is trapped: lane-coalesced J reads need
  cross-lane reductions (breaks bit-exactness) or the measured wash of
  the row-block schedule. L2 residency: MEASURED DEAD
  (2026-10-02, scripts/bench/l2_reread.cpp): whole-buffer re-read
  bandwidth is 170-303 GB/s across S = 4-64 MB with volatile-forced
  reloads -- 2-4x DRAM stream (~80-88) and beyond the bus theoretical
  (~136) at larger S, i.e. repeated reads already hit cache/memory-level
  parallelism. Pinning J for the solver's re-reads has no DRAM traffic
  left to save. The precision lever (fp16/bf16 J) halves the bytes but
  violates the accuracy contract.

Every step gates through `mjlab-sycl-test` plus a paired bench A/B
(`performance.md` reproduce block), and lands default-off behind its own
`MJLAB_SYCL_*` switch.
