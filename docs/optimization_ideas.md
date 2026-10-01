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
device-time share. Three routes, in committed order:

### T1d — oneMKL batched `potrf_batch` + `potrs_batch` (try first)

- **Mechanism.** Replace the per-world scalar LLT with oneMKL's SYCL-domain
  batched LAPACK: one call factorizes and one call solves all 4096
  20×20 SPD systems with vendor-tuned kernels.
- **Feasibility (verified, not assumed).** oneAPI 2025.3 on this machine
  ships `oneapi/mkl/lapack/lapack.hpp` with `potrf_batch` / `potrs_batch`
  (float, grouped pointer-array interface) and `mkl_sycl.lib` to link into
  `warpsycl.dll`; call it from Python through the same ctypes seam as the
  graph API (`graph_batch._api()` pattern). Layout facts checked: our L is
  `(nworld, nv_pad, nv_pad)` contiguous USM, so per-world pointers are
  `base + w·nv_pad²` — build the 4096-pointer device array once per scratch
  lifetime (the `_CHOL_STATE` cache already keys buffer lifetimes). Row-major
  lower-L ⇔ column-major upper-L on the same bytes, so `uplo=U` on the
  column-major view produces exactly our lower-triangular layout with **zero
  copies/transposes**; `n = nv = 20`, `lda = ldb = nv_pad = 20` for microduck
  (nv is a multiple of 4 here; guard `nv == nv_pad`, else fall back).
  `h = qM + JᵀDJ` is SPD (qM SPD, JᵀDJ PSD), which is potrf's contract.
- **Safety.** Kill switch: route only when `nv == nv_pad` and a new
  `MJLAB_SYCL_MKL_CHOL` is on; any query/scratchpad failure falls back to the
  flat kernel (same seam the LLT-skip already uses). Failure mode on
  degenerate (non-PD) h: oneMKL leaves garbage in the factor + info — the
  current kernel's failure mode there is NaN from `sqrt(negative)`, so this
  is not a new class of breakage.
- **Numerics.** NOT bit-identical: vendor accumulation order differs →
  ULP-level (~1e-7 relative) differences per factorization. The Newton
  solver is tolerance-driven (improvement/gradient tests at ~1e-6 scale) and
  self-correcting, so ULP-level factor differences stay far below both the
  1e-5 physics gate and solver tolerance. Validation: physics gate + a
  200-iteration paired training A/B (reward curves within run noise).
- **Expected gain (estimate).** 3–5× on the chol family → ~10–14 % of device
  time → **~5–8 % end-to-end**. Effort: 2–4 days incl. DLL rebuild and gates.

### T1c — native sub_group-cooperative LLT in the backend

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
the effort rivals T1c with less certain gain. Only worth it if both T1d and
T1c fail.

## T2 — un-serialize `_mv_jv_fused` (row-block parallelism)

- **Mechanism.** The mv+jv fusion (1 launch, 1 work-item/world doing ~1.3k
  serial MACs) was a **launch-count** optimization. Under solver graph replay
  the launch overhead is already amortized, so the trade-off has flipped:
  give the work back its parallelism — grid `(nworld, row_blocks)` with each
  item computing a block of jv rows (`J[row,:] · search`, k-ascending) and
  one item family for mv.
- **Feasibility.** Pure warp language, replaces `_mv_jv_fused` inside the
  existing `fused_linesearch` interceptor; dims stay static (njmax-based), so
  graph replay and the launch cache re-key once and continue.
- **Safety.** Kill switch already exists (`MJLAB_SYCL_FUSED_LINESEARCH=0`
  restores upstream). No new state, no cross-item dependencies.
- **Numerics.** Bit-identical by construction: every output element remains
  a single sequential dot in one work-item; only the grid changes.
- **Expected gain (estimate).** 2–3× on jv/mv (0.47–0.77 ms × 8–16/step) →
  **~2–4 % end-to-end**. Effort: ≤1 day including A/B.

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

## T3 — eliminate `update_constraint_efc` cost atomics

- **Mechanism.** Each world's ~46 efc rows `atomic_add` their cost
  contribution onto the same `ctx_cost[worldid]` address. Replace with
  per-row partial stores into a scratch `(nworld, njmax)` buffer and fold the
  reduction into the immediately-following `(nworld, 1)` `gauss_cost` kernel
  (one less launch, too). Needs a sycl rewrite of these two upstream kernels
  — the same copy-and-own pattern already used throughout.
- **Feasibility.** Both kernels are small and already understood; seam is the
  existing launch interceptor; scratch rides `SolverContext` (solver_ctx
  reuse keeps graph/cache keys stable).
- **Safety.** Kill switch; falls back to the upstream pair. Contention today
  is same-address same-world only — the rewrite removes it, it does not
  move it.
- **Numerics.** Today's atomic sum order is **non-deterministic run-to-run**;
  a fixed ascending-row reduction is strictly more reproducible. Differences
  vs any particular atomic interleaving are ULP-level. One honest edge: a
  ULP change can flip a borderline `solve_done` tolerance comparison for an
  individual world (iteration count differs) — physically equivalent, but
  validate with the physics gate + a paired training A/B, not just unit
  equality.
- **Expected gain (estimate).** 0.43 ms × ~9/step partially → **1–3 %
  end-to-end**. Effort: ~1 day.

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

## Suggested order

1. **T1d** (oneMKL batched chol) — best gain/effort, verified API, 2–4 d.
2. **T2** (jv row-block) — ≤1 d, bit-identical.
3. **T5** (substep graphs) — safe-by-construction arming, 1–2 d.
4. **T3** (efc atomics) — ~1 d, deterministic-order bonus.
5. **T1c** (sub_group LLT) if T1d disappoints or gates demand bit-exactness.
6. **T4** (compaction) — last; the biggest and only genuinely intricate one.

Every step gates through `mjlab-sycl-test` plus a paired bench A/B
(`performance.md` reproduce block), and lands default-off behind its own
`MJLAB_SYCL_*` switch.
