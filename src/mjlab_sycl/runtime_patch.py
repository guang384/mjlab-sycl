# SPDX-License-Identifier: Apache-2.0
"""Runtime patch that runs mjlab/mujoco_warp physics on the Intel SYCL device.

Physics (warp arrays) live on the ``sycl`` device while every torch tensor
stays on ``cpu``: the SYCL device allocates USM shared memory, which
``wp.to_torch`` wraps zero-copy. Kernel submissions are asynchronous, so the
queue is drained wherever host code reads or writes USM that kernels touch:
the solver's convergence poll (once per solve), the forward/reset/recompute
boundaries, and sense (via the sensor context's finalize, or the boundary
itself for bare callers).  Host-side Bvh refit drains too.

Also installs the barrier-free flat rewrites of mujoco_warp's tiled hot
kernels (see flat_kernels.py) and verifies the vendored SYCL backend overlay
is in sync with this package before anything runs (see install.py) -- a
``uv sync``/``uv run`` wipes the overlay, and a stale one fails here with a
clear remediation instead of a cryptic missing-device error later.
``patch_simulation_for_sycl()`` is called right after ``wp.init()`` by every
bundled entry point (train/bench/train_viewer); custom entries and probes
call it the same way.
"""

from __future__ import annotations

import os

import warp as wp

from mjlab_sycl import flat_kernels
from mjlab_sycl import fused_linesearch as _sycl_fused_ls
from mjlab_sycl import fused_solver as _sycl_fused_solver
from mjlab_sycl import fused_tree as _sycl_fused_tree
from mjlab_sycl import install as _sycl_install
from mjlab_sycl import launch_cache as _sycl_launch_cache
from mjlab_sycl import loop_poll as _sycl_loop
from mjlab_sycl import skip_empty as _sycl_skip_empty
from mjlab_sycl import solver_ctx as _sycl_solver_ctx
from mjlab_sycl import act_fuse as _sycl_act_fuse
from mjlab_sycl import graph_batch as _gb


_PATCHED = False


def patch_simulation_for_sycl() -> None:
  """Physics on the sycl device, torch on cpu, queue drained at host-read points.

  The sycl device reports is_cpu=True for torch/numpy interop (USM shared
  memory, zero-copy views), so only components that allocate *warp* arrays
  from the scene's device string need re-pointing to sycl: the SensorContext
  (render context feeds BVH refit kernels) and RayCastSensor ray buffers.

  Idempotent: a second call is a no-op, because re-wrapping would stack the
  drained() wrappers (double sync at every boundary) and capture the already-
  patched functions as originals.
  """
  global _PATCHED
  if _PATCHED:
    return

  # Hard guard, not a warning: a missing/stale backend overlay silently
  # corrupts physics (or makes the device vanish) — abort before iteration 0
  # with the exact remediation instead.
  _sycl_install.ensure_overlay_synced()
  _PATCHED = True  # past here a retry would stack wrappers -- commit

  from mjlab.sim import sim as sim_mod
  from mjlab.sensor import raycast_sensor as rs_mod
  from mjlab.sensor import sensor_context as sc_mod

  def drain():
    # No-arg synchronize_device() targets the ambient (cpu) device.
    wp.synchronize_device("sycl")

  orig_init = sim_mod.Simulation.__init__
  orig_step = sim_mod.Simulation.step
  orig_forward = sim_mod.Simulation.forward
  orig_reset = sim_mod.Simulation.reset
  orig_sense = sim_mod.Simulation.sense
  orig_recompute = sim_mod.Simulation.recompute_constants

  def init(self, num_envs, cfg, model, device):
    # Adaptive njmax: the default velocity env cfg sets njmax=1500 which
    # wastes 99% of solver work-items for small robots (actual nefc ~14-106).
    # Measure the baseline nefc with mj_forward, then scale it up for
    # contact-heavy transients -- formula and rationale below.  Never raises
    # a caller-set value.  Override with MJLAB_SYCL_NJMAX=N (force) or
    # =0 (disable).
    import mujoco as _mj
    njmax_override = os.environ.get("MJLAB_SYCL_NJMAX", "")
    if njmax_override == "0":
      pass  # explicitly disabled
    elif njmax_override:
      cfg.njmax = int(njmax_override)
    else:
      # measure baseline constraint count from initial pose
      _d = _mj.MjData(model)
      _mj.mj_forward(model, _d)
      baseline_nefc = max(_d.nefc, 1)
      # The static baseline can't predict contact-dense transients (falls,
      # rolls): microduck walks at nefc=14 but rolls at 122 (~9x).  Use
      # baseline * 16 (joint limits + contact spike headroom) floored at
      # nq * 8 (every joint limit active + contact rows), min 96.  This is
      # conservative: it reduces the 1500 default by ~10x for small robots
      # while leaving large models (humanoid, nv=50+) nearly unchanged.
      nq = model.nq
      njmax_est = max(baseline_nefc * 16, nq * 8, 96)
      # only reduce if configured value exceeds the estimate
      if cfg.njmax is None or cfg.njmax > njmax_est:
        cfg.njmax = njmax_est

    # Adaptive solver iterations: mjlab's velocity env cfg sets iterations=10
    # and ls_iterations=20, but small-robot locomotion typically converges in
    # 2-5 iterations with 5-10 line-search steps.  Reducing these on the SYCL
    # device cuts solver time ~2.5x with no NaN (verified on microduck walking
    # and side_roll).  Only applies when the configured values exceed the
    # adaptive defaults; user can override with MJLAB_SYCL_LS_ITER / _ITER.
    ls_override = os.environ.get("MJLAB_SYCL_LS_ITER", "")
    if ls_override:
      cfg.mujoco.ls_iterations = int(ls_override)
    elif cfg.mujoco.ls_iterations and cfg.mujoco.ls_iterations > 10:
      cfg.mujoco.ls_iterations = 10
    iter_override = os.environ.get("MJLAB_SYCL_ITER", "")
    if iter_override:
      cfg.mujoco.iterations = int(iter_override)
    elif cfg.mujoco.iterations and cfg.mujoco.iterations > 8:
      cfg.mujoco.iterations = 8

    orig_init(self, num_envs, cfg, model, "sycl")
    # torch-facing device: mjlab allocates all torch tensors with this string,
    # and torch has no sycl backend here. wp_device stays on sycl.
    self.device = "cpu"

  def drained(fn):
    def wrapper(self, *args, **kwargs):
      fn(self, *args, **kwargs)
      drain()

    return wrapper

  # Lite forward: skip solver in the final sim.forward() call.
  #
  # env.step() runs 4 physics substeps (each with full solver), then one
  # sim.forward() to refresh derived quantities for observations.  That final
  # forward re-runs the full pipeline including the constraint solver (~80ms
  # on 4096 envs), but obs/reward for typical locomotion tasks only read
  # kinematics (xpos/xquat/cvel from fwd_position+fwd_velocity) and contact
  # flags (from collision detection inside fwd_position) — none read solver
  # outputs (qacc, efc.force, efc.Ma).
  #
  # This patch replaces the final sim.forward() with a lite version that
  # runs fwd_position + fwd_velocity + sensors but skips the solver.
  # Controlled by MJLAB_SYCL_LITE_FORWARD (default "1" = on, "0" = off).
  _lite_forward = os.environ.get("MJLAB_SYCL_LITE_FORWARD", "1").strip().lower() not in (
      "0", "false", "off"
  )
  if _lite_forward:
    import mujoco_warp as _mjwarp
    from mujoco_warp._src import forward as _mj_fwd
    from mujoco_warp._src import sensor as _mj_sensor

    def lite_forward(self):
      with wp.ScopedDevice(self.wp_device):
        if self.use_cuda_graph and self.forward_graph is not None:
          # Can't lite-forward a captured graph (it has the full solver).
          # Fall back to full forward for graph mode.
          wp.capture_launch(self.forward_graph)
        else:
          m = self._wp_model
          d = self._wp_data

          def _seq():
            _mj_fwd.fwd_position(m, d, factorize=False)
            d.sensordata.zero_()
            _mj_sensor.sensor_pos(m, d)
            _mj_fwd.fwd_velocity(m, d)
            _mj_sensor.sensor_vel(m, d)

          # static kernel sequence: after two plain runs it replays as ONE
          # graph submission (see graph_batch.run_sequence)
          if not _gb.run_sequence("lite_forward", _seq, m):
            _seq()
        # drain so host reads in reward/obs see completed kernels
        wp.synchronize_device("sycl")

    sim_mod.Simulation.forward = lite_forward
    print("[sycl-lite] lite forward (skip solver in final forward) installed "
          "(MJLAB_SYCL_LITE_FORWARD=0 to disable)")
  else:
    sim_mod.Simulation.forward = drained(orig_forward)

  # SensorContext's render-context arrays are kernel inputs/outputs and must
  # live on the sim device. It only derives wp arrays from the device string,
  # so "sycl" is safe: torch views wrap USM zero-copy.
  orig_sc_init = sc_mod.SensorContext.__init__

  def sc_init(self, mj_model, model, data, camera_sensors, raycast_sensors, device):
    sc_dev = "cpu" if os.environ.get("MJLAB_SYCL_SC_CPU") else "sycl"
    orig_sc_init(
      self, mj_model, model, data, camera_sensors, raycast_sensors, sc_dev
    )

  sc_mod.SensorContext.__init__ = sc_init

  # RayCastSensor.initialize allocates wp buffers with the scene device
  # ("cpu") but its kernels launch on the sim device: move them to sycl.
  orig_rs_init = rs_mod.RayCastSensor.initialize

  def rs_init(self, mj_model, model, data, device):
    orig_rs_init(self, mj_model, model, data, device)
    for name in (
      "_ray_pnt",
      "_ray_vec",
      "_ray_dist",
      "_ray_geomid",
      "_ray_normal",
      "_ray_bodyexclude",
    ):
      cpu_arr = getattr(self, name)
      setattr(self, name, wp.array(cpu_arr.numpy(), dtype=cpu_arr.dtype, device="sycl"))
    self._wp_device = wp.get_device("sycl")

  rs_mod.RayCastSensor.initialize = rs_init

  # finalize() reads ray outputs from the host via zero-copy USM: drain the
  # queue before postprocess_rays() touches them.
  orig_finalize = sc_mod.SensorContext.finalize

  def finalize(self):
    drain()
    orig_finalize(self)

  sc_mod.SensorContext.finalize = finalize

  # wp.Bvh builds/refits via HOST code (wp_bvh_create_host /
  # wp_bvh_refit_host) that reads rc.lower/rc.upper USM which async device
  # kernels may still be writing.  The overlay's Bvh.__init__/refit already
  # drain the sycl queue for sycl-device instances (types.py is_sycl
  # branch); only non-sycl instances reading USM need the external flush.
  orig_bvh_init = wp.Bvh.__init__

  def bvh_init(self, lowers, uppers, constructor=None, groups=None, leaf_size=1):
    if not getattr(lowers.device, "is_sycl", False):
      drain()
    orig_bvh_init(self, lowers, uppers, constructor, groups, leaf_size)

  wp.Bvh.__init__ = bvh_init

  orig_bvh_refit = wp.Bvh.refit

  def bvh_refit(self):
    if not getattr(self.device, "is_sycl", False):
      drain()
    orig_bvh_refit(self)

  wp.Bvh.refit = bvh_refit

  sim_mod.Simulation.__init__ = init
  # sim.step needs no drained wrapper: capture_while's convergence poll
  # drains at solve end (1 per substep with poll_every=8 + initial-drain
  # skip), and every host read mjlab makes in or after the substep loop
  # observes pre-solve outputs -- scene.update's air-time tracking reads
  # d.sensordata, termination/reward read xpos/xquat/cvel/sensordata -- all
  # flushed by that poll drain.  Kernels launched after it (accel/integrate)
  # are only consumed by later kernels, which the same queue orders
  # correctly; the host writes that follow (reset's state writes) are
  # covered by drained(reset)'s sync, which runs before _reset_idx applies
  # the new state through torch.
  sim_mod.Simulation.step = orig_step
  sim_mod.Simulation.reset = drained(orig_reset)
  # sense: when a sensor context exists, finalize() drains before its host
  # reads (postprocess_rays is torch-only) and sense() launches nothing
  # afterwards -- an outer drain would be a redundant sync.  Without a
  # context sense() returns immediately; drain only in that case to keep
  # the boundary contract for bare sense() callers.
  def sense(self, *args, **kwargs):
    out = orig_sense(self, *args, **kwargs)
    if getattr(self, "_sensor_context", None) is None:
      drain()
    return out
  sim_mod.Simulation.sense = sense
  sim_mod.Simulation.recompute_constants = drained(orig_recompute)
  # forward is already set above (lite_forward at line 163, or drained at 167).
  # Do NOT re-assign here — a blanket drained(orig_forward) would overwrite
  # the lite_forward that skips the solver in the final forward call.

  # Interceptor stack (launch wrappers + solver re-points). The order is
  # load-bearing: each layer only sees what the layers above it left in
  # place, and several fusions go silently dead if a row moves. The full
  # ordered table with per-row rationale lives in _INTERCEPTOR_LAYERS at the
  # bottom of this module.
  for _installer, _why in _INTERCEPTOR_LAYERS:
    _installer()


def install_skip_decoration_sites() -> None:
  """Skip mjlab's decorative terrain sites (env origins, flat patches...).

  These are viewer-only markers: one sphere site per environment plus flat-
  patch boxes, so a 4096-env scene carries ~4100 sites that mujoco_warp then
  transforms every step (_site_local_to_global: ~140ms/step serialized at
  4096 envs). Pure training does not render them; robot sites are untouched.
  """
  from mjlab.terrains import terrain_entity as _te

  cls = _te.TerrainEntity
  for name in ("_add_env_origin_sites", "_add_terrain_origin_sites", "_add_flat_patch_sites"):
    setattr(cls, name, lambda self: None)


def _install_fused_set_const() -> None:
  from mjlab_sycl import fused_set_const

  fused_set_const.install()


def _install_fused_solver_tail_if_wanted() -> None:
  cpu_sim = os.environ.get("MJLAB_SYCL_SIM_DEVICE", "sycl").strip().lower() == "cpu"
  if cpu_sim or os.environ.get("MJLAB_SYCL_FUSED_SOLVER_TAIL", "").strip().lower() in (
      "1",
      "true",
      "on",
  ):
    from mjlab_sycl import fused_solver_tail

    fused_solver_tail.install()


def _install_batched_polling() -> None:
  # default poll_every=8 >= the solver iteration cap, so a whole solve polls
  # once -- at solve end, which doubles as the sync covering in-loop host
  # reads (see the step/sense wrapper comments above)
  if not os.environ.get("MJLAB_SYCL_POLL_EVERY"):
    os.environ["MJLAB_SYCL_POLL_EVERY"] = "8"
  _sycl_loop.install_poll_batching()


def _install_step_graph() -> None:
  """Capture the whole mjwarp.step substep as one command graph.

  Every kernel the substep submits (collision -> constraints -> solve ->
  integrate) has static, buffer-sized grids, so after one plain count and
  one verified record the entire substep replays as a single queue
  submission -- the same amortization the solver batch got, extended to the
  ~900 remaining per-kernel submits of each step. The solver batch graph
  runs plain while the substep counts/records (graph_batch's scoped mode)
  so the outer graph absorbs its kernels as nodes; nested replay inside a
  recording would break the outer count-verification. Any non-static
  launch sequence fails the count check and falls back to plain forever
  (safe by construction). MJLAB_SYCL_GRAPH=0 disables together with the
  solver-batch graphs.
  """
  import mujoco_warp as _mjw

  from mjlab_sycl import graph_batch

  if os.environ.get("MJLAB_SYCL_STEP_GRAPH", "1").strip().lower() in (
      "0", "false", "off",
  ):
    return

  orig_step = _mjw.step

  def step_graphed(m, d, *args, **kwargs):
    if graph_batch.run_sequence(
        "mjwarp_step", lambda: orig_step(m, d, *args, **kwargs), ident=m,
        scoped=True,
    ):
      # the substep ran (plain count, recorded, or replayed). On the plain
      # count the internal poll drain already synced; on record/replay it
      # was skipped (illegal during capture), so this boundary drain keeps
      # the substep's host-read contract at exactly one sync per substep.
      import warp as _wp

      _wp.synchronize_device("sycl")
      return
    orig_step(m, d, *args, **kwargs)

  _mjw.step = step_graphed
  print(
    "[sycl-step-graph] mjwarp.step captured as one command graph "
    "(MJLAB_SYCL_GRAPH=0 to disable)"
  )


# The interceptor stack, in the one order that keeps every fusion live.
# Layers wrap wp.launch (or re-point solver internals), so each layer only
# sees what the rows above it left in place; the note on each row records
# why it sits exactly here. Do not reorder without re-reading the notes --
# several fusions go silently dead (correct output, lost speedup) if a row
# moves. Installed by patch_simulation_for_sycl().
_INTERCEPTOR_LAYERS = (
    (
        flat_kernels.install,
        "Barrier-free flat rewrites of mujoco_warp's hottest tiled kernels "
        "(JTDAJ, contact_jac, cholesky variants). First, so every launch "
        "wrapper below intercepts the re-routed kernels, not the originals.",
    ),
    (
        install_skip_decoration_sites,
        "Drop mjlab's decorative terrain sites at model build time (~4100 "
        "sites at 4096 envs, ~140 ms/step of _site_local_to_global). Not a "
        "launch wrapper; placement is free.",
    ),
    (
        _sycl_launch_cache.install,
        "Cached wp.launch for repeated (kernel, args, dim) -- ~1500/step in "
        "the physics hot path where ~48% of step time is host-side submit "
        "overhead. After flat_kernels so its re-routes also benefit.",
    ),
    (
        _sycl_fused_tree.install,
        "Fused tree chains: one launch per chain invocation instead of one "
        "per kinematic depth level (7 levels per invocation, ~210 -> 30 "
        "launches/step).",
    ),
    (
        _sycl_fused_solver.install,
        "The four per-iteration zero/rotate launches fold into "
        "linesearch_jaref's tail (~160 launches/memsets per step). MUST sit "
        "after launch_cache (outside the cache) so the cache sees the FUSED "
        "jaref kernel -- a cache hit on the unfused one would skip the "
        "zeroing while the suppressed launches stay suppressed.",
    ),
    (
        _sycl_solver_ctx.install,
        "Reuse the per-solve SolverContext/step_size_cost/nsolving scratch "
        "so every solver kernel keeps a stable launch-cache key across "
        "solves (~28% of launches would otherwise rebuild packed args every "
        "step, ~65 ms/step of host submit at 4096 envs). Must run BEFORE "
        "fused_linesearch: its solve shim wraps THIS solve replacement, or "
        "its mul_m/jv/teardown fusions go dead.",
    ),
    (
        _sycl_act_fuse.install,
        "Rollout act path: closed-form Gaussian sample/log_prob instead of "
        "torch.distributions (~9 ms -> ~1 ms per act call at 4096 envs). "
        "Independent of the launch chain.",
    ),
    (
        _sycl_fused_ls.install,
        "Linesearch fusions (teardown, mv+jv, prepare-gauss+quad), after "
        "launch_cache/fused_solver so the cache still sees the fused "
        "kernels.",
    ),
    (
        _sycl_skip_empty.install,
        "Skip 0-dim launches (flex/tendon/equality/limit rows the model "
        "doesn't have, ~120 wasted dispatches/step) as the OUTERMOST layer: "
        "installed last among the launch wrappers so its _prev_launch "
        "captures the whole interceptor chain and no inner interceptor ever "
        "sees a suppressed launch.",
    ),
    (
        _install_fused_set_const,
        "Selective per-world set_const recompute after domain rand: the "
        "reset path (fall -> event -> recompute_constants) recomputes ALL "
        "worlds for a handful of reset ones; 50 -> 10 ms/step at 4096 envs "
        "(all-worlds case is a wash). MJLAB_SYCL_FUSED_SET_CONST=0 falls "
        "back.",
    ),
    (
        _install_fused_solver_tail_if_wanted,
        "Fuse the per-iteration CG-tail kernels (prev_grad/beta/zero/"
        "search_update/done -> 1 launch). CPU-sim only by default: it "
        "serializes the per-world tail and the GPU prefers the original "
        "parallel kernels (CG is rare here); MJLAB_SYCL_FUSED_SOLVER_TAIL=1 "
        "opts in on any device.",
    ),
    (
        _install_batched_polling,
        "Batched convergence polling for the sycl capture_while fallback "
        "(SYCL has no graph capture; warp's emulation drains per iteration, "
        "up to 8 iter x 4 substeps = 32 drains/step just to poll). Default "
        "poll_every=8 makes a whole solve one batch: exactly one drain per "
        "solve, bit-identical physics (extra iterations are guarded no-ops "
        "via ctx.done). MJLAB_SYCL_POLL_EVERY=1 restores warp's original "
        "per-iteration polling.",
    ),
    (
        _install_step_graph,
        "Whole-substep command graph: mjwarp.step (collision -> constraints "
        "-> solve -> integrate) replays as one queue submission after a "
        "plain count and a verified record. Scoped mode forces the solver "
        "batch graph plain while the substep establishes itself, so the "
        "outer graph absorbs it and the launch counts match. Sits after "
        "every launch wrapper so the recorded sequence is the final fused "
        "one.",
    ),
)
