# SPDX-License-Identifier: Apache-2.0
"""Runtime patch that runs mjlab/mujoco_warp physics on the Intel SYCL device.

Physics (warp arrays) live on the ``sycl`` device while every torch tensor
stays on ``cpu``: the SYCL device allocates USM shared memory, which
``wp.to_torch`` wraps zero-copy. Kernel submissions are asynchronous, so every
sim call boundary (step/forward/reset/sense/recompute_constants) drains the
queue -- exactly the points where host code reads or writes USM that kernels
touch.

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
from mjlab_sycl import fused_solver as _sycl_fused_solver
from mjlab_sycl import fused_tree as _sycl_fused_tree
from mjlab_sycl import install as _sycl_install
from mjlab_sycl import launch_cache as _sycl_launch_cache
from mjlab_sycl import loop_poll as _sycl_loop


def patch_simulation_for_sycl() -> None:
  """Physics on the sycl device, torch on cpu, queue drained at call boundaries.

  The sycl device reports is_cpu=True for torch/numpy interop (USM shared
  memory, zero-copy views), so only components that allocate *warp* arrays
  from the scene's device string need re-pointing to sycl: the SensorContext
  (render context feeds BVH refit kernels) and RayCastSensor ray buffers.
  """
  # Hard guard, not a warning: a missing/stale backend overlay silently
  # corrupts physics (or makes the device vanish) — abort before iteration 0
  # with the exact remediation instead.
  _sycl_install.ensure_overlay_synced()

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
    # Use mj_forward to measure the baseline nefc, then add headroom for
    # contact-heavy transients (falls, multi-body contact during acrobatics):
    #   njmax_est = baseline_nefc * 8  (covers 8x spike from 14 → 112)
    # Capped at the configured value (don't increase beyond what caller set).
    # Override with MJLAB_SYCL_NJMAX=N (force) or =0 (disable).
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
          _mj_fwd.fwd_position(m, d, factorize=False)
          d.sensordata.zero_()
          _mj_sensor.sensor_pos(m, d)
          _mj_fwd.fwd_velocity(m, d)
          _mj_sensor.sensor_vel(m, d)
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
    import os

    sc_dev = "cpu" if os.environ.get("BENCH_SC_CPU") else "sycl"
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

  # wp.Bvh on the sycl device (is_cpu=True) builds via the HOST constructor
  # and refits via wp_bvh_refit_host: both read rc.lower/rc.upper USM that
  # async device kernels just wrote — drain before each host read.
  orig_bvh_init = wp.Bvh.__init__

  def bvh_init(self, lowers, uppers, constructor=None, groups=None, leaf_size=1):
    drain()
    orig_bvh_init(self, lowers, uppers, constructor, groups, leaf_size)

  wp.Bvh.__init__ = bvh_init

  orig_bvh_refit = wp.Bvh.refit

  def bvh_refit(self):
    drain()
    orig_bvh_refit(self)

  wp.Bvh.refit = bvh_refit

  sim_mod.Simulation.__init__ = init
  sim_mod.Simulation.step = drained(orig_step)
  sim_mod.Simulation.forward = drained(orig_forward)
  sim_mod.Simulation.reset = drained(orig_reset)
  sim_mod.Simulation.sense = drained(orig_sense)
  sim_mod.Simulation.recompute_constants = drained(orig_recompute)

  # barrier-free flat rewrites of the hottest tiled kernels
  flat_kernels.install()
  install_skip_decoration_sites()

  # cached wp.launch: skip pack_arg/invoke/ArgsStruct for repeated
  # (kernel, args, dim) launches (~1500/step in the physics hot path,
  # ~48% of step time is host-side launch overhead). Installed after
  # flat_kernels so its wp.launch re-routes also benefit from the cache.
  _sycl_launch_cache.install()

  # Fused tree chains: one launch per chain instead of one per kinematic
  # depth level (5 chains x 7 levels = 210 launches/step -> 30).
  _sycl_fused_tree.install()

  # Fused solver zero/rotate launches: the four per-iteration zero kernels
  # fold into the tail of linesearch_jaref.  MUST be installed after
  # launch_cache (outer layer) so the cache sees the fused jaref kernel —
  # a cache hit on the unfused kernel would skip the zeroing while the
  # suppressed launches stay suppressed.
  _sycl_fused_solver.install()

  # Fused set_const_0 + solver_tail: only beneficial on CPU (SYCL GPU
  # prefers the original parallel kernels — the fused versions serialize
  # per-world work and lose parallelism).  Enable explicitly on CPU with
  #   MJLAB_SYCL_FUSED_SET_CONST=1 / MJLAB_SYCL_FUSED_SOLVER_TAIL=1
  # (auto-enabled when the sim device is CPU, not sycl).
  _is_cpu_sim = os.environ.get("MJLAB_SYCL_SIM_DEVICE", "sycl") == "cpu"
  if _is_cpu_sim or os.environ.get("MJLAB_SYCL_FUSED_SET_CONST", "").strip().lower() in ("1", "true", "on"):
    from mjlab_sycl import fused_set_const
    fused_set_const.install()
  if _is_cpu_sim or os.environ.get("MJLAB_SYCL_FUSED_SOLVER_TAIL", "").strip().lower() in ("1", "true", "on"):
    from mjlab_sycl import fused_solver_tail
    fused_solver_tail.install()

  # batched convergence polling for the sycl capture_while fallback
  # (MJLAB_SYCL_POLL_EVERY=N; off by default -> original per-iteration polls)
  _sycl_loop.install_poll_batching()


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
