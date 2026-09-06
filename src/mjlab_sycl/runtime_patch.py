# SPDX-License-Identifier: Apache-2.0
"""Runtime patch that runs mjlab/mujoco_warp physics on the Intel SYCL device.

Physics (warp arrays) live on the ``sycl`` device while every torch tensor
stays on ``cpu``: the SYCL device allocates USM shared memory, which
``wp.to_torch`` wraps zero-copy. Kernel submissions are asynchronous, so every
sim call boundary (step/forward/reset/sense/recompute_constants) drains the
queue -- exactly the points where host code reads or writes USM that kernels
touch.

Also installs the barrier-free flat rewrites of mujoco_warp's tiled hot
kernels (see flat_kernels.py). Enable from the training entry point with
MJLAB_SYCL=1 (wired in train_hook), or call ``patch_simulation_for_sycl()``
directly from probes/benchmarks.
"""

from __future__ import annotations

import os

import warp as wp

from mjlab_sycl import flat_kernels


def patch_simulation_for_sycl() -> None:
  """Physics on the sycl device, torch on cpu, queue drained at call boundaries.

  The sycl device reports is_cpu=True for torch/numpy interop (USM shared
  memory, zero-copy views), so only components that allocate *warp* arrays
  from the scene's device string need re-pointing to sycl: the SensorContext
  (render context feeds BVH refit kernels) and RayCastSensor ray buffers.
  """
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
    orig_init(self, num_envs, cfg, model, "sycl")
    # torch-facing device: mjlab allocates all torch tensors with this string,
    # and torch has no sycl backend here. wp_device stays on sycl.
    self.device = "cpu"

  def drained(fn):
    def wrapper(self, *args, **kwargs):
      fn(self, *args, **kwargs)
      drain()

    return wrapper

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
