# SPDX-License-Identifier: Apache-2.0
"""Runtime API call census for one mjlab training step on the sycl device.

Counts wp.launch submissions, queue drains (synchronize), USM allocs/frees,
memsets, memtiles, and host<->array copies per env step, plus wall time of
each category, to rank optimization targets.

Usage:
    python scripts/probe_sycl_profile.py --num-envs 4096 --steps 5
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import warp as wp

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402

STATS = Counter()
TIMES = Counter()


def install_counters() -> None:
  rt = wp._src.context.runtime
  sycl = rt.sycl

  orig_sync = sycl.wp_sycl_synchronize

  def sync():
    t0 = time.perf_counter()
    orig_sync()
    STATS["drain"] += 1
    TIMES["drain"] += time.perf_counter() - t0

  sycl.wp_sycl_synchronize = sync

  orig_alloc = sycl.wp_sycl_alloc_shared

  def alloc(size):
    t0 = time.perf_counter()
    out = orig_alloc(size)
    STATS["alloc"] += 1
    TIMES["alloc"] += time.perf_counter() - t0
    return out

  sycl.wp_sycl_alloc_shared = alloc

  orig_free = sycl.wp_sycl_free

  def free(ptr):
    t0 = time.perf_counter()
    orig_free(ptr)
    STATS["free"] += 1
    TIMES["free"] += time.perf_counter() - t0

  sycl.wp_sycl_free = free

  # device memset/memtile wrappers (bound as Device methods)
  for dev_name in ("sycl", "sycl:0"):
    dev = rt.device_map.get(dev_name)
    if dev is None:
      continue
    if getattr(dev, "memset", None) is None:
      continue
    orig_memset = dev.memset

    def memset(ptr, value, size, _orig=orig_memset):
      t0 = time.perf_counter()
      _orig(ptr, value, size)
      STATS["memset"] += 1
      TIMES["memset"] += time.perf_counter() - t0

    dev.memset = memset

    orig_memtile = getattr(dev, "memtile", None)
    if orig_memtile is not None:

      def memtile(ptr, src, srcsize, reps, _orig=orig_memtile):
        t0 = time.perf_counter()
        _orig(ptr, src, srcsize, reps)
        STATS["memtile"] += 1
        TIMES["memtile"] += time.perf_counter() - t0

      dev.memtile = memtile

  # launch census
  orig_launch = wp.launch

  def launch(kernel, dim, *args, **kwargs):
    t0 = time.perf_counter()
    out = orig_launch(kernel, dim, *args, **kwargs)
    STATS["launch"] += 1
    TIMES["launch"] += time.perf_counter() - t0
    STATS[f"launch:{kernel.key}"] += 1
    return out

  wp.launch = launch


def install_sim_phase_timers() -> None:
  from mjlab.sim import sim as sim_mod

  for name in ("step", "forward", "reset", "sense", "recompute_constants"):
    fn = getattr(sim_mod.Simulation, name)

    def wrapper(self, *args, _fn=fn, _name=name, **kwargs):
      t0 = time.perf_counter()
      out = _fn(self, *args, **kwargs)
      TIMES[f"sim:{_name}"] += time.perf_counter() - t0
      STATS[f"sim:{_name}"] += 1
      return out

    setattr(sim_mod.Simulation, name, wrapper)

  from mjlab.sensor import sensor_context as sc_mod

  for name in ("finalize", "update"):
    if not hasattr(sc_mod.SensorContext, name):
      continue
    fn = getattr(sc_mod.SensorContext, name)

    def sc_wrapper(self, *args, _fn=fn, _name=name, **kwargs):
      t0 = time.perf_counter()
      out = _fn(self, *args, **kwargs)
      TIMES[f"sensor:{_name}"] += time.perf_counter() - t0
      STATS[f"sensor:{_name}"] += 1
      return out

    setattr(sc_mod.SensorContext, name, sc_wrapper)

  orig_refit = wp.Bvh.refit

  def bvh_refit(self):
    t0 = time.perf_counter()
    orig_refit(self)
    TIMES["bvh:refit"] += time.perf_counter() - t0
    STATS["bvh:refit"] += 1

  wp.Bvh.refit = bvh_refit

  orig_mesh_refit = wp.Mesh.refit

  def mesh_refit(self):
    t0 = time.perf_counter()
    orig_mesh_refit(self)
    TIMES["mesh:refit"] += time.perf_counter() - t0
    STATS["mesh:refit"] += 1

  wp.Mesh.refit = mesh_refit


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--steps", type=int, default=5)
  args = parser.parse_args()

  wp.init()
  patch_simulation_for_sycl()
  install_counters()
  install_sim_phase_timers()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  task = "Mjlab-Velocity-Flat-MicroDuck"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  print(f"[profile] env constructed ({args.num_envs} envs)")
  print("[profile] census during construction:")
  for k in ("launch", "drain", "alloc", "free", "memset", "memtile"):
    print(f"  {k:9s} {STATS[k]:8d}  {TIMES[k]:8.2f}s")

  STATS.clear()
  TIMES.clear()

  obs, _ = env.reset()
  n_actions = env.action_manager.total_action_dim

  import torch

  t0 = time.perf_counter()
  STATS.clear()
  TIMES.clear()
  for _ in range(args.steps):
    action = torch.zeros((env.num_envs, n_actions))
    obs, rew, term, trunc, info = env.step(action)
  total = time.perf_counter() - t0

  print(f"\n[profile] {args.steps} steps in {total:.2f}s "
        f"({total / args.steps * 1000:.1f} ms/step)")
  print("[profile] census per-phase (constructed env, stepping):")
  wall = 0.0
  for k in ("launch", "drain", "alloc", "free", "memset", "memtile"):
    per = STATS[k] / args.steps
    print(f"  {k:9s} {STATS[k]:8d} total ({per:6.1f}/step)  {TIMES[k]:8.2f}s")
    wall += TIMES[k]
  print(f"  (sum of timed categories: {wall:.2f}s of {total:.2f}s wall)")

  print("[profile] phase wall time:")
  for k in sorted(TIMES):
    if ":" not in k:
      continue
    per = TIMES[k] / args.steps
    print(f"  {k:22s} {STATS.get(k, 0):8d} calls  {TIMES[k]:8.2f}s total ({per * 1000:7.1f} ms/step)")

  top = [
    (k, v)
    for k, v in STATS.most_common()
    if k.startswith("launch:") and v >= max(2, STATS["launch"] // 100)
  ][:20]
  print("\n[profile] top kernels by launch count:")
  for k, v in top:
    print(f"  {v:6d}  {k[7:]}")


if __name__ == "__main__":
  main()
