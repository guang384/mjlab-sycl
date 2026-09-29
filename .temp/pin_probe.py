# SPDX-License-Identifier: Apache-2.0
"""Measure how much USM the launch cache pins via its keys (stale solver
contexts from finished substeps are the concern)."""

import os
import sys

sys.path.insert(0, r"D:\mjlab-sycl\src")

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import warp as wp

wp.init()

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

patch_simulation_for_sycl()

import torch

torch.set_num_threads(2)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
env_cfg.scene.num_envs = 4096
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
env.reset()
action = torch.zeros(
  (4096, env.action_manager.total_action_dim), dtype=torch.float32, device="cpu"
)
for _ in range(5):
  env.step(action)

from mjlab_sycl import launch_cache as lc


def pinned_bytes():
  seen = set()
  n_entries = 0
  for key in lc._cache:
    n_entries += 1
    for a in key:  # (kernel, args_tuple, dim)
      if isinstance(a, tuple):
        for x in a:
          if isinstance(x, wp.array) and id(x) not in seen:
            seen.add(id(x))
      elif isinstance(a, wp.array) and id(a) not in seen:
        seen.add(id(a))
  return n_entries, len(seen)


import time

t0 = time.perf_counter()
for step in range(30):
  env.step(action)
  if step % 10 == 9:
    n, na = pinned_bytes()
    # rough MB estimate: most pinned arrays are (4096, 168) float = 2.75MB,
    # (4096, 20, 20) = 6.5MB; assume avg ~2MB per array
    est_mb = na * 2
    print(
      f"[pin] after {step + 1} steps: cache_entries={n} "
      f"distinct_arrays={na} (~{est_mb} MB if avg 2MB each)",
      flush=True,
    )
elapsed = time.perf_counter() - t0
print(f"[pin] 30 steps in {elapsed:.1f}s")
