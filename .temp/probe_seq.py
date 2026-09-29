# SPDX-License-Identifier: Apache-2.0
"""Probe: record the exact launch sequence of ONE env.step (kernel key, dim,
call site) to find fusion candidates."""

import os
import sys

sys.path.insert(0, r"D:\mjlab-sycl\src")

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import warp as wp

wp.init()

# Disable the launch cache so we see EVERY launch through the original path
# (the cache would short-circuit hits and hide nothing, but recording through
# one choke point is simpler).
os.environ["MJLAB_SYCL_LAUNCH_CACHE"] = "0"

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

patch_simulation_for_sycl()

import torch

torch.set_num_threads(2)

seq = []
_recording = [False]

# Hook BELOW fused_solver's interceptor so suppressed launches (which never
# execute) are not recorded: wrap the innermost layer the interceptor chains
# to.
import mjlab_sycl.fused_solver as _fs

_inner = _fs._prev_launch


def recording_launch(kernel, dim, inputs=(), outputs=(), **kwargs):
  if _recording[0]:
    import inspect

    stack = inspect.stack()
    caller = "?"
    for f in stack[1:8]:
      fn = f.filename.replace("\\", "/")
      if "/warp/_src/" in fn:
        continue
      caller = f"{fn.split('/')[-1]}:{f.lineno}"
      break
    d = dim if not isinstance(dim, (list, tuple)) else tuple(dim)
    n = len(tuple(inputs)) + len(tuple(outputs))
    seq.append((getattr(kernel, "key", str(kernel)), d, n, caller))
  return _inner(kernel, dim, inputs, outputs, **kwargs)


_fs._prev_launch = recording_launch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
env_cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", "512"))
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
env.reset()

action = torch.zeros(
  (env_cfg.scene.num_envs, env.action_manager.total_action_dim),
  dtype=torch.float32,
  device="cpu",
)

# warmup: 3 steps (compile everything, stabilize allocations)
for _ in range(3):
  env.step(action)

# record exactly ONE step
seq.clear()
_recording[0] = True
env.step(action)
_recording[0] = False

print(f"\n=== {len(seq)} launches in ONE step (nworld={env_cfg.scene.num_envs}) ===\n")
for i, (key, dim, nargs, caller) in enumerate(seq):
  print(f"{i:4d} | {key:70.70} | dim={str(dim):18.18} | args={nargs:2d} | {caller}")
