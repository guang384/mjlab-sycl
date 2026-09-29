# SPDX-License-Identifier: Apache-2.0
"""Precise physics-only A/B: time N env steps with the launch cache ON vs
OFF in the same process conditions, reporting ms/step for each."""

import os
import sys

sys.path.insert(0, r"D:\mjlab-sycl\src")

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import warp as wp

wp.init()

if os.environ.get("USE_CACHE", "1") != "0":
    from mjlab_sycl import launch_cache

    launch_cache.install()

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

patch_simulation_for_sycl()

import time

import torch

torch.set_num_threads(2)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = "Mjlab-Velocity-Flat-MicroDuck"
env_cfg = load_env_cfg(task)
env_cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", "4096"))

env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
env.reset()

action = torch.zeros(
    (env_cfg.scene.num_envs, env.action_manager.total_action_dim),
    dtype=torch.float32,
    device="cpu",
)

# Warmup: 5 steps (fills the cache, compiles everything)
for _ in range(5):
    env.step(action)

n_steps = int(os.environ.get("N_STEPS", "50"))
t0 = time.perf_counter()
for _ in range(n_steps):
    env.step(action)
elapsed = time.perf_counter() - t0

label = "CACHE ON " if os.environ.get("USE_CACHE", "1") != "0" else "CACHE OFF"
ms = elapsed / n_steps * 1000
print(f"\n[ab] {label}: {ms:.1f} ms/step over {n_steps} steps ({env_cfg.scene.num_envs} envs)")

if os.environ.get("USE_CACHE", "1") != "0":
    from mjlab_sycl import launch_cache as lc

    print(f"[ab] stats: {lc.stats()}")
