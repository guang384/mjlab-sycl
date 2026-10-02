"""Single-arm runner: one env, 10 steps, dump per-step qacc snapshots.
Invoke twice (different MJLAB_SYCL_NATIVE_MVJV + out paths) and compare the
files -- separate processes guarantee identical fresh RNG state."""
import os, sys
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
tag, out_path = sys.argv[1], sys.argv[2]
import torch
torch.manual_seed(42)
import warp as wp
wp.init()
from mjlab_sycl.runtime_patch import patch_simulation_for_sycl
patch_simulation_for_sycl()
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
cfg.scene.num_envs = 256
cfg.seed = 42
env = ManagerBasedRlEnv(cfg, device="cpu")
env.reset()
a = torch.zeros((256, env.action_manager.total_action_dim))
snaps = []
for i in range(10):
    env.step(a)
    snaps.append(env.sim._wp_data.qacc.numpy().copy())
np.save(out_path, np.stack(snaps))
print(f"{tag}: dumped {len(snaps)} snapshots -> {out_path}", flush=True)
