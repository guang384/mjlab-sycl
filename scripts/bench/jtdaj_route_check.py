import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import torch
import warp as wp
wp.init()
from mjlab_sycl.runtime_patch import patch_simulation_for_sycl
patch_simulation_for_sycl()

from mjlab_sycl import native_kernels
native_kernels._api()
print("jtdaj export resolved:", native_kernels._fn_jtdaj is not None)

# wrap the native entry to see if the pipeline calls it
calls = {"n": 0, "ok": 0}
orig = native_kernels.jtdaj
def spy(*a, **k):
    r = orig(*a, **k)
    calls["n"] += 1
    calls["ok"] += int(r)
    return r
native_kernels.jtdaj = spy

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg, device="cpu")
env.reset()
env.step(torch.zeros((64, env.action_manager.total_action_dim)))
print("jtdaj route calls:", calls)
