# SPDX-License-Identifier: Apache-2.0
"""SYCL training with a live MuJoCo viewer window.

Launches the native MuJoCo viewer (env 0) while training runs on the Intel
GPU. Watch the duck: it starts flailing and (over iterations) starts walking.

How it works: mjlab 1.3.0 has no interactive native viewer (its render modes
are ``None``/``rgb_array`` only), so this entry mirrors env 0's state into a
plain cpu ``mujoco.MjData`` and syncs the native passive viewer after every
env step. The passive viewer does NOT simulate on its own (verified), so what
you see is exactly env 0's state. Mirroring is a per-step numpy copy of one
env (a few hundred floats) -- negligible against the physics cost.

Usage:
    python -m mjlab_sycl.train_viewer <TASK_ID> --num-envs 1024
"""

import argparse
import dataclasses
from pathlib import Path

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import torch  # noqa: E402
import warp as wp  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task")
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=1000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--ppo-device", default=None)
  parser.add_argument("--run-name", default="viewer")
  args = parser.parse_args()

  wp.init()
  patch_simulation_for_sycl()

  import mujoco  # noqa: E402
  import mujoco.viewer  # noqa: E402  (binds mujoco.viewer for launch_passive)

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.upload_model = False
  agent_cfg.logger = "tensorboard"
  agent_cfg.experiment_name = f"{args.task}-sycl-viewer"

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ppo_device = args.ppo_device or ("xpu" if torch.xpu.is_available() else "cpu")
  print(f"[train-viewer] PPO on {ppo_device}; native viewer opens on env 0", flush=True)

  # -- native viewer mirror of env 0 -----------------------------------------
  mj_model = env.unwrapped.sim.mj_model
  mj_data = mujoco.MjData(mj_model)

  def mirror_env0() -> None:
    d = env.unwrapped.sim.data
    mj_data.qpos[:] = d.qpos.numpy()[0]
    mj_data.qvel[:] = d.qvel.numpy()[0]
    mj_data.ctrl[:] = d.ctrl.numpy()[0]
    t = d.time
    mj_data.time = float(t.numpy()[0]) if hasattr(t, "numpy") else float(t)

  env.reset()
  mirror_env0()

  try:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
  except Exception as e:
    print(f"[train-viewer] WARN: could not open the native viewer ({e!r}); "
          "training continues headless", flush=True)
    viewer = None

  orig_step = env.step

  def stepping(action):
    out = orig_step(action)
    if viewer is not None:
      try:
        if viewer.is_running():
          mirror_env0()
          viewer.sync()
      except Exception:
        pass  # viewer closed mid-run: training continues headless
    return out

  env.step = stepping

  log_dir = Path("logs") / agent_cfg.experiment_name / args.run_name
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), str(log_dir), ppo_device)
  try:
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  finally:
    if viewer is not None:
      viewer.close()


if __name__ == "__main__":
  main()
