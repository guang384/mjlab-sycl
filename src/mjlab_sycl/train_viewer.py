# SPDX-License-Identifier: Apache-2.0
"""SYCL training with a live MuJoCo viewer window.

Launches the native MuJoCo viewer rendering env 0 while training runs on the
Intel GPU. Watch the duck: it starts flailing and (over iterations) starts
walking. Physics on sycl, PPO on torch.xpu, rendering on your screen.

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

  import mujoco.viewer

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
  agent_cfg.experiment_name = f"{args.task}-sycl"
  agent_cfg.experiment_name += "-viewer"

  # viewer env: render_mode 'human' opens the native MuJoCo window on env 0
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="human")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ppo_device = args.ppo_device or ("xpu" if torch.xpu.is_available() else "cpu")
  print(f"[train-viewer] PPO on {ppo_device}; viewer window should be open", flush=True)

  log_dir = Path("logs") / agent_cfg.experiment_name / args.run_name
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), str(log_dir), ppo_device)
  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
  main()