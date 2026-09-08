# SPDX-License-Identifier: Apache-2.0
"""Train on the Intel SYCL device without mjlab's CUDA GPU selection.

mjlab's `train` calls select_gpus() which indexes torch's CUDA device list --
on a machine with only an Intel iGPU that list is empty and training dies
before iteration 0. This entry builds the same mjlab env + rsl_rl runner as
mjlab's own trainer while keeping every training feature: checkpoints,
tensorboard logging, and full-length runs.

Usage:
    mjlab-sycl-train Mjlab-Velocity-Flat-MicroDuck \
        --num-envs 1024 --max-iterations 1000 --save-interval 200
"""

import argparse
import dataclasses
from pathlib import Path

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import torch  # noqa: E402
from mjlab_sycl._bootstrap import configure_torch_threads  # noqa: E402
configure_torch_threads()  # cap CPU threads (MJLAB_TORCH_THREADS)
import warp as wp  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task")
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=1000)
  parser.add_argument("--save-interval", type=int, default=200)
  parser.add_argument("--checkpoint", default=None,
                      help="resume from logs/<exp>/<run>/model_XXXX.pt")
  parser.add_argument("--run-name", default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--ppo-device", default=None,
                      help="torch device for PPO (default: xpu if available, else cpu)")
  args = parser.parse_args()

  wp.init()
  patch_simulation_for_sycl()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)

  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.upload_model = False
  agent_cfg.logger = "tensorboard"
  agent_cfg.experiment_name = f"{args.task}-sycl"
  if args.seed is not None:
    agent_cfg.seed = args.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ppo_device = args.ppo_device or ("xpu" if torch.xpu.is_available() else "cpu")
  print(f"[train-sycl] PPO on {ppo_device}")

  log_dir = Path("logs") / agent_cfg.experiment_name
  if args.run_name:
    log_dir = log_dir / args.run_name
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), str(log_dir), ppo_device)

  if args.checkpoint:
    runner.load(args.checkpoint)

  print(f"[train-sycl] task={args.task} num_envs={args.num_envs} "
        f"iters={args.max_iterations} log_dir={log_dir}")
  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
  main()