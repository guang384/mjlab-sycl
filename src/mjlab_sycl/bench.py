# SPDX-License-Identifier: Apache-2.0
"""End-to-end PPO iteration benchmark at scale: CPU vs Intel-GPU (SYCL) physics.

Runs the real training loop (mjlab env + rsl_rl PPO) for a task at a given
num_envs and reports per-iteration rollout / update wall time.

With ``--device sycl`` the Simulation's warp device is swapped to ``sycl``
while every torch tensor stays on ``cpu``: the SYCL device allocates USM
shared memory, which ``wp.to_torch`` wraps zero-copy, so the rest of mjlab
runs unmodified. Kernel submissions are asynchronous, so every sim call
boundary (step/forward/reset/sense/recompute_constants) drains the queue --
those are exactly the points where host code reads or writes USM that
kernels touch. Without this, torch reads race in-flight kernels.

Usage:
    python -m mjlab_sycl.bench --device cpu   --num-envs 4096 --iters 6
    python -m mjlab_sycl.bench --device sycl  --num-envs 4096 --iters 6
"""

import argparse
import dataclasses
import os
import time
from pathlib import Path

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import torch  # noqa: E402
import warp as wp  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", choices=["cpu", "sycl"], default="sycl")
  parser.add_argument(
    "--task", default="Mjlab-Velocity-Flat-MicroDuck"
  )
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iters", type=int, default=6)
  parser.add_argument("--skip-warmup", type=int, default=1,
                      help="iterations to exclude from the mean (compile/warmup)")
  args = parser.parse_args()

  wp.init()
  if args.device == "sycl":
    patch_simulation_for_sycl()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)

  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.iters
  agent_cfg.logger = "tensorboard"
  agent_cfg.save_interval = 10**9  # no checkpoint I/O in the timed region
  agent_cfg.upload_model = False
  agent_cfg.experiment_name = "bench_sycl_train"

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  log_dir = Path("logs/bench_sycl_train") / f"{args.device}_{args.num_envs}"
  # match train's default: PPO on xpu when available (bench previously
  # defaulted to cpu, understating the realistic iteration time by ~3 s)
  ppo_device = os.environ.get("MJLAB_PPO_DEVICE") or (
    "xpu" if torch.xpu.is_available() else "cpu"
  )
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), str(log_dir), ppo_device)

  # -- timing instrumentation ------------------------------------------------
  step_times, act_times, update_times = [], [], []

  orig_env_step = runner.env.step
  orig_act = runner.alg.act
  orig_update = runner.alg.update

  def timed_env_step(action):
    t0 = time.perf_counter()
    out = orig_env_step(action)
    step_times.append(time.perf_counter() - t0)
    return out

  def timed_act(obs):
    t0 = time.perf_counter()
    out = orig_act(obs)
    act_times.append(time.perf_counter() - t0)
    return out

  def timed_update():
    t0 = time.perf_counter()
    out = orig_update()
    update_times.append(time.perf_counter() - t0)
    return out

  runner.env.step = timed_env_step
  runner.alg.act = timed_act
  runner.alg.update = timed_update

  print(
    f"[bench] task={args.task} device={args.device} num_envs={args.num_envs} "
    f"iters={args.iters}"
  )
  t_start = time.perf_counter()
  runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
  total = time.perf_counter() - t_start

  # -- report -----------------------------------------------------------------
  spe = agent_cfg.num_steps_per_env
  n_iters = len(update_times)
  skip = min(args.skip_warmup, max(n_iters - 1, 0))

  def bucket(xs, i):
    chunk = xs[i * spe : (i + 1) * spe]
    return sum(chunk) if chunk else float("nan")

  rows = []
  for i in range(n_iters):
    rollout = bucket(step_times, i) + bucket(act_times, i)
    update = update_times[i]
    rows.append((rollout, update, rollout + update))

  steady = rows[skip:]
  n = len(steady)
  mean_rollout = sum(r[0] for r in steady) / n
  mean_update = sum(r[1] for r in steady) / n
  mean_iter = sum(r[2] for r in steady) / n

  print("\n== per-iteration wall time (s) ==")
  print(f"{'iter':>5} | {'rollout':>8} | {'ppo':>8} | {'total':>8}")
  for i, (r, u, t) in enumerate(rows):
    print(f"{i:5d} | {r:8.2f} | {u:8.2f} | {t:8.2f}")
  print("-" * 42)
  print(
    f"mean (skip {skip}) | {mean_rollout:8.2f} | {mean_update:8.2f} | {mean_iter:8.2f}"
  )
  print(f"\nenv stepping throughput: {spe / mean_rollout * args.num_envs:,.0f} env-steps/s")
  print(f"total bench wall time (incl. warmup): {total:.1f} s")


if __name__ == "__main__":
  main()
