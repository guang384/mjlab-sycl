# SPDX-License-Identifier: Apache-2.0
"""Drain call-site attribution for one mjlab training step on the sycl device.

Wraps ``wp_sycl_synchronize`` and records, per drain, the innermost
non-warp call frames -- answers "which wrapper drains how often per step"
for the per-step drain census (probe_sycl_profile.py counts totals only).

Usage:
    python scripts/probe_drain_sites.py --num-envs 4096 --steps 20
"""

import argparse
import traceback
from collections import Counter


def _sig(stack) -> str:
  """Leaf-side signature: the 3 innermost non-warp frames."""
  frames = []
  for fr in reversed(stack):
    fn = fr.filename.replace("\\", "/")
    if "/warp/" in fn or fn.endswith("probe_drain_sites.py"):
      continue
    frames.append(f"{fn.split('/')[-1]}:{fr.lineno} {fr.name}")
    if len(frames) >= 3:
      break
  return " <- ".join(frames) if frames else "<unknown>"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--steps", type=int, default=20)
  args = parser.parse_args()

  # sycl8.dll PATH ordering -- must run before torch/warp come up (see
  # mjlab_sycl._bootstrap); this script imports torch through mjlab below.
  from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

  prepare_sycl_runtime_path()

  import warp as wp

  from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

  wp.init()
  patch_simulation_for_sycl()

  sycl = wp._src.context.runtime.sycl
  orig_sync = sycl.wp_sycl_synchronize

  sites = Counter()
  totals = Counter()

  def sync():
    sites[_sig(traceback.extract_stack()[:-1])] += 1
    totals["drain"] += 1
    orig_sync()

  sycl.wp_sycl_synchronize = sync

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  task = "Mjlab-Velocity-Flat-MicroDuck"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  obs, _ = env.reset()
  n_actions = env.action_manager.total_action_dim

  import torch

  sites.clear()
  totals.clear()
  per_step = []
  for _ in range(args.steps):
    before = totals["drain"]
    action = torch.zeros((env.num_envs, n_actions))
    obs, rew, term, trunc, info = env.step(action)
    per_step.append(totals["drain"] - before)

  print(f"\n[drains] {totals['drain']} drains over {args.steps} steps")
  print(f"[drains] per step: min={min(per_step)} max={max(per_step)} "
        f"mean={totals['drain'] / args.steps:.2f}")
  print("[drains] by call site:")
  for site, n in sites.most_common():
    print(f"  {n:5d}  ({n / args.steps:5.2f}/step)  {site}")


if __name__ == "__main__":
  main()
