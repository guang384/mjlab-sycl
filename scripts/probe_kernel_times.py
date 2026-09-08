# SPDX-License-Identifier: Apache-2.0
"""Per-kernel GPU execution time attribution on the sycl device.

Patches wp.launch so every submission is immediately followed by a queue
drain, attributing the elapsed time to the kernel's name. This serializes
execution (no pipelining), so absolute wall time is inflated, but the
*relative* ranking identifies which mujoco_warp kernels dominate device
time on the Intel GPU.

Usage:
    python scripts/probe_kernel_times.py --num-envs 1024 --steps 2
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import warp as wp

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument("--top", type=int, default=25)
  args = parser.parse_args()

  # sycl8.dll PATH ordering -- must run before torch/warp come up (see
  # _bootstrap); this script imports torch through mjlab below.
  from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

  prepare_sycl_runtime_path()

  wp.init()
  patch_simulation_for_sycl()

  KEPT = Counter()      # total (submit+wait) time per kernel
  WAITS = Counter()     # wait-only time per kernel
  COUNTS = Counter()    # launches per kernel
  DIMS = {}             # last seen dim per kernel

  orig_launch = wp.launch

  def timed_launch(kernel, dim, *args, **kwargs):
    dev = kwargs.get("device")
    if dev is None and len(args) >= 6:
      dev = args[5]
    try:
      is_sycl = wp.get_device(dev).is_sycl
    except Exception:
      is_sycl = False
    if is_sycl:
      wp.synchronize_device("sycl")  # pre-drain: attribute only this kernel's work
      t0 = time.perf_counter()
      out = orig_launch(kernel, dim, *args, **kwargs)
      t1 = time.perf_counter()
      wp.synchronize_device("sycl")
      t2 = time.perf_counter()
      KEPT[kernel.key] += t2 - t0
      WAITS[kernel.key] += t2 - t1
      COUNTS[kernel.key] += 1
      DIMS[kernel.key] = tuple(dim) if not isinstance(dim, int) else dim
      return out
    return orig_launch(kernel, dim, *args, **kwargs)

  wp.launch = timed_launch

  from mjlab.envs import ManagerBasedRlEnv

  env_cfg = None
  from mjlab.tasks.registry import load_env_cfg
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

  import torch
  obs, _ = env.reset()
  action = torch.zeros((args.num_envs, env.action_manager.total_action_dim))

  KEPT.clear()  # drop construction/warmup kernels
  for _ in range(args.steps):
    env.step(action)

  total = sum(KEPT.values())
  print(f"\n[profile] {args.steps} env steps, {len(KEPT)} kernels, "
        f"{total:.2f}s serialized device time "
        f"({total / args.steps * 1000:.0f} ms/step)")
  print(f"\n{'kernel':<60} {'total':>10} {'wait':>10} {'ms/launch':>10} {'launches':>9}  dim")
  for name, t in KEPT.most_common(args.top):
    n = COUNTS[name]
    w = WAITS[name] / args.steps * 1000
    print(f"{name[:60]:<60} {t / args.steps * 1000:10.1f} {w:10.1f} {t / max(n,1) * 1000:10.2f} {n // args.steps:9d}  {DIMS[name]}")


if __name__ == "__main__":
  main()
