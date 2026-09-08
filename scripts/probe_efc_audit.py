# SPDX-License-Identifier: Apache-2.0
"""Audit real constraint usage vs the compiled dense buffers.

mujoco_warp sizes its dense efc arrays to the compiled njmax (the (nworld,
njmax) grids seen in the constraint/solver kernels). If the real per-world
efc count is far below njmax, those wide kernels pay for idle padding lanes
on every launch. This probe reports the compiled njmax/nconmax and the peak
per-world nefc/ncon observed while stepping a task, so a buffer-redundancy
decision can be made on data.

Read-only: it never touches model files or the mjlab install -- it only reads
what a task already compiles. Usage:

    python scripts/probe_efc_audit.py --task Mjlab-Velocity-Flat-MicroDuck \
        --num-envs 512 --steps 40
"""

import argparse

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import torch  # noqa: E402
import warp as wp  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--steps", type=int, default=40)
  args = parser.parse_args()

  wp.init()
  patch_simulation_for_sycl()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

  m = env.sim.mj_model
  print(f"[audit] mj_model: njmax={m.njmax} nconmax={m.nconmax}")
  d = env.sim.data
  print(f"[audit] compiled buffers: nefc cap seen as kernel dim (njmax pad); "
        f"warp d.efc arrays: {[f'{k}={getattr(d.efc, k).shape}' for k in ('J',) if hasattr(getattr(d.efc, 'J', None), 'shape')]}")

  action = torch.zeros((args.num_envs, env.action_manager.total_action_dim))
  peak_efc = peak_con = peak_active = 0
  env.reset()
  for i in range(args.steps):
    env.step(action)
    nefc = d.nefc.numpy()
    peak_efc = max(peak_efc, int(nefc.max()))
    ncon = getattr(d, "ncon", None)
    if ncon is not None and hasattr(ncon, "numpy"):
      peak_con = max(peak_con, int(ncon.numpy().max()))
    # worlds with zero contacts (fallen/inactive constraints are still rows)
    nz = int((nefc > 0).sum())
    peak_active = max(peak_active, nz)

  print(f"[audit] over {args.steps} steps ({args.num_envs} envs):")
  print(f"  peak per-world nefc   = {peak_efc}")
  print(f"  peak per-world ncon   = {peak_con}")
  print(f"  worlds with nefc>0 at peak = {peak_active}/{args.num_envs}")
  # kernel grids use njmax_pad from the compiled model; report if visible
  j = getattr(d.efc, "J", None)
  if j is not None and hasattr(j, "shape"):
    buf = j.shape[1]
    print(f"  efc_J buffer rows     = {buf}  -> ratio buffer/peak = "
          f"{buf / max(peak_efc, 1):.1f}x")
  else:
    print("  efc_J buffer rows     = <not exposed>")


if __name__ == "__main__":
  main()
