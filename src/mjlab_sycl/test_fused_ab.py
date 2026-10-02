# SPDX-License-Identifier: Apache-2.0
"""Unit-level correctness test for the fused tree chains + 1-step
end-to-end sanity for the fused solver zeros.

Chain tests run the FUSED and ORIGINAL versions of com_pos / crb /
_rne_cfrc_backward / subtree_vel on IDENTICAL inputs and compare outputs.
The only legitimate difference is float accumulation order (the originals
use wp.atomic_add over siblings, which is order-nondeterministic), so the
tolerance is 1e-4 absolute on O(1)-magnitude buffers.

End-to-end fusion equivalence lives in the patched-stack physics gate
(test_patched: fused pipeline vs cpu reference at ~2e-08). An earlier
cross-build one-step qacc comparison was removed here: it compared
separately constructed envs, where construction nondeterminism + chaotic
contact make the fused-vs-unfused delta indistinguishable from build
noise (multi-step comparisons are meaningless — 1e-7 seed noise amplifies to O(10) qacc error by step 8).
"""

import os
import sys

# Ensure the package is importable when run directly (e.g. python -m mjlab_sycl.test_fused_ab
# from the repo root); if already installed in the venv, this is a no-op.
_repo_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import warp as wp

wp.init()

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

patch_simulation_for_sycl()

import numpy as np
import torch

torch.set_num_threads(2)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TOL = 1e-4


def make_env(n_envs: int = 256):
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
  env_cfg.scene.num_envs = n_envs
  env_cfg.seed = 12345
  return ManagerBasedRlEnv(cfg=env_cfg, device="cpu")


def sync():
  wp.synchronize_device("sycl")


def compare(tag, a, b, tol=TOL):
  a = np.asarray(a)
  b = np.asarray(b)
  diff = np.abs(a - b)
  print(f"[unit] {tag}: max_abs={diff.max():.3e} shape={a.shape}")
  assert diff.max() < tol, f"{tag}: diff {diff.max():.3e} >= {tol}"
  assert not np.isnan(a).any(), f"{tag}: NaN in fused output"


def chain_tests():
  from mujoco_warp._src import smooth

  env = make_env()
  env.reset()
  action = torch.zeros(
    (256, env.action_manager.total_action_dim), dtype=torch.float32, device="cpu"
  )
  action[:, 0] = 0.1
  action[:, 5] = -0.2
  for _ in range(3):
    env.step(action)
  sync()

  m = env.sim._wp_model
  d = env.sim._wp_data

  def snap(*arrs):
    sync()
    return tuple(a.numpy().copy() for a in arrs)

  # ---- com_pos: inputs (xipos/xmat/xanchor/xaxis) are read-only ----
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"
  with wp.ScopedDevice("sycl"):
    smooth.com_pos(m, d)
  com_f, cinert_f, cdof_f = snap(d.subtree_com, d.cinert, d.cdof)

  os.environ["MJLAB_SYCL_FUSED_TREE"] = "0"
  with wp.ScopedDevice("sycl"):
    smooth.com_pos(m, d)
  com_o, cinert_o, cdof_o = snap(d.subtree_com, d.cinert, d.cdof)
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"

  compare("com_pos.subtree_com", com_f, com_o)
  compare("com_pos.cinert", cinert_f, cinert_o)
  compare("com_pos.cdof", cdof_f, cdof_o)

  # ---- crb: reads cinert (unchanged), writes crb + qM ----
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"
  with wp.ScopedDevice("sycl"):
    smooth.crb(m, d)
  crb_f, qm_f = snap(d.crb, d.qM)

  os.environ["MJLAB_SYCL_FUSED_TREE"] = "0"
  with wp.ScopedDevice("sycl"):
    smooth.crb(m, d)
  crb_o, qm_o = snap(d.crb, d.qM)
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"

  compare("crb.crb", crb_f, crb_o)
  compare("crb.qM", qm_f, qm_o)

  # ---- _rne_cfrc_backward: IN-PLACE on cfrc_int — save/restore ----
  saved = wp.clone(d.cfrc_int)
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"
  with wp.ScopedDevice("sycl"):
    smooth._rne_cfrc_backward(m, d)
  cfrc_f = snap(d.cfrc_int)[0]

  wp.copy(d.cfrc_int, saved)
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "0"
  with wp.ScopedDevice("sycl"):
    smooth._rne_cfrc_backward(m, d)
  cfrc_o = snap(d.cfrc_int)[0]
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"

  compare("cfrc_backward.cfrc_int", cfrc_f, cfrc_o)

  # ---- subtree_vel: reads read-only inputs, writes linvel/angmom ----
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"
  with wp.ScopedDevice("sycl"):
    smooth.subtree_vel(m, d)
  lv_f, am_f = snap(d.subtree_linvel, d.subtree_angmom)

  os.environ["MJLAB_SYCL_FUSED_TREE"] = "0"
  with wp.ScopedDevice("sycl"):
    smooth.subtree_vel(m, d)
  lv_o, am_o = snap(d.subtree_linvel, d.subtree_angmom)
  os.environ["MJLAB_SYCL_FUSED_TREE"] = "1"

  compare("subtree_vel.subtree_linvel", lv_f, lv_o)
  compare("subtree_vel.subtree_angmom", am_f, am_o)


def main() -> None:
  chain_tests()
  print("\nALL FUSION TESTS PASSED")


if __name__ == "__main__":
  main()

