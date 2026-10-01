# SPDX-License-Identifier: Apache-2.0
"""Env-0 mirror helpers shared by the native-viewer entries (train_viewer,
play): snapshot one training env's physics state after the sim queue has
drained, and apply it to a plain cpu ``mujoco.MjData`` the native passive
viewer renders. mjlab 1.3.0 has no interactive native viewer, so the entries
mirror env 0 this way (see train_viewer's module docstring for the watcher-
thread architecture both entries use).

Import this AFTER ``_bootstrap.prepare_sycl_runtime_path()`` like any module
that pulls in mujoco (mujoco itself is sycl8-independent, but the entries'
import order contract is uniform).
"""

import mujoco


def snapshot_env0(env) -> dict:
  # Call after env.step (sim queue drained): fresh small arrays, swapped in
  # atomically by the caller (under its lock) so a concurrent watcher never
  # sees a torn state.
  d = env.unwrapped.sim.data
  t = d.time
  return {
    "qpos": d.qpos.numpy()[0].copy(),
    "qvel": d.qvel.numpy()[0].copy(),
    "ctrl": d.ctrl.numpy()[0].copy(),
    "t": float(t.numpy()[0]) if hasattr(t, "numpy") else float(t),
  }


def apply_state(mj_model, mj_data, state) -> None:
  mj_data.qpos[:] = state["qpos"]
  mj_data.qvel[:] = state["qvel"]
  mj_data.ctrl[:] = state["ctrl"]
  mj_data.time = state["t"]
  mujoco.mj_forward(mj_model, mj_data)  # refresh visual transforms
