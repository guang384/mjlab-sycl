# SPDX-License-Identifier: Apache-2.0
"""SYCL training with a live MuJoCo viewer window.

Launches the native MuJoCo viewer (env 0) while training runs on the Intel
GPU. Watch the duck: it starts flailing and (over iterations) starts walking.

How it works: mjlab 1.3.0 has no interactive native viewer (its render modes
are ``None``/``rgb_array`` only), so this entry mirrors env 0's state into a
plain cpu ``mujoco.MjData`` and drives the native passive viewer. Rendering is
decoupled from the training loop so it cannot stall physics:

  - the training thread takes a cheap snapshot of env 0 after every env step
    (a few hundred floats copied after the sim queue is drained);
  - a dedicated watcher thread copies the newest snapshot into the cpu
    MjData, runs mj_forward and calls viewer.sync() at ``--viewer-fps``
    (~20 fps default). The passive viewer does NOT simulate on its own
    (verified), so what you see is exactly env 0.

Usage:
    python -m mjlab_sycl.train_viewer <TASK_ID> --num-envs 1024
"""

import argparse
import dataclasses
import threading
import time
from pathlib import Path

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import torch  # noqa: E402
from mjlab_sycl._bootstrap import configure_torch_threads  # noqa: E402
configure_torch_threads()  # cap CPU threads (MJLAB_TORCH_THREADS)
import warp as wp  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402  (binds mujoco.viewer for launch_passive)

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def _snapshot_env0(env) -> dict:
  # Call after env.step (sim queue drained): fresh small arrays, swapped in
  # atomically so a concurrent watcher never sees a torn state.
  d = env.unwrapped.sim.data
  t = d.time
  return {
    "qpos": d.qpos.numpy()[0].copy(),
    "qvel": d.qvel.numpy()[0].copy(),
    "ctrl": d.ctrl.numpy()[0].copy(),
    "t": float(t.numpy()[0]) if hasattr(t, "numpy") else float(t),
  }


def _apply_state(mj_model, mj_data, state) -> None:
  mj_data.qpos[:] = state["qpos"]
  mj_data.qvel[:] = state["qvel"]
  mj_data.ctrl[:] = state["ctrl"]
  mj_data.time = state["t"]
  mujoco.mj_forward(mj_model, mj_data)  # refresh visual transforms


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task")
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=1000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--ppo-device", default=None)
  parser.add_argument("--run-name", default="viewer")
  parser.add_argument("--viewer-fps", type=float, default=20.0,
                      help="watcher-thread refresh rate (lower = less CPU, "
                           "choppier; higher = smoother but more overhead)")
  args = parser.parse_args()

  wp.init()
  patch_simulation_for_sycl()

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

  # -- env-0 snapshot plumbing -------------------------------------------------
  mj_model = env.unwrapped.sim.mj_model
  mj_data = mujoco.MjData(mj_model)
  state = _snapshot_env0(env)
  lock = threading.Lock()
  stop = threading.Event()

  try:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
  except Exception as e:
    print(f"[train-viewer] WARN: could not open the native viewer ({e!r}); "
          "training continues headless", flush=True)
    viewer = None

  if viewer is not None:
    # envs live on a ~60 m terrain grid (env 0 is NOT at the origin). Point
    # the free camera at env 0; the watcher keeps it centered as env 0 moves.
    _apply_state(mj_model, mj_data, state)
    try:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      viewer.cam.azimuth = 120.0
      viewer.cam.elevation = -25.0
      viewer.cam.distance = 0.9
      viewer.cam.lookat[:] = (float(state["qpos"][0]), float(state["qpos"][1]), 0.15)
    except Exception:
      pass
    print("[train-viewer] native viewer open (free camera on env 0): "
          "drag = rotate, wheel = zoom, right-drag = pan", flush=True)

  def watcher_loop():
    interval = 1.0 / max(1.0, args.viewer_fps)
    while not stop.is_set():
      if viewer is None or not viewer.is_running():
        stop.set()
        break
      t0 = time.perf_counter()
      with lock:
        snap = state
      try:
        _apply_state(mj_model, mj_data, snap)
        viewer.cam.lookat[:] = (float(snap["qpos"][0]), float(snap["qpos"][1]), 0.15)
        viewer.sync()
      except Exception:
        pass  # viewer closed mid-run: training continues headless
      elapsed = time.perf_counter() - t0
      if elapsed < interval:
        stop.wait(interval - elapsed)

  watcher = None
  if viewer is not None:
    watcher = threading.Thread(target=watcher_loop, daemon=True)
    watcher.start()

  orig_step = env.step

  def stepping(action):
    nonlocal state
    out = orig_step(action)
    try:
      with lock:
        state = _snapshot_env0(env)  # cheap: ~300 floats after the sim drain
    except Exception:
      pass  # snapshot failure must never stall training
    return out

  env.step = stepping

  log_dir = Path("logs") / agent_cfg.experiment_name / args.run_name
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), str(log_dir), ppo_device)
  try:
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  finally:
    stop.set()
    if watcher is not None:
      watcher.join(timeout=2.0)
    if viewer is not None:
      viewer.close()


if __name__ == "__main__":
  main()
