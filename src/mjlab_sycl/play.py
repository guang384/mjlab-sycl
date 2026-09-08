# SPDX-License-Identifier: Apache-2.0
"""Play a checkpoint in the native MuJoCo viewer WITHOUT training.

Loads a saved policy (rsl_rl ``model_XXXX.pt``) and rolls it out in real time
on the Intel GPU, mirroring env 0 into the native viewer -- no learning, no
policy updates, tiny physics load, so the window stays smooth. Commands come
from the task's own command manager (the duck walks/stops/turns on its own
schedule, like in training).

Usage:
    python -m mjlab_sycl.play <TASK_ID> --checkpoint <path to model_XXXX.pt>
        [--num-envs 4] [--no-realtime] [--ppo-device xpu|cpu]
"""

import argparse
import dataclasses
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
import mujoco.viewer  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402
from mjlab_sycl.train_viewer import _apply_state, _snapshot_env0  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task")
  parser.add_argument("--checkpoint", required=True,
                      help="path to a saved model_XXXX.pt checkpoint")
  parser.add_argument("--num-envs", type=int, default=4,
                      help="parallel envs rolled out (physics load stays tiny)")
  parser.add_argument("--ppo-device", default=None)
  parser.add_argument("--no-realtime", action="store_true",
                      help="step as fast as physics allows instead of real time")
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
  agent_cfg.upload_model = False
  agent_cfg.logger = "tensorboard"
  agent_cfg.experiment_name = "play"
  agent_cfg.save_interval = 10 ** 9

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ppo_device = args.ppo_device or ("xpu" if torch.xpu.is_available() else "cpu")

  # build the runner purely to host the policy, then load the checkpoint
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), None, ppo_device)
  runner.load(args.checkpoint)
  print(f"[play] checkpoint loaded: {args.checkpoint} | PPO on {ppo_device}", flush=True)

  # viewer mirror of env 0
  mj_model = env.unwrapped.sim.mj_model
  mj_data = mujoco.MjData(mj_model)
  env.reset()
  state = _snapshot_env0(env)
  _apply_state(mj_model, mj_data, state)

  try:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
  except Exception as e:
    print(f"[play] WARN: could not open the native viewer ({e!r}); "
          "rolling out headless", flush=True)
    viewer = None

  if viewer is not None:
    try:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      viewer.cam.azimuth = 120.0
      viewer.cam.elevation = -25.0
      viewer.cam.distance = 1.2
      viewer.cam.lookat[:] = (float(state["qpos"][0]), float(state["qpos"][1]), 0.15)
    except Exception:
      pass
    print("[play] native viewer open on env 0 -- close the window to stop", flush=True)

  # Decoupled watcher thread (same architecture as train_viewer, which keeps
  # the window alive and smooth): this thread refreshes the viewer, the main
  # loop below only rolls out + snapshots.
  stop = threading.Event()
  lock = threading.Lock()

  def watcher_loop():
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
        pass  # viewer closed mid-run: keep rolling out headless
      dt = time.perf_counter() - t0
      if dt < 1.0 / 30.0:
        stop.wait(1.0 / 30.0 - dt)

  watcher = None
  if viewer is not None:
    watcher = threading.Thread(target=watcher_loop, daemon=True)
    watcher.start()

  obs = env.get_observations().to(ppo_device)
  step_dt = float(env.unwrapped.step_dt)  # env dt (e.g. 20 ms)

  n = 0
  with torch.inference_mode():
    while not stop.is_set():
      t0 = time.perf_counter()
      actions = runner.alg.act(obs)
      obs, _rew, _dones, _extras = env.step(actions.to("cpu"))
      obs = obs.to(ppo_device)

      try:
        with lock:
          state = _snapshot_env0(env)
      except Exception:
        pass

      n += 1
      if not args.no_realtime:
        elapsed = time.perf_counter() - t0
        if elapsed < step_dt:
          time.sleep(step_dt - elapsed)
      if n % 500 == 0:
        print(f"[play] step={n} env0 x={float(state['qpos'][0]):.2f} "
              f"y={float(state['qpos'][1]):.2f}", flush=True)

  stop.set()
  if watcher is not None:
    watcher.join(timeout=2.0)
  if viewer is not None:
    viewer.close()
  print(f"[play] done after {n} steps", flush=True)


if __name__ == "__main__":
  main()
