# SPDX-License-Identifier: Apache-2.0
"""Smooth CPU playback of a checkpoint (approximate physics).

The sycl/mjlab pipeline costs ~120-140 ms per env.step regardless of env
count (fixed kernel-launch overhead), which caps live viewing at ~8 fps.
This module plays a checkpoint in PLAIN CPU MuJoCo instead -- mj_step for a
single duck is sub-ms, so motion is smooth realtime.

IMPORTANT fidelity caveat: plain MuJoCo lacks the BAM voltage-actuator model
and the exact observation stack, so this is an APPROXIMATE replay (position
servos + a hand-built 61-D obs): good for watching smooth behavior and rough
sim2real sanity, NOT a faithful re-run of the trained environment.

Usage:
    python -m mjlab_sycl.cpu_replay <TASK_ID> --checkpoint <model_XXXX.pt>
        [--vx 0.4] [--max-s 120]
"""

import argparse
import dataclasses
import re
import time
from pathlib import Path

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402


def _servo_names(m: mujoco.MjModel):
  return [m.jnt(j).name for j in range(m.njnt)
          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def build_cpu_model(robot_xml: str):
  """Robot model + a position actuator per servo (approximates BAM).

  Returns (model, ordered servo names, name -> ctrl index). If the xml
  already carries one actuator per servo (some revisions of the robot file
  do), it is reused; otherwise plain position actuators are injected.
  """
  text = Path(robot_xml).read_text(encoding="utf-8")
  meshdir = str(Path(robot_xml).resolve().parent / "assets")
  text = re.sub(r'meshdir="[^"]*"', lambda m: f'meshdir="{meshdir}"', text, count=1)
  # robot_walk.xml carries no light/floor/background (scene_walk.xml adds
  # them) -- inject a headlight, a directional light and a floor.
  text = text.replace(
    "</mujoco>",
    '  <visual>\n    <headlight diffuse="0.6 0.6 0.6" ambient="0.45 0.45 0.45" '
    'specular="0 0 0"/>\n    <global azimuth="160" elevation="-20"/>\n  </visual>\n</mujoco>',
  )
  wb = text.index("<worldbody>") + len("<worldbody>")
  text = (
    text[:wb]
    + '\n  <light pos="0 0 4" dir="0 0 -1" directional="true"/>\n'
    + '  <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"/>\n'
    + text[wb:]
  )
  base = mujoco.MjModel.from_xml_string(text)
  names = _servo_names(base)

  if base.nu == len(names):
    # reuse the xml's own actuators; find ctrl index per servo joint name
    idx = {}
    for a in range(base.nu):
      jid = base.actuator_trnid[a, 0] if base.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT else -1
      if jid >= 0:
        idx[base.jnt(jid).name] = a
    missing = [n for n in names if n not in idx]
    if not missing:
      return base, names, idx
  elif base.nu != 0:
    raise RuntimeError(f"robot xml already has {base.nu} actuators for "
                       f"{len(names)} servos -- cannot infer ctrl layout")

  act = "".join(
    f'    <position joint="{n}" kp="6" kv="0.25" ctrlrange="-1 1"/>\n' for n in names
  )
  text = text.replace("</mujoco>", f"  <actuator>\n{act}  </actuator>\n</mujoco>")
  model = mujoco.MjModel.from_xml_string(text)
  idx = {n: i for i, n in enumerate(names)}
  return model, names, idx


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--vx", type=float, default=0.4)
  parser.add_argument("--max-s", type=float, default=120.0)
  parser.add_argument("--robot-xml", default=None)
  args = parser.parse_args()

  wp.init()
  patch_simulation_for_sycl()  # sycl only to host the mjlab policy loader

  import importlib.util
  if args.robot_xml is None:
    md_dir = Path(importlib.util.find_spec("mjlab_microduck").origin).parent
    robot_xml = str(md_dir / "robot" / "microduck" / "robot_walk.xml")
  else:
    robot_xml = args.robot_xml

  # ---- load policy through mjlab (this env is never stepped) ---------------
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  agent_cfg.upload_model = False
  agent_cfg.logger = "tensorboard"
  agent_cfg.save_interval = 10 ** 9
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), None, "cpu")
  runner.load(args.checkpoint)
  policy = runner.alg
  print(f"[cpu_replay] checkpoint loaded: {args.checkpoint}", flush=True)

  # ---- cpu mujoco model -----------------------------------------------------
  model, names, act_idx = build_cpu_model(robot_xml)
  data = mujoco.MjData(model)
  n_servo = len(names)

  # standing start (STAND keyframe) as the HOME pose
  key = None
  for i in range(model.nkey):
    if model.key(i).name == "STAND":
      key = i
      break
  if key is not None:
    data.qpos[:] = model.key_qpos[key]
    home = np.array(model.key_qpos[key][7:], dtype=np.float64)
  else:
    data.qpos[2] = 0.12
    data.qpos[3] = 1.0
    home = np.zeros(n_servo)
  mujoco.mj_forward(model, data)
  print(f"[cpu_replay] cpu model: {model.nbody} bodies, {n_servo} servos "
        f"@ {model.opt.timestep*1000:.1f} ms", flush=True)

  # servo qpos column / dof index per name
  qcol = {name: None for name in names}
  dof = {name: None for name in names}
  for j in range(model.njnt):
    n = model.jnt(j).name
    if n in qcol:
      qcol[n] = model.jnt_qposadr[j]
      dof[n] = model.jnt_dofadr[j]

  # ---- viewer ---------------------------------------------------------------
  try:
    viewer = mujoco.viewer.launch_passive(model, data)
  except Exception as e:
    print(f"[cpu_replay] WARN no viewer: {e!r}", flush=True)
    viewer = None
  if viewer is not None:
    try:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      viewer.cam.azimuth = 120.0
      viewer.cam.elevation = -25.0
      viewer.cam.distance = 1.4
      viewer.cam.lookat[:] = (0.0, 0.0, 0.2)
    except Exception:
      pass
    print("[cpu_replay] viewer open -- close window to stop", flush=True)

  dt = float(model.opt.timestep)
  t_start = time.perf_counter()
  nsteps = 0
  action = np.zeros(n_servo)
  prev_q = data.qpos[3:7].copy()

  def _body_angvel():
    # crude gyro: trunk angular velocity in body frame from quat finite diff
    q = data.qpos[3:7]
    q /= np.linalg.norm(q)
    dq = q - prev_q
    ang_world = 2.0 * np.array([q[0] * dq[1] - q[1] * dq[0] - q[2] * dq[3] + q[3] * dq[2],
                                q[0] * dq[2] + q[1] * dq[3] - q[2] * dq[0] - q[3] * dq[1],
                                q[0] * dq[3] - q[1] * dq[2] + q[2] * dq[1] - q[3] * dq[0]]) / dt
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ ang_world

  def _obs():
    q = data.qpos[3:7]
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    grav = (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
    pos = np.array([data.qpos[qcol[n]] for n in names], dtype=np.float32)
    vel = np.array([data.qvel[dof[n]] for n in names], dtype=np.float32)
    ang = _body_angvel().astype(np.float32)
    cmd = np.array([args.vx, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([ang, grav, pos - home, vel, action, cmd,
                           np.zeros(4, np.float32), np.zeros(6, np.float32)]).astype(np.float32)

  with torch.inference_mode():
    from tensordict import TensorDict
    critic_zero = torch.zeros(1, 76, device="cpu")
    while True:
      if viewer is not None and not viewer.is_running():
        break
      if time.perf_counter() - t_start > args.max_s:
        break
      obs = _obs()
      assert obs.shape[0] == 61, obs.shape
      td = TensorDict(
        {"actor": torch.tensor(obs[None], device="cpu"), "critic": critic_zero},
        batch_size=(1,),
      )
      a = policy.actor(td, stochastic_output=True)
      action = np.clip(a[0].numpy(), -1.0, 1.0)
      for i, n in enumerate(names):
        data.ctrl[act_idx[n]] = np.clip(home[i] + 0.5 * action[i], -1.0, 1.0)

      prev_q = data.qpos[3:7].copy()
      mujoco.mj_step(model, data)

      if viewer is not None:
        try:
          viewer.sync()
        except Exception:
          pass
      nsteps += 1
      if nsteps % 400 == 0:
        print(f"[cpu_replay] step={nsteps} x={data.qpos[0]:.2f} z={data.qpos[2]:.2f} "
              f"nsteps/s={nsteps/(time.perf_counter()-t_start):.0f}", flush=True)
      # pace to real time (50 Hz physics)
      want = nsteps * dt
      now = time.perf_counter() - t_start
      if now < want:
        time.sleep(want - now)

  if viewer is not None:
    viewer.close()
  print(f"[cpu_replay] done after {nsteps} steps", flush=True)


if __name__ == "__main__":
  main()
