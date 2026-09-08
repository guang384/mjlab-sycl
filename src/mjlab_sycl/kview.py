# SPDX-License-Identifier: Apache-2.0
"""Watch MANY ducks at once -- K parallel envs rendered side by side.

mjlab's CPU model holds a single env (GPU envs are batched arrays, not cloned
bodies), so the plain viewer shows env 0 only. This module builds a K-duck CPU
model by duplicating the robot's body subtree K times (names prefixed so the
compiler accepts them) and, every frame, mirrors the joint state of K chosen
envs from the sim into the K copies -- a theater row of ducks that are all
training/acting in parallel.

Pure viewer/adaptor code: it never modifies microduck_rl sources; the K-model
is compiled in memory from the robot XML text.

Usage:
    python -m mjlab_sycl.kview <TASK_ID> [--checkpoint path] [--show-envs K]
        [--num-envs N] [--no-realtime]
  (without --checkpoint the actor is random noise: what training looks like
   at iteration 0)
"""

import argparse
import os
import re
import threading
import time
from pathlib import Path

# sycl8.dll PATH ordering -- must run before torch/warp come up (see _bootstrap)
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import torch  # noqa: E402
import warp as wp  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl  # noqa: E402

_ATTR_NAME = re.compile(r'(?<=\bname=")[^"]+')


def _extract_duck_body(xml: str) -> str:
  """Return the duck's top-level <body ...>...</body> subtree (root trunk_base)."""
  start = xml.index('<body name="trunk_base"')
  i = start
  depth = 0
  while i < len(xml):
    o = xml.find("<body", i)
    c = xml.find("</body>", i)
    if o == -1 or (c != -1 and c < o):
      depth -= 1
      if depth == 0:
        return xml[start:c + len("</body>")]
      i = c + len("</body>")
    else:
      depth += 1
      i = o + 5
  raise RuntimeError("could not extract duck <body> subtree")


def build_k_model(robot_xml: str, k: int, spacing: float = 0.75):
  """Compile an in-memory model with k ducks in a row.

  Returns (mj_model, per-duck column maps, per-duck slot x). Each duck copy is
  renamed with a per-slot prefix; duck 0 keeps its original names. The maps
  give, for every duck, the (model qpos index -> sim qpos column) pairs to
  copy so the mirror is exact regardless of compiler ordering.
  """
  text = Path(robot_xml).read_text(encoding="utf-8")
  # mesh assets resolve relative to meshdir; make it absolute so compiling
  # from a string works from any cwd. (lambda replacement: the Windows path
  # contains backslashes that re would otherwise treat as escapes)
  meshdir = str(Path(robot_xml).resolve().parent / "assets")
  text = re.sub(r'meshdir="[^"]*"', lambda m: f'meshdir="{meshdir}"', text, count=1)

  # compile the single-duck model once to learn the original joint layout
  single = mujoco.MjModel.from_xml_string(text)

  def _jnt_qwidth(j):
    t = single.jnt_type[j]
    if t == mujoco.mjtJoint.mjJNT_FREE:
      return 7
    if t == mujoco.mjtJoint.mjJNT_BALL:
      return 4
    return 1

  # original joint names in qpos order + each joint's qpos start in the
  # single-duck model (columns inside a joint block share jnt_qposadr, so the
  # mirror must add the intra-block offset)
  jnt_names_by_col = {}
  jnt_qpos_start = {}
  for j in range(single.njnt):
    for q in range(single.jnt_qposadr[j], single.jnt_qposadr[j] + _jnt_qwidth(j)):
      jnt_names_by_col[q] = single.jnt(j).name
      jnt_qpos_start[q] = single.jnt_qposadr[j]
  nq_env = single.nq

  if k == 1:
    return single, [{c: c for c in range(nq_env)}], [0.0], nq_env

  # robot_walk.xml carries no light/floor/background (scene_walk.xml adds
  # them), so the k-model would render pitch black -- inject a headlight,
  # a light and a VISIBLE checkered floor into the composed model.
  text = text.replace(
    "</mujoco>",
    '  <asset>\n'
    '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" '
    'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" '
    'width="300" height="300"/>\n'
    '    <material name="groundplane" texture="groundplane" texuniform="true" '
    'texrepeat="5 5" reflectance="0.2"/>\n'
    '  </asset>\n'
    '  <visual>\n    <headlight diffuse="0.6 0.6 0.6" ambient="0.45 0.45 0.45" '
    'specular="0 0 0"/>\n    <global azimuth="160" elevation="-20"/>\n  </visual>\n'
    '</mujoco>',
  )
  wb = text.index("<worldbody>") + len("<worldbody>")
  text = (
    text[:wb]
    + '\n  <light pos="0 0 4" dir="0 0 -1" directional="true"/>\n'
    + '  <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" '
    + 'material="groundplane"/>\n'
    + text[wb:]
  )

  # Only the feet have collision geoms in this model (mesh geoms do not
  # collide in MuJoCo), so a fallen duck would pass through the floor: add
  # invisible body capsules as an approximate shell so any fall pose rests
  # ON the ground.
  body_caps = (
    '    <geom type="capsule" fromto="0 0 -0.04  0 0 0.02" size="0.055" '
    'contype="1" conaffinity="1" group="2" rgba="0 0 0 0"/>\n'
    '    <geom type="capsule" fromto="-0.10 0 0  0.11 0 0" size="0.05" '
    'contype="1" conaffinity="1" group="2" rgba="0 0 0 0"/>\n'
    '    <geom type="capsule" fromto="0 -0.055 0  0 0.055 0" size="0.05" '
    'contype="1" conaffinity="1" group="2" rgba="0 0 0 0"/>\n'
  )
  idx = text.index('<freejoint name="trunk_base_freejoint"/>')
  idx = text.index("/>", idx) + 2
  text = text[:idx] + "\n" + body_caps + text[idx:]

  body = _extract_duck_body(text)
  clones = []
  for i in range(1, k):
    pre = f"d{i}_"
    dup = _ATTR_NAME.sub(lambda m: pre + m.group(0), body)
    clones.append(dup)  # top-level free-joint bodies (MuJoCo requirement)
  multi_text = text.replace("</worldbody>", "\n".join(clones) + "\n</worldbody>")
  multi = mujoco.MjModel.from_xml_string(multi_text)

  maps = []
  for i in range(k):
    pre = "" if i == 0 else f"d{i}_"
    col_map = []
    for col in range(nq_env):
      name = pre + jnt_names_by_col[col]
      # find the joint by name
      for j in range(multi.njnt):
        if multi.jnt(j).name == name:
          # model index = multi block start + intra-block offset from the
          # single-duck layout
          col_map.append((multi.jnt_qposadr[j] + col - jnt_qpos_start[col], col))
          break
      else:
        raise RuntimeError(f"joint {name} not found in multi model")
    maps.append(col_map)
  slots = [spacing * i for i in range(k)]
  return multi, maps, slots, nq_env


def mirror_k(mj_model, mj_data, qpos_row, maps, slots, nq_env):
  """Copy env states (one qpos row per duck) into the k-duck model.

  Free-joint bodies must sit at world x/y from qpos, and the sim envs live on
  a 60 m terrain grid -- so each duck's local x/y is replaced by its theater
  slot (x=slot_i, y=0). Everything else mirrors the env state.
  """
  mj_data.qpos[:] = 0.0
  for state, col_map, slot in zip(qpos_row, maps, slots):
    for adr, col in col_map:
      val = state[col]
      if col == 0:
        val = slot
      elif col == 1:
        val = 0.0
      mj_data.qpos[adr] = val
  mujoco.mj_forward(mj_model, mj_data)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--show-envs", type=int, default=6,
                      help="how many ducks to render side by side (<= num-envs)")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--spacing", type=float, default=0.8)
  parser.add_argument("--ppo-device", default=None)
  parser.add_argument("--no-realtime", action="store_true")
  parser.add_argument("--robot-xml", default=None,
                      help="robot xml used to build the k-duck model "
                           "(auto-resolved from the task registry if omitted)")
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
  agent_cfg.save_interval = 10 ** 9

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ppo_device = args.ppo_device or ("xpu" if torch.xpu.is_available() else "cpu")
  policy = None
  if args.checkpoint:
    runner = MjlabOnPolicyRunner(env, dict(agent_cfg), None, ppo_device)
    runner.load(args.checkpoint)
    policy = runner.alg
    print(f"[kview] checkpoint loaded: {args.checkpoint}", flush=True)
  else:
    print("[kview] no checkpoint: random-noise actor (iteration-0 look)", flush=True)

  # -- build the k-duck model -------------------------------------------------
  robot_xml = args.robot_xml
  if robot_xml is None:
    # default: the walk-model robot xml shipped with the microduck task package
    import importlib.util
    md_dir = Path(importlib.util.find_spec("mjlab_microduck").origin).parent
    robot_xml = str(md_dir / "robot" / "microduck" / "robot_walk.xml")
  if not Path(robot_xml).exists():
    raise SystemExit(f"--robot-xml not found: {robot_xml}")

  k = max(1, min(args.show_envs, args.num_envs))
  mj_model, maps, slots, nq_env = build_k_model(robot_xml, k, args.spacing)
  mj_data = mujoco.MjData(mj_model)
  print(f"[kview] k-duck model built: {k} ducks x {nq_env} qpos | "
        f"bodies={mj_model.nbody}", flush=True)

  # -- rollout + viewer -------------------------------------------------------
  env.reset()
  sim = env.unwrapped.sim.data
  env_ids = list(range(k))

  def grab_rows():
    return sim.qpos.numpy()[env_ids]  # (k, nq_env)

  viewer = None
  try:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
  except Exception as e:
    print(f"[kview] WARN: no native viewer ({e!r}); headless rollout", flush=True)

  if viewer is not None:
    mirror_k(mj_model, mj_data, grab_rows(), maps, slots, nq_env)
    try:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      viewer.cam.azimuth = 120.0
      viewer.cam.elevation = -25.0
      viewer.cam.distance = 0.5 * k * args.spacing
      viewer.cam.lookat[:] = ((k - 1) * args.spacing / 2.0, 0.0, 0.15)
    except Exception:
      pass
    print(f"[kview] native viewer open: {k} ducks (drag/wheel to look around; "
          "close window to stop)", flush=True)

  stop = threading.Event()
  lock = threading.Lock()
  last_rows = grab_rows()

  def watcher_loop():
    while not stop.is_set():
      if viewer is None or not viewer.is_running():
        stop.set()
        break
      t0 = time.perf_counter()
      with lock:
        rows = last_rows
      try:
        mirror_k(mj_model, mj_data, rows, maps, slots, nq_env)
        viewer.sync()
      except Exception:
        pass
      dt = time.perf_counter() - t0
      if dt < 1.0 / 30.0:
        stop.wait(1.0 / 30.0 - dt)

  watcher = None
  if viewer is not None:
    watcher = threading.Thread(target=watcher_loop, daemon=True)
    watcher.start()

  obs = env.get_observations().to(ppo_device)
  step_dt = float(env.unwrapped.step_dt)
  if sim.qpos.shape[1] != nq_env:
    raise SystemExit(f"sim qpos width {sim.qpos.shape[1]} != robot model {nq_env}")
  n = 0
  with torch.inference_mode():
    while not stop.is_set():
      t0 = time.perf_counter()
      if policy is not None:
        actions = policy.act(obs)
      else:
        actions = torch.randn(args.num_envs, 14, device=ppo_device) * 0.35
      obs, _r, _d, _e = env.step(actions.to("cpu"))
      obs = obs.to(ppo_device)
      try:
        with lock:
          last_rows = grab_rows()
      except Exception:
        pass
      n += 1
      if not args.no_realtime:
        elapsed = time.perf_counter() - t0
        if elapsed < step_dt:
          time.sleep(step_dt - elapsed)
      if n % 400 == 0:
        print(f"[kview] step={n}", flush=True)

  stop.set()
  if watcher is not None:
    watcher.join(timeout=2.0)
  if viewer is not None:
    viewer.close()
  print(f"[kview] done after {n} steps", flush=True)


if __name__ == "__main__":
  main()
