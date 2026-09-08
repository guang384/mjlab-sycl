import sys, numpy as np, dataclasses
# historical debug probe: requires mjlab-sycl importable (editable install)
import warp as wp
from mjlab_sycl.runtime_patch import patch_simulation_for_sycl
patch_simulation_for_sycl()
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import mjlab.tasks.cartpole
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

cfg = load_env_cfg('Mjlab-Cartpole-Balance')
cfg.scene.num_envs = 64
acfg = load_rl_cfg('Mjlab-Cartpole-Balance')
acfg.max_iterations = 1
acfg.logger = 'tensorboard'; acfg.save_interval = 10**9; acfg.upload_model = False
acfg.experiment_name = 'hunt3'
env = ManagerBasedRlEnv(cfg=cfg, device='cpu')

d = env.sim.data
def scan(tag):
    bad = []
    for name in dir(d):
        if name.startswith('_'): continue
        try:
            arr = getattr(d, name)
            if type(arr).__name__ == 'array' and arr.size > 0:
                a = arr.numpy()
                n = int((~np.isfinite(a)).sum())
                if n: bad.append((name, n, arr.size, hex(arr.ptr)))
        except Exception: pass
    print(f"[{tag}] fields with non-finite: {bad}", flush=True)

scan('post-construct')
env.seed(0); torch.manual_seed(0)
env.reset()
scan('post-reset')

# hunt the wrapper/runner construction window
STATE = {'on': False, 'data': d}
STATE['on'] = True
wrap = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
STATE['on'] = False
scan('post-wrapper')
runner = MjlabOnPolicyRunner(wrap, dataclasses.asdict(acfg), 'logs/hunt3', 'cpu')
scan('post-runner')
# keep the hunt ON through the entire learn (alg.act + obs compute + steps)
print('[hunt] v8 staying armed through learn (both bindings)', flush=True)
runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
print('[hunt] learn finished clean', flush=True)
