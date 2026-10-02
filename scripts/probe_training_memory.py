# SPDX-License-Identifier: Apache-2.0
"""Training-loop memory census (both sides of the unified pool).

The env-stepping census (probe_memory.py) covers the warp/USM side; the
PPO update runs on torch.xpu and has its own pools. This probe mirrors
bench's full training loop and reports, per phase: process RSS, torch
XPU allocated/reserved, and the warpsycl USM pool census. On Lunar Lake
all of it draws from the same physical LPDDR5X; "GPU memory" in Task
Manager is the USM + xpu pools.

Usage:
    python scripts/probe_training_memory.py --num-envs 4096 --iters 2
"""

import argparse
import dataclasses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()

    from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

    prepare_sycl_runtime_path()
    import psutil
    import torch
    from mjlab_sycl._bootstrap import configure_torch_threads

    configure_torch_threads()
    import warp as wp

    wp.init()
    from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

    patch_simulation_for_sycl()
    proc = psutil.Process()

    def mem(tag):
        from mjlab_sycl import native_kernels

        lv, fb, pd = native_kernels.pool_stats()
        xa = torch.xpu.memory_allocated() / 2**20
        xr = torch.xpu.memory_reserved() / 2**20
        ca = cr = 0.0
        try:
            st = torch.cpu.memory_stats()
            ca = st.get("allocated_bytes.all.current", 0) / 2**20
            cr = st.get("reserved_bytes.all.current", 0) / 2**20
        except Exception:
            pass
        print(f"[{tag:14s}] rss={proc.memory_info().rss / 2**20:7.0f} MB"
              f"   xpu: alloc={xa:6.0f} res={xr:6.0f}"
              f"   cpu-torch: alloc={ca:6.0f} res={cr:6.0f}"
              f"   usm: live={lv / 2**20:.0f} free={fb / 2**20:.0f} MB",
              flush=True)

    mem("start")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    agent_cfg = load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck")
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.max_iterations = args.iters
    agent_cfg.logger = "tensorboard"  # bench's choice; wandb needs login

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    mem("env built")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_device = "xpu" if torch.xpu.is_available() else "cpu"
    import tempfile

    log_dir = tempfile.mkdtemp(prefix="mjlab_memprobe_")
    runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent_cfg), log_dir, ppo_device)
    mem("runner init")

    obs, _ = env.reset()
    mem("reset")

    # one learn iteration triggers rollout storage + policy + optimizer allocs
    import tracemalloc

    tracemalloc.start(25)
    snap0 = tracemalloc.take_snapshot()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    mem("after learn")
    snap1 = tracemalloc.take_snapshot()
    diff = snap1.compare_to(snap0, "lineno")
    print("[py-heap delta top10]")
    for stat in diff[:10]:
        print(f"  {stat.size_diff / 2**20:7.1f} MB  {stat}")
    tracemalloc.stop()

    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    try:
        torch.cpu.empty_cache()
    except Exception:
        pass
    mem("after empty_cache")


if __name__ == "__main__":
    main()
