# SPDX-License-Identifier: Apache-2.0
"""Memory census for the training stack at scale.

Reports where runtime memory goes at N envs: process RSS per phase, a
field-level breakdown of the warp Model/Data arrays, and the total live
torch CPU tensors. Read-only.

Usage:
    python scripts/probe_memory.py --num-envs 4096 --steps 3
"""

import argparse
import dataclasses


def array_bytes(a) -> int:
    try:
        n = 1
        for s in a.shape:
            n *= s
        import numpy as _np

        return n * _np.dtype(a.dtype).itemsize
    except Exception:
        return 0


def census_struct(obj, buckets: dict) -> int:
    total = 0
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name, None)
        if v is None:
            continue
        if hasattr(v, "shape") and hasattr(v, "dtype") and not isinstance(v, (list, tuple)):
            b = array_bytes(v)
            total += b
            buckets[f"{type(obj).__name__}.{f.name}"] = b
        elif dataclasses.is_dataclass(v) and not isinstance(v, type):
            total += census_struct(v, buckets)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3)
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
        print(f"[{tag:12s}] rss={proc.memory_info().rss / 2**20:8.0f} MB"
              f"   pool: live={lv / 2**20:.0f} free={fb / 2**20:.0f}"
              f" pend={pd / 2**20:.0f} MB", flush=True)

    mem("start")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    mem("env built")

    env.reset()
    mem("reset")

    action = torch.zeros((args.num_envs, env.action_manager.total_action_dim))
    for _ in range(args.steps):
        env.step(action)
    mem(f"{args.steps} steps")

    # precise breakdown: warp Model / Data fields
    buckets = {}
    mt = census_struct(env.sim._wp_model, buckets)
    dt = census_struct(env.sim._wp_data, buckets)
    print(f"\n[warp Model] {mt / 2**20:.0f} MB   [warp Data] {dt / 2**20:.0f} MB")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])[:18]:
        if v > 2**20:
            print(f"  {v / 2**20:8.1f} MB  {k}")

    # live torch CPU tensors (mjlab manager buffers etc.)
    import gc

    t_total = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                t_total += obj.element_size() * obj.nelement()
        except Exception:
            continue
    print(f"\n[torch cpu tensors] live total ~{t_total / 2**20:.0f} MB")

    # trim the USM pool free lists (construction churn) and re-measure
    from mjlab_sycl import native_kernels

    native_kernels.pool_trim()
    mem("after trim")
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "empty_cache"):
        torch.cpu.empty_cache()
        mem("after torch cpu")
    mem("final")


if __name__ == "__main__":
    main()
