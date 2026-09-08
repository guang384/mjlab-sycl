# mjlab-sycl performance baseline (measured archive)

One place for every throughput/latency number measured on this machine, the
tooling to reproduce it, and the verdict on each optimization attempted. Use
it as the baseline when tuning on this box or benchmarking any other machine.

Measured on: Intel Arc 130T iGPU (16 GB, Lunar Lake) · Windows · Python 3.12
· warp 1.12.0 (sycl overlay) · mujoco_warp 3.8.1 · mjlab 1.3.0 · torch 2.9.1+xpu
Task: `Mjlab-Velocity-Flat-MicroDuck` (14 servos, nv=20), 50 Hz control.

## Reproduce

```powershell
# end-to-end training loop timing (rollout | ppo | total per iteration)
.venv\Scripts\mjlab-sycl-bench.exe --device sycl  --num-envs 4096 --iters 3
.venv\Scripts\mjlab-sycl-bench.exe --device cpu   --num-envs 1024 --iters 2   # pure warp-cpu, no sycl patch
# per-step API census (launches / drains / allocs per env step)
python scripts\probe_sycl_profile.py --num-envs 4096 --steps 20
# per-kernel device time attribution (serialized; relative ranking valid)
python scripts\probe_kernel_times.py --num-envs 4096 --steps 4
# real efc usage vs compiled buffer (njmax padding audit)
python scripts\probe_efc_audit.py --num-envs 512 --steps 30
```

## Headline numbers

| config (4096 envs unless noted) | per-iteration wall | env-steps/s |
|---|---|---|
| sycl, PPO on **cpu** | 23.7 s (rollout 19.1 / ppo 4.5) | 5,140 |
| sycl, PPO on **xpu** (default) | **19.2 s** (rollout 17.9 / ppo 1.2) | 5,485 |
| sycl, 8192 envs, PPO xpu | 37.3 s | 5,634 (+2.7 %) |
| **pure warp-CPU device** (1024 envs) | 100.2 s | 246 |
| sycl (1024 envs) | 7.6 s | 3,436 |

- PPO xpu vs cpu: update 4.5 s -> 1.2 s/iter (~19 % faster iterations).
- sycl vs same-stack pure CPU device: **~14x** (3436 vs 246 env-steps/s @1024).
- Env-count scaling is flat beyond ~4096 (device saturated; doubling envs doubles
  per-step wall -> same samples/s).

## Where the ~670 ms/env.step @4096 goes (census)

- sim:step x4 ~454 ms + env-level forward ~111 ms (5 full Newton solves/step)
- ~1,600 kernel launches/step; host submit ~300 ms/step (overlaps device)
- ~62 queue drains/step (~55 are the capture-while convergence polls),
  ~230 ms/step waiting on real device work
- residual host (managers/obs/etc.) ~120 ms/step

## Optimization attempts + verdicts

| change | result | verdict |
|---|---|---|
| PPO on torch.xpu | 4.5 -> 1.2 s/iter | kept (default) |
| torch CPU threads cap (MJLAB_TORCH_THREADS=2) | CPU 4.6 -> 1.9 cores, wall unchanged (~670 ms/step) | kept (default) |
| batched convergence polling (MJLAB_SYCL_POLL_EVERY) | 716 vs 671 ms/step (guard no-op iterations cost more than the polls saved) | dead end, kept off |
| kernel args by-value capture (codegen) | ~1 % (noise); DPC++ also requires const kernel lambdas | dead end, reverted |
| 8192 envs | +2.7 % | not worth wall 2x |
| solver micro-kernel fusion (Route C) | infeasible here: fusion points need intra-world sync (barriers) which must never be added on the iGPU; only tiny per-world scalar merges remain (~1-2 %) | not viable, documented |
| viewer rendering | GL runs on the same iGPU as physics (hardware GL confirmed); Windows KnownDLLs block Mesa software-GL override; real-sim viewers cap at ~8 updates/s (per-step fixed cost) | structural |

## Known device-side costs (out of adapter reach)

- **efc buffer padding**: peak real efc/world = 46 vs compiled buffer 1504 rows
  (~33x idle). Shrinking needs model-level `<size njmax>` (microduck MJCF) or a
  mujoco_warp compile change; see scripts/probe_efc_audit.py.
- Kernel time leaders (serialized ranking @4096): update_constraint_efc,
  cholesky solves, linesearch family (jv/prepare_quad/jaref), flat JTDAJ/contact.

## Hardware context (web, for wall-clock planning)

- Same task, 4096 envs: RTX 5080 ~1.13 s/iter (community course,
  https://forum.d-robotics.cc/t/topic/35668) -> ~17x faster than this iGPU.
- PassMark GPU compute: Arc 130V ~2.2k vs RTX 5090 ~24.4k ops/s (~11x);
  memory bandwidth GDDR7 ~15-18x LPDDR5X-class iGPU.
- Wall-clock projection (tricks ~1000 iters, gaits 4000-6000 iters):
  this iGPU ~5.5 h / 21-32 h; RTX 5090-class ~12 min / 50-75 min.
