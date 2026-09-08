# SPDX-License-Identifier: Apache-2.0
"""One-command environment preflight: can this machine run mjlab-sycl?

Runs every check that decides whether mjlab-sycl can train here and prints a
[PASS]/[FAIL] line per check with the exact fix when one is needed:

  1. platform: Windows, Python 3.12
  2. warp 1.12.0 installed (the version the vendored backend targets)
  3. warp SYCL backend overlay in sync with this package (see install.py)
  4. Intel oneAPI runtime present (sycl8.dll / icx on PATH for kernel builds)
  5. the `sycl` device comes up and names an Intel GPU
  6. a real device kernel compiles and matches the cpu device (small, cached)
  7. torch XPU available (the PPO device used by the training entries)
  8. (soft) the venv carries an mjlab task package (e.g. microduck_rl) whose
     task registry loads

Exit code: 0 when nothing FAILs, 1 otherwise. No files are modified; nothing
is installed -- this is read-only diagnosis. For the full numerical gates run
`mjlab-sycl-test` afterwards.

Usage:
    mjlab-sycl-check                # console entry (all checks)
    python -m mjlab_sycl.doctor     # same, as a module
    mjlab-sycl-check --no-kernel    # skip the compile+run device kernel test
"""

import argparse
import glob
import os
import sys

# sycl8.dll PATH ordering -- must run before torch/warp come up (see
# _bootstrap); the torch-XPU check below imports torch.
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

_ONEAPI_ROOTS = (
    "C:/Program Files (x86)/Intel/oneAPI",
    "C:/Program Files/Intel/oneAPI",
)


def _check(label: str, ok: bool, detail: str = "", fix: str | None = None) -> bool:
  print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
  if not ok and fix:
    print(f"       fix: {fix}")
  return ok


def _find_oneapi_bin() -> str | None:
  env = os.environ.get("WARP_SYCL_ONEAPI_BIN", "")
  if env and os.path.isdir(env):
    return env
  for root in _ONEAPI_ROOTS:
    hits = sorted(glob.glob(os.path.join(root, "compiler", "*", "bin")))
    if hits:
      return hits[-1]  # newest compiler version
  return None


def _find_pip_sycl_bin() -> str | None:
  for site in [p for p in sys.path if p.endswith("site-packages")]:
    hits = sorted(glob.glob(os.path.join(site, "Library", "bin")))
    if hits:
      return hits[0]
  return None


def _device_kernel_test() -> tuple[bool, str]:
  import numpy as np
  import warp as wp

  @wp.kernel
  def saxpy(a: float, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a * x[i] + y[i]

  n = 1 << 20
  a = 2.0
  x = np.random.default_rng(0).random(n, dtype=np.float32)
  y = np.random.default_rng(1).random(n, dtype=np.float32)
  outs = {}
  for dev in ("sycl", "cpu"):
    xd = wp.array(x, dtype=wp.float32, device=dev)
    yd = wp.array(y, dtype=wp.float32, device=dev)
    od = wp.empty(n, dtype=wp.float32, device=dev)
    with wp.ScopedDevice(dev):
      wp.launch(saxpy, dim=n, inputs=[a, xd, yd], outputs=[od])
    outs[dev] = od.numpy()
  max_err = float(np.max(np.abs(outs["sycl"] - outs["cpu"])))
  ok = max_err < 1e-5
  return ok, f"sycl vs cpu max err = {max_err:.2e}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--no-kernel", action="store_true",
    help="skip the compile-and-run device kernel test (check 6)",
  )
  args = parser.parse_args()

  ok_all = True

  # 1. platform ---------------------------------------------------------------
  ok_all &= _check(
    "Windows + Python 3.12",
    sys.platform == "win32" and (3, 12) <= sys.version_info[:2] < (3, 13),
    f"{sys.platform}, Python {sys.version.split()[0]}",
    "mjlab-sycl requires Windows + Python >=3.12,<3.13",
  )

  # 2. warp version ------------------------------------------------------------
  try:
    import warp as wp
  except ImportError:
    ok_all &= _check("warp installed", False, fix="`uv sync` in your mjlab project (warp-lang==1.12.0)")
    print("\nRESULT: FAIL -- see fixes above")
    raise SystemExit(1)

  ok_all &= _check(
    "warp version",
    wp.__version__ == "1.12.0",
    wp.__version__,
    "the vendored SYCL backend targets warp 1.12.0 (warp-lang==1.12.0 in pyproject)",
  )

  # 3. overlay sync -------------------------------------------------------------
  from mjlab_sycl.install import overlay_problems

  problems = overlay_problems()
  fix = None if not problems else "python -m mjlab_sycl install"
  ok_all &= _check(
    "warp SYCL overlay in sync",
    not problems,
    "ok" if not problems else f"{len(problems)} problem(s): " + problems[0].splitlines()[0],
    fix,
  )
  if problems:
    for p in problems[1:]:
      print(f"         - {p.splitlines()[0]}")

  # 4. oneAPI runtime -----------------------------------------------------------
  oneapi_bin = _find_oneapi_bin()
  if oneapi_bin:
    has_runtime = os.path.isfile(os.path.join(oneapi_bin, "sycl8.dll")) or os.path.isfile(
      os.path.join(oneapi_bin, "icx.exe")
    )
  else:
    has_runtime = False
  ok_all &= _check(
    "Intel oneAPI runtime",
    has_runtime,
    oneapi_bin or "not found",
    "install Intel oneAPI 2025.x (or set WARP_SYCL_ONEAPI_BIN to its compiler bin dir)",
  )

  # 5. sycl device --------------------------------------------------------------
  wp.init()
  try:
    dev = wp.get_device("sycl")
    dev_name = str(dev.name)
    is_intel = "Intel" in dev_name
    ok_all &= _check("Intel GPU (sycl device)", is_intel, dev_name)
  except Exception as e:
    ok_all &= _check("Intel GPU (sycl device)", False, repr(e), "see oneAPI/overlay fixes above")

  # 6. device kernel test --------------------------------------------------------
  if not args.no_kernel:
    try:
      ok_k, detail = _device_kernel_test()
      ok_all &= _check("device kernel compile+run (vs cpu)", ok_k, detail)
    except Exception as e:
      ok_all &= _check("device kernel compile+run (vs cpu)", False, repr(e), "re-run after fixing the checks above")

  # 7. torch XPU -----------------------------------------------------------------
  try:
    import torch

    ok_xpu = torch.xpu.is_available()
    ok_all &= _check(
      "torch XPU",
      ok_xpu,
      f"{torch.__version__}, xpu available: {ok_xpu}",
      'pip install "torch==2.9.1+xpu" --index-url https://download.pytorch.org/whl/xpu '
      "(per-machine step, see README-SYCL-TRAINING.md)",
    )
  except ImportError:
    ok_all &= _check("torch XPU", False, "torch not importable", "install torch==2.9.1 (project dep) then the +xpu wheel")

  # 8. mjlab task package (soft) ---------------------------------------------------
  try:
    from mjlab.tasks.registry import load_env_cfg  # noqa: F401

    tasks_ok = True
    detail = "mjlab registry importable"
  except Exception as e:
    tasks_ok = False
    detail = f"{type(e).__name__}: {e}"
  _check("mjlab task package present (e.g. microduck_rl)", tasks_ok, detail,
         "run inside your mjlab project's venv (uv sync) so its tasks register")

  # 9. summary ---------------------------------------------------------------------
  print("\nRESULT: " + ("ALL CHECKS PASSED" if ok_all else "FAILURES ABOVE -- fix them, re-run, then:"
       " `python -m mjlab_sycl install` (if told) and `mjlab-sycl-test` before long runs"))
  raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
  main()
