"""One-time backend installation: overlay the SYCL backend onto this
environment's warp package and verify the device comes up.

Usage:  python -m mjlab_sycl install
"""

import os
import shutil
import sys


def main() -> int:
  try:
    import warp
  except ImportError:
    print("error: warp is not installed in this environment", file=sys.stderr)
    return 1

  warp_dir = os.path.dirname(warp.__file__)
  backend = os.path.join(os.path.dirname(__file__), "backend")

  # 1) overlay the patched files
  pairs = [("_src", "_src"), ("native", "native")]
  n = 0
  for src_sub, dst_sub in pairs:
    src_dir = os.path.join(backend, src_sub)
    for root, _dirs, files in os.walk(src_dir):
      rel = os.path.relpath(root, src_dir)
      dst_dir = os.path.join(warp_dir, dst_sub, rel)
      os.makedirs(dst_dir, exist_ok=True)
      for f in files:
        shutil.copy2(os.path.join(root, f), os.path.join(dst_dir, f))
        n += 1
  print(f"[mjlab-sycl] overlaid {n} backend files onto {warp_dir}")

  # 2) place the prebuilt warpsycl.dll in warp's kernel cache
  ver = warp.__version__
  cache = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "NVIDIA", "warp", "Cache", ver)
  os.makedirs(cache, exist_ok=True)
  dll = os.path.join(backend, "warpsycl.dll")
  shutil.copy2(dll, os.path.join(cache, "warpsycl.dll"))
  print(f"[mjlab-sycl] warpsycl.dll -> {cache}")

  # 3) verify
  import warp as wp  # re-import picks up the overlaid modules

  wp.init()
  try:
    dev = wp.get_device("sycl")
    print(f"[mjlab-sycl] sycl device OK: {dev.name}")
  except Exception as e:
    print(f"[mjlab-sycl] WARN: sycl device not available ({e!r})", file=sys.stderr)
    return 1

  try:
    import torch

    ok = torch.xpu.is_available()
    print(f"[mjlab-sycl] torch {torch.__version__} | xpu available: {ok}")
    if not ok:
      print('  install it per-machine: pip install "torch==2.9.1+xpu" '
            '--index-url https://download.pytorch.org/whl/xpu')
  except ImportError:
    print("[mjlab-sycl] torch not installed (PPO needs torch.xpu)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
