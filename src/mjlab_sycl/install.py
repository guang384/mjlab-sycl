"""One-time backend installation: overlay the SYCL backend onto this
environment's warp package, verify the device comes up, and confirm the
overlay it just applied is exactly what this package ships.

Usage:  python -m mjlab_sycl install

The overlay is *not* tracked by uv: any ``uv sync`` / ``uv run`` re-installs
warp from the lock file and silently wipes it. ``overlay_problems()`` /
``ensure_overlay_synced()`` therefore compare every file this package ships
against what is actually on disk (the venv's warp package + warp's kernel
cache) — the training entries call ``ensure_overlay_synced()`` before physics
starts, so a stale or missing overlay aborts with a clear remediation instead
of a cryptic missing-device error or silently corrupt physics.
"""

import hashlib
import os
import shutil
import sys

# Mirrors install()'s copy layout: backend/<src_sub>/<rel> -> warp/<dst_sub>/<rel>.
_OVERLAY_PAIRS = (("_src", "_src"), ("native", "native"))
# The vendored backend is patched against exactly this warp release; a
# different warp needs its own backend build (the overlay is version-coupled
# through the SYCL codegen + tile.h changes).
WARP_VERSION = "1.12.0"


def _file_sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _warp_layout():
  """(warp package dir, warp kernel cache dir) for the *active* environment."""
  import warp as wp

  warp_dir = os.path.dirname(wp.__file__)
  cache = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "NVIDIA", "warp", "Cache", wp.__version__,
  )
  return warp_dir, cache


def _shipped_files(backend: str):
  """Yield (shipped_path, installed_path) for every file install() copies."""
  warp_dir, cache_dir = _warp_layout()
  for src_sub, dst_sub in _OVERLAY_PAIRS:
    src_dir = os.path.join(backend, src_sub)
    for root, dirs, files in os.walk(src_dir):
      # bytecode caches are build artifacts of whoever imported the backend
      # source last; each tree compiles its own .pyc, so comparing them
      # byte-wise false-fails. They are never part of the shipped overlay.
      dirs[:] = [d for d in dirs if d != "__pycache__"]
      rel = os.path.relpath(root, src_dir)
      dst_dir = os.path.join(warp_dir, dst_sub, rel)
      for f in files:
        if f.endswith((".pyc", ".pyo")):
          continue
        yield os.path.join(root, f), os.path.join(dst_dir, f)
  yield os.path.join(backend, "warpsycl.dll"), os.path.join(cache_dir, "warpsycl.dll")


def overlay_problems() -> list:
  """Concrete problems with the installed warp SYCL overlay, [] when healthy.

  Every file this package ships must be present in the venv's warp package
  (or kernel cache, for warpsycl.dll) with a byte-identical hash. Catches a
  uv-sync-wiped overlay, a partial copy, a backend edit that was never
  re-installed, and a warp version that drifted from the one the backend was
  patched against.
  """
  problems = []
  try:
    import warp as wp
  except ImportError:
    return [
      "warp is not installed in this environment — run `uv sync` first, "
      "then `python -m mjlab_sycl install`",
    ]

  if wp.__version__ != WARP_VERSION:
    problems.append(
      f"warp version is {wp.__version__}, expected {WARP_VERSION} — the SYCL "
      f"backend overlay targets {WARP_VERSION}; a different warp needs its "
      f"own backend build"
    )

  backend = os.path.join(os.path.dirname(__file__), "backend")
  for shipped, installed in _shipped_files(backend):
    if not os.path.isfile(installed):
      problems.append(f"missing overlay file: {installed}")
    elif _file_sha256(shipped) != _file_sha256(installed):
      problems.append(
        f"overlay file differs from the bundled backend: {installed}\n"
        f"    (either the backend changed after install, or only the license "
        f"header differs — re-install to sync)"
      )
  return problems


def ensure_overlay_synced() -> None:
  """Abort (exit 2) with the remediation when the overlay is not healthy.

  Called at the top of ``patch_simulation_for_sycl()`` so every SYCL entry
  point fails fast and loudly instead of running on a stale backend.
  """
  problems = overlay_problems()
  if not problems:
    return
  print(
    "\n[mjlab-sycl] ERROR: the warp SYCL backend overlay is out of sync with "
    "this package.\n"
    "  A `uv sync` / `uv run` re-installs warp from the lock file and wipes "
    "the overlay;\n"
    "  running on a stale overlay silently corrupts physics or hides the "
    "sycl device.\n"
    "\n"
    "  Fix: re-run\n"
    "      python -m mjlab_sycl install\n"
    "  (it re-applies the overlay and self-checks). If it persists, the "
    "environment's warp\n"
    "  version or the overlay target changed unexpectedly.\n"
    "\n"
    "  Detected problems:",
    file=sys.stderr,
  )
  for p in problems:
    print(f"    - {p}", file=sys.stderr)
  raise SystemExit(2)


def _warmup(task: str | None) -> int:
  """Compile the kernel modules now so the first training run doesn't pay
  the one-time JIT (~3 min at 4096 envs) mid-session.

  Builds a 2-env instance of ``task`` (or the first registered task) and
  steps it twice through the patched stack -- the full kernel set for
  that task's shapes gets compiled and cached. Never fails the install:
  returns 0 on success, 1 with a message otherwise."""
  import time

  t0 = time.time()
  print("[mjlab-sycl] warmup: compiling kernel modules (one-time, ~3 min)...")
  try:
    from mjlab_sycl._bootstrap import configure_torch_threads

    configure_torch_threads()
    import torch

    from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

    patch_simulation_for_sycl()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    if not task:
      names = sorted(registry.list_tasks())
      if not names:
        print("[mjlab-sycl] warmup skipped: no tasks registered")
        return 1
      task = names[0]
    cfg = registry.load_env_cfg(task)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    for _ in range(2):
      env.step(torch.zeros((2, env.action_manager.total_action_dim)))
    print(f"[mjlab-sycl] warmup done in {time.time() - t0:.0f}s "
          f"({task}, 2 envs) -- later runs start warm")
    return 0
  except Exception as e:
    print(f"[mjlab-sycl] warmup failed ({e!r}) -- training still works, "
          "the first run will compile instead")
    return 1


def main() -> int:
  import argparse

  parser = argparse.ArgumentParser(
      prog="mjlab-sycl-install",
      description="Overlay the SYCL backend and self-check")
  # the documented `python -m mjlab_sycl install` spells the subcommand out;
  # the console script omits it. Accept (and ignore) exactly that token.
  parser.add_argument("cmd", nargs="?", default="install", choices=["install"])
  parser.add_argument("--warmup", nargs="?", const="", default=None, metavar="TASK",
                      help="also JIT-compile the kernel modules now (first "
                           "registered task by default) so the first training "
                           "run starts warm")
  args = parser.parse_args()

  # sycl8.dll PATH ordering -- must run before torch/warp come up (see
  # _bootstrap); the torch-XPU availability probe below imports torch.
  from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

  prepare_sycl_runtime_path()

  # Locate warp's package dir WITHOUT executing it: on a fresh environment
  # an import here would load stock warp (no sycl device), and the copied
  # overlay would then be verified against already-imported modules — a
  # false failure on exactly the machines this installer exists for.
  import importlib.util

  spec = importlib.util.find_spec("warp")
  if spec is None or spec.origin is None:
    print("error: warp is not installed in this environment", file=sys.stderr)
    return 1

  warp_dir = os.path.dirname(spec.origin)
  backend = os.path.join(os.path.dirname(__file__), "backend")

  # 1) overlay the patched files
  n = 0
  for src_sub, dst_sub in _OVERLAY_PAIRS:
    src_dir = os.path.join(backend, src_sub)
    for root, dirs, files in os.walk(src_dir):
      dirs[:] = [d for d in dirs if d != "__pycache__"]
      rel = os.path.relpath(root, src_dir)
      dst_dir = os.path.join(warp_dir, dst_sub, rel)
      os.makedirs(dst_dir, exist_ok=True)
      for f in files:
        if f.endswith((".pyc", ".pyo")):
          continue
        shutil.copy2(os.path.join(root, f), os.path.join(dst_dir, f))
        n += 1
  print(f"[mjlab-sycl] overlaid {n} backend files onto {warp_dir}")

  # First import of warp now executes the overlaid files.
  import warp

  # 2) place the prebuilt warpsycl.dll in warp's kernel cache. A live
  # process holding warp loaded keeps the cached DLL open (WinError 32 on
  # rewrite): skip the copy when it is already byte-identical, and only
  # fail when a genuinely different DLL cannot be replaced.
  cache = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "NVIDIA", "warp", "Cache", warp.__version__)
  os.makedirs(cache, exist_ok=True)
  dll = os.path.join(backend, "warpsycl.dll")
  cached = os.path.join(cache, "warpsycl.dll")
  if not (os.path.isfile(cached) and _file_sha256(dll) == _file_sha256(cached)):
    try:
      shutil.copy2(dll, cached)
    except PermissionError as e:
      print(f"[mjlab-sycl] ERROR: cannot replace {cached} ({e}); close the "
            "process holding it and re-run install", file=sys.stderr)
      return 1
  print(f"[mjlab-sycl] warpsycl.dll -> {cache}")

  # 3) verify the sycl device comes up on the overlaid backend
  wp = warp

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

  # 4) self-check the overlay just applied (byte-identical to this package)
  problems = overlay_problems()
  if problems:
    print("\n[mjlab-sycl] ERROR: overlay self-check failed right after install:",
          file=sys.stderr)
    for p in problems:
      print(f"    - {p}", file=sys.stderr)
    return 1
  print(f"[mjlab-sycl] overlay verified: {n} files + warpsycl.dll match this package")
  if args.warmup is not None:
    _warmup(args.warmup or None)
  return 0


if __name__ == "__main__":
  sys.exit(main())
