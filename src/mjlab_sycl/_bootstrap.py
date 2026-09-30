# SPDX-License-Identifier: Apache-2.0
"""PATH bootstrap for the SYCL runtime DLLs -- call before torch/warp come up.

torch's XPU wheel and oneAPI both ship a DLL named sycl8.dll at incompatible
versions, and warpsycl.dll (built against oneAPI) must resolve oneAPI's copy
first or warp's device registration fails with WinError 127. torch also needs
its pip-provided runtime dir on PATH to import at all.

PATH ordering (first match wins):
  1. oneAPI's compiler bin (WARP_SYCL_ONEAPI_BIN, else the newest install
     found under the default roots -- see find_oneapi_bin)
  2. everything already on PATH
  3. the pip SYCL runtime dir, auto-detected from site-packages
     (dpcpp-cpp-rt's Library/bin layout) or set via WARP_SYCL_PIP_BIN --
     appended last so its older sycl8.dll can never shadow oneAPI's
"""

import ctypes
import glob
import os
import re
import sys

# DLLs shipped by BOTH oneAPI and the pip runtime stack under the same name.
# The Windows loader reuses whatever module is already in the process by name,
# so whichever copy loads first wins for the whole process. warpsycl.dll is
# built against oneAPI 2025.3 and dies with WinError 127 (missing exports) if
# it resolves the pip intel-sycl-rt's older sycl8/ur_loader; torch-xpu (built
# against 2025.2) is forward-compatible with the newer oneAPI runtime, so
# pinning the oneAPI copies first serves both.
_COMPILER_DLLS = [
    "sycl8.dll",
    "ur_loader.dll",
    "ur_adapter_level_zero.dll",
    "ur_adapter_level_zero_v2.dll",
    "ur_adapter_opencl.dll",
    "ur_win_proxy_loader.dll",
    "tcm.dll",
    "umf.dll",
    "sycl-jit.dll",
    "common_clang64.dll",
    "OpenCL.dll",
    "omptarget.dll",
    "omptarget.sycl.wrap.dll",
    "omptarget.rtl.level0.dll",
    "omptarget.rtl.opencl.dll",
    "omptarget.rtl.unified_runtime.dll",
]
# MKL SYCL domain libs are only named identically within the same MKL major
# (mkl_sycl_*.5.dll in both oneAPI 2025.3 and onemkl-sycl-* 2025.2 wheels).
_MKL_SYCL_DLLS = [
    "mkl_sycl_blas.5.dll",
    "mkl_sycl_dft.5.dll",
    "mkl_sycl_lapack.5.dll",
    "mkl_sycl_rng.5.dll",
    "mkl_sycl_sparse.5.dll",
]

# Keep the ctypes handles alive for the process lifetime (a plain LoadLibrary
# in a dropped temp would let the loader unload the DLL and lose the pin).
_PRELOADED = []

_ONEAPI_ROOTS = (
    "C:/Program Files (x86)/Intel/oneAPI",
    "C:/Program Files/Intel/oneAPI",
)


def _version_key(path: str) -> list[int]:
  """Sort key ordering digit runs numerically, so 2025.10 > 2025.9."""
  return [int(x) for x in re.findall(r"\d+", path)]


def find_oneapi_bin() -> str:
  """Resolve the oneAPI compiler bin dir for PATH prepends and preloads.

  WARP_SYCL_ONEAPI_BIN wins when set (callers verify the dir exists);
  otherwise pick the newest compiler/<ver>/bin under the default install
  roots. Returns "" when nothing is found. The PATH bootstrap and the
  doctor share this so they can never disagree about which runtime is
  active.
  """
  env = os.environ.get("WARP_SYCL_ONEAPI_BIN", "")
  if env:
    return env
  hits = [
      p
      for root in _ONEAPI_ROOTS
      for p in glob.glob(os.path.join(root, "compiler", "*", "bin"))
      if os.path.isdir(p)
  ]
  return max(hits, key=_version_key) if hits else ""


def preload_oneapi_sycl_runtime(oneapi_bin: str) -> None:
  """Load oneAPI's SYCL-stack DLLs before torch/warp can load the pip copies."""
  dirs = [oneapi_bin]
  # MKL ships beside the compiler under mkl/<ver>/bin; glob instead of
  # pinning a version so a newer oneAPI install keeps working unedited.
  mkl_hits = [
      p
      for p in glob.glob(
          os.path.normpath(os.path.join(os.path.dirname(oneapi_bin), "..", "mkl", "*", "bin"))
      )
      if os.path.isdir(p)
  ]
  if mkl_hits:
    dirs.append(max(mkl_hits, key=_version_key))
  for dll in _COMPILER_DLLS + _MKL_SYCL_DLLS:
    for d in dirs:
      p = os.path.join(d, dll)
      if not os.path.exists(p):
        continue
      try:
        _PRELOADED.append(ctypes.WinDLL(p))
        break
      except OSError:
        pass  # optional component: leave the loader free to find another copy


def configure_torch_threads(default: int = 2) -> None:
  """Cap torch intra-op threads (default 2, override MJLAB_TORCH_THREADS).

  The env managers run hundreds of SMALL torch ops per sim step (obs/reward
  over N envs); with torch's default = all cores, each tiny op fans out to 14
  threads -- a large CPU tax for almost no speed. Measured at 4096 envs:
  14 threads -> ~4.6 cores busy, 2 threads -> ~1.9 cores, wall time unchanged
  (~670 ms/step both); 1 thread costs ~+16% wall. PPO runs on torch.xpu, so
  capping CPU threads does not slow the update.
  """
  try:
    import torch

    n = int(os.environ.get("MJLAB_TORCH_THREADS", str(default)))
    if n > 0:
      torch.set_num_threads(n)
    try:
      torch.set_num_interop_threads(1)
    except Exception:
      pass  # already set in this process
  except Exception:
    pass


def prepare_sycl_runtime_path() -> None:
  oneapi = find_oneapi_bin()
  pip_bin = os.environ.get("WARP_SYCL_PIP_BIN", "")
  if not pip_bin:
    # pip installs a wheel's data files relative to the PREFIX root, i.e.
    # <venv>/Library/bin -- not under site-packages. Fall back to a
    # site-packages-relative probe for --prefix layouts.
    for site in [p for p in sys.path if p.endswith("site-packages")]:
      prefix = os.path.dirname(os.path.dirname(site))  # <venv>/Lib/site-packages
      hits = sorted(glob.glob(os.path.join(prefix, "Library", "bin")))
      if hits:
        pip_bin = hits[0]
        break
  if oneapi and os.path.isdir(oneapi):
    # prepend: warp's warpsycl.dll must resolve the oneAPI sycl8.dll
    os.environ["PATH"] = oneapi + os.pathsep + os.environ["PATH"]
    preload_oneapi_sycl_runtime(oneapi)
  if pip_bin and os.path.isdir(pip_bin):
    # append: torch's other DLL dependencies (libuv etc.) as a fallback only
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + pip_bin