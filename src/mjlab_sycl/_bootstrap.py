# SPDX-License-Identifier: Apache-2.0
"""PATH bootstrap for the SYCL runtime DLLs -- call before torch/warp come up.

torch's XPU wheel and oneAPI both ship a DLL named sycl8.dll at incompatible
versions, and warpsycl.dll (built against oneAPI) must resolve oneAPI's copy
first or warp's device registration fails with WinError 127. torch also needs
its pip-provided runtime dir on PATH to import at all.

PATH ordering (first match wins):
  1. oneAPI's compiler bin (WARP_SYCL_ONEAPI_BIN, or the default install)
  2. everything already on PATH
  3. the pip SYCL runtime dir, auto-detected from site-packages
     (dpcpp-cpp-rt's Library/bin layout) or set via WARP_SYCL_PIP_BIN --
     appended last so its older sycl8.dll can never shadow oneAPI's
"""

import glob
import os
import sys


def prepare_sycl_runtime_path() -> None:
  oneapi = os.environ.get(
      "WARP_SYCL_ONEAPI_BIN",
      "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3/bin",
  )
  pip_bin = os.environ.get("WARP_SYCL_PIP_BIN", "")
  if not pip_bin:
    for site in [p for p in sys.path if p.endswith("site-packages")]:
      hits = sorted(glob.glob(os.path.join(site, "Library", "bin")))
      if hits:
        pip_bin = hits[0]
        break
  if oneapi and os.path.isdir(oneapi):
    # prepend: warp's warpsycl.dll must resolve the oneAPI sycl8.dll
    os.environ["PATH"] = oneapi + os.pathsep + os.environ["PATH"]
  if pip_bin and os.path.isdir(pip_bin):
    # append: torch's other DLL dependencies (libuv etc.) as a fallback only
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + pip_bin