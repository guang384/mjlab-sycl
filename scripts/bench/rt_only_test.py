"""Runtime-only validation: can warpsycl.dll run on JUST the pip SYCL
runtime (no oneAPI toolkit on PATH)?

Usage (via a subprocess whose PATH is scrubbed of oneAPI):
    python rt_only_test.py <runtime-bin-dir>

Prints the device name on success; the loader error on failure.
"""
import ctypes
import os
import sys

rt_bin = sys.argv[1]
os.environ["PATH"] = rt_bin + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(rt_bin)

# dependency order (sycl8 needs its own imports in memory first)
for dll in ("libmmd.dll", "ur_win_proxy_loader.dll", "sycl8.dll", "ur_loader.dll",
            "ur_adapter_level_zero.dll", "ur_adapter_level_zero_v2.dll",
            "tcm.dll", "umf.dll"):
    p = os.path.join(rt_bin, dll)
    if os.path.isfile(p):
        try:
            ctypes.WinDLL(p)
        except OSError as e:
            print(f"preload {dll}: {e}")

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(os.path.dirname(here))  # scripts/bench -> repo root
dll = ctypes.WinDLL(os.path.join(repo, "src", "mjlab_sycl", "backend", "warpsycl.dll"))
dll.wp_sycl_device_name.restype = ctypes.c_char_p
name = dll.wp_sycl_device_name().decode()
print(f"RUNTIME OK: {name}")
