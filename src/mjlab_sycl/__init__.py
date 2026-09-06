"""Intel GPU (SYCL) training support for mjlab.

Runtime patch + barrier-free flat kernels + a select_gpus-free train entry.
Requires a warp build with the SYCL backend (the warp-sycl fork) overlaid
onto the environment's warp package; see the project README.
"""

from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

__all__ = ["patch_simulation_for_sycl"]
