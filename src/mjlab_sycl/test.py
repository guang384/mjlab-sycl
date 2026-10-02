# SPDX-License-Identifier: Apache-2.0
"""Run all verification gates in order: overlay sync (host-only), then the
raw backend, physics, extended fusion/cache tests, and the patched stack.
backend e2e, then mujoco_warp physics vs cpu (both GPU).

Usage:
    mjlab-sycl-test                    # console entry (all gates)
    python -m mjlab_sycl.test          # same, as a module

Each gate raises SystemExit(1) on failure, so the run stops at the first
failing gate. test_overlay runs first and needs no GPU: it aborts fast with a
clear remediation if the warp SYCL backend overlay is missing or drifted.
"""

from mjlab_sycl.test_overlay import main as overlay_main
from mjlab_sycl.test_e2e import main as e2e_main
from mjlab_sycl.test_mujoco import main as mujoco_main
from mjlab_sycl.test_patched import main as patched_main


def main() -> int:
  overlay_main()
  e2e_main()
  mujoco_main()
  # extended GPU tests: imported LAZILY because test_fused_ab installs the
  # runtime patch at import time -- the raw-backend gates above must run
  # before that. test_launch_cache exercises cache hit/miss/fallback
  # semantics; test_fused_ab's chain tests compare every fused kernel
  # against its original on identical inputs.
  from mjlab_sycl.test_launch_cache import main as launch_cache_main

  launch_cache_main()
  from mjlab_sycl.test_fused_ab import main as fused_ab_main

  fused_ab_main()
  # LAST: test_patched installs the full runtime patch (the code the other
  # gates deliberately stay clear of); it must not run before them
  patched_main()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
