# SPDX-License-Identifier: Apache-2.0
"""Run all verification gates in order: overlay sync (host-only), then the
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


def main() -> int:
  overlay_main()
  e2e_main()
  mujoco_main()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
