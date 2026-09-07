# SPDX-License-Identifier: Apache-2.0
"""Run both verification gates in order: backend e2e, then mujoco_warp vs cpu.

Usage:
    mjlab-sycl-test                    # console entry (both gates)
    python -m mjlab_sycl.test          # same, as a module

Each gate raises SystemExit(1) on failure, so the run stops at the first
failing gate.
"""

from mjlab_sycl.test_e2e import main as e2e_main
from mjlab_sycl.test_mujoco import main as mujoco_main


def main() -> int:
  e2e_main()
  mujoco_main()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())