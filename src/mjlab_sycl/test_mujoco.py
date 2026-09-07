# SPDX-License-Identifier: Apache-2.0
"""mujoco_warp smoke gate: run real physics steps on the Warp SYCL device.

This is the real workload shape of mjlab training: mujoco_warp compiles its
physics kernels as regular warp modules, so they must flow through the SYCL
codegen + icx chain just like user kernels. The test model is the
pendula.xml shipped with mujoco_warp (two pendulums + constraints).

Requires `python -m mjlab_sycl install` to have been run in this environment.

Usage:
    python -m mjlab_sycl.test_mujoco    # this gate alone
    mjlab-sycl-test                     # both gates, in order

Checks:
  1. put_model / put_data succeed on the sycl device
  2. mjw.forward + mjw.step complete without NaN
  3. qpos after N steps is deterministic across two runs
  4. results agree with the CPU device running the same steps
"""

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import pathlib  # noqa: E402

import mujoco  # noqa: E402
import mujoco_warp as mjw  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.init()

MODEL_XML = "pendula.xml"
NSTEP = 20


def check(failures, label, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def load_model():
    xml = pathlib.Path(mjw.__file__).parent / "test_data" / MODEL_XML
    return mujoco.MjModel.from_xml_path(str(xml))


def run_steps(device: str, mjm):
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mujoco.MjData(mjm))
        mjw.forward(m, d)
        for _ in range(NSTEP):
            mjw.step(m, d)
        return d.qpos.numpy()


def main():
    failures = []

    mjm = load_model()

    # -- 1/2. model + data on the sycl device, forward + step ------------------
    dev = wp.get_device("sycl")
    print(f"[1] running mujoco_warp on {dev.name!r}")
    qpos_sycl = run_steps("sycl", mjm)

    if np.isnan(qpos_sycl).any():
        failures.append("NaN in qpos after stepping on sycl")
    print(f"[2] qpos after {NSTEP} steps on sycl (first 4): {qpos_sycl[:4]}")

    # -- 3. determinism ----------------------------------------------------------
    qpos_sycl_2 = run_steps("sycl", mjm)
    check(failures, "determinism across runs (atol=1e-6)", np.allclose(qpos_sycl, qpos_sycl_2, atol=1e-6))

    # -- 4. compare against CPU ---------------------------------------------------
    qpos_cpu = run_steps("cpu", mjm)
    max_err = float(np.max(np.abs(qpos_sycl - qpos_cpu)))
    print(f"[4] max |sycl - cpu| = {max_err:.3e}")
    check(failures, "sycl vs cpu agreement (atol=1e-5)", max_err < 1e-5)

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)

    print("ALL OK")


if __name__ == "__main__":
    main()