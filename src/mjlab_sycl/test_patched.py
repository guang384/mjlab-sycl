# SPDX-License-Identifier: Apache-2.0
"""Physics gate for the PATCHED stack (the blind spot of test_mujoco).

test_mujoco calls mjw.step directly, so it exercises only the raw
backend: none of the runtime patch layers (native kernels, fusions,
launch cache, step graphs) run under it. A broken route there -- e.g. a
function-level seam that silently skipped a launch -- passed every gate
until now. This gate closes the hole:

    patched-sycl (patch active, all defaults) vs unpatched-cpu reference

on a model with contacts AND equality/limit constraints (collision.xml:
nv=12 -> the small-nv chol path, nefc=8, ncon=11, Newton + pyramidal ->
the incremental gradient path), i.e. every route landed so far is
exercised. Both runs are plain mjw.step loops on the same model; the
native routes may differ from warp kernels at ULP level (documented in
performance.md), so the comparison uses the physics contract 1e-5
scaled by run length.

Usage:
    python -m mjlab_sycl.test_patched
"""

import pathlib

import mujoco

MODEL_XML = "collision.xml"
NSTEP = 40
# ULP-level route differences accumulate over steps; 20 steps of pendula
# measure 3e-6 in test_mujoco. Keep the same contract here, and only
# widen with measurement evidence.
ATOL = 1e-5


def load_model():
    import mujoco_warp as mjw

    xml = pathlib.Path(mjw.__file__).parent / "test_data" / MODEL_XML
    return mujoco.MjModel.from_xml_path(str(xml))


def run_steps(device: str, mjm):
    import mujoco_warp as mjw
    import warp as wp

    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mujoco.MjData(mjm))
        mjw.forward(m, d)
        for _ in range(NSTEP):
            mjw.step(m, d)
        return d.qpos.numpy()


def check_hinc_contract(mjm) -> list:
    """The hinc route replaces _update_gradient_incremental; assert the
    wrapper issues the original's launches (minus the one kernel the
    native export substitutes for)."""
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import solver as S

    out = []
    with wp.ScopedDevice("sycl"):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mujoco.MjData(mjm))
        mjw.forward(m, d)
        mjw.step(m, d)  # reach a mid-solve-ish state
        ctx = S.create_solver_context(m, d)

    orig = getattr(S, "_orig_hinc_for_gate", None)
    # the wrapper keeps the original reachable through flat_kernels
    from mjlab_sycl import flat_kernels as fk

    rec = {}
    real_launch = wp.launch

    def spy(kernel, *args, **kwargs):
        rec.setdefault(rec["arm"], []).append(getattr(kernel, "key", str(kernel)))
        return real_launch(kernel, *args, **kwargs)

    wp.launch = spy
    try:
        rec["arm"] = "orig"
        rec["orig"] = []
        S.update_gradient_zero_grad_dot  # noqa: B018 -- exists check
        # call the pristine implementation body via a fresh reference chain
        from mujoco_warp._src import solver as S2

        # flat_kernels captured the pre-patch function; re-derive by
        # replaying the documented three launches here as the reference
        rec["arm"] = "ref"
        rec["ref"] = []
        wp.launch(S2.update_gradient_zero_grad_dot, dim=(d.nworld,),
                  inputs=[ctx.done], outputs=[ctx.grad_dot], device="sycl")
        wp.launch(S2.update_gradient_grad, dim=(d.nworld, m.nv),
                  inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done],
                  outputs=[ctx.grad, ctx.grad_dot], device="sycl")
        wp.launch(S2.update_gradient_h_incremental,
                  dim=(d.nworld, m.nv * (m.nv + 1) // 2),
                  inputs=[d.efc.J, d.efc.D, d.efc.state,
                          ctx.changed_efc_ids, ctx.changed_efc_count],
                  outputs=[ctx.h], device="sycl")
        ref_keys = sorted(rec["ref"])

        rec["arm"] = "patched"
        rec["patched"] = []
        with wp.ScopedDevice("sycl"):  # the sim steps run under this scope
            S._update_gradient_incremental(m, d, ctx)
        pat_keys = sorted(rec["patched"])
    finally:
        wp.launch = real_launch

    # the patched sequence must launch the two gradient kernels; the third
    # (h_incremental) is legitimately replaced by the native export, which
    # bypasses wp.launch entirely
    required = ["update_gradient_zero_grad_dot", "update_gradient_grad"]
    for k in required:
        ok = any(k in key for key in pat_keys)
        print(f"[{'PASS' if ok else 'FAIL'}] hinc contract: wrapper launches {k}")
        if not ok:
            out.append(f"hinc wrapper missing {k}")
    return out


def main():
    from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

    prepare_sycl_runtime_path()
    import numpy as np
    import warp as wp

    wp.init()

    mjm = load_model()
    d0 = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d0)
    print(f"[patched] model {MODEL_XML}: nv={mjm.nv} nefc={d0.nefc} "
          f"ncon={d0.ncon} solver={mjm.opt.solver} cone={mjm.opt.cone}")
    failures = []

    # the cpu reference MUST run before the patch: the interceptors cover
    # every device, and the native routes read raw pointers on the sycl
    # queue (they now guard against non-sycl arrays, but ordering is the
    # honest setup for a reference run)
    qpos_cpu = run_steps("cpu", mjm)

    # install the full interceptor stack -- the code under test (test_mujoco
    # runs before this gate and stays clean of it)
    from mjlab_sycl.runtime_patch import patch_simulation_for_sycl

    patch_simulation_for_sycl()
    qpos_patched = run_steps("sycl", mjm)

    if not np.isfinite(qpos_patched).all():
        failures.append("NaN/Inf in patched qpos")
        print("[FAIL] finite patched qpos")
    else:
        print("[PASS] finite patched qpos")

    # determinism of the patched stack (the natives must be reproducible)
    qpos_patched_2 = run_steps("sycl", mjm)
    ok = np.array_equal(qpos_patched, qpos_patched_2)
    print(f"[{'PASS' if ok else 'FAIL'}] patched determinism (bit-equal across runs)")
    if not ok:
        failures.append("patched stack is not deterministic")

    err = float(np.max(np.abs(qpos_patched - qpos_cpu)))
    ok = err < ATOL
    print(f"[patched] max |sycl - cpu| = {err:.3e}")
    print(f"[{'PASS' if ok else 'FAIL'}] patched vs cpu agreement (atol={ATOL:.0e})")
    if not ok:
        failures.append(f"patched vs cpu {err:.3e}")

    # -- wrapper contract: function-level seams must replicate the replaced
    # function's FULL launch sequence (trajectory checks are not enough: a
    # skipped launch can be trajectory-neutral on some models -- measured
    # with collision.xml -- while silently dropping physics on others).
    failures += check_hinc_contract(mjm)

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
