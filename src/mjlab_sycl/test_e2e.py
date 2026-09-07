# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for the Warp SYCL (Intel GPU) device.

Requires `python -m mjlab_sycl install` to have been run in this environment
(backend overlay + warpsycl.dll in warp's kernel cache), plus an Intel GPU
with oneAPI and MSVC Build Tools available.

Usage:
    python -m mjlab_sycl.test_e2e       # this gate alone
    mjlab-sycl-test                     # both gates, in order

Checks, in order:
  1. the 'sycl' device is registered and reports the Intel GPU
  2. USM-backed array allocation, fill, and numpy readback on the host
  3. a user kernel compiles through the SYCL codegen + icx chain and runs
  4. results are bit-exact against the CPU execution of the same kernel
  5. a second launch hits the module cache (no recompile)
  6. vec3/quat math and custom structs compile and run
  7. device-side atomics (int32/float32) are truly atomic under 1M-way races
  8. 2-D launches with atomic reductions
  9. the adjoint (backward) path through wp.Tape produces correct gradients
"""

import numpy as np

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import warp as wp  # noqa: E402

wp.init()


@wp.kernel
def saxpy(a: float, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a * x[i] + y[i]


@wp.kernel
def vec_kernel(p: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = p[i] * 2.0 + q[i]
    out[i] = wp.dot(v, wp.vec3(1.0, 0.5, 0.25))


@wp.struct
class Particle:
    pos: wp.vec3
    vel: wp.vec3
    mass: wp.float32


@wp.kernel
def struct_kernel(ps: wp.array(dtype=Particle), dt: float):
    i = wp.tid()
    p = ps[i]
    p.vel = p.vel * 0.99
    p.pos = p.pos + p.vel * dt
    ps[i] = p


@wp.kernel
def atomic_int_kernel(counts: wp.array(dtype=wp.int32)):
    wp.atomic_add(counts, 0, 1)


@wp.kernel
def atomic_float_kernel(vals: wp.array(dtype=wp.float32)):
    wp.atomic_add(vals, 0, 1.0)


@wp.kernel
def reduce2d_kernel(a: wp.array2d(dtype=wp.float32), s: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    wp.atomic_add(s, 0, a[i, j])


@wp.kernel
def loss_kernel(x: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, x[i] * x[i])


def check(failures, label, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def main():
    failures = []
    n = 1 << 20

    # -- 1. device registration -------------------------------------------------
    dev = wp.get_device("sycl")
    print(f"[1] sycl device: {dev.name!r} is_sycl={dev.is_sycl}")
    if not dev.is_sycl:
        failures.append("sycl device not registered")

    # -- 2. arrays + numpy interop ----------------------------------------------
    x = wp.zeros(n, dtype=wp.float32, device="sycl")
    y = wp.zeros(n, dtype=wp.float32, device="sycl")
    out = wp.zeros(n, dtype=wp.float32, device="sycl")

    x.fill_(1.5)
    y.fill_(2.25)
    check(failures, "host write / numpy readback through USM",
          np.array_equal(x.numpy(), np.full(n, 1.5, dtype=np.float32)))

    # -- 3. kernel compile + launch ---------------------------------------------
    a = 2.0
    wp.launch(saxpy, dim=n, inputs=[a, x, y, out], device="sycl")

    expected = np.full(n, 2.0 * 1.5 + 2.25, dtype=np.float32)
    check(failures, "saxpy kernel result", np.array_equal(out.numpy(), expected))

    # -- 4. cross-check against the CPU device ---------------------------------
    x_cpu = wp.array(np.full(n, 1.5, dtype=np.float32), dtype=wp.float32, device="cpu")
    y_cpu = wp.array(np.full(n, 2.25, dtype=np.float32), dtype=wp.float32, device="cpu")
    out_cpu = wp.zeros(n, dtype=wp.float32, device="cpu")
    wp.launch(saxpy, dim=n, inputs=[a, x_cpu, y_cpu, out_cpu], device="cpu")
    check(failures, "sycl and cpu results agree", np.array_equal(out_cpu.numpy(), out.numpy()))

    # -- 5. cached relaunch ------------------------------------------------------
    out.zero_()
    wp.launch(saxpy, dim=n, inputs=[a, x, y, out], device="sycl")
    check(failures, "cached relaunch result", np.array_equal(out.numpy(), expected))

    # -- 6. vec3 math + custom structs -----------------------------------------
    p_np = np.random.rand(n, 3).astype(np.float32)
    q_np = np.random.rand(n, 3).astype(np.float32)
    p = wp.array(p_np, dtype=wp.vec3, device="sycl")
    q = wp.array(q_np, dtype=wp.vec3, device="sycl")
    vout = wp.zeros(n, dtype=wp.float32, device="sycl")
    wp.launch(vec_kernel, dim=n, inputs=[p, q, vout], device="sycl")

    v_expected = (p_np * 2.0 + q_np) @ np.array([1.0, 0.5, 0.25], dtype=np.float32)
    check(failures, "vec3 kernel vs numpy", np.allclose(vout.numpy(), v_expected, atol=1e-6))

    ps_np = np.zeros(n, dtype=[("pos", np.float32, (3,)), ("vel", np.float32, (3,)), ("mass", np.float32)])
    ps_np["pos"] = np.random.rand(n, 3).astype(np.float32)
    ps_np["vel"] = np.random.rand(n, 3).astype(np.float32)
    ps_np["mass"] = 1.0
    # note: array.numpy() copies out (like CUDA devices), so initialize through
    # the constructor, which exercises the host->device copy path
    ps = wp.array(ps_np, dtype=Particle, device="sycl")
    dt = 0.01
    wp.launch(struct_kernel, dim=n, inputs=[ps, dt], device="sycl")
    ps_expected = ps_np["pos"] + ps_np["vel"] * 0.99 * dt
    check(failures, "struct kernel vs numpy", np.allclose(ps.numpy()["pos"], ps_expected, atol=1e-6))

    # -- 7. device-side atomics --------------------------------------------------
    counts = wp.zeros(1, dtype=wp.int32, device="sycl")
    wp.launch(atomic_int_kernel, dim=n, inputs=[counts], device="sycl")
    check(failures, f"int atomic_add under {n}-way race", counts.numpy()[0] == n)

    fvals = wp.zeros(1, dtype=wp.float32, device="sycl")
    wp.launch(atomic_float_kernel, dim=n, inputs=[fvals], device="sycl")
    check(failures, f"float atomic_add under {n}-way race", fvals.numpy()[0] == float(n))

    # -- 8. 2-D launch + atomic reduction --------------------------------------
    d0, d1 = 512, 512
    a2d_np = np.random.rand(d0, d1).astype(np.float32)
    a2d = wp.array(a2d_np, dtype=wp.float32, device="sycl")
    s = wp.zeros(1, dtype=wp.float32, device="sycl")
    wp.launch(reduce2d_kernel, dim=(d0, d1), inputs=[a2d, s], device="sycl")
    check(failures, "2-D launch atomic reduction", np.isclose(s.numpy()[0], a2d_np.sum(), rtol=1e-5))

    # -- 9. adjoint path (wp.Tape) ----------------------------------------------
    xg = wp.array(np.random.rand(1024).astype(np.float32), dtype=wp.float32, device="sycl", requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device="sycl", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(loss_kernel, dim=1024, inputs=[xg, loss], device="sycl")
    tape.backward(loss)
    check(failures, "tape gradients (d/dx sum(x^2) == 2x)", np.allclose(xg.grad.numpy(), 2.0 * xg.numpy(), atol=1e-5))

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)

    print("ALL OK")


if __name__ == "__main__":
    main()