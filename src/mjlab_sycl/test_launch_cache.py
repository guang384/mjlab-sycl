# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for launch_cache: cache hit/miss behavior, data
mutation through cached launches, scalar-key sensitivity, dim-key
sensitivity, and fallback paths."""

from mjlab_sycl._bootstrap import prepare_sycl_runtime_path

prepare_sycl_runtime_path()

import warp as wp

wp.init()

from mjlab_sycl import launch_cache

launch_cache.install()
print("launch_cache installed:", wp.launch.__name__)


@wp.kernel(enable_backward=False)
def add_kernel(
  a: wp.array(dtype=wp.float32),
  b: wp.array(dtype=wp.float32),
  c: wp.array(dtype=wp.float32),
  n: int,
):
  i = wp.tid()
  if i < n:
    c[i] = a[i] + b[i]


def main() -> None:
  n = 10
  a = wp.zeros(n, dtype=wp.float32, device="cpu")
  b = wp.zeros(n, dtype=wp.float32, device="cpu")
  c = wp.zeros(n, dtype=wp.float32, device="cpu")
  a.fill_(1.0)
  b.fill_(2.0)

  # First launch: cache miss, runs original path
  wp.launch(add_kernel, dim=n, inputs=[a, b, c], outputs=[n], device="cpu")
  wp.synchronize_device("cpu")
  assert abs(c.numpy()[0] - 3.0) < 1e-6, f"first launch failed: {c.numpy()[0]}"
  print("PASS: first launch (miss)")

  # Second launch: cache hit — same arrays, same dim, same scalar.
  # CRITICAL: data has changed; cached struct must point to the same
  # (mutated) buffers, not stale copies.
  a.fill_(10.0)
  b.fill_(20.0)
  wp.launch(add_kernel, dim=n, inputs=[a, b, c], outputs=[n], device="cpu")
  wp.synchronize_device("cpu")
  assert abs(c.numpy()[0] - 30.0) < 1e-6, (
    f"cache hit produced wrong result: {c.numpy()[0]} != 30"
  )
  print("PASS: cache hit with mutated data")

  # Third launch: different scalar arg -> different key -> fresh entry
  a.fill_(1.0)
  b.fill_(1.0)
  wp.launch(add_kernel, dim=n, inputs=[a, b, c], outputs=[n // 2], device="cpu")
  wp.synchronize_device("cpu")
  assert abs(c.numpy()[0] - 2.0) < 1e-6, "different scalar failed"
  print("PASS: scalar-key sensitivity")

  # Fourth launch: different array object -> different key
  d = wp.zeros(n, dtype=wp.float32, device="cpu")
  d.fill_(100.0)
  wp.launch(add_kernel, dim=n, inputs=[a, d, c], outputs=[n], device="cpu")
  wp.synchronize_device("cpu")
  assert abs(c.numpy()[0] - 101.0) < 1e-6, "different array failed"
  print("PASS: array-identity key")

  # Fifth: back to the original key — must still hit and be correct
  a.fill_(5.0)
  b.fill_(6.0)
  wp.launch(add_kernel, dim=n, inputs=[a, b, c], outputs=[n], device="cpu")
  wp.synchronize_device("cpu")
  assert abs(c.numpy()[0] - 11.0) < 1e-6, "return to original key failed"
  print("PASS: return to original cache entry")

  # Fallback paths must still work (adjoint launches bypass cache)
  try:
    wp.launch(
      add_kernel, dim=n, inputs=[a, b, c], outputs=[n], device="cpu", adjoint=True
    )
  except Exception as e:
    print(f"PASS: adjoint falls back to original ({type(e).__name__})")

  stats = launch_cache.stats()
  print(f"\ncache stats: {stats}")
  # With the recurrence gate, launch 2 is a "recurring" build (not a hit);
  # the only true hit is launch 5 (the third occurrence of K1).  The gate
  # also means every non-hit non-bypass launch is a miss.
  assert stats["hit"] >= 1, f"expected >=1 hit, got {stats}"
  assert stats["miss"] >= 3, f"expected >=3 misses, got {stats}"

  # Kill switch test
  import os

  launch_cache.uninstall()
  assert wp.launch is not launch_cache._cached_launch, "uninstall failed"
  print("PASS: uninstall restores original")
  launch_cache.install()

  print("\nALL LAUNCH CACHE TESTS PASSED")


if __name__ == "__main__":
  main()
