# SPDX-License-Identifier: Apache-2.0
"""Coarser convergence polling for the sycl capture_while fallback.

On CUDA, warp's ``wp.capture_while`` (used by mujoco_warp's solver loop) runs
as a conditional graph node -- zero host round-trips. On the sycl device there
is no graph capture, so the patched warp falls back to an emulated loop that
drains the whole queue and reads the 1-int condition **every iteration**
(warp/_src/context.py, capture_while non-graph branch). With ~11 solver
iterations per solve and 5 solves per env step that is ~55 full queue drains
per step just to poll convergence.

This module replaces that emulation with batched polling:

  - while many worlds are still solving, run ``POLL_EVERY`` body iterations
    between polls (the queue only drains at the poll);
  - once few worlds remain (<= ``POLL_TAIL``), poll every iteration so the
    loop still stops exactly when they converge.

Correctness is bit-identical to the original: the host only decides *when to
stop launching more iterations*. Iterations launched after a world converged
are guarded no-ops (every solver body kernel early-returns on ``ctx.done``;
``solve_done`` increments ``solver_niter`` and decrements ``nsolving`` only on
the not-done path), so stopping later changes nothing but the number of
no-op launches. Enabled with ``MJLAB_SYCL_POLL_EVERY=N`` (default off -> the
original per-iteration polling runs untouched); ``MJLAB_SYCL_POLL_TAIL``
(default 512) sets the remaining-worlds threshold below which polling reverts
to every iteration.
"""

import ctypes
import os
import struct

import warp as wp

_orig = None


def _remaining(condition) -> int:
  # USM shared memory is host-visible; the caller drains before calling so the
  # read observes kernels submitted so far.
  return struct.unpack("<i", ctypes.string_at(condition.ptr, 4))[0]


def install_poll_batching() -> None:
  """Wrap wp.capture_while (sycl only) with batched convergence polling."""
  global _orig
  if _orig is not None:
    return

  import warp._src.context as wctx

  _orig = wctx.capture_while

  def patched(condition, while_body, stream=None, **kwargs):
    dev = getattr(condition, "device", None)
    is_sycl = dev is not None and getattr(dev, "is_sycl", False)
    poll_every_raw = os.environ.get("MJLAB_SYCL_POLL_EVERY", "")
    if not is_sycl or not poll_every_raw:
      return _orig(condition, while_body, stream=stream, **kwargs)

    poll_every = max(1, int(poll_every_raw))
    try:
      tail = max(1, int(os.environ.get("MJLAB_SYCL_POLL_TAIL", "512")))
    except ValueError:
      tail = 512

    while True:
      # drain so the raw USM read below observes all kernels submitted so far
      wp.synchronize_device("sycl")
      if _remaining(condition) <= 0:
        return
      # coarse batches while many worlds solve; per-iteration once few remain
      batch = 1 if _remaining(condition) <= tail else poll_every
      for _ in range(batch):
        while_body(**kwargs)

  wctx.capture_while = patched
  if hasattr(wp, "capture_while"):
    wp.capture_while = patched


def uninstall_poll_batching() -> None:
  global _orig
  if _orig is None:
    return
  import warp._src.context as wctx

  wctx.capture_while = _orig
  if hasattr(wp, "capture_while"):
    wp.capture_while = _orig
  _orig = None
