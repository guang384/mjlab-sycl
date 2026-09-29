# SPDX-License-Identifier: Apache-2.0
"""Skip mujoco_warp launches whose grid dimension is 0 (no work to do).

mujoco_warp unconditionally launches every kernel in fwd_position /
fwd_velocity / fwd_actuation / sensor regardless of whether the model
has the relevant features.  For microduck (nflex=0, ntendon=0, neq=0,
no ball joints, no touch/tactile/limit sensors) that means 24 distinct
kernels per call launch with a 0-sized grid axis — at 5 calls per env
step, 120 wasted dispatches per step.

This module installs a SINGLE global filter at the outermost wp.launch
layer.  It checks the ACTUAL dim argument passed to each wp.launch call;
if any axis is 0 the launch is skipped.  This is:

  - Per-call, not per-model: a different model with tendons/flex will
    produce non-zero dims for those kernels, so they are NOT skipped.
  - Semantically equivalent to executing: warp schedules zero work
    items for a 0-axis grid, so the kernel body never runs and no memory
    is read or written.  Skipping the launch saves the host-side
    dispatch + pack_args overhead (~0.1 ms × 120 = 12 ms/step) with
    zero change in output.

Safety: the only conceivable side effect of a 0-dim launch would be
writing to output arrays, but (a) callers zero_() outputs before launch,
(b) the kernel body starts with ``if idx >= count: return`` so even with
work items it would skip, and (c) the outputs are never read when the
corresponding model dimension is 0.

Kill switch: ``MJLAB_SYCL_SKIP_EMPTY=0``.
"""

from __future__ import annotations

import os

import warp as wp

_prev_launch = None


def _enabled() -> bool:
  return os.environ.get("MJLAB_SYCL_SKIP_EMPTY", "1").strip().lower() not in (
    "0",
    "false",
    "off",
  )


def _has_zero_axis(dim) -> bool:
  if isinstance(dim, (list, tuple)):
    return any(axis == 0 for axis in dim)
  return dim == 0


def _filtered_launch(kernel, dim, inputs=(), outputs=(), *args, **kwargs):
  if _has_zero_axis(dim):
    return None  # no work: every work item would return immediately
  return _prev_launch(kernel, dim, inputs, outputs, *args, **kwargs)


def install() -> None:
  global _prev_launch
  if _prev_launch is not None:
    return
  if not _enabled():
    return
  _prev_launch = wp.launch
  wp.launch = _filtered_launch
  from warp._src import context as _ctx

  _ctx.launch = _filtered_launch
  print(
    "[sycl-skip-empty] skipping 0-dim launches "
    "(MJLAB_SYCL_SKIP_EMPTY=0 to disable)"
  )


def uninstall() -> None:
  global _prev_launch
  if _prev_launch is None:
    return
  wp.launch = _prev_launch
  from warp._src import context as _ctx

  _ctx.launch = _prev_launch
  _prev_launch = None
