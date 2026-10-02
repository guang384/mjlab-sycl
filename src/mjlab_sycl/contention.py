# SPDX-License-Identifier: Apache-2.0
"""GPU-contention check for the training and benching entries.

Physics on the iGPU is bus-bound: background GPU work (video playback,
browsers, compositing bursts) measurably costs 10-45% of training
throughput -- a ~50%-duty ~85 GB/s bandwidth hog dropped a 4096-env bench
from 14,453 to 8,000 env-steps/s (2026-10-02, docs/performance.md), and
the historical "cold vs warm" session variance is attributed to exactly
this, not thermals. This module samples the Windows "GPU Engine"
performance counters and reports OTHER processes with measurable GPU
utilization, so a multi-hour run never starts against a silent
bandwidth thief.

Implementation notes:

- Sampling goes through the `typeperf` command line (same PDH data) rather
  than raw Pdh* ctypes calls: on this machine the formatted/raw counter
  ARRAY functions reject the wildcard GPU Engine counter (0xC0000BBD)
  while typeperf handles the identical path fine. Startup cost ~2-3 s.
- English counter paths resolve on non-English Windows via typeperf
  (verified on zh-CN).
- "Utilization Percentage" is a rate counter: two samples bracket the
  window; the second line carries the utilization over the interval.
- Instance names encode the process: pid_1234_luid_..._engtype_3D.
  Instances are summed per pid across engines (a process can hold 3D,
  Copy, VideoDecode engines simultaneously); our own pid and the system
  ids are excluded.

Best-effort by contract: any failure returns quietly -- the check must
never block training. Kill switch: MJLAB_SYCL_CONTENTION_CHECK=0.

Detection boundary (measured 2026-10-02): the GPU Engine counter set
attributes D3D-class loads -- video playback (VideoDecode/VideoProcessing),
browsers, compositing (3D) -- but NOT Level-Zero compute submissions: a
full-rate torch.xpu copy loop registered ~0%. The checker therefore covers
the desktop-app contention that motivated it (the historical "cold vs
warm" variance) and cannot see competing SYCL/L0 compute jobs; for those,
queue utilization remains observable only through the bench itself.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess

_PID_RE = re.compile(r"pid_(\d+)_")
_SYSTEM_PIDS = frozenset({0, 4})  # idle, system


def utilization_by_pid(sample_s: int = 1) -> dict[int, float]:
  """Per-pid GPU utilization (%) sampled over one interval. Windows only;
  {} on any failure."""
  try:
    out = subprocess.run(
        ["typeperf", "\\GPU Engine(*)\\Utilization Percentage",
         "-sc", "2", "-si", str(max(int(sample_s), 1))],
        capture_output=True, timeout=30, check=True,
    ).stdout.decode("utf-8", errors="replace")  # summary lines are GBK on zh-CN
  except Exception:
    return {}

  # rows: header (first cell "(PDH-CSV 4.0)"), 2 sample rows, then localized
  # summary lines. Keep rows with the full field count; drop empty/summary.
  rows = [r for r in csv.reader(out.splitlines()) if len(r) > 5]
  if len(rows) < 3:  # header + 2 samples
    return {}
  header, values = rows[0], rows[-1]
  if len(header) != len(values):
    return {}
  per_pid: dict[int, float] = defaultdict(float)
  for path, value in zip(header[1:], values[1:]):
    m = _PID_RE.search(path)
    if not m:
      continue
    try:
      pct = float(value)
    except ValueError:
      continue
    if pct > 0:
      per_pid[int(m.group(1))] += pct
  return dict(per_pid)


def _process_name(pid: int) -> str:
  try:
    import psutil

    return psutil.Process(pid).name()
  except Exception:
    return "?"


def check(threshold_pct: float = 1.0, top: int = 5) -> None:
  """Warn on stdout when other processes hold measurable GPU utilization.

  Never raises; prints nothing on a quiet machine."""
  if os.environ.get("MJLAB_SYCL_CONTENTION_CHECK", "1").strip().lower() in (
      "0", "false", "off",
  ):
    return
  try:
    util = utilization_by_pid()
  except Exception:
    return
  if not util:
    return
  mine = os.getpid()
  others = sorted(
      ((pid, pct) for pid, pct in util.items()
       if pid != mine and pid not in _SYSTEM_PIDS and pct >= threshold_pct),
      key=lambda kv: -kv[1],
  )[:top]
  if not others:
    return
  rows = ", ".join(f"{_process_name(pid)} (pid {pid}, {pct:.0f}%)" for pid, pct in others)
  print(
      f"[mjlab-sycl] WARNING: other processes are using the GPU: {rows}. "
      f"Physics on the iGPU is bus-bound -- expect 10-45% slower training. "
      f"Close them for full throughput (silence: MJLAB_SYCL_CONTENTION_CHECK=0).",
      flush=True,
  )
