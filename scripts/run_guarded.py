# SPDX-License-Identifier: Apache-2.0
"""Run a command under a wall-clock watchdog (process-level fuse).

The warp-sycl dll already carries an in-process device watchdog (see
WARP_SYCL_SYNC_TIMEOUT_S): a hung GPU kernel aborts the process within
seconds and names the culprit kernel, letting Windows TDR reset the iGPU.
This script is the outer backstop for everything the dll cannot see --
endless host-side loops, stuck compiles, wedged subprocesses.

Usage:
    python scripts/run_guarded.py --timeout 900 -- python -m mjlab_sycl.train ...
    python scripts/run_guarded.py --timeout 900 -- uv run train <TASK_ID> ...

On timeout the child process tree is killed and the exit code is 3.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--timeout", type=float, default=1800.0,
                      help="hard wall-clock limit in seconds (default 1800)")
  parser.add_argument("cmd", nargs=argparse.REMAINDER,
                      help="command to run (everything after '--')")
  args = parser.parse_args()

  cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
  if not cmd:
    parser.error("no command given (use: run_guarded.py --timeout N -- <cmd...>)")

  started = time.monotonic()
  proc = subprocess.Popen(cmd)

  def watchdog() -> None:
    time.sleep(args.timeout)
    if proc.poll() is None:
      print(f"\n[run_guarded] TIMEOUT after {args.timeout:.0f}s -- killing "
            f"process tree (pid {proc.pid}): {' '.join(cmd)}", file=sys.stderr)
      try:
        # kill the whole tree, not just the shell
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
      except OSError:
        proc.kill()
      # exit hard so a stuck parent cannot linger
      os._exit(3)

  threading.Thread(target=watchdog, daemon=True).start()

  code = proc.wait()
  wall = time.monotonic() - started
  if code != 0:
    print(f"[run_guarded] command exited with code {code} after {wall:.0f}s",
          file=sys.stderr)
  sys.exit(code)


if __name__ == "__main__":
  main()
