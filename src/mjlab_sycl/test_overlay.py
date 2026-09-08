# SPDX-License-Identifier: Apache-2.0
"""Host-side overlay-sync gate + unit tests for install.overlay_problems().

Runs with no GPU and no sycl device: every file this package ships must be
byte-identical to what ``python -m mjlab_sycl install`` placed in the
environment's warp package and kernel cache. It is the FIRST gate of
``mjlab-sycl-test`` so a uv-sync-wiped or drifted overlay is reported before
the GPU gates run. Also pytest-discoverable for the pure file-comparison
logic (``pytest src/mjlab_sycl/test_overlay.py``).

Checks:
  1. the real environment's overlay is in sync (run install if not)
  2. a byte-identical mirror of the shipped backend reports zero problems
  3. a tampered file is reported ("differs from the bundled backend")
  4. a deleted file is reported ("missing overlay file")
  5. ensure_overlay_synced() exits 2 when problems exist
  6. a warp version drift is reported

Usage:
    python -m mjlab_sycl.test_overlay    # this gate alone
    mjlab-sycl-test                      # overlay gate, then e2e, then mujoco
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

from mjlab_sycl import install as _install


def _mirror_backend(tmp_warp: str, tmp_cache: str) -> None:
  """Copy the shipped backend files into throwaway dirs, mirroring install().

  Uses install._shipped_files() (the single source of the install layout) so
  the test cannot drift from what install() actually copies. The layout must
  point at the throwaway dirs during the copy, or the files land in the real
  venv's warp package.
  """
  backend = os.path.join(os.path.dirname(_install.__file__), "backend")
  os.makedirs(tmp_warp, exist_ok=True)
  os.makedirs(tmp_cache, exist_ok=True)
  prev = _install._warp_layout
  _install._warp_layout = lambda: (tmp_warp, tmp_cache)
  try:
    for shipped, installed in _install._shipped_files(backend):
      installed = os.path.normpath(installed)  # strip the "\.\.." joins
      os.makedirs(os.path.dirname(installed), exist_ok=True)
      shutil.copy2(shipped, installed)
  finally:
    _install._warp_layout = prev


def _problems_with(tmp_warp: str, tmp_cache: str) -> list:
  """overlay_problems() against throwaway dirs instead of the real venv."""
  prev = _install._warp_layout
  _install._warp_layout = lambda: (tmp_warp, tmp_cache)
  try:
    return _install.overlay_problems()
  finally:
    _install._warp_layout = prev


def _tmp_layout():
  d = tempfile.mkdtemp(prefix="mjlab_sycl_overlay_")
  return d, os.path.join(d, "warp"), os.path.join(d, "cache")


# ---------------------------------------------------------------------------
# checks (raise AssertionError on failure; used by both main() and pytest)
# ---------------------------------------------------------------------------

def check_real_env_synced() -> None:
  problems = _install.overlay_problems()
  assert not problems, (
    "the real environment's overlay is out of sync — run `python -m "
    "mjlab_sycl install`:\n  " + "\n  ".join(problems)
  )


def check_identical_mirror_is_clean() -> None:
  d, tmp_warp, tmp_cache = _tmp_layout()
  try:
    _mirror_backend(tmp_warp, tmp_cache)
    problems = _problems_with(tmp_warp, tmp_cache)
    assert problems == [], f"byte-identical mirror reported problems: {problems}"
  finally:
    shutil.rmtree(d, ignore_errors=True)


def check_tamper_is_reported() -> None:
  d, tmp_warp, tmp_cache = _tmp_layout()
  try:
    _mirror_backend(tmp_warp, tmp_cache)
    tampered = os.path.join(tmp_warp, "native", "tile.h")
    with open(tampered, "ab") as f:
      f.write(b"// tamper-test\n")
    problems = _problems_with(tmp_warp, tmp_cache)
    assert len(problems) == 1, f"expected exactly 1 problem, got {problems}"
    assert "differs from the bundled backend" in problems[0], problems[0]
    assert "tile.h" in problems[0], problems[0]
  finally:
    shutil.rmtree(d, ignore_errors=True)


def check_missing_is_reported() -> None:
  d, tmp_warp, tmp_cache = _tmp_layout()
  try:
    _mirror_backend(tmp_warp, tmp_cache)
    os.remove(os.path.join(tmp_cache, "warpsycl.dll"))
    problems = _problems_with(tmp_warp, tmp_cache)
    assert len(problems) == 1, f"expected exactly 1 problem, got {problems}"
    assert "missing overlay file" in problems[0], problems[0]
    assert "warpsycl.dll" in problems[0], problems[0]
  finally:
    shutil.rmtree(d, ignore_errors=True)


def check_ensure_exits_on_problems() -> None:
  d, tmp_warp, tmp_cache = _tmp_layout()
  try:
    _mirror_backend(tmp_warp, tmp_cache)
    with open(os.path.join(tmp_warp, "native", "tile.h"), "ab") as f:
      f.write(b"// tamper-test\n")
    prev = _install._warp_layout
    _install._warp_layout = lambda: (tmp_warp, tmp_cache)
    try:
      try:
        # redirect the remediation text: it is expected output here
        with contextlib.redirect_stderr(io.StringIO()):
          _install.ensure_overlay_synced()
      except SystemExit as e:
        assert e.code == 2, f"expected exit code 2, got {e.code}"
      else:
        raise AssertionError("ensure_overlay_synced() did not exit on problems")
    finally:
      _install._warp_layout = prev
  finally:
    shutil.rmtree(d, ignore_errors=True)


def check_warp_version_drift_is_reported() -> None:
  d, tmp_warp, tmp_cache = _tmp_layout()
  try:
    os.makedirs(tmp_warp, exist_ok=True)
    os.makedirs(tmp_cache, exist_ok=True)
    class _StubWarp:
      __version__ = "9.9.9"
    prev_module = sys.modules.get("warp")
    prev_layout = _install._warp_layout
    sys.modules["warp"] = _StubWarp()
    _install._warp_layout = lambda: (tmp_warp, tmp_cache)
    try:
      problems = _install.overlay_problems()
    finally:
      if prev_module is not None:
        sys.modules["warp"] = prev_module
      else:
        del sys.modules["warp"]
      _install._warp_layout = prev_layout
    assert any("warp version is 9.9.9" in p for p in problems), problems
  finally:
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

def test_real_env_synced():
  # CI runners have no installed sycl overlay (no Intel GPU): the real-env
  # check needs `python -m mjlab_sycl install` to have been run. The pure
  # file-comparison tests above cover the logic anywhere.
  if os.environ.get("CI"):
    import pytest
    pytest.skip("real-overlay check needs an installed sycl overlay (GPU)")
  check_real_env_synced()


def test_identical_mirror_is_clean():
  check_identical_mirror_is_clean()


def test_tamper_is_reported():
  check_tamper_is_reported()


def test_missing_is_reported():
  check_missing_is_reported()


def test_ensure_exits_on_problems():
  check_ensure_exits_on_problems()


def test_warp_version_drift_is_reported():
  check_warp_version_drift_is_reported()


# ---------------------------------------------------------------------------
# standalone gate entry
# ---------------------------------------------------------------------------

def main() -> None:
  checks = [
    ("real-env overlay is in sync", check_real_env_synced),
    ("identical mirror is clean", check_identical_mirror_is_clean),
    ("tampered file is reported", check_tamper_is_reported),
    ("missing file is reported", check_missing_is_reported),
    ("ensure_overlay_synced exits 2 on problems", check_ensure_exits_on_problems),
    ("warp version drift is reported", check_warp_version_drift_is_reported),
  ]
  failures = []
  for label, fn in checks:
    try:
      fn()
      print(f"[PASS] {label}")
    except AssertionError as e:
      print(f"[FAIL] {label}")
      print(f"       {e}")
      failures.append(label)
  if failures:
    for f in failures:
      print(f"FAIL: {f}")
    raise SystemExit(1)
  print("ALL OK")


if __name__ == "__main__":
  main()
