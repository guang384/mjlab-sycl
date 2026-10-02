# Releasing mjlab-sycl

Maintainer checklist. Users install from a git tag or the release's wheel
(both paths below are validated — the wheel ships the prebuilt
`warpsycl.dll` and the full backend).

## Versioning

`0.x.y` — bump **y** for fixes, **x** for feature/optimization releases.
Version lives in exactly one place: `pyproject.toml` → `[project].version`.

## Release steps

```powershell
# 1. bump the version
#    pyproject.toml: version = "0.3.0"

# 2. fold the changelog: [Unreleased] -> [0.3.0] - YYYY-MM-DD,
#    leaving an empty [Unreleased] at the top (CHANGELOG.md)

# 3. build and verify the artifact
uv build                       # or: python -m pip wheel . -w dist --no-deps
#    the wheel MUST contain mjlab_sycl/backend/warpsycl.dll (the prebuilt
#    micro-driver) -- a wheel without it is broken, check package data.

# 4. smoke it end to end before tagging (2 min if the kernel cache is warm)
<some-test-project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps dist\mjlab_sycl-<ver>-py3-none-any.whl
<some-test-project>\.venv\Scripts\mjlab-sycl-install.exe
<some-test-project>\.venv\Scripts\mjlab-sycl-check.exe
<some-test-project>\.venv\Scripts\mjlab-sycl-test.exe

# 5. commit + tag + push (the tag name is the release name)
git commit -am "release: 0.3.0"
git tag v0.3.0
git push origin master v0.3.0

# 6. GitHub -> Releases -> create from tag v0.3.0
#    body: paste the CHANGELOG section for this version
#    assets: attach dist/*.whl (optional but convenient)
```

CI (`.github/workflows/ci.yml`) gates the commit with the host-side unit
tests and build; `.github/workflows/gpu-gates.yml` runs the GPU gates on
the self-hosted runner.

## What users do (keep in sync with the READMEs)

Install into their **mjlab project's venv** (`--no-deps`: the project's own
lockfile owns the dependency pins; `--isolated`: guards against a global
pip `target=` redirect):

```powershell
# released version (from the git tag)
<project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps "git+https://github.com/guang384/mjlab-sycl@v0.3.0"

# or the release's wheel
<project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps dist\mjlab_sycl-0.3.0-py3-none-any.whl

# then: overlay + precompile kernels, and verify the machine
<project>\.venv\Scripts\mjlab-sycl-install.exe --warmup <TASK>
<project>\.venv\Scripts\mjlab-sycl-check.exe
```

Contributors instead clone the repo and install editable
(`pip install --isolated --no-deps -e .`); `mjlab-sycl-install` re-syncs the
overlay after any source change.

Not published to PyPI (the package is built around a pinned
warp/mjlab/mujoco-warp stack that lives in the user's project lockfile);
the git tag and release wheel are the distribution channels.
