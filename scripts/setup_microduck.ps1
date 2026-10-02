# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
  One-command setup of mjlab-sycl inside any mjlab project venv (microduck_rl
  or any other package that registers mjlab tasks).

.DESCRIPTION
  Runs the documented five steps against a target mjlab project:
    1. `uv sync` the project (only when its .venv is missing)
    2. install mjlab-sycl INTO the project venv (isolated from any global
       pip `target=` redirect) so its console scripts land in .venv\Scripts\
    3. optionally install torch XPU for this machine (default: skip; the
       doctor reports the exact command when it is missing)
    4. `python -m mjlab_sycl install` -- overlay the warp SYCL backend and
       self-check it
    5. `mjlab-sycl-check` -- the read-only environment preflight

  Nothing in the target project's source is modified: this only installs into
  its .venv and overlays the venv's warp package (both wiped by the next
  `uv sync`, re-run this script afterwards).

.EXAMPLE
  .\scripts\setup_microduck.ps1 -Repo C:\path\to\microduck_rl
  .\scripts\setup_microduck.ps1 -Repo C:\path\to\microduck_rl -InstallTorchXpu
  .\scripts\setup_microduck.ps1 -Repo . -PipIndex https://mirrors.aliyun.com/pypi/simple/
#>
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Repo,

  # Download+install the torch XPU wheel for THIS machine (per-machine step,
  # ~2 GB). Off by default: the doctor prints the exact command when missing.
  [switch]$InstallTorchXpu,

  # pip index for the mjlab-sycl build deps (uv_build). Defaults to PyPI;
  # use an aliyun-style mirror if PyPI is slow/blocked on this machine.
  [string]$PipIndex = ""
)

$ErrorActionPreference = "Stop"

function Step-Hint($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Step-Ok($msg)    { Write-Host "    $msg" -ForegroundColor Green }
function Step-Fail($msg)  { Write-Host "    FAIL: $msg" -ForegroundColor Red }

# A pip config file pointed at by PIP_CONFIG_FILE cannot be bypassed by
# --isolated -- clear the env vars so installs land in the venv itself.
Remove-Item Env:PIP_CONFIG_FILE, Env:PIP_TARGET -ErrorAction SilentlyContinue

$RepoPath = (Resolve-Path -LiteralPath $Repo).Path
$ProjectPy = Join-Path $RepoPath "pyproject.toml"
if (-not (Test-Path -LiteralPath $ProjectPy)) {
  Step-Fail "$RepoPath is not a python project (no pyproject.toml). Point -Repo at your mjlab project, e.g. a microduck_rl clone."
  exit 1
}

$Py = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
  Step-Hint "1/6  .venv missing -> uv sync $RepoPath"
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step-Fail "uv is required to create the project venv. Install uv (https://docs.astral.sh/uv/) then re-run."
    exit 1
  }
  Push-Location $RepoPath
  try { uv sync }
  finally { Pop-Location }
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  Step-Hint "1/6  project venv present: $Py"
}
Step-Ok "using venv python: $Py"

# 2. install this package into the venv (editable, no deps: the project
# already provides mjlab/warp/torch; this keeps the package's source live).
$Pkg = Split-Path -Parent $PSScriptRoot   # mjlab-sycl repo root (parent of scripts/)
Step-Hint "2/6  installing mjlab-sycl (editable) into the project venv"
$pipArgs = @("-m", "pip", "install", "--isolated", "--no-deps", "-e", $Pkg)
if ($PipIndex) { $pipArgs += @("--index-url", $PipIndex) }
& $Py @pipArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3. per-machine torch XPU (optional here; doctor reports it when missing)
if ($InstallTorchXpu) {
  Step-Hint "3/6  installing torch==2.9.1+xpu from the PyTorch XPU index (~2 GB)"
  & $Py -m pip install --isolated --no-deps "torch==2.9.1+xpu" --index-url https://download.pytorch.org/whl/xpu
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  Step-Hint "3/6  skipping torch XPU install (use -InstallTorchXpu to fetch it)"
  # the CPU-wheel trap: a plain `pip install torch` installs the CPU build
  # and PPO silently runs ~3x slower with no error. Detect and name it.
  & $Py -c "import torch, sys; sys.exit(0 if torch.xpu.is_available() else 1)" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "    WARNING: torch has no XPU support -- PPO will silently run ~3x slower." -ForegroundColor Yellow
    Write-Host "    A plain 'pip install torch' installs the CPU wheel (the trap)." -ForegroundColor Yellow
    Write-Host "    Fix: re-run with -InstallTorchXpu, or:" -ForegroundColor Yellow
    Write-Host "      $Py -m pip install `"torch==2.9.1+xpu`" --index-url https://download.pytorch.org/whl/xpu" -ForegroundColor Yellow
  }
}

# 4. overlay the warp SYCL backend + self-check
# SYCL 2025.3+ runtime: warpsycl.dll needs sycl8.dll 2025.3+; torch-xpu
  # pins the 2025.2 wheels (one export too old, WinError 127). The ~50 MB
  # pip wheels replace the multi-GB oneAPI toolkit at runtime -- install
  # them only when no oneAPI toolkit is present (its runtime wins PATH).
  $oneapi = @("C:\Program Files (x86)\Intel\oneAPI", "C:\Program Files\Intel\oneAPI") |
    Where-Object { Test-Path (Join-Path $_ "compiler") }
  if (-not $oneapi) {
    Step-Hint "4/6  no oneAPI toolkit -> installing the ~50 MB SYCL 2025.3 runtime wheels"
    & $Py -m pip install --isolated "intel-sycl-rt==2025.3.3" "dpcpp-cpp-rt==2025.3.3"
    if ($LASTEXITCODE -ne 0) {
      Step-Fail "could not install the SYCL runtime wheels (see README-SYCL-TRAINING.md Requirements)"
      exit 1
    }
    Step-Ok "SYCL runtime 2025.3 installed from pip wheels"
  } else {
    Step-Hint "4/6  oneAPI toolkit found -> runtime covered (pip wheels not needed)"
  }

  Step-Hint "5/6  python -m mjlab_sycl install (warp backend overlay)"
& $Py -m mjlab_sycl install
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 5. read-only environment preflight
Step-Hint "6/6  mjlab-sycl-check (environment preflight)"
& $Py -m mjlab_sycl.doctor
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
  Step-Ok "Environment ready. Task ids come from the project's registry (e.g. uv run list-envs inside the project)."
  $trainExe = Join-Path (Split-Path -Parent $Py) "mjlab-sycl-train.exe"
  Write-Host ""
  Write-Host "    & '$trainExe' Mjlab-Velocity-Flat-MicroDuck --num-envs 64 --max-iterations 5" -ForegroundColor Yellow
  Write-Host ""
  Write-Host "NOTE: any 'uv sync' / 'uv run' wipes the overlay and this install - re-run this script afterwards." -ForegroundColor DarkYellow
} else {
  Step-Fail "doctor found problems - fix them per its messages, then re-run this script (or 'python -m mjlab_sycl install' + 'mjlab-sycl-check')."
}
exit $code
