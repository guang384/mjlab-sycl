---
name: Bug report
about: Something is broken or behaves wrong with mjlab-sycl
title: ''
labels: bug
assignees: ''
---

**Environment**
- OS: (Windows version)
- GPU: (e.g. Intel Arc 130T iGPU)
- Python: 3.12.x
- mjlab-sycl version / commit:
- Host mjlab project + version (e.g. microduck_rl @ <commit>)
- warp / mujoco-warp / torch versions in the project venv

**Describe the bug**
A clear and concise description.

**To reproduce**
Steps, including the exact command (e.g. `mjlab-sycl-train ...`).

**Expected vs actual**
What you expected and what happened (paste key log lines).

**Diagnostics**
Run `mjlab-sycl-check` and paste its output (all lines, [PASS]/[FAIL]).
If relevant, also `python -m mjlab_sycl.test`.

**Additional context**
Anything else (overlay re-install state, `uv sync` history, screenshots).
