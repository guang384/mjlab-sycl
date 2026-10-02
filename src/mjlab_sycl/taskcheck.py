# SPDX-License-Identifier: Apache-2.0
"""Task-name validation for the training entries.

Task names are plain strings registered by whatever mjlab task package
lives in the venv (``register_mjlab_task(task_id=...)`` -- e.g.
microduck_rl's ``src/mjlab_microduck/tasks/__init__.py``). Nothing in
mjlab-sycl knows any particular name; these helpers turn the registry's
raw KeyError into a listing and did-you-mean suggestions.
"""

from __future__ import annotations


def registered_tasks() -> list[str]:
    from mjlab.tasks import registry

    return sorted(registry.list_tasks())


def print_task_list(names: list[str]) -> None:
    print(f"{len(names)} registered tasks in this venv:")
    for name in names:
        print(f"  {name}")


def resolve_task_or_exit(task: str | None, prog: str) -> str:
    """Return a registered task name, or exit with a helpful listing."""
    import difflib

    names = registered_tasks()
    if task is None:
        print(f"{prog}: missing task name.", flush=True)
        print_task_list(names)
        raise SystemExit(1)
    if task in names:
        return task
    print(f"{prog}: unknown task: {task!r}", flush=True)
    close = difflib.get_close_matches(task, names, n=5, cutoff=0.35)
    if close:
        print("  did you mean: " + ", ".join(close), flush=True)
    print(f"  {len(names)} tasks are registered in this venv (names come "
          "from the task package's register_mjlab_task calls); "
          "list them with:", flush=True)
    print(f"    {prog} --list-tasks", flush=True)
    raise SystemExit(1)
