from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlx.core.compare import CompareError, RunComparison, load_run_comparison
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.paths import RUNS_DIR


class RunListError(Exception):
    """Raised when RLCLI runs cannot be listed."""


@dataclass(frozen=True)
class RunList:
    project_root: Path
    runs: tuple[RunComparison, ...]


def list_runs(path: Path | None = None) -> RunList:
    candidate = (path or Path.cwd()).resolve()

    try:
        project_root = find_project_root(candidate)
    except ProjectLookupError as exc:
        raise RunListError(str(exc)) from exc

    runs_root = project_root / RUNS_DIR
    if not runs_root.is_dir():
        raise RunListError(f"Runs directory not found: {runs_root}")

    runs = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            runs.append(load_run_comparison(run_dir))
        except CompareError:
            continue

    runs.sort(key=lambda run: run.run_id, reverse=True)
    return RunList(project_root=project_root, runs=tuple(runs))
