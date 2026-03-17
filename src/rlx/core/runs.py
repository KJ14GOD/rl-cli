import re
from dataclasses import dataclass
from pathlib import Path

from rlx.config.schema import ExperimentConfig
from rlx.core.metadata import build_run_metadata, write_metadata
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.paths import (
    CONFIG_SNAPSHOT_NAME,
    METADATA_NAME,
    METRICS_NAME,
    RUN_ARTIFACT_DIRS,
    RUNS_DIR,
)


class RunPreparationError(Exception):
    """Raised when a training run scaffold cannot be created."""


@dataclass(frozen=True)
class RunPreparationResult:
    project_root: Path
    run_dir: Path
    config_snapshot: Path
    metadata_path: Path
    metrics_path: Path
    artifact_dirs: tuple[Path, ...]


def prepare_run(config: ExperimentConfig, config_path: Path) -> RunPreparationResult:
    """Create the standard RLCLI run directory and core tracking files."""

    source_config_path = config_path.expanduser().resolve()

    try:
        project_root = find_project_root(source_config_path)
    except ProjectLookupError as exc:
        raise RunPreparationError(str(exc)) from exc

    runs_root = project_root / RUNS_DIR
    run_dir = runs_root / _next_run_id(runs_root, config.run_name)
    run_dir.mkdir()

    artifact_dirs = []
    for dirname in RUN_ARTIFACT_DIRS:
        path = run_dir / dirname
        path.mkdir()
        artifact_dirs.append(path)

    config_snapshot = run_dir / CONFIG_SNAPSHOT_NAME
    config_snapshot.write_text(source_config_path.read_text(encoding="utf-8"), encoding="utf-8")

    metrics_path = run_dir / METRICS_NAME
    metrics_path.write_text("", encoding="utf-8")

    metadata_path = run_dir / METADATA_NAME
    metadata = build_run_metadata(
        project_root=project_root,
        run_dir=run_dir,
        config=config,
        source_config_path=source_config_path,
        status="prepared",
    )
    write_metadata(metadata_path, metadata)

    return RunPreparationResult(
        project_root=project_root,
        run_dir=run_dir,
        config_snapshot=config_snapshot,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        artifact_dirs=tuple(artifact_dirs),
    )


def _next_run_id(runs_root: Path, run_name: str) -> str:
    base = _slugify_run_name(run_name)
    pattern = re.compile(rf"^{re.escape(base)}_(\d{{3}})$")
    existing = []

    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            existing.append(int(match.group(1)))

    next_index = max(existing, default=0) + 1
    return f"{base}_{next_index:03d}"


def _slugify_run_name(run_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", run_name.strip()).strip("_").lower()
    return slug or "run"
