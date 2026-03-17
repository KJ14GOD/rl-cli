import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlx.config.schema import ExperimentConfig
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME, METRICS_NAME, RUN_ARTIFACT_DIRS


def build_run_metadata(
    *,
    project_root: Path,
    run_dir: Path,
    config: ExperimentConfig,
    source_config_path: Path,
    status: str,
) -> dict[str, Any]:
    """Build the canonical metadata payload for a prepared run."""

    return {
        "run_id": run_dir.name,
        "run_name": config.run_name,
        "status": status,
        "created_at": _utc_now_iso(),
        "seed": config.seed,
        "device": config.device,
        "environment": config.env.id,
        "algorithm": config.algo.name,
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "source_config_path": str(source_config_path),
        "git_commit": get_git_commit(project_root),
        "artifacts": {
            "config_snapshot": CONFIG_SNAPSHOT_NAME,
            "metadata": METADATA_NAME,
            "metrics": METRICS_NAME,
            **{f"{name}_dir": name for name in RUN_ARTIFACT_DIRS},
        },
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write run metadata as stable, readable JSON."""

    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def update_metadata(path: Path, **updates: Any) -> dict[str, Any]:
    """Update an existing metadata file with new top-level keys."""

    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(updates)
    write_metadata(path, metadata)
    return metadata


def get_git_commit(cwd: Path) -> str | None:
    """Return the current git commit hash if the path is inside a git repository."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
