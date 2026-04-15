from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.core.compare import CompareError, RunComparison, load_run_comparison, resolve_run_ref
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME, METRICS_NAME


class RunInfoError(Exception):
    """Raised when a run cannot be inspected."""


@dataclass(frozen=True)
class RunInfo:
    run: RunComparison
    project_root: str | None
    source_config_path: str | None
    git_commit: str | None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    failed_at: str | None
    interrupted_at: str | None
    error: str | None
    last_eval_at: str | None
    last_eval_result: str | None
    last_eval_checkpoint: str | None
    eval_log: str | None
    last_video_at: str | None
    last_video_dir: str | None
    last_video_manifest: str | None
    last_video_checkpoint: str | None
    last_plot_at: str | None
    last_plot_dir: str | None
    last_plot_manifest: str | None
    config_snapshot: str
    metadata_file: str
    metrics_file: str
    config_summary: tuple[tuple[str, str], ...]


def load_run_info(run_ref: str, cwd: Path | None = None) -> RunInfo:
    working_dir = (cwd or Path.cwd()).resolve()

    try:
        run_dir = resolve_run_ref(run_ref, working_dir)
        run = load_run_comparison(run_dir)
    except CompareError as exc:
        raise RunInfoError(str(exc)) from exc

    metadata = _load_metadata(run_dir / METADATA_NAME)

    return RunInfo(
        run=run,
        project_root=_maybe_str(metadata.get("project_root")),
        source_config_path=_maybe_str(metadata.get("source_config_path")),
        git_commit=_maybe_str(metadata.get("git_commit")),
        created_at=_maybe_str(metadata.get("created_at")),
        started_at=_maybe_str(metadata.get("started_at")),
        completed_at=_maybe_str(metadata.get("completed_at")),
        failed_at=_maybe_str(metadata.get("failed_at")),
        interrupted_at=_maybe_str(metadata.get("interrupted_at")),
        error=_maybe_str(metadata.get("error")),
        last_eval_at=_maybe_str(metadata.get("last_eval_at")),
        last_eval_result=_maybe_str(metadata.get("last_eval_result")),
        last_eval_checkpoint=_maybe_str(metadata.get("last_eval_checkpoint")),
        eval_log=_maybe_str(metadata.get("eval_log")),
        last_video_at=_maybe_str(metadata.get("last_video_at")),
        last_video_dir=_maybe_str(metadata.get("last_video_dir")),
        last_video_manifest=_maybe_str(metadata.get("last_video_manifest")),
        last_video_checkpoint=_maybe_str(metadata.get("last_video_checkpoint")),
        last_plot_at=_maybe_str(metadata.get("last_plot_at")),
        last_plot_dir=_maybe_str(metadata.get("last_plot_dir")),
        last_plot_manifest=_maybe_str(metadata.get("last_plot_manifest")),
        config_snapshot=CONFIG_SNAPSHOT_NAME,
        metadata_file=METADATA_NAME,
        metrics_file=METRICS_NAME,
        config_summary=_build_config_summary(run),
    )


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunInfoError(f"Could not read run metadata: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunInfoError(f"Run metadata is not valid JSON: {path}") from exc


def _build_config_summary(run: RunComparison) -> tuple[tuple[str, str], ...]:
    keys = (
        ("seed", "seed"),
        ("device", "device"),
        ("envs", "env.num_envs"),
        ("timesteps", "algo.total_timesteps"),
        ("rollout", "algo.rollout_steps"),
        ("batch", "algo.batch_size"),
        ("lr", "algo.learning_rate"),
        ("checkpoint", "checkpoint.save_every"),
        ("eval every", "eval.every"),
        ("eval eps", "eval.episodes"),
        ("policy", "policy.hidden_sizes"),
    )

    rows = []
    for label, key in keys:
        value = run.config_values.get(key)
        if value is None:
            continue
        rows.append((label, value))
    return tuple(rows)


def _maybe_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
