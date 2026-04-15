from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME, METRICS_NAME


class CompareError(Exception):
    """Raised when one or more runs cannot be compared."""


@dataclass(frozen=True)
class EvalSummary:
    source: str
    mean_reward: float | None
    mean_episode_length: float | None


@dataclass(frozen=True)
class RunComparison:
    run_dir: Path
    run_id: str
    run_name: str | None
    tags: tuple[str, ...]
    status: str | None
    environment: str | None
    requested_device: str | None
    resolved_device: str | None
    total_timesteps: int | None
    final_rollout_reward: float | None
    best_rollout_reward: float | None
    latest_checkpoint: str | None
    best_checkpoint: str | None
    latest_eval: EvalSummary | None
    best_eval: EvalSummary | None
    last_video_manifest: str | None
    config_values: dict[str, str]


def load_comparisons(run_refs: list[str], cwd: Path | None = None) -> list[RunComparison]:
    if len(run_refs) < 2:
        raise CompareError("Pass at least two runs to compare.")

    working_dir = (cwd or Path.cwd()).resolve()
    return [load_run_comparison(resolve_run_ref(ref, working_dir)) for ref in run_refs]


def resolve_run_ref(run_ref: str, cwd: Path) -> Path:
    candidate = Path(run_ref).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        if _is_run_dir(resolved):
            return resolved
        raise CompareError(f"Run path is not a valid RLCLI run directory: {resolved}")

    direct = (cwd / "runs" / run_ref).resolve()
    if _is_run_dir(direct):
        return direct

    matches = _search_run_matches(run_ref, cwd)
    if not matches:
        raise CompareError(f"Run not found: {run_ref}")
    if len(matches) > 1:
        options = ", ".join(str(path) for path in matches[:5])
        raise CompareError(f"Run reference is ambiguous: {run_ref} ({options})")
    return matches[0]


def load_run_comparison(run_dir: Path) -> RunComparison:
    metadata = json.loads((run_dir / METADATA_NAME).read_text(encoding="utf-8"))
    config_values = _flatten_config_snapshot(run_dir / CONFIG_SNAPSHOT_NAME)
    metrics_summary = _read_metrics_summary(run_dir / METRICS_NAME)
    latest_eval, best_eval = _read_eval_summaries(run_dir / "eval")

    latest_checkpoint = _metadata_relative(metadata, "latest_checkpoint")
    best_checkpoint = _metadata_relative(metadata, "best_checkpoint")

    return RunComparison(
        run_dir=run_dir,
        run_id=metadata.get("run_id", run_dir.name),
        run_name=metadata.get("run_name"),
        tags=_read_tags(metadata.get("tags")),
        status=metadata.get("status"),
        environment=metadata.get("environment"),
        requested_device=metadata.get("device"),
        resolved_device=metadata.get("resolved_device"),
        total_timesteps=_maybe_int(metadata.get("total_timesteps")),
        final_rollout_reward=metrics_summary["final_reward"],
        best_rollout_reward=metrics_summary["best_reward"],
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=best_checkpoint,
        latest_eval=latest_eval,
        best_eval=best_eval,
        last_video_manifest=_metadata_relative(metadata, "last_video_manifest"),
        config_values=config_values,
    )


def diff_config_values(runs: list[RunComparison]) -> list[tuple[str, list[str]]]:
    all_keys = sorted({key for run in runs for key in run.config_values})
    diffs: list[tuple[str, list[str]]] = []

    for key in all_keys:
        values = [run.config_values.get(key, "—") for run in runs]
        if len(set(values)) > 1:
            diffs.append((key, values))
    return diffs


def _is_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / METADATA_NAME).exists()
        and (path / CONFIG_SNAPSHOT_NAME).exists()
    )


def _search_run_matches(run_ref: str, cwd: Path) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()

    for path in cwd.rglob(run_ref):
        resolved = path.resolve()
        if resolved in seen:
            continue
        if _is_run_dir(resolved):
            seen.add(resolved)
            matches.append(resolved)

    return sorted(matches)


def _flatten_config_snapshot(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}

    flat: dict[str, str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child_prefix, value[key])
            return
        if isinstance(value, list):
            flat[prefix] = json.dumps(value)
            return
        if isinstance(value, bool):
            flat[prefix] = "true" if value else "false"
            return
        flat[prefix] = str(value)

    visit("", raw)
    return flat


def _read_metrics_summary(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"final_reward": None, "best_reward": None}

    final_reward: float | None = None
    best_reward: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        reward = payload.get("rollout/ep_rew_mean")
        if not isinstance(reward, int | float):
            continue
        reward_value = float(reward)
        final_reward = reward_value
        if best_reward is None or reward_value > best_reward:
            best_reward = reward_value

    return {"final_reward": final_reward, "best_reward": best_reward}


def _read_eval_summaries(eval_dir: Path) -> tuple[EvalSummary | None, EvalSummary | None]:
    manual = sorted(eval_dir.glob("manual_eval_*.json"))
    summaries: list[EvalSummary] = []

    for path in manual:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = _manual_eval_summary(path, payload)
        if summary is not None:
            summaries.append(summary)

    training_eval = _training_eval_summary(eval_dir / "evaluations.npz")
    if training_eval is not None:
        summaries.append(training_eval)

    if not summaries:
        return None, None

    latest = summaries[-1]
    best = max(
        summaries,
        key=lambda summary: (
            summary.mean_reward if summary.mean_reward is not None else float("-inf")
        ),
    )
    return latest, best


def _manual_eval_summary(path: Path, payload: dict[str, Any]) -> EvalSummary | None:
    if "summary" in payload and isinstance(payload["summary"], dict):
        summary = payload["summary"]
        mean_reward = _maybe_float(summary.get("mean_reward"))
        mean_episode_length = _maybe_float(summary.get("mean_episode_length"))
    else:
        mean_reward = _maybe_float(payload.get("mean_reward"))
        mean_episode_length = _maybe_float(payload.get("mean_episode_length"))

    if mean_reward is None and mean_episode_length is None:
        return None
    return EvalSummary(
        source=path.name,
        mean_reward=mean_reward,
        mean_episode_length=mean_episode_length,
    )


def _training_eval_summary(path: Path) -> EvalSummary | None:
    if not path.exists():
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    try:
        data = np.load(path)
        results = data.get("results")
        ep_lengths = data.get("ep_lengths")
    except Exception:
        return None

    if results is None or len(results) == 0:
        return None

    last_rewards = results[-1]
    mean_reward = float(last_rewards.mean()) if len(last_rewards) else None
    mean_episode_length = None
    if ep_lengths is not None and len(ep_lengths) > 0:
        last_lengths = ep_lengths[-1]
        if len(last_lengths):
            mean_episode_length = float(last_lengths.mean())

    return EvalSummary(
        source=path.name,
        mean_reward=mean_reward,
        mean_episode_length=mean_episode_length,
    )


def _metadata_relative(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _read_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    tags = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                tags.append(stripped)
    return tuple(tags)
