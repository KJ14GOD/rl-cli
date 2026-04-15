from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.core.info import RunInfo, RunInfoError, load_run_info
from rlx.paths import METRICS_NAME


class ExplainMetricsError(Exception):
    """Raised when run metrics cannot be explained."""


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class MetricSeries:
    key: str
    label: str
    description: str
    count: int
    first: float
    latest: float
    minimum: float
    maximum: float
    mean: float
    trend: str
    interpretation: str


@dataclass(frozen=True)
class MetricNote:
    severity: str
    metric: str
    note: str


@dataclass(frozen=True)
class MetricsExplanation:
    info: RunInfo
    series: tuple[MetricSeries, ...]
    notes: tuple[MetricNote, ...]
    missing: tuple[str, ...]
    total_records: int
    numeric_records: int


METRIC_DEFINITIONS = (
    MetricDefinition(
        "rollout/ep_rew_mean",
        "Reward mean",
        "Average episode reward from recent rollout episodes. This is the main progress signal.",
    ),
    MetricDefinition(
        "rollout/ep_len_mean",
        "Episode length",
        "Average episode length from recent rollout episodes. Meaning depends on the environment.",
    ),
    MetricDefinition(
        "train/approx_kl",
        "Approx KL",
        "Estimated policy shift after an update. High values can mean PPO updates are too large.",
    ),
    MetricDefinition(
        "train/clip_fraction",
        "Clip fraction",
        (
            "Fraction of policy updates clipped by PPO. High values mean many "
            "updates hit the trust region."
        ),
    ),
    MetricDefinition(
        "train/entropy_loss",
        "Entropy loss",
        (
            "Negative entropy term. Values closer to zero usually mean the "
            "policy is becoming less random."
        ),
    ),
    MetricDefinition(
        "train/explained_variance",
        "Explained variance",
        "How well the value function explains returns. Near 1 is strong; near or below 0 is weak.",
    ),
    MetricDefinition(
        "train/value_loss",
        "Value loss",
        "Critic/value-function loss. Persistent growth can mean value learning is struggling.",
    ),
    MetricDefinition(
        "train/policy_gradient_loss",
        "Policy gradient",
        "Actor optimization loss. It is noisy; spikes matter more than the raw sign.",
    ),
    MetricDefinition(
        "train/loss",
        "Total loss",
        "Combined PPO training loss. Useful as a stability clue, not as the main success metric.",
    ),
    MetricDefinition(
        "train/learning_rate",
        "Learning rate",
        "Optimizer learning rate used during training.",
    ),
    MetricDefinition(
        "train/n_updates",
        "Updates",
        "Number of PPO optimizer update rounds completed.",
    ),
    MetricDefinition(
        "progress_remaining",
        "Progress remaining",
        "Stable-Baselines training progress from 1.0 down toward 0.0.",
    ),
)

_DEFINITIONS_BY_KEY = {definition.key: definition for definition in METRIC_DEFINITIONS}


def explain_metrics(run_ref: str, cwd: Path | None = None) -> MetricsExplanation:
    try:
        info = load_run_info(run_ref, cwd=cwd)
    except RunInfoError as exc:
        raise ExplainMetricsError(str(exc)) from exc

    total_records, values = _read_metric_values(info.run.run_dir / METRICS_NAME)
    series = tuple(
        _build_metric_series(definition, values[definition.key])
        for definition in METRIC_DEFINITIONS
        if definition.key in values
    )
    missing = tuple(
        definition.key
        for definition in METRIC_DEFINITIONS
        if definition.key not in values
    )
    notes = _build_notes(series)

    return MetricsExplanation(
        info=info,
        series=series,
        notes=tuple(notes),
        missing=missing,
        total_records=total_records,
        numeric_records=sum(len(items) for items in values.values()),
    )


def _read_metric_values(path: Path) -> tuple[int, dict[str, list[float]]]:
    if not path.exists():
        return 0, {}

    total_records = 0
    values: dict[str, list[float]] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        total_records += 1
        if not isinstance(payload, dict):
            continue

        for key, definition in _DEFINITIONS_BY_KEY.items():
            value = _maybe_float(payload.get(key))
            if value is None:
                continue
            values.setdefault(definition.key, []).append(value)

    return total_records, values


def _build_metric_series(definition: MetricDefinition, values: list[float]) -> MetricSeries:
    first = values[0]
    latest = values[-1]
    trend = _trend(values)

    return MetricSeries(
        key=definition.key,
        label=definition.label,
        description=definition.description,
        count=len(values),
        first=first,
        latest=latest,
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        trend=trend,
        interpretation=_interpret_metric(definition.key, values, trend),
    )


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "single point"

    first = values[0]
    latest = values[-1]
    delta = latest - first
    scale = max(abs(first), abs(latest), 1.0)
    relative = abs(delta) / scale

    if relative < 0.05:
        return "flat"
    if delta > 0:
        return "rising"
    return "falling"


def _interpret_metric(key: str, values: list[float], trend: str) -> str:
    latest = values[-1]

    if key == "rollout/ep_rew_mean":
        if trend == "rising":
            return "Reward is improving across logged rollout summaries."
        if trend == "falling":
            return "Reward is ending below where it started; inspect best checkpoint and eval."
        return "Reward is not moving much; this may be a plateau or a short run."

    if key == "rollout/ep_len_mean":
        if trend == "rising":
            return "Episodes are lasting longer. For CartPole this usually tracks improvement."
        if trend == "falling":
            return "Episodes are shortening; check whether reward also dropped."
        return "Episode length is mostly stable."

    if key == "train/approx_kl":
        if latest > 0.05:
            return "Latest KL is high; policy updates may be too aggressive."
        if latest < 1e-5:
            return "Latest KL is tiny; updates may be very conservative."
        return "Latest KL is in a normal PPO range for many small experiments."

    if key == "train/clip_fraction":
        if latest > 0.3:
            return "Many policy updates are being clipped; consider smaller learning rate."
        if latest < 0.001:
            return "Almost no updates are clipped; PPO may be taking small policy steps."
        return "Clipping is active but not obviously excessive."

    if key == "train/entropy_loss":
        if trend == "rising":
            return "Entropy loss moved toward zero, so the policy likely became less random."
        if trend == "falling":
            return "Entropy loss became more negative, suggesting more stochastic exploration."
        return "Policy entropy is relatively steady."

    if key == "train/explained_variance":
        if latest < 0:
            return (
                "Value function fit is poor; critic predictions are worse "
                "than a simple baseline."
            )
        if latest < 0.3:
            return "Value function fit is weak; value loss and reward stability matter here."
        if latest > 0.8:
            return "Value function fit looks strong."
        return "Value function fit is moderate."

    if key == "train/value_loss":
        if trend == "rising":
            return "Value loss is rising; critic learning may be lagging policy learning."
        if trend == "falling":
            return "Value loss is falling, which usually means critic fit improved."
        return "Value loss is mostly stable."

    if key == "train/policy_gradient_loss":
        spread = max(values) - min(values)
        if spread > 0.05:
            return "Policy-gradient loss moved a lot; inspect KL and clip fraction for stability."
        return "Policy-gradient loss stayed in a narrow range."

    if key == "train/loss":
        if trend == "rising":
            return "Total loss rose; use reward, KL, clip fraction, and value loss to interpret it."
        if trend == "falling":
            return "Total loss fell, but reward/eval still matter more than this number alone."
        return "Total loss is mostly stable."

    if key == "train/learning_rate":
        return "Learning rate is tracked so schedules and sweeps can be audited."

    if key == "train/n_updates":
        return "Optimizer update count confirms PPO continued updating during the run."

    if key == "progress_remaining":
        return "Progress remaining should decline toward zero as training advances."

    return "Metric was logged, but RLCLI has no specialized explanation for it yet."


def _build_notes(series: tuple[MetricSeries, ...]) -> list[MetricNote]:
    by_key = {item.key: item for item in series}
    notes: list[MetricNote] = []

    reward = by_key.get("rollout/ep_rew_mean")
    if reward is None:
        notes.append(
            MetricNote(
                "high",
                "rollout/ep_rew_mean",
                "No reward metric was logged, so learning progress cannot be judged.",
            )
        )
    elif reward.trend == "rising":
        notes.append(
            MetricNote(
                "low",
                "rollout/ep_rew_mean",
                "Reward trend is positive.",
            )
        )
    elif reward.trend == "falling":
        notes.append(
            MetricNote(
                "high",
                "rollout/ep_rew_mean",
                "Reward ended lower than it started.",
            )
        )

    kl = by_key.get("train/approx_kl")
    if kl is not None and kl.latest > 0.05:
        notes.append(
            MetricNote(
                "medium",
                "train/approx_kl",
                "Latest KL is high enough to question update stability.",
            )
        )

    clip = by_key.get("train/clip_fraction")
    if clip is not None and clip.latest > 0.3:
        notes.append(
            MetricNote(
                "medium",
                "train/clip_fraction",
                "A large fraction of PPO updates are being clipped.",
            )
        )

    explained = by_key.get("train/explained_variance")
    if explained is not None and explained.latest < 0:
        notes.append(
            MetricNote(
                "medium",
                "train/explained_variance",
                "Critic fit is currently poor.",
            )
        )

    value_loss = by_key.get("train/value_loss")
    if value_loss is not None and value_loss.trend == "rising":
        notes.append(
            MetricNote(
                "medium",
                "train/value_loss",
                "Value loss rose over the run.",
            )
        )

    if not notes:
        notes.append(
            MetricNote(
                "low",
                "metrics",
                "No obvious metric-level warning was detected.",
            )
        )

    return notes


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
