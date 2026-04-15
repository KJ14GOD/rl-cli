from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.core.info import RunInfo, RunInfoError, load_run_info
from rlx.paths import METRICS_NAME


class AnalyzeError(Exception):
    """Raised when a run cannot be analyzed."""


@dataclass(frozen=True)
class RewardPoint:
    step: int | None
    reward: float
    episode_length: float | None


@dataclass(frozen=True)
class LearningSignal:
    points: tuple[RewardPoint, ...]
    first_reward: float | None
    final_reward: float | None
    best_reward: float | None
    best_step: int | None
    delta: float | None
    late_mean: float | None
    late_std: float | None
    best_to_final_drop: float | None
    trend: str


@dataclass(frozen=True)
class AnalysisFinding:
    category: str
    signal: str
    interpretation: str


@dataclass(frozen=True)
class AnalysisSuggestion:
    priority: str
    action: str
    reason: str


@dataclass(frozen=True)
class RunAnalysis:
    info: RunInfo
    learning: LearningSignal
    findings: tuple[AnalysisFinding, ...]
    suggestions: tuple[AnalysisSuggestion, ...]


def analyze_run(run_ref: str, cwd: Path | None = None) -> RunAnalysis:
    try:
        info = load_run_info(run_ref, cwd=cwd)
    except RunInfoError as exc:
        raise AnalyzeError(str(exc)) from exc

    points = _read_reward_points(info.run.run_dir / METRICS_NAME)
    learning = _build_learning_signal(points)
    findings = _build_findings(info, learning)
    suggestions = _build_suggestions(info, learning)

    return RunAnalysis(
        info=info,
        learning=learning,
        findings=tuple(findings),
        suggestions=tuple(suggestions),
    )


def _read_reward_points(path: Path) -> tuple[RewardPoint, ...]:
    if not path.exists():
        return ()

    points: list[RewardPoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        reward = _maybe_float(payload.get("rollout/ep_rew_mean"))
        if reward is None:
            continue

        points.append(
            RewardPoint(
                step=_maybe_int(payload.get("step")),
                reward=reward,
                episode_length=_maybe_float(payload.get("rollout/ep_len_mean")),
            )
        )

    return tuple(points)


def _build_learning_signal(points: tuple[RewardPoint, ...]) -> LearningSignal:
    if not points:
        return LearningSignal(
            points=(),
            first_reward=None,
            final_reward=None,
            best_reward=None,
            best_step=None,
            delta=None,
            late_mean=None,
            late_std=None,
            best_to_final_drop=None,
            trend="no reward data",
        )

    rewards = [point.reward for point in points]
    first_reward = rewards[0]
    final_reward = rewards[-1]
    best_index = max(range(len(points)), key=lambda index: points[index].reward)
    best_reward = rewards[best_index]
    delta = final_reward - first_reward
    best_to_final_drop = best_reward - final_reward

    late_rewards = rewards[-min(5, len(rewards)) :]
    late_mean = statistics.fmean(late_rewards)
    late_std = statistics.pstdev(late_rewards) if len(late_rewards) > 1 else 0.0

    return LearningSignal(
        points=points,
        first_reward=first_reward,
        final_reward=final_reward,
        best_reward=best_reward,
        best_step=points[best_index].step,
        delta=delta,
        late_mean=late_mean,
        late_std=late_std,
        best_to_final_drop=best_to_final_drop,
        trend=_classify_trend(
            first_reward=first_reward,
            final_reward=final_reward,
            best_reward=best_reward,
            late_mean=late_mean,
            late_std=late_std,
        ),
    )


def _classify_trend(
    *,
    first_reward: float,
    final_reward: float,
    best_reward: float,
    late_mean: float,
    late_std: float,
) -> str:
    delta = final_reward - first_reward
    best_drop = best_reward - final_reward
    relative_drop = best_drop / max(abs(best_reward), 1.0)
    relative_noise = late_std / max(abs(late_mean), 1.0)

    if relative_drop >= 0.25 and best_drop >= 10:
        return "regressed late"
    if delta >= 10 and final_reward >= best_reward * 0.9:
        return "improving"
    if relative_noise >= 0.25 and late_std >= 10:
        return "unstable late"
    if delta >= 10:
        return "improving"
    if abs(delta) < 5:
        return "flat"
    return "declining"


def _build_findings(info: RunInfo, learning: LearningSignal) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []

    if info.run.status == "completed":
        findings.append(
            AnalysisFinding(
                category="Run state",
                signal="completed",
                interpretation="Training finished and artifacts are safe to inspect.",
            )
        )
    elif info.run.status:
        findings.append(
            AnalysisFinding(
                category="Run state",
                signal=info.run.status,
                interpretation="The run did not finish cleanly; treat metrics as partial.",
            )
        )

    if learning.points:
        findings.append(
            AnalysisFinding(
                category="Learning",
                signal=learning.trend,
                interpretation=_learning_interpretation(learning),
            )
        )
        if learning.best_to_final_drop is not None and learning.best_to_final_drop >= 10:
            findings.append(
                AnalysisFinding(
                    category="Checkpoint choice",
                    signal=f"best-to-final drop {learning.best_to_final_drop:.2f}",
                    interpretation=(
                        "The best checkpoint may be more useful than latest for "
                        "eval, video, or resume."
                    ),
                )
            )
    else:
        findings.append(
            AnalysisFinding(
                category="Learning",
                signal="missing metrics",
                interpretation="No rollout reward records were found in metrics.jsonl.",
            )
        )

    if info.run.best_eval is not None and info.run.best_eval.mean_reward is not None:
        findings.append(
            AnalysisFinding(
                category="Evaluation",
                signal=f"{info.run.best_eval.mean_reward:.2f} from {info.run.best_eval.source}",
                interpretation="Best eval is available for model-quality comparison across runs.",
            )
        )
    else:
        findings.append(
                AnalysisFinding(
                    category="Evaluation",
                    signal="missing",
                    interpretation=(
                        "No evaluation artifact was found; run standalone eval "
                        "before trusting the checkpoint."
                    ),
                )
            )

    if info.sweep_name is not None:
        findings.append(
                AnalysisFinding(
                    category="Lineage",
                    signal=f"sweep {info.sweep_name}",
                    interpretation=(
                        "This run belongs to a sweep, so compare it against "
                        "sibling variants."
                    ),
                )
            )
    elif info.resumed_from_run is not None:
        findings.append(
                AnalysisFinding(
                    category="Lineage",
                    signal=f"resumed from {info.resumed_from_run}",
                    interpretation=(
                        "This run continued from an earlier checkpoint instead "
                        "of starting from scratch."
                    ),
                )
            )

    return findings


def _learning_interpretation(learning: LearningSignal) -> str:
    if learning.first_reward is None or learning.final_reward is None:
        return "No reward trend can be computed."

    delta = learning.delta or 0.0
    if learning.trend == "regressed late":
        return (
            f"Reward peaked at {learning.best_reward:.2f} but ended at "
            f"{learning.final_reward:.2f}."
        )
    if learning.trend == "unstable late":
        return f"Recent rewards are noisy; late-window std is {learning.late_std:.2f}."
    if learning.trend == "improving":
        return f"Reward improved by {delta:.2f} from first to final logged rollout."
    if learning.trend == "flat":
        return "Reward changed little across the run; this may be a plateau."
    return f"Reward ended {abs(delta):.2f} below the first logged rollout."


def _build_suggestions(info: RunInfo, learning: LearningSignal) -> list[AnalysisSuggestion]:
    suggestions: list[AnalysisSuggestion] = []
    run_ref = info.run.run_id

    if info.run.status in {"failed", "interrupted"}:
        suggestions.append(
            AnalysisSuggestion(
                priority="high",
                action=f"rlx resume {run_ref}",
                reason=(
                    "Continue from the latest checkpoint instead of discarding "
                    "partial progress."
                ),
            )
        )

    if info.run.best_eval is None:
        suggestions.append(
            AnalysisSuggestion(
                priority="high",
                action=f"rlx eval --run {run_ref}",
                reason=(
                    "A standalone eval creates a stable artifact for compare "
                    "and later analysis."
                ),
            )
        )

    if info.last_plot_manifest is None:
        suggestions.append(
            AnalysisSuggestion(
                priority="medium",
                action=f"rlx plot {run_ref}",
                reason="Plots make the reward curve and eval trend easier to inspect visually.",
            )
        )

    if info.last_video_manifest is None and info.run.best_checkpoint is not None:
        suggestions.append(
            AnalysisSuggestion(
                priority="medium",
                action=f"rlx video {info.run.run_dir / info.run.best_checkpoint}",
                reason="Video confirms what the best checkpoint is actually doing in the env.",
            )
        )

    if learning.best_to_final_drop is not None and learning.best_to_final_drop >= 10:
        suggestions.append(
            AnalysisSuggestion(
                priority="medium",
                action=f"rlx eval {info.run.run_dir / 'checkpoints' / 'best.zip'}",
                reason="Latest underperformed the run peak; validate the best checkpoint directly.",
            )
        )

    if _looks_undertrained(info, learning):
        suggestions.append(
            AnalysisSuggestion(
                priority="medium",
                action="rlx sweep configs/<sweep>.yaml",
                reason=(
                    "Performance is still low; vary seed, learning rate, "
                    "entropy, or train budget."
                ),
            )
        )

    if not suggestions:
        suggestions.append(
            AnalysisSuggestion(
                priority="low",
                action=f"rlx tag {run_ref} reviewed",
                reason="The run has enough artifacts and no obvious analysis gaps.",
            )
        )

    return suggestions


def _looks_undertrained(info: RunInfo, learning: LearningSignal) -> bool:
    best_observed = _max_optional(
        [
            learning.best_reward,
            info.run.best_eval.mean_reward
            if info.run.best_eval is not None
            else None,
        ]
    )
    if best_observed is None:
        return False
    if info.run.environment == "CartPole-v1":
        return best_observed < 400
    return learning.trend in {"flat", "declining"} and best_observed < 100


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None
