from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlx.core.analyze import AnalyzeError, RunAnalysis, analyze_run
from rlx.core.explain_metrics import ExplainMetricsError, MetricsExplanation, explain_metrics


class DiagnoseError(Exception):
    """Raised when a run cannot be diagnosed."""


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    area: str
    issue: str
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class RunDiagnosis:
    analysis: RunAnalysis
    metrics: MetricsExplanation
    health: str
    diagnostics: tuple[Diagnostic, ...]


def diagnose_run(run_ref: str, cwd: Path | None = None) -> RunDiagnosis:
    try:
        analysis = analyze_run(run_ref, cwd=cwd)
        metrics = explain_metrics(run_ref, cwd=cwd)
    except (AnalyzeError, ExplainMetricsError) as exc:
        raise DiagnoseError(str(exc)) from exc

    diagnostics = _build_diagnostics(analysis, metrics)
    return RunDiagnosis(
        analysis=analysis,
        metrics=metrics,
        health=_classify_health(diagnostics),
        diagnostics=tuple(diagnostics),
    )


def _build_diagnostics(
    analysis: RunAnalysis,
    metrics: MetricsExplanation,
) -> list[Diagnostic]:
    info = analysis.info
    learning = analysis.learning
    diagnostics: list[Diagnostic] = []

    if info.run.status in {"failed", "interrupted"}:
        diagnostics.append(
            Diagnostic(
                severity="high",
                area="Run state",
                issue=f"Run status is {info.run.status}",
                evidence=info.error or "Training did not complete cleanly.",
                recommendation=(
                    f"Resume from latest checkpoint with `rlx resume {info.run.run_id}`."
                ),
            )
        )

    if not learning.points:
        diagnostics.append(
            Diagnostic(
                severity="high",
                area="Metrics",
                issue="No reward curve was logged",
                evidence="metrics.jsonl has no rollout/ep_rew_mean records.",
                recommendation=(
                    "Rerun training or inspect metrics logging before comparing this run."
                ),
            )
        )
    elif learning.trend == "declining":
        diagnostics.append(
            Diagnostic(
                severity="high",
                area="Learning",
                issue="Reward declined over training",
                evidence=_reward_evidence(analysis),
                recommendation="Evaluate best.zip and lower learning rate or try a different seed.",
            )
        )
    elif learning.trend == "regressed late":
        diagnostics.append(
            Diagnostic(
                severity="medium",
                area="Learning",
                issue="Reward regressed after a better checkpoint",
                evidence=_reward_evidence(analysis),
                recommendation=(
                    "Prefer best.zip over latest.zip and consider shorter training or LR decay."
                ),
            )
        )
    elif learning.trend == "unstable late":
        diagnostics.append(
            Diagnostic(
                severity="medium",
                area="Stability",
                issue="Recent reward is noisy",
                evidence=f"Late-window std is {_fmt_number(learning.late_std)}.",
                recommendation="Run more eval episodes and sweep seeds before trusting one run.",
            )
        )
    elif learning.trend == "flat":
        diagnostics.append(
            Diagnostic(
                severity="medium",
                area="Learning",
                issue="Reward plateaued",
                evidence=_reward_evidence(analysis),
                recommendation="Try more timesteps, learning-rate sweep, or entropy-coef sweep.",
            )
        )

    best_observed = _best_observed_reward(analysis)
    if info.run.environment == "CartPole-v1" and best_observed is not None and best_observed < 400:
        diagnostics.append(
            Diagnostic(
                severity="medium",
                area="Performance",
                issue="CartPole is not solved yet",
                evidence=f"Best observed reward is {best_observed:.2f}; CartPole max is 500.",
                recommendation=(
                    "Increase training budget or sweep seed, learning rate, and entropy."
                ),
            )
        )

    if info.run.best_eval is None:
        diagnostics.append(
            Diagnostic(
                severity="high",
                area="Evaluation",
                issue="No evaluation artifact found",
                evidence="No manual eval JSON or training-time evaluations.npz was detected.",
                recommendation=f"Run `rlx eval --run {info.run.run_id}`.",
            )
        )

    for note in metrics.notes:
        if note.severity not in {"high", "medium"}:
            continue
        diagnostics.append(
            Diagnostic(
                severity=note.severity,
                area="PPO metric",
                issue=note.note,
                evidence=note.metric,
                recommendation=_metric_recommendation(note.metric),
            )
        )

    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                severity="low",
                area="Overall",
                issue="No obvious problem detected",
                evidence="Reward, eval, and PPO metric checks did not trigger warnings.",
                recommendation="Compare this run against alternatives before treating it as final.",
            )
        )

    return diagnostics


def _classify_health(diagnostics: list[Diagnostic]) -> str:
    severities = {item.severity for item in diagnostics}
    if "high" in severities:
        return "needs attention"
    if "medium" in severities:
        return "watch"
    return "clean"


def _reward_evidence(analysis: RunAnalysis) -> str:
    learning = analysis.learning
    return (
        f"first={_fmt_number(learning.first_reward)}, "
        f"final={_fmt_number(learning.final_reward)}, "
        f"best={_fmt_number(learning.best_reward)}"
    )


def _best_observed_reward(analysis: RunAnalysis) -> float | None:
    candidates = [analysis.learning.best_reward]
    if analysis.info.run.best_eval is not None:
        candidates.append(analysis.info.run.best_eval.mean_reward)
    present = [value for value in candidates if value is not None]
    if not present:
        return None
    return max(present)


def _metric_recommendation(metric: str) -> str:
    if metric == "train/approx_kl":
        return "Lower learning rate or reduce update aggressiveness."
    if metric == "train/clip_fraction":
        return "Lower learning rate or inspect clip_range."
    if metric == "train/explained_variance":
        return "Inspect value loss; try more rollout steps or a smaller learning rate."
    if metric == "train/value_loss":
        return "Try lower learning rate, more timesteps, or a value-coef sweep."
    if metric == "rollout/ep_rew_mean":
        return "Evaluate best checkpoint and compare against seed variants."
    return "Inspect this metric alongside reward and eval performance."


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}"
