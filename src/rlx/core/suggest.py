from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlx.core.diagnose import DiagnoseError, RunDiagnosis, diagnose_run


class SuggestError(Exception):
    """Raised when suggestions cannot be generated."""


@dataclass(frozen=True)
class SuggestedAction:
    priority: str
    kind: str
    suggestion: str
    reason: str


@dataclass(frozen=True)
class RunSuggestions:
    diagnosis: RunDiagnosis
    actions: tuple[SuggestedAction, ...]


def suggest_next_steps(run_ref: str, cwd: Path | None = None) -> RunSuggestions:
    try:
        diagnosis = diagnose_run(run_ref, cwd=cwd)
    except DiagnoseError as exc:
        raise SuggestError(str(exc)) from exc

    actions = _build_actions(diagnosis)
    return RunSuggestions(diagnosis=diagnosis, actions=tuple(actions))


def _build_actions(diagnosis: RunDiagnosis) -> list[SuggestedAction]:
    info = diagnosis.analysis.info
    learning = diagnosis.analysis.learning
    run_ref = info.run.run_id
    actions: list[SuggestedAction] = []

    if info.run.status in {"failed", "interrupted"}:
        actions.append(
            SuggestedAction(
                "high",
                "command",
                f"rlx resume {run_ref}",
                "Continue from the last checkpoint instead of restarting from zero.",
            )
        )

    if info.run.best_eval is None:
        actions.append(
            SuggestedAction(
                "high",
                "command",
                f"rlx eval --run {run_ref}",
                "Create a stable eval artifact before comparing or diagnosing further.",
            )
        )

    if learning.best_to_final_drop is not None and learning.best_to_final_drop >= 10:
        actions.append(
            SuggestedAction(
                "medium",
                "command",
                f"rlx eval {info.run.run_dir / 'checkpoints' / 'best.zip'}",
                "The latest checkpoint underperformed the best logged training point.",
            )
        )

    if info.last_plot_manifest is None:
        actions.append(
            SuggestedAction(
                "medium",
                "command",
                f"rlx plot {run_ref}",
                "Generate reward/eval plots so the learning curve is easier to inspect.",
            )
        )

    if info.last_video_manifest is None and info.run.best_checkpoint is not None:
        actions.append(
            SuggestedAction(
                "medium",
                "command",
                f"rlx video {info.run.run_dir / info.run.best_checkpoint}",
                "Render behavior from the best checkpoint instead of judging by reward only.",
            )
        )

    actions.extend(_experiment_actions(diagnosis))

    if not actions:
        actions.append(
            SuggestedAction(
                "low",
                "organization",
                f"rlx tag {run_ref} reviewed",
                "No urgent gaps were detected; mark this run before moving on.",
            )
        )

    return _dedupe_actions(actions)


def _experiment_actions(diagnosis: RunDiagnosis) -> list[SuggestedAction]:
    info = diagnosis.analysis.info
    learning = diagnosis.analysis.learning
    actions: list[SuggestedAction] = []
    issues = {item.issue for item in diagnosis.diagnostics}

    if "CartPole is not solved yet" in issues:
        actions.append(
            SuggestedAction(
                "medium",
                "sweep",
                "grid: seed=[42,123,999], algo.learning_rate=[0.0003,0.001]",
                "The run is learning but not solved; seed/LR variation is the next cheap test.",
            )
        )

    if learning.trend in {"flat", "declining"}:
        actions.append(
            SuggestedAction(
                "medium",
                "config",
                "increase algo.total_timesteps by 2x",
                "A flat or declining curve needs more budget only if eval still improves.",
            )
        )
        actions.append(
            SuggestedAction(
                "medium",
                "sweep",
                "grid: algo.entropy_coef=[0.0,0.01,0.02]",
                "Entropy changes can test whether exploration is limiting progress.",
            )
        )

    if _has_metric_issue(diagnosis, "train/approx_kl") or _has_metric_issue(
        diagnosis,
        "train/clip_fraction",
    ):
        actions.append(
            SuggestedAction(
                "high",
                "config",
                "lower algo.learning_rate",
                "KL or clipping warnings mean PPO updates may be too aggressive.",
            )
        )

    if _has_metric_issue(diagnosis, "train/explained_variance") or _has_metric_issue(
        diagnosis,
        "train/value_loss",
    ):
        actions.append(
            SuggestedAction(
                "medium",
                "config",
                "try larger algo.rollout_steps or lower algo.learning_rate",
                "Critic instability often improves with cleaner rollouts or gentler updates.",
            )
        )

    if info.sweep_name is not None:
        actions.append(
            SuggestedAction(
                "low",
                "command",
                f"rlx compare <siblings from sweep {info.sweep_name}>",
                "This run is one variant; the useful conclusion comes from sibling comparison.",
            )
        )

    return actions


def _has_metric_issue(diagnosis: RunDiagnosis, metric: str) -> bool:
    return any(item.evidence == metric for item in diagnosis.diagnostics)


def _dedupe_actions(actions: list[SuggestedAction]) -> list[SuggestedAction]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SuggestedAction] = []
    for action in actions:
        key = (action.kind, action.suggestion)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped
