from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rlx.config import ConfigError, ExperimentConfig, load_config
from rlx.core.diagnose import DiagnoseError, RunDiagnosis, diagnose_run
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.core.runs import RunPreparationError, prepare_run
from rlx.llm.env import get_env_value
from rlx.llm.planner import (
    LLM_ALLOWED_MUTATION_KEYS,
    LLM_DEFAULT_MODEL,
    LLM_DEFAULT_PROVIDER,
    LLMPlannerError,
    generate_llm_proposals,
)
from rlx.paths import CONFIG_SNAPSHOT_NAME
from rlx.rl import EvaluationError, TrainingError, evaluate_checkpoint, train_ppo


class AdvisorError(Exception):
    """Raised when RLCLI cannot build or execute an advisor plan."""


@dataclass(frozen=True)
class AdvisorProposal:
    mutations: dict[str, Any]
    signal: str
    rationale: str
    priority: str


@dataclass(frozen=True)
class AdvisorVariantResult:
    index: int
    config_path: Path
    mutations: dict[str, Any]
    signal: str
    rationale: str
    priority: str
    status: str
    run_id: str | None
    run_dir: Path | None
    score: float | None
    score_source: str | None
    error: str | None


@dataclass(frozen=True)
class AdvisorResult:
    project_root: Path
    bundle_dir: Path
    manifest_path: Path
    plan_path: Path
    baseline_run_id: str
    baseline_run_dir: Path
    baseline_score: float | None
    baseline_score_source: str | None
    mode: str
    variants: tuple[AdvisorVariantResult, ...]
    best_variant: AdvisorVariantResult | None
    context_actions: tuple[str, ...]
    diagnosis: RunDiagnosis


def run_advisor(
    run_ref: str,
    *,
    variants: int = 4,
    execute: bool = False,
    timesteps: int | None = None,
    locked_mutations: dict[str, Any] | None = None,
    allowed_mutation_keys: tuple[str, ...] | None = None,
    excluded_mutation_signatures: set[str] | None = None,
    signature_ignored_keys: tuple[str, ...] = (),
    require_eval_score: bool = False,
    planner: str = "rules",
    llm_provider: str | None = None,
    llm_model: str | None = None,
    cwd: Path | None = None,
) -> AdvisorResult:
    if variants < 1:
        raise AdvisorError("Advisor must create at least one variant.")
    if timesteps is not None and timesteps < 1:
        raise AdvisorError("Advisor timesteps override must be positive.")
    fixed_mutations = dict(locked_mutations or {})
    if timesteps is not None:
        fixed_mutations["algo.total_timesteps"] = timesteps
    planner_name = planner.lower()
    if planner_name not in {"rules", "llm"}:
        raise AdvisorError("Advisor planner must be either 'rules' or 'llm'.")

    try:
        diagnosis = diagnose_run(run_ref, cwd=cwd)
    except DiagnoseError as exc:
        raise AdvisorError(str(exc)) from exc

    info = diagnosis.analysis.info
    baseline_run = info.run
    config_path = baseline_run.run_dir / CONFIG_SNAPSHOT_NAME
    base_payload = _load_yaml_mapping(config_path, "baseline config snapshot")

    try:
        project_root = find_project_root(baseline_run.run_dir)
    except ProjectLookupError as exc:
        raise AdvisorError(str(exc)) from exc

    resolved_llm_provider = _resolve_llm_setting(
        "RLX_LLM_PROVIDER",
        llm_provider,
        default=LLM_DEFAULT_PROVIDER,
        project_root=project_root,
    )
    resolved_llm_model = _resolve_llm_setting(
        "RLX_LLM_MODEL",
        llm_model,
        default=LLM_DEFAULT_MODEL,
        project_root=project_root,
    )

    bundle_dir = _next_bundle_dir(
        project_root / "analysis" / "advisor",
        f"{baseline_run.run_id}_advisor",
    )
    configs_dir = bundle_dir / "configs"
    bundle_dir.mkdir(parents=True, exist_ok=False)
    configs_dir.mkdir()

    resolved_allowed_keys = allowed_mutation_keys
    if planner_name == "llm" and resolved_allowed_keys is None:
        resolved_allowed_keys = LLM_ALLOWED_MUTATION_KEYS

    try:
        source_proposals = _planner_proposals(
            planner=planner_name,
            base_payload=base_payload,
            diagnosis=diagnosis,
            project_root=project_root,
            variants=variants,
            locked_mutations=fixed_mutations,
            allowed_mutation_keys=resolved_allowed_keys,
            excluded_mutation_signatures=excluded_mutation_signatures or set(),
            signature_ignored_keys=signature_ignored_keys,
            llm_provider=resolved_llm_provider,
            llm_model=resolved_llm_model,
        )
    except LLMPlannerError as exc:
        raise AdvisorError(str(exc)) from exc

    proposals = _select_proposals(
        source_proposals,
        limit=variants,
        base_payload=base_payload,
        locked_mutations=fixed_mutations,
        allowed_mutation_keys=resolved_allowed_keys,
        excluded_mutation_signatures=excluded_mutation_signatures or set(),
        signature_ignored_keys=signature_ignored_keys,
    )
    if not proposals:
        raise AdvisorError("Advisor could not create any valid variant proposals.")

    context_actions = _context_actions(diagnosis)
    baseline_score, baseline_score_source = _score_run(
        baseline_run,
        require_eval_score=require_eval_score,
    )

    variant_results: list[AdvisorVariantResult] = []
    for index, proposal in enumerate(proposals, start=1):
        variant_payload = copy.deepcopy(base_payload)
        _apply_overrides(variant_payload, proposal.mutations)
        variant_config_path = configs_dir / f"variant_{index:03d}.yaml"
        variant_config_path.write_text(
            yaml.safe_dump(variant_payload, sort_keys=False),
            encoding="utf-8",
        )

        result = AdvisorVariantResult(
            index=index,
            config_path=variant_config_path,
            mutations=proposal.mutations,
            signal=proposal.signal,
            rationale=proposal.rationale,
            priority=proposal.priority,
            status="proposed",
            run_id=None,
            run_dir=None,
            score=None,
            score_source=None,
            error=None,
        )

        if execute:
            result = _execute_variant(
                result,
                project_root=project_root,
                baseline_run_id=baseline_run.run_id,
                bundle_dir=bundle_dir,
                require_eval_score=require_eval_score,
            )
        variant_results.append(result)

    best_variant = _best_variant(variant_results)
    mode = "executed" if execute else "dry_run"
    manifest_path = bundle_dir / "manifest.json"
    plan_path = bundle_dir / "plan.md"
    _write_manifest(
        manifest_path=manifest_path,
        project_root=project_root,
        bundle_dir=bundle_dir,
        mode=mode,
        baseline_run_id=baseline_run.run_id,
        baseline_run_dir=baseline_run.run_dir,
        baseline_score=baseline_score,
        baseline_score_source=baseline_score_source,
        diagnosis=diagnosis,
        context_actions=context_actions,
        variants=variant_results,
        best_variant=best_variant,
        fixed_mutations=_effective_locked_mutations(base_payload, fixed_mutations),
        allowed_mutation_keys=resolved_allowed_keys,
        excluded_mutation_signatures=excluded_mutation_signatures or set(),
        require_eval_score=require_eval_score,
        planner=planner_name,
        llm_provider=resolved_llm_provider if planner_name == "llm" else None,
        llm_model=resolved_llm_model if planner_name == "llm" else None,
    )
    _write_plan(
        plan_path=plan_path,
        mode=mode,
        planner=planner_name,
        baseline_run_id=baseline_run.run_id,
        diagnosis=diagnosis,
        context_actions=context_actions,
        variants=variant_results,
        best_variant=best_variant,
    )

    return AdvisorResult(
        project_root=project_root,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        plan_path=plan_path,
        baseline_run_id=baseline_run.run_id,
        baseline_run_dir=baseline_run.run_dir,
        baseline_score=baseline_score,
        baseline_score_source=baseline_score_source,
        mode=mode,
        variants=tuple(variant_results),
        best_variant=best_variant,
        context_actions=tuple(context_actions),
        diagnosis=diagnosis,
    )


def _resolve_llm_setting(
    env_name: str,
    explicit: str | None,
    *,
    default: str,
    project_root: Path,
) -> str:
    if explicit:
        return explicit
    return get_env_value(env_name, project_root=project_root, default=default) or default


def _planner_proposals(
    *,
    planner: str,
    base_payload: dict[str, Any],
    diagnosis: RunDiagnosis,
    project_root: Path,
    variants: int,
    locked_mutations: dict[str, Any],
    allowed_mutation_keys: tuple[str, ...] | None,
    excluded_mutation_signatures: set[str],
    signature_ignored_keys: tuple[str, ...],
    llm_provider: str,
    llm_model: str,
) -> list[AdvisorProposal]:
    if planner == "rules":
        return _build_proposals(base_payload, diagnosis)

    proposals = generate_llm_proposals(
        provider=llm_provider,
        model=llm_model,
        project_root=project_root,
        base_payload=base_payload,
        diagnosis=diagnosis,
        allowed_mutation_keys=allowed_mutation_keys or LLM_ALLOWED_MUTATION_KEYS,
        locked_mutations=locked_mutations,
        excluded_mutation_signatures=excluded_mutation_signatures,
        signature_ignored_keys=signature_ignored_keys,
        variants=variants,
    )
    return [
        AdvisorProposal(
            mutations=item.mutations,
            signal=item.signal,
            rationale=item.rationale,
            priority=item.priority,
        )
        for item in proposals
    ]


def _build_proposals(
    payload: dict[str, Any],
    diagnosis: RunDiagnosis,
) -> list[AdvisorProposal]:
    config = _ConfigView(payload)
    learning = diagnosis.analysis.learning
    metrics = {item.key: item for item in diagnosis.metrics.series}
    proposals: list[AdvisorProposal] = []

    approx_kl = metrics.get("train/approx_kl")
    clip_fraction = metrics.get("train/clip_fraction")
    explained = metrics.get("train/explained_variance")
    value_loss = metrics.get("train/value_loss")
    entropy = metrics.get("train/entropy_loss")

    conservative_updates = (
        (approx_kl is not None and approx_kl.latest < 1e-4)
        or (clip_fraction is not None and clip_fraction.latest < 0.001)
    )
    aggressive_updates = (
        (approx_kl is not None and approx_kl.latest > 0.03)
        or (clip_fraction is not None and clip_fraction.latest > 0.2)
    )
    weak_value_function = (
        (explained is not None and explained.latest < 0.3)
        or (value_loss is not None and value_loss.trend == "rising")
    )

    if aggressive_updates:
        proposals.append(
            AdvisorProposal(
                mutations={"algo.learning_rate": _scale_float(config.learning_rate, 0.5)},
                signal=_metric_signal(approx_kl, clip_fraction),
                rationale=(
                    "PPO updates look too large, so this halves the learning rate relative "
                    "to the baseline."
                ),
                priority="high",
            )
        )
        proposals.append(
            AdvisorProposal(
                mutations={
                    "algo.learning_rate": _scale_float(config.learning_rate, 0.75),
                    "algo.clip_range": _scale_float(config.clip_range, 0.75, floor=0.05),
                },
                signal="large policy update guardrail",
                rationale=(
                    "Pair a gentler learning rate with a tighter PPO clip range to test "
                    "whether instability is update-size related."
                ),
                priority="medium",
            )
        )

    if conservative_updates and learning.trend in {"flat", "improving", "no reward data"}:
        proposals.append(
            AdvisorProposal(
                mutations={"algo.learning_rate": _scale_float(config.learning_rate, 2.0)},
                signal=_metric_signal(approx_kl, clip_fraction),
                rationale=(
                    "Policy movement is very small, so this doubles the learning rate "
                    "relative to the baseline."
                ),
                priority="high" if learning.trend == "flat" else "medium",
            )
        )

    if learning.trend in {"flat", "declining", "regressed late", "unstable late"}:
        proposals.append(
            AdvisorProposal(
                mutations={"algo.entropy_coef": _increase_entropy(config.entropy_coef)},
                signal=f"reward trend is {learning.trend}",
                rationale=(
                    "Reward is not cleanly improving, so this adds exploration pressure "
                    "through entropy regularization."
                ),
                priority="medium",
            )
        )
        proposals.append(
            AdvisorProposal(
                mutations={"seed": _next_seed(config.seed, 81)},
                signal=f"reward trend is {learning.trend}",
                rationale=(
                    "A seed variant checks whether the current result is noise before "
                    "changing model or optimizer settings."
                ),
                priority="medium",
            )
        )

    if weak_value_function:
        rollout = _scale_int(config.rollout_steps, 2, minimum=16)
        proposals.append(
            AdvisorProposal(
                mutations=_rollout_mutation(config, rollout),
                signal=_metric_signal(explained, value_loss),
                rationale=(
                    "The critic/value signal looks weak, so this collects longer rollouts "
                    "while keeping batch size valid."
                ),
                priority="high" if learning.trend != "improving" else "medium",
            )
        )
        proposals.append(
            AdvisorProposal(
                mutations={
                    "algo.learning_rate": _scale_float(config.learning_rate, 0.75),
                    "algo.value_coef": _scale_float(config.value_coef, 1.5),
                },
                signal=_metric_signal(explained, value_loss),
                rationale=(
                    "This gently lowers actor step size and gives the value loss more "
                    "weight to test critic-limited learning."
                ),
                priority="medium",
            )
        )

    if learning.best_to_final_drop is not None and learning.best_to_final_drop >= 10:
        shorter_budget = _shorter_budget(config.total_timesteps, learning.best_step)
        if shorter_budget is not None:
            proposals.append(
                AdvisorProposal(
                    mutations={"algo.total_timesteps": shorter_budget},
                    signal=f"best-to-final reward drop {learning.best_to_final_drop:.2f}",
                    rationale=(
                        "The run peaked before the end, so this tests a shorter training "
                        "budget near the best observed step."
                    ),
                    priority="medium",
                )
            )

    if diagnosis.analysis.info.run.best_eval is None:
        proposals.append(
            AdvisorProposal(
                mutations={"eval.episodes": max(config.eval_episodes, 20)},
                signal="missing eval artifact",
                rationale=(
                    "The baseline lacks a stable eval summary, so this increases eval "
                    "episodes for clearer advisor comparisons."
                ),
                priority="high",
            )
        )

    if _is_cartpole_unsolved(diagnosis):
        proposals.append(
            AdvisorProposal(
                mutations={
                    "algo.total_timesteps": _scale_int(config.total_timesteps, 2),
                    "seed": _next_seed(config.seed, 81),
                },
                signal="CartPole best observed reward below solved range",
                rationale=(
                    "CartPole is still below the solved range, so this combines a longer "
                    "budget with a seed check."
                ),
                priority="medium",
            )
        )

    if entropy is not None and entropy.trend == "rising" and learning.trend != "declining":
        proposals.append(
            AdvisorProposal(
                mutations={"algo.entropy_coef": _increase_entropy(config.entropy_coef)},
                signal="entropy moved toward deterministic policy",
                rationale=(
                    "Policy entropy decreased during training, so this tests whether more "
                    "exploration improves final reward."
                ),
                priority="low",
            )
        )

    for offset in (37, 81, 123, 211, 307, 503, 997):
        proposals.append(
            AdvisorProposal(
                mutations={"seed": _next_seed(config.seed, offset)},
                signal="baseline reproducibility check",
                rationale=(
                    "A seed variant checks whether the observed outcome is repeatable "
                    "before adding more complex changes."
                ),
                priority="low",
            )
        )

    return proposals


def _select_proposals(
    proposals: list[AdvisorProposal],
    *,
    limit: int,
    base_payload: dict[str, Any],
    locked_mutations: dict[str, Any],
    allowed_mutation_keys: tuple[str, ...] | None,
    excluded_mutation_signatures: set[str],
    signature_ignored_keys: tuple[str, ...],
) -> list[AdvisorProposal]:
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    selected: list[AdvisorProposal] = []
    seen: set[str] = set()
    allowed = set(allowed_mutation_keys) if allowed_mutation_keys is not None else None
    locked_effective = _effective_locked_mutations(base_payload, locked_mutations)

    for proposal in sorted(
        proposals,
        key=lambda item: (priority_rank.get(item.priority, 3), len(item.mutations)),
    ):
        if allowed is not None and any(key not in allowed for key in proposal.mutations):
            continue

        mutations = {**proposal.mutations, **locked_effective}
        signature = mutation_signature(mutations, ignored_keys=signature_ignored_keys)
        if not signature:
            continue
        if signature in seen or signature in excluded_mutation_signatures:
            continue
        if not _is_valid_variant_mutation(base_payload, mutations):
            continue
        seen.add(signature)
        selected.append(
            AdvisorProposal(
                mutations=mutations,
                signal=proposal.signal,
                rationale=proposal.rationale,
                priority=proposal.priority,
            )
        )
        if len(selected) >= limit:
            break

    return selected


def _is_valid_variant_mutation(
    base_payload: dict[str, Any],
    mutations: dict[str, Any],
) -> bool:
    variant_payload = copy.deepcopy(base_payload)
    _apply_overrides(variant_payload, mutations)
    try:
        config = ExperimentConfig.model_validate(variant_payload)
    except ValidationError:
        return False
    return config.algo.batch_size <= config.algo.rollout_steps * config.env.num_envs


def _execute_variant(
    variant: AdvisorVariantResult,
    *,
    project_root: Path,
    baseline_run_id: str,
    bundle_dir: Path,
    require_eval_score: bool,
) -> AdvisorVariantResult:
    try:
        config = load_config(variant.config_path)
        run = prepare_run(config, variant.config_path)
        train_result = train_ppo(
            config,
            run,
            lineage_metadata={
                "advisor_parent_run": baseline_run_id,
                "advisor_variant_index": variant.index,
                "advisor_bundle": str(_relative_to(project_root, bundle_dir)),
                "advisor_mutations": variant.mutations,
                "tags": ["advisor", f"advisor:{baseline_run_id}"],
            },
        )
        if require_eval_score:
            eval_result = evaluate_checkpoint(train_result.latest_checkpoint)
            score = eval_result.mean_reward
            source = f"standalone eval ({eval_result.result_path.name})"
        else:
            comparison = diagnose_run(run.run_dir.name, cwd=project_root).analysis.info.run
            score, source = _score_run(comparison, require_eval_score=require_eval_score)
        return AdvisorVariantResult(
            index=variant.index,
            config_path=variant.config_path,
            mutations=variant.mutations,
            signal=variant.signal,
            rationale=variant.rationale,
            priority=variant.priority,
            status="completed",
            run_id=run.run_dir.name,
            run_dir=run.run_dir,
            score=score,
            score_source=source,
            error=None,
        )
    except (
        ConfigError,
        RunPreparationError,
        TrainingError,
        EvaluationError,
        AdvisorError,
        DiagnoseError,
    ) as exc:
        return AdvisorVariantResult(
            index=variant.index,
            config_path=variant.config_path,
            mutations=variant.mutations,
            signal=variant.signal,
            rationale=variant.rationale,
            priority=variant.priority,
            status="failed",
            run_id=None,
            run_dir=None,
            score=None,
            score_source=None,
            error=str(exc),
        )


def _write_manifest(
    *,
    manifest_path: Path,
    project_root: Path,
    bundle_dir: Path,
    mode: str,
    baseline_run_id: str,
    baseline_run_dir: Path,
    baseline_score: float | None,
    baseline_score_source: str | None,
    diagnosis: RunDiagnosis,
    context_actions: list[str],
    variants: list[AdvisorVariantResult],
    best_variant: AdvisorVariantResult | None,
    fixed_mutations: dict[str, Any],
    allowed_mutation_keys: tuple[str, ...] | None,
    excluded_mutation_signatures: set[str],
    require_eval_score: bool,
    planner: str,
    llm_provider: str | None,
    llm_model: str | None,
) -> None:
    payload = {
        "kind": "advisor_bundle",
        "created_at": _utc_now_iso(),
        "mode": mode,
        "bundle": str(_relative_to(project_root, bundle_dir)),
        "baseline": {
            "run_id": baseline_run_id,
            "run_dir": str(_relative_to(project_root, baseline_run_dir)),
            "score": baseline_score,
            "score_source": baseline_score_source,
            "health": diagnosis.health,
            "trend": diagnosis.analysis.learning.trend,
        },
        "context_actions": context_actions,
        "protocol": {
            "planner": planner,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "fixed_mutations": fixed_mutations,
            "allowed_mutation_keys": list(allowed_mutation_keys)
            if allowed_mutation_keys is not None
            else None,
            "excluded_mutation_signatures": sorted(excluded_mutation_signatures),
            "require_eval_score": require_eval_score,
        },
        "diagnostics": [
            {
                "severity": item.severity,
                "area": item.area,
                "issue": item.issue,
                "evidence": item.evidence,
                "recommendation": item.recommendation,
            }
            for item in diagnosis.diagnostics
        ],
        "metric_notes": [
            {
                "severity": item.severity,
                "metric": item.metric,
                "note": item.note,
            }
            for item in diagnosis.metrics.notes
        ],
        "variants": [
            {
                "index": variant.index,
                "config": str(_relative_to(project_root, variant.config_path)),
                "mutations": variant.mutations,
                "signal": variant.signal,
                "rationale": variant.rationale,
                "priority": variant.priority,
                "status": variant.status,
                "run_id": variant.run_id,
                "run_dir": (
                    str(_relative_to(project_root, variant.run_dir))
                    if variant.run_dir is not None
                    else None
                ),
                "score": variant.score,
                "score_source": variant.score_source,
                "error": variant.error,
            }
            for variant in variants
        ],
        "best_variant": best_variant.index if best_variant is not None else None,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_plan(
    *,
    plan_path: Path,
    mode: str,
    planner: str,
    baseline_run_id: str,
    diagnosis: RunDiagnosis,
    context_actions: list[str],
    variants: list[AdvisorVariantResult],
    best_variant: AdvisorVariantResult | None,
) -> None:
    lines = [
        f"# RLCLI Advisor Plan: {baseline_run_id}",
        "",
        f"- Mode: {mode}",
        f"- Planner: {planner}",
        f"- Health: {diagnosis.health}",
        f"- Learning trend: {diagnosis.analysis.learning.trend}",
        f"- Best variant: {best_variant.index:03d}" if best_variant else "- Best variant: n/a",
        "",
        "## Context Actions",
    ]
    if context_actions:
        lines.extend(f"- `{action}`" for action in context_actions)
    else:
        lines.append("- No missing context actions detected.")

    lines.extend(["", "## Proposed Variants"])
    for variant in variants:
        mutation_text = "; ".join(
            f"{key}={value}" for key, value in variant.mutations.items()
        )
        lines.extend(
            [
                f"- Variant {variant.index:03d}: {mutation_text}",
                f"  Signal: {variant.signal}",
                f"  Rationale: {variant.rationale}",
                f"  Status: {variant.status}",
            ]
        )
        if variant.run_id:
            lines.append(f"  Run: {variant.run_id}")
        if variant.score is not None:
            lines.append(f"  Score: {variant.score:.4g} ({variant.score_source})")
        if variant.error:
            lines.append(f"  Error: {variant.error}")

    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _context_actions(diagnosis: RunDiagnosis) -> list[str]:
    info = diagnosis.analysis.info
    actions: list[str] = []
    if info.run.best_eval is None:
        actions.append(f"rlx eval --run {info.run.run_id}")
    if info.last_plot_manifest is None:
        actions.append(f"rlx plot {info.run.run_id}")
    if info.last_video_manifest is None and info.run.best_checkpoint is not None:
        actions.append(f"rlx video {info.run.run_dir / info.run.best_checkpoint}")
    return actions


def _score_run(run: Any, *, require_eval_score: bool) -> tuple[float | None, str | None]:
    if run.best_eval is not None and run.best_eval.mean_reward is not None:
        return run.best_eval.mean_reward, f"best eval ({run.best_eval.source})"
    if run.latest_eval is not None and run.latest_eval.mean_reward is not None:
        return run.latest_eval.mean_reward, f"latest eval ({run.latest_eval.source})"
    if require_eval_score:
        return None, None
    if run.best_rollout_reward is not None:
        return run.best_rollout_reward, "best rollout reward"
    if run.final_rollout_reward is not None:
        return run.final_rollout_reward, "final rollout reward"
    return None, None


def _best_variant(variants: list[AdvisorVariantResult]) -> AdvisorVariantResult | None:
    scored = [variant for variant in variants if variant.score is not None]
    if not scored:
        return None
    return max(scored, key=lambda variant: variant.score or float("-inf"))


class _ConfigView:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    @property
    def seed(self) -> int:
        return int(self._get("seed", 0))

    @property
    def num_envs(self) -> int:
        return int(self._get("env.num_envs", 1))

    @property
    def total_timesteps(self) -> int:
        return int(self._get("algo.total_timesteps", 1))

    @property
    def rollout_steps(self) -> int:
        return int(self._get("algo.rollout_steps", 1))

    @property
    def batch_size(self) -> int:
        return int(self._get("algo.batch_size", 1))

    @property
    def learning_rate(self) -> float:
        return float(self._get("algo.learning_rate", 0.0003))

    @property
    def clip_range(self) -> float:
        return float(self._get("algo.clip_range", 0.2))

    @property
    def entropy_coef(self) -> float:
        return float(self._get("algo.entropy_coef", 0.0))

    @property
    def value_coef(self) -> float:
        return float(self._get("algo.value_coef", 0.5))

    @property
    def eval_episodes(self) -> int:
        return int(self._get("eval.episodes", 1))

    def _get(self, dotted_key: str, default: Any) -> Any:
        current: Any = self._payload
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def mutation_signature(
    mutations: dict[str, Any],
    *,
    ignored_keys: tuple[str, ...] = (),
) -> str:
    effective = {
        key: value for key, value in mutations.items() if key not in set(ignored_keys)
    }
    if not effective:
        return ""
    return json.dumps(effective, sort_keys=True)


def _effective_locked_mutations(
    base_payload: dict[str, Any],
    locked_mutations: dict[str, Any],
) -> dict[str, Any]:
    effective: dict[str, Any] = {}
    for key, value in locked_mutations.items():
        if _get_dotted_value(base_payload, key) == value:
            continue
        effective[key] = value
    return effective


def _get_dotted_value(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _rollout_mutation(config: _ConfigView, rollout_steps: int) -> dict[str, Any]:
    buffer_size = max(rollout_steps * config.num_envs, 2)
    batch_size = min(config.batch_size, buffer_size)
    return {
        "algo.rollout_steps": rollout_steps,
        "algo.batch_size": max(batch_size, 2),
    }


def _shorter_budget(total_timesteps: int, best_step: int | None) -> int | None:
    if best_step is None or best_step <= 0:
        return None
    candidate = max(best_step, total_timesteps // 2)
    if candidate >= total_timesteps:
        return None
    return candidate


def _is_cartpole_unsolved(diagnosis: RunDiagnosis) -> bool:
    info = diagnosis.analysis.info
    if info.run.environment != "CartPole-v1":
        return False
    candidates = [diagnosis.analysis.learning.best_reward]
    if info.run.best_eval is not None:
        candidates.append(info.run.best_eval.mean_reward)
    present = [value for value in candidates if value is not None]
    return bool(present) and max(present) < 475


def _metric_signal(*series: Any) -> str:
    parts = []
    for item in series:
        if item is None:
            continue
        parts.append(f"{item.key} latest={item.latest:.4g}, trend={item.trend}")
    return "; ".join(parts) or "diagnostic signal"


def _increase_entropy(value: float) -> float:
    if value <= 0:
        return 0.01
    return _scale_float(value, 2.0, ceiling=0.1)


def _next_seed(seed: int, offset: int) -> int:
    return max(seed + offset, 0)


def _scale_int(value: int, factor: int, *, minimum: int = 1) -> int:
    return max(int(value * factor), minimum)


def _scale_float(
    value: float,
    factor: float,
    *,
    floor: float | None = None,
    ceiling: float | None = None,
) -> float:
    scaled = value * factor
    if floor is not None:
        scaled = max(scaled, floor)
    if ceiling is not None:
        scaled = min(scaled, ceiling)
    return float(f"{scaled:.12g}")


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AdvisorError(f"Could not read {label}: {path}") from exc
    except yaml.YAMLError as exc:
        raise AdvisorError(f"Failed to parse {label} {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise AdvisorError(f"{label.capitalize()} must be a top-level mapping: {path}")
    return raw


def _apply_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        _set_dotted_value(payload, dotted_key, value)


def _set_dotted_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _next_bundle_dir(root: Path, name: str) -> Path:
    slug = _slugify(name) or "advisor"
    pattern = re.compile(rf"^{re.escape(slug)}_(\d{{3}})$")
    existing = []

    if root.exists():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                existing.append(int(match.group(1)))

    next_index = max(existing, default=0) + 1
    return root / f"{slug}_{next_index:03d}"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
