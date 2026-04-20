from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlx.config import ConfigError, load_config
from rlx.core.advisor import (
    AdvisorError,
    AdvisorExhaustedError,
    mutation_signature,
    run_advisor,
)
from rlx.core.diagnose import DiagnoseError, diagnose_run
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.llm.env import get_env_value
from rlx.llm.planner import LLM_ALLOWED_MUTATION_KEYS, LLM_DEFAULT_MODEL, LLM_DEFAULT_PROVIDER
from rlx.paths import CONFIG_SNAPSHOT_NAME
from rlx.rl import EvaluationError, evaluate_checkpoint
from rlx.rl.ppo import _prepare_matplotlib_cache


class ResearchError(Exception):
    """Raised when RLCLI cannot run a bounded research loop."""


OBJECTIVE_METRIC = "eval_mean_reward"
RESEARCH_PROTOCOL_VERSION = 2
EXECUTED_SCORE_MODE = "standalone_eval_latest_checkpoint"
DRY_RUN_SCORE_MODE = "existing_eval_or_rollout_signal"
ALLOWED_RESEARCH_MUTATION_KEYS = LLM_ALLOWED_MUTATION_KEYS
SIGNATURE_IGNORED_KEYS = (
    "algo.total_timesteps",
    "eval.every",
    "eval.episodes",
    "eval.deterministic",
)


@dataclass(frozen=True)
class ResearchVariantScore:
    index: int
    run_id: str | None
    status: str
    mutations: dict[str, Any]
    score: float | None
    score_source: str | None
    promoted: bool


@dataclass(frozen=True)
class ResearchRound:
    index: int
    baseline_run_id: str
    advisor_bundle: Path
    advisor_manifest: Path
    champion_before: str
    champion_score_before: float | None
    candidate_run_id: str | None
    candidate_score: float | None
    candidate_score_source: str | None
    improvement: float | None
    promoted: bool
    variants: tuple[ResearchVariantScore, ...]
    champion_after: str
    champion_score_after: float | None
    stop_reason: str | None


@dataclass(frozen=True)
class ResearchResult:
    project_root: Path
    bundle_dir: Path
    manifest_path: Path
    report_path: Path
    score_plot_path: Path | None
    progress_plot_path: Path | None
    initial_run_id: str
    initial_score: float | None
    initial_score_source: str | None
    champion_run_id: str
    champion_score: float | None
    champion_score_source: str | None
    mode: str
    rounds: tuple[ResearchRound, ...]
    stop_reason: str


def run_research(
    run_ref: str,
    *,
    rounds: int = 3,
    variants: int = 4,
    execute: bool = False,
    timesteps: int | None = None,
    min_improvement: float = 0.0,
    planner: str = "rules",
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_strict: bool = False,
    cwd: Path | None = None,
) -> ResearchResult:
    if rounds < 1:
        raise ResearchError("Research must run at least one round.")
    if variants < 1:
        raise ResearchError("Research must create at least one variant per round.")
    if timesteps is not None and timesteps < 1:
        raise ResearchError("Research timesteps override must be positive.")
    if min_improvement < 0:
        raise ResearchError("Research min improvement must be non-negative.")
    planner_name = planner.lower()
    if planner_name not in {"rules", "llm"}:
        raise ResearchError("Research planner must be either 'rules' or 'llm'.")

    try:
        initial_diagnosis = diagnose_run(run_ref, cwd=cwd)
    except DiagnoseError as exc:
        raise ResearchError(str(exc)) from exc

    initial_run = initial_diagnosis.analysis.info.run
    try:
        initial_config = load_config(initial_run.run_dir / CONFIG_SNAPSHOT_NAME)
    except ConfigError as exc:
        raise ResearchError(str(exc)) from exc
    try:
        project_root = find_project_root(initial_run.run_dir)
    except ProjectLookupError as exc:
        raise ResearchError(str(exc)) from exc

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
        project_root / "analysis" / "research",
        f"{initial_run.run_id}_research",
    )
    bundle_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = bundle_dir / "manifest.json"
    report_path = bundle_dir / "report.md"

    budget_timesteps = timesteps or initial_config.algo.total_timesteps
    locked_mutations = {
        "algo.total_timesteps": budget_timesteps,
        "eval.every": initial_config.eval.every,
        "eval.episodes": initial_config.eval.episodes,
        "eval.deterministic": initial_config.eval.deterministic,
    }
    require_eval_score = execute
    champion_run_id = initial_run.run_id
    if execute:
        champion_score, champion_score_source = _standalone_eval_score(initial_run.run_dir)
    else:
        champion_score, champion_score_source = _score_run(
            initial_run,
            require_eval_score=require_eval_score,
        )
    initial_score = champion_score
    initial_score_source = champion_score_source

    protocol = _build_protocol(
        budget_timesteps=budget_timesteps,
        budget_source="cli override" if timesteps is not None else "baseline config",
        locked_mutations=locked_mutations,
        require_eval_score=require_eval_score,
        planner=planner_name,
        llm_provider=resolved_llm_provider if planner_name == "llm" else None,
        llm_model=resolved_llm_model if planner_name == "llm" else None,
        llm_strict=llm_strict,
    )

    return _continue_research(
        project_root=project_root,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        initial_run_id=initial_run.run_id,
        initial_score=initial_score,
        initial_score_source=initial_score_source,
        champion_run_id=champion_run_id,
        champion_score=champion_score,
        champion_score_source=champion_score_source,
        execute=execute,
        target_rounds=rounds,
        variants=variants,
        min_improvement=min_improvement,
        protocol=protocol,
        locked_mutations=locked_mutations,
        research_rounds=[],
        tried_signatures=set(),
    )


def resume_research(
    resume_path: str | Path,
    *,
    rounds: int | None = None,
    variants: int | None = None,
    execute: bool | None = None,
    timesteps: int | None = None,
    min_improvement: float | None = None,
    planner: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_strict: bool | None = None,
) -> ResearchResult:
    manifest_path = _resolve_research_manifest(resume_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResearchError(f"Could not read research manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchError(f"Research manifest is not valid JSON: {manifest_path}") from exc

    if not isinstance(payload, dict) or payload.get("kind") != "research_bundle":
        raise ResearchError(f"Not an RLCLI research bundle: {manifest_path}")

    project_root = _project_root_for_manifest(manifest_path)
    bundle_dir = manifest_path.parent
    report_path = bundle_dir / "report.md"
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ResearchError("Research bundle is missing protocol data and cannot be resumed.")

    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ResearchError("Research bundle is missing settings and cannot be resumed.")

    target_rounds = _coerce_positive_int(
        rounds if rounds is not None else settings.get("rounds"),
        field_name="rounds",
    )
    variant_count = _coerce_positive_int(
        variants if variants is not None else settings.get("variants"),
        field_name="variants",
    )
    improvement = _coerce_non_negative_float(
        min_improvement if min_improvement is not None else settings.get("min_improvement"),
        field_name="min_improvement",
    )
    if (
        planner is not None
        or llm_provider is not None
        or llm_model is not None
        or llm_strict is not None
    ):
        raise ResearchError(
            "Resumed research keeps the original planner/provider/model. "
            "Start a new research bundle to change LLM settings."
        )

    locked_mutations = _locked_mutations_from_protocol(protocol)
    if timesteps is not None:
        locked_timesteps = locked_mutations.get("algo.total_timesteps")
        if locked_timesteps != timesteps:
            raise ResearchError(
                "Resumed research cannot change the locked timestep budget. "
                f"Existing budget is {locked_timesteps}."
            )

    prior_mode = _mode_from_payload(payload)
    should_execute = prior_mode == "executed" if execute is None else execute
    if ("executed" if should_execute else "dry_run") != prior_mode:
        raise ResearchError(
            "Resumed research must keep the original mode. Start a new research bundle "
            "to switch between dry-run and execute."
        )

    initial = payload.get("initial")
    champion = payload.get("champion")
    if not isinstance(initial, dict) or not isinstance(champion, dict):
        raise ResearchError("Research bundle is missing initial or champion state.")

    research_rounds = _rounds_from_payload(payload, project_root=project_root)
    tried_signatures = _tried_signatures_from_rounds(
        research_rounds,
        ignored_keys=_signature_ignored_keys_from_protocol(protocol),
    )

    return _continue_research(
        project_root=project_root,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        initial_run_id=_required_str(initial.get("run_id"), "initial.run_id"),
        initial_score=_maybe_float(initial.get("score")),
        initial_score_source=_maybe_str(initial.get("score_source")),
        champion_run_id=_required_str(champion.get("run_id"), "champion.run_id"),
        champion_score=_maybe_float(champion.get("score")),
        champion_score_source=_maybe_str(champion.get("score_source")),
        execute=should_execute,
        target_rounds=target_rounds,
        variants=variant_count,
        min_improvement=improvement,
        protocol=protocol,
        locked_mutations=locked_mutations,
        research_rounds=list(research_rounds),
        tried_signatures=tried_signatures,
    )


def _continue_research(
    *,
    project_root: Path,
    bundle_dir: Path,
    manifest_path: Path,
    report_path: Path,
    initial_run_id: str,
    initial_score: float | None,
    initial_score_source: str | None,
    champion_run_id: str,
    champion_score: float | None,
    champion_score_source: str | None,
    execute: bool,
    target_rounds: int,
    variants: int,
    min_improvement: float,
    protocol: dict[str, Any],
    locked_mutations: dict[str, Any],
    research_rounds: list[ResearchRound],
    tried_signatures: set[str],
) -> ResearchResult:
    if target_rounds < len(research_rounds):
        raise ResearchError(
            f"Research bundle already has {len(research_rounds)} round(s); "
            f"pass --rounds {len(research_rounds) + 1} or higher to continue."
        )

    stop_reason = "completed requested rounds"
    score_plot_path: Path | None = None
    progress_plot_path: Path | None = None
    signature_ignored_keys = _signature_ignored_keys_from_protocol(protocol)
    require_eval_score = _require_eval_score_from_protocol(protocol)
    planner = _planner_from_protocol(protocol)
    llm_provider = _maybe_str(protocol.get("llm_provider")) or LLM_DEFAULT_PROVIDER
    llm_model = _maybe_str(protocol.get("llm_model")) or LLM_DEFAULT_MODEL
    llm_strict = bool(protocol.get("llm_strict", False))

    for round_index in range(len(research_rounds) + 1, target_rounds + 1):
        try:
            advisor = run_advisor(
                champion_run_id,
                variants=variants,
                execute=execute,
                locked_mutations=locked_mutations,
                allowed_mutation_keys=tuple(_allowed_mutation_keys_from_protocol(protocol)),
                excluded_mutation_signatures=tried_signatures,
                signature_ignored_keys=tuple(signature_ignored_keys),
                require_eval_score=require_eval_score,
                planner=planner,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_strict=llm_strict,
                cwd=project_root,
            )
        except AdvisorExhaustedError as exc:
            stop_reason = str(exc)
            if not stop_reason:
                stop_reason = "proposal space exhausted"
            score_plot_path, progress_plot_path = _write_outputs(
                manifest_path=manifest_path,
                report_path=report_path,
                project_root=project_root,
                bundle_dir=bundle_dir,
                initial_run_id=initial_run_id,
                initial_score=initial_score,
                initial_score_source=initial_score_source,
                champion_run_id=champion_run_id,
                champion_score=champion_score,
                champion_score_source=champion_score_source,
                mode="executed" if execute else "dry_run",
                settings={
                    "rounds": target_rounds,
                    "variants": variants,
                    "min_improvement": min_improvement,
                },
                protocol=protocol,
                rounds=research_rounds,
                stop_reason=stop_reason,
            )
            break
        except AdvisorError as exc:
            raise ResearchError(str(exc)) from exc

        candidate = advisor.best_variant
        candidate_score = candidate.score if candidate is not None else None
        candidate_run_id = candidate.run_id if candidate is not None else None
        candidate_source = candidate.score_source if candidate is not None else None
        improvement = _improvement(candidate_score, champion_score)
        promoted = _should_promote(
            candidate_score,
            champion_score,
            min_improvement=min_improvement,
        )

        champion_before = champion_run_id
        score_before = champion_score

        if execute and promoted and candidate_run_id is not None:
            champion_run_id = candidate_run_id
            champion_score = candidate_score
            champion_score_source = candidate_source

        variant_scores = _variant_scores(
            advisor.variants,
            promoted_run_id=candidate_run_id if execute and promoted else None,
        )
        _record_tried_mutations(
            tried_signatures,
            advisor.variants,
            ignored_keys=signature_ignored_keys,
        )
        research_rounds.append(
            ResearchRound(
                index=round_index,
                baseline_run_id=advisor.baseline_run_id,
                advisor_bundle=advisor.bundle_dir,
                advisor_manifest=advisor.manifest_path,
                champion_before=champion_before,
                champion_score_before=score_before,
                candidate_run_id=candidate_run_id,
                candidate_score=candidate_score,
                candidate_score_source=candidate_source,
                improvement=improvement,
                promoted=promoted,
                variants=variant_scores,
                champion_after=champion_run_id,
                champion_score_after=champion_score,
                stop_reason=None,
            )
        )

        score_plot_path, progress_plot_path = _write_outputs(
            manifest_path=manifest_path,
            report_path=report_path,
            project_root=project_root,
            bundle_dir=bundle_dir,
            initial_run_id=initial_run_id,
            initial_score=initial_score,
            initial_score_source=initial_score_source,
            champion_run_id=champion_run_id,
            champion_score=champion_score,
            champion_score_source=champion_score_source,
            mode="executed" if execute else "dry_run",
            settings={
                "rounds": target_rounds,
                "variants": variants,
                "min_improvement": min_improvement,
            },
            protocol=protocol,
            rounds=research_rounds,
            stop_reason=stop_reason,
        )

    if not score_plot_path:
        score_plot_path, progress_plot_path = _write_outputs(
            manifest_path=manifest_path,
            report_path=report_path,
            project_root=project_root,
            bundle_dir=bundle_dir,
            initial_run_id=initial_run_id,
            initial_score=initial_score,
            initial_score_source=initial_score_source,
            champion_run_id=champion_run_id,
            champion_score=champion_score,
            champion_score_source=champion_score_source,
            mode="executed" if execute else "dry_run",
            settings={
                "rounds": target_rounds,
                "variants": variants,
                "min_improvement": min_improvement,
            },
            protocol=protocol,
            rounds=research_rounds,
            stop_reason=stop_reason,
        )

    return ResearchResult(
        project_root=project_root,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        score_plot_path=score_plot_path,
        progress_plot_path=progress_plot_path,
        initial_run_id=initial_run_id,
        initial_score=initial_score,
        initial_score_source=initial_score_source,
        champion_run_id=champion_run_id,
        champion_score=champion_score,
        champion_score_source=champion_score_source,
        mode="executed" if execute else "dry_run",
        rounds=tuple(research_rounds),
        stop_reason=stop_reason,
    )


def _improvement(candidate: float | None, champion: float | None) -> float | None:
    if candidate is None or champion is None:
        return None
    return candidate - champion


def _standalone_eval_score(run_dir: Path) -> tuple[float, str]:
    latest_checkpoint = run_dir / "checkpoints" / "latest.zip"
    try:
        result = evaluate_checkpoint(latest_checkpoint)
    except EvaluationError as exc:
        raise ResearchError(
            "Executed research could not create a standalone eval for the baseline "
            f"{run_dir.name}: {exc}"
        ) from exc
    return result.mean_reward, f"standalone eval ({result.result_path.name})"


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


def _should_promote(
    candidate: float | None,
    champion: float | None,
    *,
    min_improvement: float,
) -> bool:
    if candidate is None:
        return False
    if champion is None:
        return True
    return candidate > champion + min_improvement


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


def _variant_scores(
    variants: tuple[Any, ...],
    *,
    promoted_run_id: str | None,
) -> tuple[ResearchVariantScore, ...]:
    scores = []
    for variant in variants:
        scores.append(
            ResearchVariantScore(
                index=variant.index,
                run_id=variant.run_id,
                status=variant.status,
                mutations=variant.mutations,
                score=variant.score,
                score_source=variant.score_source,
                promoted=(
                    promoted_run_id is not None
                    and variant.run_id is not None
                    and variant.run_id == promoted_run_id
                ),
            )
        )
    return tuple(scores)


def _record_tried_mutations(
    seen: set[str],
    variants: tuple[Any, ...],
    *,
    ignored_keys: tuple[str, ...],
) -> None:
    for variant in variants:
        signature = mutation_signature(
            variant.mutations,
            ignored_keys=ignored_keys,
        )
        if signature:
            seen.add(signature)


def _build_protocol(
    *,
    budget_timesteps: int,
    budget_source: str,
    locked_mutations: dict[str, Any],
    require_eval_score: bool,
    planner: str,
    llm_provider: str | None,
    llm_model: str | None,
    llm_strict: bool,
) -> dict[str, Any]:
    score_mode = EXECUTED_SCORE_MODE if require_eval_score else DRY_RUN_SCORE_MODE
    return {
        "version": RESEARCH_PROTOCOL_VERSION,
        "score_mode": score_mode,
        "objective": {
            "metric": OBJECTIVE_METRIC,
            "higher_is_better": True,
            "score_source": (
                "standalone latest-checkpoint eval when executed; existing eval/rollout "
                "signals for dry-run planning"
            ),
            "requires_eval_score": require_eval_score,
        },
        "budget": {
            "timesteps_per_variant": budget_timesteps,
            "source": budget_source,
        },
        "allowed_mutation_keys": list(ALLOWED_RESEARCH_MUTATION_KEYS),
        "planner": planner,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_strict": llm_strict,
        "locked_mutations": locked_mutations,
        "signature_ignored_keys": list(SIGNATURE_IGNORED_KEYS),
        "duplicate_guard": "exact mutation signatures are not repeated across rounds",
    }


def _write_outputs(
    *,
    manifest_path: Path,
    report_path: Path,
    project_root: Path,
    bundle_dir: Path,
    initial_run_id: str,
    initial_score: float | None,
    initial_score_source: str | None,
    champion_run_id: str,
    champion_score: float | None,
    champion_score_source: str | None,
    mode: str,
    settings: dict[str, Any],
    protocol: dict[str, Any],
    rounds: list[ResearchRound],
    stop_reason: str,
) -> tuple[Path | None, Path | None]:
    payload = {
        "kind": "research_bundle",
        "updated_at": _utc_now_iso(),
        "mode": mode,
        "bundle": str(_relative_to(project_root, bundle_dir)),
        "initial": {
            "run_id": initial_run_id,
            "score": initial_score,
            "score_source": initial_score_source,
        },
        "initial_run_id": initial_run_id,
        "champion": {
            "run_id": champion_run_id,
            "score": champion_score,
            "score_source": champion_score_source,
        },
        "settings": settings,
        "protocol": protocol,
        "stop_reason": stop_reason,
        "rounds": [
            {
                "index": item.index,
                "baseline_run_id": item.baseline_run_id,
                "advisor_bundle": str(_relative_to(project_root, item.advisor_bundle)),
                "advisor_manifest": str(_relative_to(project_root, item.advisor_manifest)),
                "champion_before": item.champion_before,
                "champion_score_before": item.champion_score_before,
                "candidate_run_id": item.candidate_run_id,
                "candidate_score": item.candidate_score,
                "candidate_score_source": item.candidate_score_source,
                "improvement": item.improvement,
                "promoted": item.promoted,
                "variants": [
                    {
                        "index": variant.index,
                        "run_id": variant.run_id,
                        "status": variant.status,
                        "mutations": variant.mutations,
                        "score": variant.score,
                        "score_source": variant.score_source,
                        "promoted": variant.promoted,
                    }
                    for variant in item.variants
                ],
                "champion_after": item.champion_after,
                "champion_score_after": item.champion_score_after,
                "stop_reason": item.stop_reason,
            }
            for item in rounds
        ],
    }

    score_plot_path = _plot_scoreboard(payload, bundle_dir / "scoreboard.png")
    progress_plot_path = _plot_progress(payload, bundle_dir / "progress.png")
    artifacts = []
    if score_plot_path is not None:
        artifacts.append(
            {
                "key": "scoreboard",
                "file": str(_relative_to(project_root, score_plot_path)),
            }
        )
    if progress_plot_path is not None:
        artifacts.append(
            {
                "key": "progress",
                "file": str(_relative_to(project_root, progress_plot_path)),
            }
        )
    payload["artifacts"] = artifacts

    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_build_report(payload), encoding="utf-8")
    return score_plot_path, progress_plot_path


def _build_report(payload: dict[str, Any]) -> str:
    champion = payload["champion"]
    artifacts = payload.get("artifacts", [])
    lines = [
        f"# RLCLI Research Report: {payload['initial_run_id']}",
        "",
        f"- Mode: {payload['mode']}",
        (
            f"- Protocol: v{payload.get('protocol', {}).get('version', 'unknown')} "
            f"({payload.get('protocol', {}).get('score_mode', 'unknown score mode')})"
        ),
        f"- Stop reason: {payload['stop_reason']}",
        f"- Champion: {champion['run_id']}",
        f"- Champion score: {_fmt_score(champion['score'], champion['score_source'])}",
        "",
    ]
    if artifacts:
        lines.append("## Artifacts")
        for artifact in artifacts:
            lines.append(f"- {artifact['key']}: `{artifact['file']}`")
        lines.append("")

    lines.extend(
        [
            "## Scoreboard",
            "",
            "| Run | Score | Source | Promoted |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in _scoreboard_rows(payload):
        promoted = "yes" if row["promoted"] else "no"
        lines.append(
            "| "
            f"{row['run_id']} | "
            f"{_fmt_optional_number(row['score'])} | "
            f"{row['source'] or 'n/a'} | "
            f"{promoted} |"
        )
    lines.extend(["", "## Rounds"])

    rounds = payload["rounds"]
    if not rounds:
        lines.append("- No rounds were recorded.")
    for item in rounds:
        promoted = "yes" if item["promoted"] else "no"
        lines.extend(
            [
                f"- Round {item['index']:03d}",
                f"  Baseline: {item['baseline_run_id']}",
                f"  Advisor bundle: {item['advisor_bundle']}",
                f"  Candidate: {item['candidate_run_id'] or 'n/a'}",
                (
                    "  Candidate score: "
                    f"{_fmt_score(item['candidate_score'], item['candidate_score_source'])}"
                ),
                f"  Improvement: {_fmt_optional_number(item['improvement'])}",
                f"  Promoted: {promoted}",
                f"  Champion after: {item['champion_after']}",
            ]
        )
        if item["stop_reason"]:
            lines.append(f"  Stop: {item['stop_reason']}")

    return "\n".join(lines) + "\n"


def _plot_scoreboard(payload: dict[str, Any], output_path: Path) -> Path | None:
    rows = [row for row in _scoreboard_rows(payload) if row["score"] is not None]
    if not rows:
        return None

    _prepare_matplotlib_cache()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [row["run_id"] for row in rows]
    scores = [float(row["score"]) for row in rows]
    colors = [_score_color(row, payload["champion"]["run_id"]) for row in rows]
    width = max(8.5, min(16.0, 0.72 * len(rows) + 3.5))

    fig, ax = plt.subplots(figsize=(width, 5.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.bar(labels, scores, color=colors, edgecolor="#0f172a", linewidth=0.7)
    for bar, score in zip(bars, scores, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )

    ax.set_title("Research Scoreboard", fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Score (eval reward preferred)", color="#334155")
    ax.set_xlabel("Baseline and trained advisor variants", color="#334155")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, color="#d4d4d8", alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(axis="x", colors="#475569", labelrotation=35)
    ax.tick_params(axis="y", colors="#475569")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_progress(payload: dict[str, Any], output_path: Path) -> Path | None:
    rows = [row for row in _progress_rows(payload) if row["score"] is not None]
    if not rows:
        return None

    _prepare_matplotlib_cache()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    experiments = [int(row["experiment"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    running_best = []
    best = float("-inf")
    for score in scores:
        best = max(best, score)
        running_best.append(best)

    promoted_x = [
        int(row["experiment"])
        for row in rows
        if row["promoted"] or row["role"] == "baseline"
    ]
    promoted_y = [
        float(row["score"])
        for row in rows
        if row["promoted"] or row["role"] == "baseline"
    ]
    discarded_x = [
        int(row["experiment"])
        for row in rows
        if not row["promoted"] and row["role"] != "baseline"
    ]
    discarded_y = [
        float(row["score"])
        for row in rows
        if not row["promoted"] and row["role"] != "baseline"
    ]

    width = max(10.5, min(18.0, 0.18 * len(rows) + 9.5))
    fig, ax = plt.subplots(figsize=(width, 7.0), dpi=150)
    fig.patch.set_facecolor("#fbfbf8")
    ax.set_facecolor("#fbfbf8")

    if discarded_x:
        ax.scatter(
            discarded_x,
            discarded_y,
            s=24,
            color="#cbd5e1",
            alpha=0.68,
            label="Discarded",
            edgecolors="none",
        )
    ax.plot(
        experiments,
        running_best,
        color="#2f9e55",
        linewidth=1.8,
        alpha=0.7,
        label="Running best",
    )
    ax.scatter(
        promoted_x,
        promoted_y,
        s=46,
        color="#43a047",
        edgecolors="#1b5e20",
        linewidths=0.7,
        label="Kept",
        zorder=3,
    )

    for row in rows:
        if not _should_label_progress_row(row):
            continue
        label = str(row["label"])
        if not label:
            continue
        ax.annotate(
            label,
            (int(row["experiment"]), float(row["score"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6.2,
            rotation=22,
            color="#2f7d32" if row["promoted"] or row["role"] == "baseline" else "#64748b",
            alpha=0.9,
        )

    kept = sum(1 for row in rows if row["promoted"])
    ax.set_title(
        f"RLX Research Progress: {len(rows) - 1} Experiments, {kept} Kept Improvements",
        fontsize=14,
        fontweight="bold",
        color="#1f2937",
    )
    ax.set_xlabel("Experiment #", color="#334155")
    ax.set_ylabel("Eval reward (higher is better)", color="#334155")
    ax.grid(True, axis="both", linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
    ax.spines["top"].set_color("#64748b")
    ax.spines["right"].set_color("#64748b")
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.tick_params(axis="both", colors="#475569")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _scoreboard_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    initial = payload["initial"]
    rows = [
        {
            "run_id": initial["run_id"],
            "score": initial["score"],
            "source": initial["score_source"],
            "role": "baseline",
            "promoted": False,
        }
    ]
    seen = {initial["run_id"]}

    for research_round in payload["rounds"]:
        for variant in research_round.get("variants", []):
            run_id = variant.get("run_id")
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            rows.append(
                {
                    "run_id": run_id,
                    "score": variant.get("score"),
                    "source": variant.get("score_source"),
                    "role": "variant",
                    "promoted": bool(variant.get("promoted")),
                }
            )
    return rows


def _progress_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    initial = payload["initial"]
    rows = [
        {
            "experiment": 0,
            "run_id": initial["run_id"],
            "score": initial["score"],
            "source": initial["score_source"],
            "role": "baseline",
            "promoted": True,
            "candidate": False,
            "label": "baseline",
        }
    ]
    candidate_ids = {
        item.get("candidate_run_id")
        for item in payload.get("rounds", [])
        if item.get("candidate_run_id")
    }
    seen = {initial["run_id"]}
    experiment = 0

    for research_round in payload["rounds"]:
        for variant in research_round.get("variants", []):
            run_id = variant.get("run_id")
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            experiment += 1
            rows.append(
                {
                    "experiment": experiment,
                    "run_id": run_id,
                    "score": variant.get("score"),
                    "source": variant.get("score_source"),
                    "role": "variant",
                    "promoted": bool(variant.get("promoted")),
                    "candidate": run_id in candidate_ids,
                    "label": _short_mutation_label(variant.get("mutations", {})),
                }
            )
    return rows


def _should_label_progress_row(row: dict[str, Any]) -> bool:
    return bool(row["promoted"] or row["candidate"] or row["role"] == "baseline")


def _short_mutation_label(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""

    labels = []
    for key, raw_value in value.items():
        labels.append(f"{_short_key(key)}={_compact_value(raw_value)}")
        if len(labels) >= 3:
            break
    remaining = len(value) - len(labels)
    if remaining > 0:
        labels.append(f"+{remaining}")
    return ", ".join(labels)


def _short_key(key: str) -> str:
    names = {
        "seed": "seed",
        "device": "device",
        "env.num_envs": "envs",
        "algo.rollout_steps": "rollout",
        "algo.batch_size": "batch",
        "algo.learning_rate": "lr",
        "algo.gamma": "gamma",
        "algo.gae_lambda": "gae",
        "algo.clip_range": "clip",
        "algo.entropy_coef": "ent",
        "algo.value_coef": "vf",
        "algo.update_epochs": "epochs",
        "policy.hidden_sizes": "hidden",
        "checkpoint.save_every": "ckpt",
    }
    return names.get(key, key.rsplit(".", maxsplit=1)[-1])


def _compact_value(value: Any) -> str:
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _score_color(row: dict[str, Any], champion_run_id: str) -> str:
    if row["run_id"] == champion_run_id:
        return "#10b981"
    if row["role"] == "baseline":
        return "#64748b"
    if row["promoted"]:
        return "#22c55e"
    return "#06b6d4"


def _resolve_research_manifest(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists() or not path.is_file():
        raise ResearchError(f"Research manifest not found: {path}")
    return path


def _project_root_for_manifest(manifest_path: Path) -> Path:
    try:
        return find_project_root(manifest_path)
    except ProjectLookupError as exc:
        raise ResearchError(str(exc)) from exc


def _rounds_from_payload(
    payload: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[ResearchRound, ...]:
    raw_rounds = payload.get("rounds")
    if not isinstance(raw_rounds, list):
        raise ResearchError("Research bundle has invalid rounds data.")

    rounds = []
    for item in raw_rounds:
        if not isinstance(item, dict):
            raise ResearchError("Research round entries must be objects.")
        variants = []
        for raw_variant in item.get("variants", []):
            if not isinstance(raw_variant, dict):
                continue
            variants.append(
                ResearchVariantScore(
                    index=_coerce_positive_int(
                        raw_variant.get("index"),
                        field_name="variant.index",
                    ),
                    run_id=_maybe_str(raw_variant.get("run_id")),
                    status=_maybe_str(raw_variant.get("status")) or "unknown",
                    mutations=_maybe_dict(raw_variant.get("mutations")),
                    score=_maybe_float(raw_variant.get("score")),
                    score_source=_maybe_str(raw_variant.get("score_source")),
                    promoted=bool(raw_variant.get("promoted")),
                )
            )

        rounds.append(
            ResearchRound(
                index=_coerce_positive_int(item.get("index"), field_name="round.index"),
                baseline_run_id=_required_str(
                    item.get("baseline_run_id"),
                    "round.baseline_run_id",
                ),
                advisor_bundle=_resolve_project_path(
                    project_root,
                    _required_str(item.get("advisor_bundle"), "round.advisor_bundle"),
                ),
                advisor_manifest=_resolve_project_path(
                    project_root,
                    _required_str(item.get("advisor_manifest"), "round.advisor_manifest"),
                ),
                champion_before=_required_str(
                    item.get("champion_before"),
                    "round.champion_before",
                ),
                champion_score_before=_maybe_float(item.get("champion_score_before")),
                candidate_run_id=_maybe_str(item.get("candidate_run_id")),
                candidate_score=_maybe_float(item.get("candidate_score")),
                candidate_score_source=_maybe_str(item.get("candidate_score_source")),
                improvement=_maybe_float(item.get("improvement")),
                promoted=bool(item.get("promoted")),
                variants=tuple(variants),
                champion_after=_required_str(
                    item.get("champion_after"),
                    "round.champion_after",
                ),
                champion_score_after=_maybe_float(item.get("champion_score_after")),
                stop_reason=_maybe_str(item.get("stop_reason")),
            )
        )
    return tuple(rounds)


def _tried_signatures_from_rounds(
    rounds: tuple[ResearchRound, ...],
    *,
    ignored_keys: tuple[str, ...],
) -> set[str]:
    signatures = set()
    for research_round in rounds:
        for variant in research_round.variants:
            signature = mutation_signature(variant.mutations, ignored_keys=ignored_keys)
            if signature:
                signatures.add(signature)
    return signatures


def _locked_mutations_from_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    locked = protocol.get("locked_mutations")
    if not isinstance(locked, dict):
        raise ResearchError("Research protocol is missing locked_mutations.")
    return dict(locked)


def _allowed_mutation_keys_from_protocol(protocol: dict[str, Any]) -> tuple[str, ...]:
    keys = protocol.get("allowed_mutation_keys")
    if not isinstance(keys, list) or not all(isinstance(item, str) for item in keys):
        raise ResearchError("Research protocol is missing allowed_mutation_keys.")
    return tuple(keys)


def _signature_ignored_keys_from_protocol(protocol: dict[str, Any]) -> tuple[str, ...]:
    keys = protocol.get("signature_ignored_keys")
    if not isinstance(keys, list) or not all(isinstance(item, str) for item in keys):
        raise ResearchError("Research protocol is missing signature_ignored_keys.")
    return tuple(keys)


def _require_eval_score_from_protocol(protocol: dict[str, Any]) -> bool:
    objective = protocol.get("objective")
    if not isinstance(objective, dict):
        raise ResearchError("Research protocol is missing objective data.")
    return bool(objective.get("requires_eval_score"))


def _planner_from_protocol(protocol: dict[str, Any]) -> str:
    planner = protocol.get("planner")
    if planner is None:
        return "rules"
    if planner not in {"rules", "llm"}:
        raise ResearchError("Research protocol has invalid planner.")
    return str(planner)


def _mode_from_payload(payload: dict[str, Any]) -> str:
    mode = payload.get("mode")
    if mode not in {"dry_run", "executed"}:
        raise ResearchError("Research bundle has invalid mode.")
    return str(mode)


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ResearchError(f"Research {field_name} must be a positive integer.")
    return value


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, int | float) or value < 0:
        raise ResearchError(f"Research {field_name} must be non-negative.")
    return float(value)


def _required_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchError(f"Research {field_name} must be a non-empty string.")
    return value


def _maybe_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _maybe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _fmt_score(score: float | None, source: str | None) -> str:
    if score is None:
        return "n/a"
    if source:
        return f"{score:.4g} ({source})"
    return f"{score:.4g}"


def _fmt_optional_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _next_bundle_dir(root: Path, name: str) -> Path:
    slug = _slugify(name) or "research"
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
