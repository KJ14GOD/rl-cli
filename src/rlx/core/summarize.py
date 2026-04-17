from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.core.analyze import AnalyzeError, RunAnalysis, analyze_run
from rlx.core.compare import CompareError, RunComparison, load_run_comparison
from rlx.core.list_runs import RunList, RunListError, list_runs
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME


class SummarizeError(Exception):
    """Raised when a path or experiment cannot be summarized."""


@dataclass(frozen=True)
class SweepVariantSummary:
    index: int
    status: str
    run_id: str | None
    mutations: dict[str, Any]
    run: RunComparison | None
    error: str | None


@dataclass(frozen=True)
class SweepSummary:
    project_root: Path
    bundle_dir: Path
    manifest_path: Path
    name: str
    variants: tuple[SweepVariantSummary, ...]


@dataclass(frozen=True)
class SummaryResult:
    kind: str
    target: Path | str
    run: RunAnalysis | None = None
    project: RunList | None = None
    sweep: SweepSummary | None = None


def summarize_target(target: str, cwd: Path | None = None) -> SummaryResult:
    working_dir = (cwd or Path.cwd()).resolve()
    candidate = Path(target).expanduser()

    if candidate.exists():
        resolved = candidate.resolve()
        if _is_run_dir(resolved):
            return _summarize_run(str(resolved), working_dir)
        if _is_sweep_manifest(resolved):
            return _summarize_sweep(resolved)
        if _is_sweep_bundle(resolved):
            return _summarize_sweep(resolved / "manifest.json")
        if _looks_like_project(resolved):
            return _summarize_project(resolved)
        raise SummarizeError(f"Unsupported summary target: {resolved}")

    try:
        return _summarize_run(target, working_dir)
    except SummarizeError:
        pass

    raise SummarizeError(f"Summary target not found: {target}")


def _summarize_run(run_ref: str, cwd: Path) -> SummaryResult:
    try:
        return SummaryResult(kind="run", target=run_ref, run=analyze_run(run_ref, cwd=cwd))
    except AnalyzeError as exc:
        raise SummarizeError(str(exc)) from exc


def _summarize_project(path: Path) -> SummaryResult:
    try:
        return SummaryResult(kind="project", target=path, project=list_runs(path))
    except RunListError as exc:
        raise SummarizeError(str(exc)) from exc


def _summarize_sweep(manifest_path: Path) -> SummaryResult:
    try:
        project_root = find_project_root(manifest_path)
    except ProjectLookupError as exc:
        raise SummarizeError(str(exc)) from exc

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummarizeError(f"Could not read sweep manifest: {manifest_path}") from exc

    if not isinstance(payload, dict) or payload.get("kind") != "sweep_bundle":
        raise SummarizeError(f"Not an RLCLI sweep manifest: {manifest_path}")

    variants = []
    raw_variants = payload.get("variants")
    if isinstance(raw_variants, list):
        for item in raw_variants:
            if not isinstance(item, dict):
                continue
            run = _load_variant_run(project_root, item.get("run_dir"))
            mutations = item.get("mutations")
            variants.append(
                SweepVariantSummary(
                    index=_maybe_int(item.get("index")) or len(variants) + 1,
                    status=str(item.get("status") or "unknown"),
                    run_id=_maybe_str(item.get("run_id")),
                    mutations=mutations if isinstance(mutations, dict) else {},
                    run=run,
                    error=_maybe_str(item.get("error")),
                )
            )

    summary = SweepSummary(
        project_root=project_root,
        bundle_dir=manifest_path.parent,
        manifest_path=manifest_path,
        name=str(payload.get("name") or manifest_path.parent.name),
        variants=tuple(variants),
    )
    return SummaryResult(kind="sweep", target=manifest_path.parent, sweep=summary)


def best_final_run(runs: tuple[RunComparison, ...]) -> RunComparison | None:
    present = [run for run in runs if run.final_rollout_reward is not None]
    if not present:
        return None
    return max(present, key=lambda run: run.final_rollout_reward or float("-inf"))


def best_eval_run(runs: tuple[RunComparison, ...]) -> RunComparison | None:
    present = [
        run
        for run in runs
        if run.best_eval is not None and run.best_eval.mean_reward is not None
    ]
    if not present:
        return None
    return max(present, key=lambda run: run.best_eval.mean_reward or float("-inf"))


def _load_variant_run(project_root: Path, value: Any) -> RunComparison | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return load_run_comparison((project_root / value).resolve())
    except CompareError:
        return None


def _is_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / METADATA_NAME).exists()
        and (path / CONFIG_SNAPSHOT_NAME).exists()
    )


def _looks_like_project(path: Path) -> bool:
    return path.is_dir() and (path / "runs").is_dir() and (path / "configs").is_dir()


def _is_sweep_bundle(path: Path) -> bool:
    return path.is_dir() and _is_sweep_manifest(path / "manifest.json")


def _is_sweep_manifest(path: Path) -> bool:
    if not path.is_file() or path.name != "manifest.json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("kind") == "sweep_bundle"


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _maybe_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
