from __future__ import annotations

# ruff: noqa: E501
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rlx.core.compare import CompareError, RunComparison, load_run_comparison, resolve_run_ref
from rlx.core.explain_metrics import METRIC_DEFINITIONS, explain_metrics
from rlx.core.list_runs import RunListError, list_runs
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME, METRICS_NAME


class ReportError(Exception):
    """Raised when RLX cannot build a web report or dashboard."""


@dataclass(frozen=True)
class ReportResult:
    kind: str
    project_root: Path
    output_dir: Path
    index_path: Path
    title: str
    target: str


@dataclass(frozen=True)
class DashboardResult:
    project_root: Path
    output_dir: Path
    index_path: Path
    runs: int
    research_bundles: int


@dataclass(frozen=True)
class WebAppProject:
    project_root: Path
    demo: bool


_METRIC_LABELS = {definition.key: definition.label for definition in METRIC_DEFINITIONS}
_METRIC_DESCRIPTIONS = {
    definition.key: definition.description for definition in METRIC_DEFINITIONS
}
_WEB_METRIC_ORDER = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "train/explained_variance",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/loss",
    "train/learning_rate",
    "train/n_updates",
    "progress_remaining",
)
_WEB_METRIC_EXCLUDED = {
    "total_timesteps",
    "event",
}


def build_report(target: str, cwd: Path | None = None) -> ReportResult:
    """Build a static HTML report for one run or one research bundle."""

    working_dir = (cwd or Path.cwd()).resolve()

    try:
        run_dir = resolve_run_ref(target, working_dir)
    except CompareError:
        run_dir = None

    if run_dir is not None:
        return _build_run_report(run_dir)

    manifest_path = _resolve_research_manifest(target, working_dir)
    return _build_research_report(manifest_path)


def build_dashboard(path: Path | None = None) -> DashboardResult:
    """Build a static local dashboard for the nearest RLX project."""

    candidate = (path or Path.cwd()).resolve()
    try:
        project_root = find_project_root(candidate)
    except ProjectLookupError as exc:
        raise ReportError(str(exc)) from exc

    try:
        run_list = list_runs(project_root)
    except RunListError as exc:
        raise ReportError(str(exc)) from exc

    research_bundles = _load_research_bundles(project_root)
    output_dir = project_root / "analysis" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"

    html_text = _dashboard_html(
        project_root=project_root,
        runs=list(run_list.runs),
        research_bundles=research_bundles,
    )
    index_path.write_text(html_text, encoding="utf-8")

    return DashboardResult(
        project_root=project_root,
        output_dir=output_dir,
        index_path=index_path,
        runs=len(run_list.runs),
        research_bundles=len(research_bundles),
    )


def build_preview_report(path: Path | None = None) -> ReportResult:
    """Build a design preview report without requiring a trained run."""

    root = ((path or Path.cwd()).resolve() / ".rlx_preview")
    output_dir = root / "analysis" / "reports" / "preview_run_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"

    data = _preview_run_data(root)
    index_path.write_text(_run_report_html(data), encoding="utf-8")

    return ReportResult(
        kind="preview",
        project_root=root,
        output_dir=output_dir,
        index_path=index_path,
        title="Preview Run Report",
        target="preview_run",
    )


def build_preview_research_report(path: Path | None = None) -> ReportResult:
    """Build a design preview research report without requiring trained variants."""

    root = ((path or Path.cwd()).resolve() / ".rlx_preview")
    output_dir = root / "analysis" / "reports" / "preview_research_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"

    manifest = _preview_research_manifest()
    report_payload = {
        "kind": "research",
        "generated_at": _utc_now_iso(),
        "project_root": str(root),
        "manifest": manifest,
        "bundle_dir": "analysis/research/preview_cartpole_001_research_001",
        "images": {},
        "score_rows": _research_score_rows(manifest),
    }
    index_path.write_text(_research_report_html(report_payload), encoding="utf-8")

    return ReportResult(
        kind="preview_research",
        project_root=root,
        output_dir=output_dir,
        index_path=index_path,
        title="Preview Research Report",
        target="preview_research",
    )


def build_preview_dashboard(path: Path | None = None) -> DashboardResult:
    """Build a design preview dashboard without requiring an initialized project."""

    root = ((path or Path.cwd()).resolve() / ".rlx_preview")
    output_dir = root / "analysis" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"

    html_text = _dashboard_page_html(
        project_root=root,
        run_payloads=_preview_runs(),
        research_bundles=_preview_research_bundles(),
    )
    index_path.write_text(html_text, encoding="utf-8")

    return DashboardResult(
        project_root=root,
        output_dir=output_dir,
        index_path=index_path,
        runs=3,
        research_bundles=1,
    )


def resolve_web_app_project(path: Path | None = None, *, demo: bool = False) -> WebAppProject:
    """Resolve the project that backs the connected dashboard app."""

    candidate = (path or Path.cwd()).resolve()
    if demo:
        root = candidate / ".rlx_preview"
        root.mkdir(parents=True, exist_ok=True)
        return WebAppProject(project_root=root, demo=True)

    try:
        project_root = find_project_root(candidate)
    except ProjectLookupError as exc:
        raise ReportError(str(exc)) from exc
    return WebAppProject(project_root=project_root, demo=False)


def web_app_html() -> str:
    """Return the connected dashboard app shell."""

    body = """
    <section class="webapp-topbar">
      <div class="brand-lockup">
        <div class="brand-mark">rx</div>
        <div>
          <p class="eyebrow">RLX Workbench</p>
          <h1>Experiment Console</h1>
          <p class="topbar-subtitle">A local command center for runs, evals, artifacts, and research loops.</p>
        </div>
      </div>
      <div class="topbar-panel">
        <span id="connection-state" class="status-badge">connecting</span>
        <button id="refresh-app" class="app-button primary">Refresh data</button>
      </div>
    </section>

    <section class="webapp-shell">
      <aside class="webapp-sidebar">
        <div class="sidebar-card">
          <span>Project</span>
          <strong id="project-root">loading...</strong>
        </div>
        <div class="nav-label">Views</div>
        <button class="nav-button active" data-view="overview">Overview</button>
        <button class="nav-button" data-view="runs">Runs</button>
        <button class="nav-button" data-view="research">Research</button>
        <div class="sidebar-note">
          Connected to local RLX artifacts through the CLI server. Refresh after training, eval, video, or research finishes.
        </div>
      </aside>

      <div class="webapp-main">
        <section class="workspace-strip">
          <div>
            <span>Mode</span>
            <strong id="project-mode">loading</strong>
          </div>
          <div>
            <span>Source</span>
            <strong>local artifacts</strong>
          </div>
          <div>
            <span>Refresh</span>
            <strong>manual, no cloud sync</strong>
          </div>
        </section>

        <section class="stat-grid app-stat-grid" id="app-stats"></section>

        <section class="panel command-panel app-panel" data-section="overview runs research">
          <div class="panel-title-row">
            <div>
              <p class="eyebrow">CLI Bridge</p>
              <h2>Commands For This Selection</h2>
            </div>
            <div id="app-command-context" class="command-context">Select a run to generate commands.</div>
          </div>
          <div id="app-command-list" class="command-list"></div>
        </section>

        <section class="panel app-panel" data-section="overview runs">
          <div class="panel-title-row">
            <div>
              <p class="eyebrow">Run Explorer</p>
              <h2>Runs and Metrics</h2>
            </div>
            <input id="app-run-search" class="search" placeholder="Search runs, envs, tags..." />
          </div>
          <div class="app-split">
            <aside class="run-pane">
              <div class="run-pane-header">
                <span class="run-pane-label">Runs</span>
                <span class="run-pane-count" id="app-run-count">0</span>
              </div>
              <div id="app-run-list" class="run-list scroll-box"></div>
            </aside>
            <div class="app-split-right">
              <div id="app-run-detail" class="detail-card"></div>
              <div class="metric-block">
                <div class="metric-header">
                  <h3>Training curves</h3>
                  <div class="metric-toolbar">
                    <select id="app-metric-select" class="select"></select>
                    <select id="app-metric-window" class="select compact-select">
                      <option value="all">All points</option>
                      <option value="25">Last 25</option>
                      <option value="50">Last 50</option>
                      <option value="100">Last 100</option>
                    </select>
                    <label class="toggle-label"><input id="app-metric-points" type="checkbox" checked /> points</label>
                  </div>
                </div>
                <div id="app-metric-meta" class="chart-meta"></div>
                <div id="app-metric-chart" class="chart-shell"></div>
              </div>
            </div>
          </div>
        </section>

        <section class="panel app-panel" data-section="overview research">
          <div class="panel-title-row">
            <div>
              <p class="eyebrow">Research Loop</p>
              <h2>Candidate Progress</h2>
            </div>
            <select id="app-research-select" class="select"></select>
          </div>
          <div class="control-strip">
            <div class="control-row">
              <select id="app-research-filter" class="select compact-select">
                <option value="all">All candidates</option>
                <option value="kept">Kept + baseline</option>
                <option value="top">Top scores</option>
                <option value="latest">Latest window</option>
              </select>
              <label class="range-label">
                visible
                <input id="app-research-window" type="range" min="5" max="80" value="40" />
                <span id="app-research-window-label">40</span>
              </label>
            </div>
            <div id="app-research-inspector" class="chart-inspector">Select a research bundle.</div>
          </div>
          <div id="app-research-chart" class="chart-shell"></div>
        </section>

        <section class="panel app-panel" data-section="runs">
          <div class="panel-title-row">
            <div>
              <p class="eyebrow">Artifacts</p>
              <h2>Selected Run Files</h2>
            </div>
          </div>
          <div id="app-artifacts" class="artifact-grid"></div>
        </section>
      </div>
    </section>
    """
    return _page(
        title="RLX Workbench",
        body=body,
        data={"kind": "webapp"},
        script=_web_app_script(),
    )


def web_project_payload(project: WebAppProject) -> dict[str, Any]:
    """Collect live project data for the connected app."""

    if project.demo:
        runs = _preview_runs()
        research_bundles = _preview_research_bundles()
    else:
        try:
            run_list = list_runs(project.project_root)
        except RunListError as exc:
            raise ReportError(str(exc)) from exc
        runs = [_run_payload(run) for run in run_list.runs]
        research_bundles = _load_research_bundles(project.project_root)

    return {
        "kind": "project",
        "generated_at": _utc_now_iso(),
        "demo": project.demo,
        "project_root": str(project.project_root),
        "runs": runs,
        "research_bundles": research_bundles,
        "metric_labels": _METRIC_LABELS,
        "metric_order": _WEB_METRIC_ORDER,
        "best_eval": _best_eval_label_from_payloads(runs),
    }


def web_run_payload(project: WebAppProject, ref: str) -> dict[str, Any]:
    """Collect live data for one run for the connected app."""

    if project.demo:
        run = next((item for item in _preview_runs() if item["run_id"] == ref), _preview_runs()[0])
        data = _preview_run_data(project.project_root)
        data["run"] = run
        data["artifacts"] = [
            {
                "label": "Preview metrics",
                "path": "preview/runs/preview_cartpole_001/metrics.jsonl",
                "href": "#",
            }
        ]
        return data

    try:
        run_dir = resolve_run_ref(ref, project.project_root)
    except CompareError as exc:
        raise ReportError(str(exc)) from exc
    if run_dir is None:
        raise ReportError(f"Run not found: {ref}")

    run = load_run_comparison(run_dir)
    metrics = _load_metric_points(run_dir / METRICS_NAME)
    return {
        "kind": "run",
        "generated_at": _utc_now_iso(),
        "project_root": str(project.project_root),
        "run": _run_payload(run),
        "metrics": metrics,
        "metric_order": _available_metric_order(metrics),
        "metric_labels": _METRIC_LABELS,
        "metric_descriptions": _METRIC_DESCRIPTIONS,
        "evals": _load_eval_rows(run_dir / "eval", project_root=project.project_root),
        "config": run.config_values,
        "artifacts": _run_artifacts_for_app(run, run_dir, project.project_root),
    }


def web_research_payload(project: WebAppProject, bundle: str | None = None) -> dict[str, Any]:
    """Collect live data for one research bundle for the connected app."""

    if project.demo:
        manifest = _preview_research_manifest()
        return {
            "kind": "research",
            "generated_at": _utc_now_iso(),
            "project_root": str(project.project_root),
            "manifest": manifest,
            "bundle_dir": manifest["bundle"],
            "images": {},
            "score_rows": _research_score_rows(manifest),
        }

    bundles = _load_research_bundles(project.project_root)
    if not bundle:
        if not bundles:
            raise ReportError("No research bundles found.")
        bundle = str(bundles[0]["bundle"])

    manifest_path = _resolve_research_manifest(bundle, project.project_root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Could not read research manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != "research_bundle":
        raise ReportError(f"Not an RLX research manifest: {manifest_path}")

    return {
        "kind": "research",
        "generated_at": _utc_now_iso(),
        "project_root": str(project.project_root),
        "manifest": manifest,
        "bundle_dir": str(manifest_path.parent.relative_to(project.project_root)),
        "images": _research_images_for_app(manifest, project.project_root),
        "score_rows": _research_score_rows(manifest),
    }


def web_file_path(project: WebAppProject, relative_path: str) -> Path:
    """Resolve a project-relative artifact path for the connected app."""

    target = (project.project_root / relative_path).resolve()
    root = project.project_root.resolve()
    if target != root and root not in target.parents:
        raise ReportError("File path escapes the RLX project.")
    if not target.is_file():
        raise ReportError(f"File not found: {relative_path}")
    return target


def _build_run_report(run_dir: Path) -> ReportResult:
    run = load_run_comparison(run_dir)
    project_root = run_dir.parent.parent
    output_dir = _next_output_dir(project_root / "analysis" / "reports", f"{run.run_id}_report")
    output_dir.mkdir(parents=True, exist_ok=False)
    index_path = output_dir / "index.html"

    metrics = _load_metric_points(run_dir / METRICS_NAME)
    evals = _load_eval_rows(run_dir / "eval", project_root=project_root)
    artifacts = _run_artifacts(run, run_dir, project_root, output_dir)

    try:
        explanation = explain_metrics(str(run_dir), cwd=project_root)
        metric_explanations = [
            {
                "key": item.key,
                "label": item.label,
                "first": item.first,
                "latest": item.latest,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "trend": item.trend,
                "interpretation": item.interpretation,
            }
            for item in explanation.series
        ]
        notes = [
            {"severity": item.severity, "metric": item.metric, "note": item.note}
            for item in explanation.notes
        ]
    except Exception:
        metric_explanations = []
        notes = []

    data = {
        "kind": "run",
        "generated_at": _utc_now_iso(),
        "project_root": str(project_root),
        "run": _run_payload(run),
        "metrics": metrics,
        "metric_order": _available_metric_order(metrics),
        "metric_labels": _METRIC_LABELS,
        "metric_descriptions": _METRIC_DESCRIPTIONS,
        "metric_explanations": metric_explanations,
        "notes": notes,
        "evals": evals,
        "config": run.config_values,
        "artifacts": artifacts,
    }

    index_path.write_text(_run_report_html(data), encoding="utf-8")
    return ReportResult(
        kind="run",
        project_root=project_root,
        output_dir=output_dir,
        index_path=index_path,
        title=f"Run Report: {run.run_id}",
        target=run.run_id,
    )


def _build_research_report(manifest_path: Path) -> ReportResult:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Could not read research manifest: {manifest_path}") from exc

    if not isinstance(payload, dict) or payload.get("kind") != "research_bundle":
        raise ReportError(f"Not an RLX research manifest: {manifest_path}")

    project_root = _infer_project_root(manifest_path)
    initial = str(payload.get("initial_run_id") or "research")
    output_dir = _next_output_dir(project_root / "analysis" / "reports", f"{initial}_research_report")
    output_dir.mkdir(parents=True, exist_ok=False)
    index_path = output_dir / "index.html"

    report_payload = {
        "kind": "research",
        "generated_at": _utc_now_iso(),
        "project_root": str(project_root),
        "manifest": payload,
        "bundle_dir": str(manifest_path.parent.relative_to(project_root)),
        "images": _research_images(payload, project_root, output_dir),
        "score_rows": _research_score_rows(payload),
    }
    index_path.write_text(_research_report_html(report_payload), encoding="utf-8")

    return ReportResult(
        kind="research",
        project_root=project_root,
        output_dir=output_dir,
        index_path=index_path,
        title=f"Research Report: {initial}",
        target=initial,
    )


def _dashboard_html(
    *,
    project_root: Path,
    runs: list[RunComparison],
    research_bundles: list[dict[str, Any]],
) -> str:
    run_payloads = [_run_payload(run) for run in runs]
    return _dashboard_page_html(
        project_root=project_root,
        run_payloads=run_payloads,
        research_bundles=research_bundles,
    )


def _dashboard_page_html(
    *,
    project_root: Path,
    run_payloads: list[dict[str, Any]],
    research_bundles: list[dict[str, Any]],
) -> str:
    data = {
        "kind": "dashboard",
        "generated_at": _utc_now_iso(),
        "project_root": str(project_root),
        "runs": run_payloads,
        "research_bundles": research_bundles,
        "metric_labels": _METRIC_LABELS,
        "metric_order": _WEB_METRIC_ORDER,
    }
    body = f"""
    <section class="hero">
      <div>
        <p class="eyebrow">Project cockpit</p>
        <h1>Experiment Dashboard</h1>
        <p class="lede">A local control room for runs, evals, research bundles, and performance signals.</p>
      </div>
      <div class="hero-card">
        <span>Project</span>
        <strong>{_escape(str(project_root))}</strong>
      </div>
    </section>

    <section class="stat-grid">
      {_stat_card("Runs", len(run_payloads))}
      {_stat_card("Completed", sum(1 for run in run_payloads if run.get("status") == "completed"))}
      {_stat_card("Research Bundles", len(research_bundles))}
      {_stat_card("Best Eval", _best_eval_label_from_payloads(run_payloads))}
    </section>

    <section class="panel split-panel">
      <div>
        <div class="panel-title-row">
          <div>
            <p class="eyebrow">Runs</p>
            <h2>Run Explorer</h2>
          </div>
          <input id="run-search" class="search" placeholder="Search runs, envs, tags..." />
        </div>
        <div id="run-list" class="run-list"></div>
      </div>
      <div id="run-detail" class="detail-card"></div>
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Scoreboard</p>
          <h2>Best Known Eval By Run</h2>
        </div>
      </div>
      <div id="dashboard-score-chart" class="chart-shell"></div>
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Research</p>
          <h2>Research Bundles</h2>
        </div>
      </div>
      <div id="research-list" class="research-list"></div>
    </section>
    """
    return _page(
        title="RLX Dashboard",
        body=body,
        data=data,
        script=_dashboard_script(),
    )


def _run_report_html(data: dict[str, Any]) -> str:
    run = data["run"]
    body = f"""
    <section class="hero">
      <div>
        <p class="eyebrow">Run report</p>
        <h1>{_escape(run["run_id"])}</h1>
        <p class="lede">{_escape(run.get("environment") or "Unknown environment")} / {_escape(run.get("status") or "unknown")} / generated {_escape(data["generated_at"])}</p>
      </div>
      <div class="hero-card">
        <span>Run directory</span>
        <strong>{_escape(run["run_dir"])}</strong>
      </div>
    </section>

    <section class="stat-grid">
      {_stat_card("Final Train", _fmt_optional(run.get("final_rollout_reward")))}
      {_stat_card("Best Train", _fmt_optional(run.get("best_rollout_reward")))}
      {_stat_card("Best Eval", _eval_label(run.get("best_eval")))}
      {_stat_card("Timesteps", _fmt_optional(run.get("total_timesteps"), digits=0))}
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Interactive Metrics</p>
          <h2>Training Curves</h2>
        </div>
        <div class="control-row">
          <select id="metric-select" class="select"></select>
          <select id="metric-window" class="select compact-select">
            <option value="all">All points</option>
            <option value="25">Last 25</option>
            <option value="50">Last 50</option>
            <option value="100">Last 100</option>
          </select>
          <label class="toggle-label">
            <input id="metric-points" type="checkbox" checked />
            points
          </label>
        </div>
      </div>
      <div id="metric-meta" class="chart-meta"></div>
      <div id="metric-chart" class="chart-shell"></div>
    </section>

    <section class="panel two-column">
      <div>
        <p class="eyebrow">Metric Notes</p>
        <h2>What Changed</h2>
        <div id="metric-notes" class="note-list"></div>
      </div>
      <div>
        <p class="eyebrow">Artifacts</p>
        <h2>Files</h2>
        <div class="artifact-grid">{_artifact_links(data["artifacts"])}</div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Evaluation</p>
          <h2>Eval Results</h2>
        </div>
      </div>
      {_eval_table(data["evals"])}
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Config Snapshot</p>
          <h2>Important Config Values</h2>
        </div>
      </div>
      {_config_table(data["config"])}
    </section>
    """
    return _page(
        title=f"RLX Run Report - {run['run_id']}",
        body=body,
        data=data,
        script=_run_report_script(),
    )


def _research_report_html(data: dict[str, Any]) -> str:
    manifest = data["manifest"]
    champion = manifest.get("champion", {})
    images = data["images"]
    body = f"""
    <section class="hero">
      <div>
        <p class="eyebrow">Research report</p>
        <h1>{_escape(str(manifest.get("initial_run_id", "research")))}</h1>
        <p class="lede">Mode {_escape(str(manifest.get("mode", "unknown")))} / stop: {_escape(str(manifest.get("stop_reason", "unknown")))}</p>
      </div>
      <div class="hero-card">
        <span>Champion</span>
        <strong>{_escape(str(champion.get("run_id", "n/a")))}</strong>
        <small>{_escape(_fmt_optional(champion.get("score")))} · {_escape(str(champion.get("score_source", "no score")))}</small>
      </div>
    </section>

    <section class="stat-grid">
      {_stat_card("Rounds", len(manifest.get("rounds", [])))}
      {_stat_card("Experiments", len(data["score_rows"]))}
      {_stat_card("Promotions", sum(1 for row in data["score_rows"] if row.get("promoted")))}
      {_stat_card("Champion Score", _fmt_optional(champion.get("score")))}
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Interactive Progress</p>
          <h2>Research Score Over Experiments</h2>
        </div>
      </div>
      <div class="control-strip">
        <div class="control-row">
          <select id="research-filter" class="select compact-select">
            <option value="all">All candidates</option>
            <option value="kept">Kept + baseline</option>
            <option value="top">Top scores</option>
            <option value="latest">Latest window</option>
          </select>
          <label class="range-label">
            visible
            <input id="research-window" type="range" min="5" max="{max(5, len(data["score_rows"]))}" value="{min(40, max(5, len(data["score_rows"])))}" />
            <span id="research-window-label">{min(40, max(5, len(data["score_rows"])))}</span>
          </label>
        </div>
        <div id="research-inspector" class="chart-inspector">Move across the chart to inspect candidates.</div>
      </div>
      <div id="research-progress-chart" class="chart-shell"></div>
    </section>

    <section class="panel image-grid">
      {_image_card("Progress Plot", images.get("progress"))}
      {_image_card("Scoreboard Plot", images.get("scoreboard"))}
    </section>

    <section class="panel">
      <div class="panel-title-row">
        <div>
          <p class="eyebrow">Rounds</p>
          <h2>Promotions And Candidates</h2>
        </div>
        <input id="round-search" class="search" placeholder="Search runs, mutations, advisors..." />
      </div>
      <div id="round-list" class="round-list"></div>
    </section>
    """
    return _page(
        title=f"RLX Research Report - {manifest.get('initial_run_id', 'research')}",
        body=body,
        data=data,
        script=_research_report_script(),
    )


def _page(*, title: str, body: str, data: dict[str, Any], script: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape(title)}</title>
  <style>{_styles()}</style>
</head>
<body>
  <main class="page-shell app-frame">
    {body}
  </main>
  <script id="rlx-data" type="application/json">{_escape_json(data)}</script>
  <script>{_shared_script()}</script>
  <script>{script}</script>
</body>
</html>
"""


def _styles() -> str:
    return """
:root {
  --bg: #f4f2ec;
  --paper: #ffffff;
  --paper-2: #faf8f2;
  --paper-3: #f0ede4;
  --ink: #171717;
  --ink-2: #3a3a38;
  --muted: #6b6a62;
  --faint: #a19f93;
  --line: #1a1a1a;
  --soft-line: #e3dfd1;
  --hairline: #ece8db;
  --accent: #245d52;
  --accent-soft: #e5ede9;
  --accent-2: #8a4b19;
  --good: #2f6c3f;
  --bad: #a13b2b;
  --focus: #245d52;
}
* { box-sizing: border-box; }
html { color-scheme: light; }
body {
  margin: 0;
  color: var(--ink);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Avenir Next", ui-sans-serif, sans-serif;
  font-feature-settings: "cv11", "ss01";
  background: var(--bg);
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
  font-size: 14px;
  line-height: 1.45;
}
.page-shell {
  width: min(1360px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 24px 0 48px;
  position: relative;
}
.app-frame { border: none; }

/* Topbar */
.webapp-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
  margin-bottom: 16px;
  padding: 20px 24px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--paper);
}
.brand-lockup {
  display: flex;
  align-items: center;
  min-width: 0;
  gap: 16px;
  flex: 1 1 auto;
}
.brand-mark {
  display: grid;
  place-items: center;
  flex: 0 0 48px;
  width: 48px;
  height: 48px;
  border-radius: 8px;
  background: var(--ink);
  color: var(--paper);
  font-weight: 800;
  font-size: 1.05rem;
  letter-spacing: -0.04em;
  text-transform: uppercase;
}
.webapp-topbar h1 {
  margin: 0;
  font-size: clamp(1.5rem, 2.2vw, 2rem);
  letter-spacing: -0.025em;
  line-height: 1.15;
  font-weight: 700;
  overflow-wrap: anywhere;
}
.topbar-subtitle {
  max-width: 680px;
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.5;
}
.topbar-panel {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
}

/* Status + buttons */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 11px;
  border: 1px solid var(--soft-line);
  border-radius: 6px;
  color: var(--good);
  background: var(--paper-2);
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0;
  text-transform: none;
  white-space: nowrap;
}
.status-badge::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  flex: 0 0 6px;
}
.status-badge.error { color: var(--bad); }
.app-button, .nav-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--line);
  background: var(--paper);
  color: var(--ink);
  border-radius: 6px;
  padding: 8px 14px;
  font-weight: 600;
  font-size: 0.86rem;
  line-height: 1.2;
  cursor: pointer;
  font-family: inherit;
  transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
  white-space: nowrap;
}
.app-button:hover, .nav-button:hover {
  background: var(--ink);
  color: var(--paper);
}
.app-button.primary {
  background: var(--ink);
  color: var(--paper);
}
.app-button.primary:hover {
  background: #000;
}

/* Shell */
.webapp-shell {
  display: grid;
  grid-template-columns: 224px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.webapp-sidebar {
  position: sticky;
  top: 16px;
  display: grid;
  gap: 8px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--paper);
  padding: 16px;
}
.sidebar-card {
  border: 1px solid var(--soft-line);
  border-radius: 6px;
  padding: 12px;
  background: var(--paper-2);
  overflow: hidden;
}
.sidebar-card span {
  display: block;
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.sidebar-card strong {
  display: block;
  margin-top: 6px;
  font-size: 0.92rem;
  font-weight: 600;
  overflow-wrap: anywhere;
  line-height: 1.4;
}
.nav-button {
  width: 100%;
  justify-content: flex-start;
  text-align: left;
  padding: 9px 12px;
  position: relative;
}
.nav-button.active {
  background: var(--ink);
  color: var(--paper);
  border-color: var(--ink);
}
.nav-label {
  margin: 6px 0 2px;
  color: var(--muted);
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.sidebar-note {
  margin-top: 4px;
  padding-top: 12px;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.55;
  border-top: 1px solid var(--soft-line);
}

.webapp-main { min-width: 0; }
.workspace-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.workspace-strip > div {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--paper);
  padding: 14px 16px;
}
.workspace-strip span {
  display: block;
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.workspace-strip strong {
  display: block;
  margin-top: 4px;
  font-size: 0.95rem;
  font-weight: 600;
  overflow-wrap: anywhere;
}
.app-stat-grid { margin-top: 0; }
.app-panel[data-hidden="true"] { display: none; }
.command-panel {
  background: #111;
  color: #f7f4ea;
  border-color: #111;
}
.command-panel .eyebrow {
  color: #9fc5b8;
}
.command-panel h2 {
  color: #fffaf0;
}
.command-context {
  color: #c8c1b2;
  font-size: 0.86rem;
  text-align: right;
}
.command-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.command-card {
  min-width: 0;
  border: 1px solid rgba(255, 250, 240, 0.18);
  border-radius: 8px;
  background: #191918;
  padding: 12px;
}
.command-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 9px;
}
.command-card strong {
  color: #fffaf0;
  font-size: 0.9rem;
}
.command-card span {
  color: #a9a294;
  font-size: 0.8rem;
}
.command-code {
  display: block;
  width: 100%;
  overflow-x: auto;
  border: 1px solid rgba(255, 250, 240, 0.12);
  border-radius: 6px;
  background: #0b0b0b;
  color: #f5e8c8;
  padding: 10px;
  font-family: "SFMono-Regular", "Cascadia Code", "Liberation Mono", monospace;
  font-size: 0.82rem;
  white-space: nowrap;
}
.copy-command {
  flex: 0 0 auto;
  border: 1px solid rgba(255, 250, 240, 0.28);
  border-radius: 6px;
  background: transparent;
  color: #fffaf0;
  padding: 5px 8px;
  font-family: inherit;
  font-size: 0.76rem;
  font-weight: 700;
  cursor: pointer;
}
.copy-command:hover {
  border-color: #fffaf0;
  background: rgba(255, 250, 240, 0.08);
}
.app-split {
  display: grid;
  grid-template-columns: minmax(260px, 0.72fr) minmax(0, 1.4fr);
  gap: 16px;
  align-items: stretch;
}
.app-split-right {
  display: grid;
  gap: 14px;
  min-width: 0;
}
.run-pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border: 1px solid var(--soft-line);
  border-radius: 8px;
  background: var(--paper-2);
  padding: 10px 12px 12px;
}
.run-pane-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 2px 10px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--soft-line);
}
.run-pane-label {
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.run-pane-count {
  min-width: 24px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--paper);
  border: 1px solid var(--soft-line);
  color: var(--ink-2);
  font-size: 0.75rem;
  font-weight: 700;
  text-align: center;
}
.run-pane .run-list.scroll-box {
  flex: 1 1 auto;
  min-height: 0;
  max-height: none;
  overflow: auto;
  padding-right: 4px;
  margin: 0;
}
.scroll-box {
  max-height: 640px;
  overflow: auto;
  padding-right: 4px;
  scrollbar-color: var(--soft-line) transparent;
}
.metric-block {
  border: 1px solid var(--soft-line);
  border-radius: 8px;
  background: var(--paper-2);
  padding: 14px 16px 16px;
}
.metric-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.metric-header h3 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--ink);
}
.metric-toolbar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
}
.metric-toolbar .select { min-width: 160px; padding: 7px 10px; font-size: 0.85rem; }
.metric-toolbar .compact-select { min-width: 120px; }

/* Hero (reports) */
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 0.5fr);
  gap: 16px;
  align-items: stretch;
  margin-bottom: 16px;
}
.hero h1 {
  margin: 0;
  max-width: 920px;
  font-size: clamp(1.8rem, 3.2vw, 2.6rem);
  letter-spacing: -0.025em;
  line-height: 1.05;
  font-weight: 700;
  text-wrap: balance;
  overflow-wrap: anywhere;
}
.lede {
  color: var(--muted);
  font-size: 0.95rem;
  max-width: 720px;
  line-height: 1.55;
  margin-top: 10px;
}
.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

/* Cards + panels */
.hero-card, .panel, .stat-card {
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: 10px;
}
.hero-card {
  padding: 20px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  overflow: hidden;
}
.hero-card span, .hero-card small { color: var(--muted); font-size: 0.82rem; }
.hero-card strong {
  word-break: break-word;
  overflow-wrap: anywhere;
  font-size: 0.95rem;
  font-weight: 600;
  margin-top: 6px;
  line-height: 1.45;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 12px 0;
}
.stat-card {
  min-height: 96px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.stat-card span {
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
}
.stat-card strong {
  display: block;
  margin-top: 10px;
  font-size: clamp(1.5rem, 2.4vw, 1.95rem);
  letter-spacing: -0.03em;
  line-height: 1;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.panel {
  padding: 20px;
  margin: 12px 0;
}
.app-panel { position: relative; }
.panel h2 {
  margin: 0;
  font-size: 1.2rem;
  letter-spacing: -0.02em;
  font-weight: 700;
}
.panel-title-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.split-panel { display: grid; grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.2fr); gap: 20px; }
.two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.image-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }

.chart-shell {
  position: relative;
  min-height: 380px;
  border-radius: 8px;
  border: 1px solid var(--soft-line);
  background: var(--paper-2);
  overflow: hidden;
}
.chart-shell svg { width: 100%; height: 100%; display: block; min-height: 380px; }

.search, .select {
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: 6px;
  padding: 9px 12px;
  min-width: 220px;
  color: var(--ink);
  outline: none;
  font-family: inherit;
  font-size: 0.88rem;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.search:focus, .select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(36, 93, 82, 0.14);
}
.compact-select { min-width: 140px; }
.control-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  flex-wrap: wrap;
}
.control-strip {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 16px;
  margin: 0 0 14px;
  padding: 12px 14px;
  border: 1px solid var(--soft-line);
  border-radius: 8px;
  background: var(--paper-2);
}
.toggle-label, .range-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 0.85rem;
}
.range-label input { accent-color: var(--accent); width: 160px; }
.chart-meta, .chart-inspector {
  color: var(--muted);
  font-size: 0.86rem;
}
.chart-meta { margin: -4px 0 12px; }
.chart-tooltip {
  position: absolute;
  pointer-events: none;
  z-index: 5;
  display: none;
  max-width: 280px;
  padding: 9px 11px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--paper);
  color: var(--ink);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
  font-size: 0.84rem;
  line-height: 1.4;
}
.search::placeholder { color: var(--faint); }
.select option { color: var(--ink); background: var(--paper); }

.run-list, .research-list, .round-list, .note-list { display: grid; gap: 8px; }
.run-button, .research-card, .round-card, .note-card, .detail-card, .artifact-link {
  border: 1px solid var(--soft-line);
  background: var(--paper-2);
  border-radius: 8px;
}
.run-button {
  position: relative;
  width: 100%;
  text-align: left;
  padding: 13px 14px 13px 16px;
  cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease;
  overflow: hidden;
  font-family: inherit;
  color: inherit;
}
.run-button:hover {
  border-color: var(--ink-2);
  background: var(--paper);
}
.run-button.active {
  border-color: var(--line);
  background: var(--paper);
}
.run-button.active::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: 3px;
  background: var(--accent);
}
.run-button strong, .research-card strong {
  display: block;
  font-size: 0.95rem;
  font-weight: 600;
}
.run-button span, .research-card span, .round-card span, .muted {
  color: var(--muted);
  font-size: 0.83rem;
}
.detail-card {
  padding: 20px;
  background: var(--paper);
  border-color: var(--line);
}
.detail-card h2 {
  margin: 0;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
  font-weight: 700;
}
.detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 16px;
}
.mini-card {
  border-radius: 6px;
  background: var(--paper-2);
  border: 1px solid var(--soft-line);
  padding: 12px;
}
.mini-card span {
  display: block;
  color: var(--muted);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}
.mini-card strong {
  display: block;
  margin-top: 6px;
  font-size: 0.95rem;
  font-weight: 600;
}

.artifact-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  align-items: stretch;
}
.artifact-link {
  display: block;
  padding: 14px;
  color: var(--ink);
  text-decoration: none;
  min-width: 0;
  overflow-wrap: anywhere;
}
.artifact-link strong { display: block; margin-bottom: 4px; font-weight: 600; }
.artifact-link:hover { border-color: var(--line); background: var(--paper); }

.table-wrap {
  overflow-x: auto;
  border-radius: 8px;
  border: 1px solid var(--soft-line);
}
table { width: 100%; border-collapse: collapse; background: var(--paper); }
th, td {
  padding: 11px 13px;
  text-align: left;
  border-bottom: 1px solid var(--hairline);
  vertical-align: top;
  font-size: 0.88rem;
}
tr:last-child td { border-bottom: none; }
th {
  color: var(--muted);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  background: var(--paper-2);
}

.pill {
  display: inline-flex;
  align-items: center;
  border-radius: 6px;
  padding: 3px 8px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.note-card, .research-card, .round-card { padding: 14px; }
.image-card img {
  width: 100%;
  display: block;
  border-radius: 8px;
  border: 1px solid var(--soft-line);
  background: #fff;
}
.empty {
  color: var(--muted);
  padding: 24px;
  text-align: center;
  border-radius: 8px;
  background: var(--paper-2);
  border: 1px dashed var(--soft-line);
  font-size: 0.9rem;
}

@media (max-width: 960px) {
  .webapp-shell, .app-split { grid-template-columns: 1fr; }
  .webapp-sidebar { position: static; }
  .webapp-topbar { flex-direction: column; align-items: flex-start; }
  .topbar-panel { width: 100%; justify-content: flex-start; }
  .workspace-strip { grid-template-columns: 1fr; }
  .command-list { grid-template-columns: 1fr; }
  .command-context { text-align: left; }
  .hero, .split-panel, .two-column, .image-grid { grid-template-columns: 1fr; }
  .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .panel-title-row { flex-direction: column; align-items: flex-start; }
  .control-strip { grid-template-columns: 1fr; }
  .control-row { justify-content: flex-start; width: 100%; }
  .search, .select { width: 100%; min-width: 0; }
}
"""


def _shared_script() -> str:
    return """
const RLX = JSON.parse(document.getElementById("rlx-data").textContent);
function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}
function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[ch]));
}
function scoreOfRun(run) {
  if (run.best_eval && run.best_eval.mean_reward !== null) return run.best_eval.mean_reward;
  if (run.final_rollout_reward !== null) return run.final_rollout_reward;
  return null;
}
function attachPointTooltip(container, scaledPoints, formatter, onHover) {
  const svg = container.querySelector("svg");
  if (!svg || !scaledPoints.length) return;
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  container.appendChild(tooltip);
  svg.addEventListener("mousemove", (event) => {
    const svgBox = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const svgX = (event.clientX - svgBox.left) * (viewBox.width / svgBox.width);
    let nearest = scaledPoints[0];
    let nearestDistance = Math.abs(nearest.x - svgX);
    for (const point of scaledPoints.slice(1)) {
      const distance = Math.abs(point.x - svgX);
      if (distance < nearestDistance) {
        nearest = point;
        nearestDistance = distance;
      }
    }
    const containerBox = container.getBoundingClientRect();
    tooltip.innerHTML = formatter(nearest.row);
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(event.clientX - containerBox.left + 14, containerBox.width - 300)}px`;
    tooltip.style.top = `${Math.max(event.clientY - containerBox.top - 18, 12)}px`;
    if (onHover) onHover(nearest.row);
  });
  svg.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
  });
}
function lineChart(container, points, opts = {}) {
  if (!container) return;
  if (!points || points.length < 2) {
    container.innerHTML = '<div class="empty">No chartable data found.</div>';
    return;
  }
  const width = 1040, height = 420;
  const pad = {left: 70, right: 30, top: 54, bottom: 62};
  const xs = points.map((p) => Number(p.x));
  const ys = points.map((p) => Number(p.y));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const xScale = (x) => pad.left + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad.left - pad.right);
  const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 1)) * (height - pad.top - pad.bottom);
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(Number(p.x)).toFixed(2)} ${yScale(Number(p.y)).toFixed(2)}`).join(" ");
  const circles = opts.showPoints === false ? "" : points.map((p) => `<circle cx="${xScale(Number(p.x)).toFixed(2)}" cy="${yScale(Number(p.y)).toFixed(2)}" r="4"><title>${escapeText(p.label || "")} step ${fmt(p.x, 0)}: ${fmt(p.y)}</title></circle>`).join("");
  const yTicks = [minY, minY + (maxY-minY)*0.25, minY + (maxY-minY)*0.5, minY + (maxY-minY)*0.75, maxY];
  const grid = yTicks.map((y) => `<line x1="${pad.left}" x2="${width-pad.right}" y1="${yScale(y)}" y2="${yScale(y)}" stroke="#d6d1c5" /><text x="18" y="${yScale(y)+4}" font-size="12" fill="#6c6a62">${fmt(y)}</text>`).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img">
    ${grid}
    <line x1="${pad.left}" x2="${width-pad.right}" y1="${height-pad.bottom}" y2="${height-pad.bottom}" stroke="#252525" />
    <line x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height-pad.bottom}" stroke="#252525" />
    <text x="${pad.left}" y="30" fill="#171717" font-size="18" font-weight="850">${escapeText(opts.title || "")}</text>
    <path d="${d}" fill="none" stroke="${opts.color || "#315f56"}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />
    <g fill="${opts.pointColor || "#8a4b19"}">${circles}</g>
    <text x="${width/2}" y="${height-16}" fill="#6c6a62" font-size="12" text-anchor="middle">${escapeText(opts.xLabel || "step")}</text>
  </svg>`;
  attachPointTooltip(
    container,
    points.map((p) => ({x: xScale(Number(p.x)), y: yScale(Number(p.y)), row: p})),
    (p) => `<strong>${escapeText(p.label || opts.title || "metric")}</strong><br>step ${fmt(p.x, 0)}<br>value ${fmt(p.y, 4)}`
  );
}
function barChart(container, rows, opts = {}) {
  if (!container) return;
  rows = (rows || []).filter((row) => row.value !== null && row.value !== undefined);
  if (!rows.length) {
    container.innerHTML = '<div class="empty">No score data found.</div>';
    return;
  }
  const width = 1040, height = 420;
  const pad = {left: 70, right: 30, top: 54, bottom: 95};
  let minY = Math.min(0, ...rows.map((row) => Number(row.value)));
  let maxY = Math.max(0, ...rows.map((row) => Number(row.value)));
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const slot = (width - pad.left - pad.right) / rows.length;
  const barW = Math.max(8, Math.min(42, slot * 0.62));
  const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 1)) * (height - pad.top - pad.bottom);
  const zeroY = yScale(0);
  const bars = rows.map((row, i) => {
    const x = pad.left + i * slot + (slot - barW) / 2;
    const y = yScale(Math.max(Number(row.value), 0));
    const h = Math.abs(yScale(Number(row.value)) - zeroY);
    const top = Number(row.value) >= 0 ? y : zeroY;
    const color = row.promoted ? "#2f6c3f" : (row.role === "baseline" ? "#8a4b19" : "#315f56");
    return `<g>
      <rect x="${x}" y="${top}" width="${barW}" height="${Math.max(h, 2)}" rx="6" fill="${color}" opacity="0.88"><title>${escapeText(row.label)}: ${fmt(row.value)}</title></rect>
      <text x="${x + barW / 2}" y="${top - 6}" text-anchor="middle" font-size="10" fill="#171717">${fmt(row.value, 1)}</text>
      <text transform="translate(${x + barW / 2}, ${height - pad.bottom + 18}) rotate(42)" font-size="10" fill="#6c6a62">${escapeText(row.label)}</text>
    </g>`;
  }).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img">
    <text x="${pad.left}" y="30" fill="#171717" font-size="18" font-weight="850">${escapeText(opts.title || "Scores")}</text>
    <line x1="${pad.left}" x2="${width-pad.right}" y1="${zeroY}" y2="${zeroY}" stroke="#252525" />
    ${bars}
  </svg>`;
}
function researchProgressChart(container, rows) {
  if (!container) return;
  rows = (rows || []).filter((row) => row.score !== null && row.score !== undefined);
  if (!rows.length) {
    container.innerHTML = '<div class="empty">No research score data found.</div>';
    return;
  }
  const width = 1040, height = 430;
  const pad = {left: 78, right: 34, top: 58, bottom: 64};
  const xs = rows.map((row) => Number(row.experiment));
  const ys = rows.map((row) => Number(row.score));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  const margin = Math.max((maxY - minY) * 0.08, 1);
  minY -= margin;
  maxY += margin;
  const xScale = (x) => pad.left + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad.left - pad.right);
  const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 1)) * (height - pad.top - pad.bottom);
  let running = -Infinity;
  const bestPoints = rows.map((row) => {
    running = Math.max(running, Number(row.score));
    return {x: Number(row.experiment), y: running};
  });
  const bestPath = bestPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(2)} ${yScale(p.y).toFixed(2)}`).join(" ");
  const yTicks = [minY, minY + (maxY-minY)*0.25, minY + (maxY-minY)*0.5, minY + (maxY-minY)*0.75, maxY];
  const grid = yTicks.map((y) => `<line x1="${pad.left}" x2="${width-pad.right}" y1="${yScale(y)}" y2="${yScale(y)}" stroke="#d6d1c5" /><text x="18" y="${yScale(y)+4}" font-size="12" fill="#6c6a62">${fmt(y, 1)}</text>`).join("");
  const bestScore = Math.max(...ys);
  const bestIndex = rows.findIndex((row) => Number(row.score) === bestScore);
  const dots = rows.map((row, index) => {
    const promoted = row.promoted || row.role === "baseline";
    const best = index === bestIndex;
    const color = best ? "#2f6c3f" : promoted ? "#315f56" : "#9a9588";
    const radius = best ? 7 : promoted ? 6 : 4;
    const opacity = promoted || best ? 1 : 0.46;
    const shouldLabel = promoted || best;
    const label = shouldLabel ? `<text x="${xScale(Number(row.experiment)) + 8}" y="${yScale(Number(row.score)) - 9}" font-size="11" fill="${color}" font-weight="750">${escapeText(row.run_id || "")} ${fmt(row.score, 1)}</text>` : "";
    return `<g><circle cx="${xScale(Number(row.experiment)).toFixed(2)}" cy="${yScale(Number(row.score)).toFixed(2)}" r="${radius}" fill="${color}" opacity="${opacity}"><title>${escapeText(row.run_id || "")}: ${fmt(row.score, 2)}</title></circle>${label}</g>`;
  }).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img">
    ${grid}
    <text x="${pad.left}" y="30" fill="#171717" font-size="18" font-weight="850">Research trajectory</text>
    <text x="${width - pad.right}" y="30" fill="#6c6a62" font-size="12" text-anchor="end">higher score is better</text>
    <path d="${bestPath}" fill="none" stroke="#2f6c3f" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.84" />
    ${dots}
    <line x1="${pad.left}" x2="${width-pad.right}" y1="${height-pad.bottom}" y2="${height-pad.bottom}" stroke="#252525" />
    <text x="${width/2}" y="${height-18}" fill="#6c6a62" font-size="12" text-anchor="middle">experiment number</text>
  </svg>`;
  attachPointTooltip(
    container,
    rows.map((row) => ({
      x: xScale(Number(row.experiment)),
      y: yScale(Number(row.score)),
      row,
    })),
    (row) => `<strong>${escapeText(row.run_id || "candidate")}</strong><br>experiment ${fmt(row.experiment, 0)}<br>score ${fmt(row.score, 3)}<br>${escapeText(row.source || row.role || "")}`,
    (row) => {
      const inspector = document.getElementById("research-inspector") || document.getElementById("app-research-inspector");
      if (inspector) {
        inspector.textContent = `${row.run_id || "candidate"} / score ${fmt(row.score, 3)} / ${row.source || row.role || "n/a"}`;
      }
    }
  );
}
"""


def _run_report_script() -> str:
    return """
const metricSelect = document.getElementById("metric-select");
const metricWindow = document.getElementById("metric-window");
const metricPoints = document.getElementById("metric-points");
const metricMeta = document.getElementById("metric-meta");
const preferredMetricKeys = RLX.metric_order || [];
const metricKeys = preferredMetricKeys
  .filter((key) => (RLX.metrics[key] || []).length >= 2)
  .concat(Object.keys(RLX.metrics || {}).filter((key) => !preferredMetricKeys.includes(key) && (RLX.metrics[key] || []).length >= 2));
metricSelect.innerHTML = metricKeys.map((key) => `<option value="${escapeText(key)}">${escapeText(RLX.metric_labels[key] || key)}</option>`).join("");
function renderMetric() {
  if (!metricKeys.length) {
    document.getElementById("metric-chart").innerHTML = '<div class="empty">No multi-point training metrics found yet.</div>';
    return;
  }
  const key = metricSelect.value || metricKeys[0];
  const rawPoints = (RLX.metrics[key] || []).map((p) => ({x: p.step, y: p.value, label: RLX.metric_labels[key] || key}));
  const windowValue = metricWindow.value;
  const points = windowValue === "all" ? rawPoints : rawPoints.slice(-Number(windowValue));
  metricMeta.textContent = `${RLX.metric_labels[key] || key} / ${points.length} of ${rawPoints.length} points shown`;
  lineChart(
    document.getElementById("metric-chart"),
    points,
    {
      title: RLX.metric_labels[key] || key,
      color: "#315f56",
      showPoints: metricPoints.checked,
    }
  );
}
metricSelect.addEventListener("change", renderMetric);
metricWindow.addEventListener("change", renderMetric);
metricPoints.addEventListener("change", renderMetric);
renderMetric();
const notes = document.getElementById("metric-notes");
const explanationRows = RLX.metric_explanations || [];
const noteRows = RLX.notes || [];
notes.innerHTML = [...noteRows.map((item) => ({title: item.severity, text: item.note})), ...explanationRows.slice(0, 6).map((item) => ({title: `${item.label} · ${item.trend}`, text: item.interpretation}))].map((item) => `<div class="note-card"><span class="pill">${escapeText(item.title)}</span><p>${escapeText(item.text)}</p></div>`).join("") || '<div class="empty">No metric notes available.</div>';
"""


def _dashboard_script() -> str:
    return """
const runList = document.getElementById("run-list");
const runDetail = document.getElementById("run-detail");
const search = document.getElementById("run-search");
let selectedRun = RLX.runs[0] || null;
function runSearchText(run) {
  return [run.run_id, run.run_name, run.environment, run.status, (run.tags || []).join(" ")].join(" ").toLowerCase();
}
function renderRuns() {
  const q = (search.value || "").toLowerCase();
  const filtered = RLX.runs.filter((run) => runSearchText(run).includes(q));
  runList.innerHTML = filtered.map((run) => `<button class="run-button ${selectedRun && selectedRun.run_id === run.run_id ? "active" : ""}" data-run="${escapeText(run.run_id)}"><strong>${escapeText(run.run_id)}</strong><span>${escapeText(run.environment || "unknown")} · ${escapeText(run.status || "unknown")} · score ${fmt(scoreOfRun(run))}</span></button>`).join("") || '<div class="empty">No runs match the current search.</div>';
  runList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      selectedRun = RLX.runs.find((run) => run.run_id === button.dataset.run);
      renderRuns();
      renderDetail();
    });
  });
}
function renderDetail() {
  if (!selectedRun) {
    runDetail.innerHTML = '<div class="empty">No runs found yet.</div>';
    return;
  }
  runDetail.innerHTML = `<p class="eyebrow">Selected Run</p><h2>${escapeText(selectedRun.run_id)}</h2><p class="muted">${escapeText(selectedRun.run_dir)}</p>
    <div class="detail-grid">
      <div class="mini-card"><span>Status</span><strong>${escapeText(selectedRun.status || "—")}</strong></div>
      <div class="mini-card"><span>Environment</span><strong>${escapeText(selectedRun.environment || "—")}</strong></div>
      <div class="mini-card"><span>Final Train</span><strong>${fmt(selectedRun.final_rollout_reward)}</strong></div>
      <div class="mini-card"><span>Best Eval</span><strong>${fmt(selectedRun.best_eval && selectedRun.best_eval.mean_reward)}</strong></div>
      <div class="mini-card"><span>Device</span><strong>${escapeText(selectedRun.resolved_device || selectedRun.requested_device || "—")}</strong></div>
      <div class="mini-card"><span>Timesteps</span><strong>${fmt(selectedRun.total_timesteps, 0)}</strong></div>
    </div>`;
}
function renderScoreChart() {
  const rows = RLX.runs.slice().reverse().map((run) => ({label: run.run_id, value: scoreOfRun(run), role: "run"}));
  barChart(document.getElementById("dashboard-score-chart"), rows, {title: "Run Scores"});
}
function renderResearch() {
  const list = document.getElementById("research-list");
  list.innerHTML = (RLX.research_bundles || []).map((item) => `<div class="research-card"><strong>${escapeText(item.initial_run_id || item.bundle)}</strong><span>${escapeText(item.mode || "unknown")} · champion ${escapeText(item.champion_run_id || "n/a")} · score ${fmt(item.champion_score)} · ${escapeText(item.bundle)}</span></div>`).join("") || '<div class="empty">No research bundles yet.</div>';
}
search.addEventListener("input", renderRuns);
renderRuns();
renderDetail();
renderScoreChart();
renderResearch();
"""


def _web_app_script() -> str:
    return """
const appState = {
  project: null,
  runDetail: null,
  researchDetail: null,
  selectedRunId: null,
  selectedBundle: null,
  view: "overview",
};
const appEls = {
  connection: document.getElementById("connection-state"),
  refresh: document.getElementById("refresh-app"),
  projectRoot: document.getElementById("project-root"),
  projectMode: document.getElementById("project-mode"),
  stats: document.getElementById("app-stats"),
  runSearch: document.getElementById("app-run-search"),
  runList: document.getElementById("app-run-list"),
  runCount: document.getElementById("app-run-count"),
  runDetail: document.getElementById("app-run-detail"),
  metricSelect: document.getElementById("app-metric-select"),
  metricWindow: document.getElementById("app-metric-window"),
  metricPoints: document.getElementById("app-metric-points"),
  metricMeta: document.getElementById("app-metric-meta"),
  metricChart: document.getElementById("app-metric-chart"),
  artifacts: document.getElementById("app-artifacts"),
  researchSelect: document.getElementById("app-research-select"),
  researchFilter: document.getElementById("app-research-filter"),
  researchWindow: document.getElementById("app-research-window"),
  researchWindowLabel: document.getElementById("app-research-window-label"),
  researchInspector: document.getElementById("app-research-inspector"),
  researchChart: document.getElementById("app-research-chart"),
  commandContext: document.getElementById("app-command-context"),
  commandList: document.getElementById("app-command-list"),
};
async function appFetch(path) {
  const response = await fetch(path, {headers: {"Accept": "application/json"}});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}
function setAppStatus(text, error = false) {
  appEls.connection.textContent = text;
  appEls.connection.classList.toggle("error", error);
}
function renderAppShell() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === appState.view);
  });
  document.querySelectorAll(".app-panel").forEach((panel) => {
    const sections = (panel.dataset.section || "").split(" ");
    panel.dataset.hidden = sections.includes(appState.view) ? "false" : "true";
  });
}
function renderAppStats() {
  const project = appState.project;
  const runs = project.runs || [];
  const completed = runs.filter((run) => run.status === "completed").length;
  appEls.stats.innerHTML = [
    ["Runs", runs.length],
    ["Completed", completed],
    ["Research", (project.research_bundles || []).length],
    ["Best Eval", project.best_eval || "—"],
  ].map(([label, value]) => `<div class="stat-card"><span>${escapeText(label)}</span><strong>${escapeText(value)}</strong></div>`).join("");
}
function quoteArg(value) {
  const text = String(value ?? "");
  if (/^[A-Za-z0-9_./:@=-]+$/.test(text)) return text;
  return `'${text.replaceAll("'", "'\\\\''")}'`;
}
function runPath(runId) {
  return `runs/${runId}`;
}
function checkpointPath(run, preferred = "best") {
  const checkpoint = preferred === "best" ? (run.best_checkpoint || run.latest_checkpoint) : (run.latest_checkpoint || run.best_checkpoint);
  return checkpoint ? `${runPath(run.run_id)}/${checkpoint}` : `${runPath(run.run_id)}/checkpoints/latest.zip`;
}
function topRunIds(limit = 3) {
  return (appState.project.runs || [])
    .filter((run) => run.run_id)
    .slice()
    .sort((a, b) => (scoreOfRun(b) ?? -Infinity) - (scoreOfRun(a) ?? -Infinity))
    .slice(0, limit)
    .map((run) => run.run_id);
}
function commandRowsForState() {
  const run = appState.runDetail && appState.runDetail.run;
  const commands = [];
  if (run) {
    commands.push(
      {title: "Inspect selected run", meta: "same data as the run detail view", command: `rlx info ${quoteArg(run.run_id)}`},
      {title: "Explain PPO metrics", meta: "deterministic metric explanations", command: `rlx explain-metrics ${quoteArg(run.run_id)}`},
      {title: "Evaluate latest checkpoint", meta: "writes a manual eval JSON", command: `rlx eval --run ${quoteArg(runPath(run.run_id))}`},
      {title: "Evaluate all checkpoints", meta: "latest, best, and saved step checkpoints", command: `rlx eval --run ${quoteArg(runPath(run.run_id))} --all-checkpoints`},
      {title: "Render best video", meta: "uses best checkpoint when available", command: `rlx video ${quoteArg(checkpointPath(run, "best"))} --episodes 2`},
      {title: "Generate run report", meta: "shareable report artifact", command: `rlx report ${quoteArg(run.run_id)} --serve`},
      {title: "Plot run bundle", meta: "metrics/eval plot artifacts", command: `rlx plot ${quoteArg(run.run_id)}`},
      {title: "Research from run", meta: "controlled advisor/research loop", command: `rlx research ${quoteArg(run.run_id)} --rounds 3 --variants 4 --execute`}
    );
  }
  const topRuns = topRunIds(3);
  if (topRuns.length >= 2) {
    commands.push({
      title: "Compare current leaders",
      meta: `${topRuns.length} best scored runs in this project`,
      command: `rlx compare ${topRuns.map(quoteArg).join(" ")}`,
    });
  }
  const bundle = appState.selectedBundle;
  if (bundle) {
    commands.push(
      {title: "Open research report", meta: "candidate trajectory and champion table", command: `rlx report ${quoteArg(bundle)} --serve`},
      {title: "Summarize research bundle", meta: "terminal summary of the bundle", command: `rlx summarize ${quoteArg(bundle)}`}
    );
  }
  commands.push({title: "Open this dashboard", meta: "connected local web app", command: "rlx dashboard --open"});
  return commands;
}
function copyCommand(button, command) {
  const finish = (text) => {
    const previous = button.textContent;
    button.textContent = text;
    setTimeout(() => { button.textContent = previous; }, 1100);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(command).then(() => finish("copied")).catch(() => finish("copy failed"));
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = command;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  try {
    document.execCommand("copy");
    finish("copied");
  } catch {
    finish("copy failed");
  } finally {
    textarea.remove();
  }
}
function renderAppCommands() {
  if (!appEls.commandList) return;
  const run = appState.runDetail && appState.runDetail.run;
  const bundle = appState.selectedBundle;
  const context = run ? `selected run: ${run.run_id}` : bundle ? `research bundle: ${bundle}` : "project commands";
  appEls.commandContext.textContent = context;
  const commands = commandRowsForState();
  appEls.commandList.innerHTML = commands.map((item, index) => `<div class="command-card">
    <div class="command-card-header">
      <div><strong>${escapeText(item.title)}</strong><br><span>${escapeText(item.meta)}</span></div>
      <button class="copy-command" data-command-index="${index}">copy</button>
    </div>
    <code class="command-code">${escapeText(item.command)}</code>
  </div>`).join("");
  appEls.commandList.querySelectorAll(".copy-command").forEach((button) => {
    button.addEventListener("click", () => {
      const item = commands[Number(button.dataset.commandIndex)];
      if (item) copyCommand(button, item.command);
    });
  });
}
function appRunSearchText(run) {
  return [run.run_id, run.run_name, run.environment, run.status, (run.tags || []).join(" ")].join(" ").toLowerCase();
}
function renderAppRunList() {
  const runs = appState.project.runs || [];
  const q = (appEls.runSearch.value || "").toLowerCase();
  const filtered = runs.filter((run) => appRunSearchText(run).includes(q));
  if (appEls.runCount) {
    appEls.runCount.textContent = q ? `${filtered.length}/${runs.length}` : String(runs.length);
  }
  appEls.runList.innerHTML = filtered.map((run) => {
    const active = appState.selectedRunId === run.run_id ? "active" : "";
    return `<button class="run-button ${active}" data-run="${escapeText(run.run_id)}"><strong>${escapeText(run.run_id)}</strong><span>${escapeText(run.environment || "unknown")} · ${escapeText(run.status || "unknown")} · score ${fmt(scoreOfRun(run))}</span></button>`;
  }).join("") || '<div class="empty">No runs match the current search.</div>';
  appEls.runList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => selectAppRun(button.dataset.run));
  });
}
async function selectAppRun(runId) {
  if (!runId) return;
  appState.selectedRunId = runId;
  renderAppRunList();
  appEls.runDetail.innerHTML = '<div class="empty">Loading run data...</div>';
  appEls.metricChart.innerHTML = '<div class="empty">Loading metrics...</div>';
  try {
    appState.runDetail = await appFetch(`/api/run?ref=${encodeURIComponent(runId)}`);
    renderAppRunDetail();
    renderAppMetricControls();
    renderAppMetric();
    renderAppArtifacts();
    renderAppCommands();
  } catch (error) {
    appEls.runDetail.innerHTML = `<div class="empty">${escapeText(error.message)}</div>`;
    renderAppCommands();
  }
}
function renderAppRunDetail() {
  const detail = appState.runDetail;
  if (!detail) {
    appEls.runDetail.innerHTML = '<div class="empty">Select a run.</div>';
    return;
  }
  const run = detail.run;
  const evalRows = (detail.evals || []).slice(-3).map((row) => `<tr><td>${escapeText(row.file)}</td><td>${escapeText(row.checkpoint)}</td><td>${fmt(row.mean_reward)}</td></tr>`).join("");
  appEls.runDetail.innerHTML = `<p class="eyebrow">Selected Run</p><h2>${escapeText(run.run_id)}</h2><p class="muted">${escapeText(run.run_dir)}</p>
    <div class="detail-grid">
      <div class="mini-card"><span>Status</span><strong>${escapeText(run.status || "—")}</strong></div>
      <div class="mini-card"><span>Environment</span><strong>${escapeText(run.environment || "—")}</strong></div>
      <div class="mini-card"><span>Final Train</span><strong>${fmt(run.final_rollout_reward)}</strong></div>
      <div class="mini-card"><span>Best Eval</span><strong>${fmt(run.best_eval && run.best_eval.mean_reward)}</strong></div>
      <div class="mini-card"><span>Device</span><strong>${escapeText(run.resolved_device || run.requested_device || "—")}</strong></div>
      <div class="mini-card"><span>Timesteps</span><strong>${fmt(run.total_timesteps, 0)}</strong></div>
    </div>
    <div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>Eval</th><th>Checkpoint</th><th>Reward</th></tr></thead><tbody>${evalRows || '<tr><td colspan="3">No standalone evals yet.</td></tr>'}</tbody></table></div>`;
}
function appMetricKeys() {
  const detail = appState.runDetail || {};
  const metrics = detail.metrics || {};
  const preferred = detail.metric_order || [];
  return preferred
    .filter((key) => (metrics[key] || []).length >= 2)
    .concat(Object.keys(metrics).filter((key) => !preferred.includes(key) && (metrics[key] || []).length >= 2));
}
function renderAppMetricControls() {
  const keys = appMetricKeys();
  const labels = (appState.runDetail && appState.runDetail.metric_labels) || {};
  appEls.metricSelect.innerHTML = keys.map((key) => `<option value="${escapeText(key)}">${escapeText(labels[key] || key)}</option>`).join("");
}
function renderAppMetric() {
  const detail = appState.runDetail || {};
  const metrics = detail.metrics || {};
  const labels = detail.metric_labels || {};
  const keys = appMetricKeys();
  if (!keys.length) {
    appEls.metricMeta.textContent = "";
    appEls.metricChart.innerHTML = '<div class="empty">No multi-point training metrics found yet.</div>';
    return;
  }
  const key = appEls.metricSelect.value || keys[0];
  const rawPoints = (metrics[key] || []).map((p) => ({x: p.step, y: p.value, label: labels[key] || key}));
  const windowValue = appEls.metricWindow.value;
  const points = windowValue === "all" ? rawPoints : rawPoints.slice(-Number(windowValue));
  appEls.metricMeta.textContent = `${labels[key] || key} / ${points.length} of ${rawPoints.length} points shown`;
  lineChart(appEls.metricChart, points, {title: labels[key] || key, showPoints: appEls.metricPoints.checked});
}
function renderAppArtifacts() {
  const artifacts = (appState.runDetail && appState.runDetail.artifacts) || [];
  appEls.artifacts.innerHTML = artifacts.map((item) => `<a class="artifact-link" href="${escapeText(item.href)}" target="_blank" rel="noreferrer"><strong>${escapeText(item.label)}</strong><br><span class="muted">${escapeText(item.path)}</span></a>`).join("") || '<div class="empty">No artifacts found.</div>';
}
function renderAppResearchPicker() {
  const bundles = appState.project.research_bundles || [];
  appEls.researchSelect.innerHTML = bundles.map((item) => `<option value="${escapeText(item.bundle)}">${escapeText(item.initial_run_id || item.bundle)} · ${fmt(item.champion_score)}</option>`).join("");
  if (!bundles.length) {
    appEls.researchChart.innerHTML = '<div class="empty">No research bundles yet.</div>';
    appEls.researchInspector.textContent = "Run `rlx research ...` to populate this panel.";
    return;
  }
  appState.selectedBundle = appState.selectedBundle || bundles[0].bundle;
  appEls.researchSelect.value = appState.selectedBundle;
}
async function selectAppResearch(bundle) {
  if (!bundle) return;
  appState.selectedBundle = bundle;
  appEls.researchChart.innerHTML = '<div class="empty">Loading research bundle...</div>';
  try {
    appState.researchDetail = await appFetch(`/api/research?bundle=${encodeURIComponent(bundle)}`);
    const rows = (appState.researchDetail.score_rows || []).filter((row) => row.score !== null && row.score !== undefined);
    appEls.researchWindow.max = String(Math.max(5, rows.length));
    appEls.researchWindow.value = String(Math.min(Number(appEls.researchWindow.value || 40), Math.max(5, rows.length)));
    renderAppResearchChart();
    renderAppCommands();
  } catch (error) {
    appEls.researchChart.innerHTML = `<div class="empty">${escapeText(error.message)}</div>`;
    renderAppCommands();
  }
}
function appResearchRows() {
  const rows = ((appState.researchDetail && appState.researchDetail.score_rows) || []).filter((row) => row.score !== null && row.score !== undefined);
  const mode = appEls.researchFilter.value;
  const limit = Number(appEls.researchWindow.value || 40);
  let selected = rows;
  if (mode === "kept") {
    selected = selected.filter((row) => row.promoted || row.role === "baseline");
  } else if (mode === "top") {
    selected = selected.slice().sort((a, b) => Number(b.score) - Number(a.score)).slice(0, limit);
  } else if (mode === "latest") {
    selected = selected.slice(-limit);
  } else {
    selected = selected.slice(0, limit);
  }
  return selected.sort((a, b) => Number(a.experiment) - Number(b.experiment));
}
function renderAppResearchChart() {
  appEls.researchWindowLabel.textContent = appEls.researchWindow.value;
  const rows = appResearchRows();
  const manifest = appState.researchDetail && appState.researchDetail.manifest;
  const champion = manifest && manifest.champion;
  appEls.researchInspector.textContent = champion ? `Champion ${champion.run_id || "n/a"} / score ${fmt(champion.score)} / ${champion.score_source || "score"}` : "Move across the chart to inspect candidates.";
  researchProgressChart(appEls.researchChart, rows);
}
async function loadApp() {
  setAppStatus("loading");
  try {
    appState.project = await appFetch("/api/project");
    appEls.projectRoot.textContent = appState.project.demo ? "demo project" : appState.project.project_root;
    appEls.projectMode.textContent = appState.project.demo ? "design preview" : "live project";
    renderAppStats();
    renderAppRunList();
    renderAppResearchPicker();
    renderAppCommands();
    if (!appState.selectedRunId && (appState.project.runs || []).length) {
      appState.selectedRunId = appState.project.runs[0].run_id;
    }
    await Promise.all([
      appState.selectedRunId ? selectAppRun(appState.selectedRunId) : Promise.resolve(),
      appState.selectedBundle ? selectAppResearch(appState.selectedBundle) : Promise.resolve(),
    ]);
    setAppStatus("connected");
  } catch (error) {
    setAppStatus("error", true);
    appEls.stats.innerHTML = `<div class="empty">${escapeText(error.message)}</div>`;
    renderAppCommands();
  }
}
document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    appState.view = button.dataset.view || "overview";
    renderAppShell();
  });
});
appEls.refresh.addEventListener("click", loadApp);
appEls.runSearch.addEventListener("input", renderAppRunList);
appEls.metricSelect.addEventListener("change", renderAppMetric);
appEls.metricWindow.addEventListener("change", renderAppMetric);
appEls.metricPoints.addEventListener("change", renderAppMetric);
appEls.researchSelect.addEventListener("change", () => selectAppResearch(appEls.researchSelect.value));
appEls.researchFilter.addEventListener("change", renderAppResearchChart);
appEls.researchWindow.addEventListener("input", renderAppResearchChart);
renderAppShell();
loadApp();
"""


def _research_report_script() -> str:
    return """
const rows = RLX.score_rows || [];
const researchFilter = document.getElementById("research-filter");
const researchWindow = document.getElementById("research-window");
const researchWindowLabel = document.getElementById("research-window-label");
function scoreValue(row) {
  return row.score === null || row.score === undefined ? -Infinity : Number(row.score);
}
function filteredResearchRows() {
  const mode = researchFilter.value;
  const limit = Number(researchWindow.value || 40);
  let selected = rows.filter((row) => row.score !== null && row.score !== undefined);
  if (mode === "kept") {
    selected = selected.filter((row) => row.promoted || row.role === "baseline");
  } else if (mode === "top") {
    selected = selected.slice().sort((a, b) => scoreValue(b) - scoreValue(a)).slice(0, limit);
  } else if (mode === "latest") {
    selected = selected.slice(-limit);
  } else {
    selected = selected.slice(0, limit);
  }
  return selected.sort((a, b) => Number(a.experiment) - Number(b.experiment));
}
function renderResearchChart() {
  researchWindowLabel.textContent = researchWindow.value;
  researchProgressChart(document.getElementById("research-progress-chart"), filteredResearchRows());
}
researchFilter.addEventListener("change", renderResearchChart);
researchWindow.addEventListener("input", renderResearchChart);
renderResearchChart();
const roundList = document.getElementById("round-list");
const roundSearch = document.getElementById("round-search");
function roundText(round) {
  return JSON.stringify(round).toLowerCase();
}
function renderRounds() {
  const q = (roundSearch.value || "").toLowerCase();
  const filtered = (RLX.manifest.rounds || []).filter((round) => roundText(round).includes(q));
  roundList.innerHTML = filtered.map((round) => {
    const variants = (round.variants || []).map((variant) => `${escapeText(variant.run_id || "n/a")} ${fmt(variant.score)} ${escapeText(JSON.stringify(variant.mutations || {}))}`).join("<br>");
    return `<div class="round-card"><span class="pill">Round ${String(round.index).padStart(3, "0")}</span><h3>${escapeText(round.baseline_run_id)} → ${escapeText(round.candidate_run_id || "no candidate")}</h3><p class="muted">Improve ${fmt(round.improvement)} · promoted ${round.promoted ? "yes" : "no"} · champion ${escapeText(round.champion_after || "n/a")}</p><p>${variants}</p></div>`;
  }).join("") || '<div class="empty">No rounds match the current search.</div>';
}
roundSearch.addEventListener("input", renderRounds);
renderRounds();
"""


def _run_payload(run: RunComparison) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "run_dir": str(run.run_dir),
        "tags": list(run.tags),
        "status": run.status,
        "environment": run.environment,
        "requested_device": run.requested_device,
        "resolved_device": run.resolved_device,
        "total_timesteps": run.total_timesteps,
        "final_rollout_reward": run.final_rollout_reward,
        "best_rollout_reward": run.best_rollout_reward,
        "latest_checkpoint": run.latest_checkpoint,
        "best_checkpoint": run.best_checkpoint,
        "latest_eval": _eval_payload(run.latest_eval),
        "best_eval": _eval_payload(run.best_eval),
        "last_video_manifest": run.last_video_manifest,
    }


def _preview_run_data(root: Path) -> dict[str, Any]:
    metrics = {
        "rollout/ep_rew_mean": [
            {"step": 512, "value": 22.1},
            {"step": 2048, "value": 41.4},
            {"step": 4096, "value": 74.8},
            {"step": 8192, "value": 128.2},
            {"step": 16384, "value": 184.5},
            {"step": 32768, "value": 214.0},
            {"step": 50176, "value": 231.7},
        ],
        "rollout/ep_len_mean": [
            {"step": 512, "value": 22.1},
            {"step": 2048, "value": 41.4},
            {"step": 4096, "value": 74.8},
            {"step": 8192, "value": 128.2},
            {"step": 16384, "value": 184.5},
            {"step": 32768, "value": 214.0},
            {"step": 50176, "value": 231.7},
        ],
        "train/approx_kl": [
            {"step": 2048, "value": 0.0004},
            {"step": 8192, "value": 0.0008},
            {"step": 16384, "value": 0.0012},
            {"step": 32768, "value": 0.0007},
            {"step": 50176, "value": 0.0006},
        ],
        "train/value_loss": [
            {"step": 2048, "value": 96.0},
            {"step": 8192, "value": 81.0},
            {"step": 16384, "value": 72.0},
            {"step": 32768, "value": 58.0},
            {"step": 50176, "value": 43.0},
        ],
    }
    return {
        "kind": "run",
        "generated_at": _utc_now_iso(),
        "project_root": str(root),
        "run": _preview_runs()[0],
        "metrics": metrics,
        "metric_order": _available_metric_order(metrics),
        "metric_labels": _METRIC_LABELS,
        "metric_descriptions": _METRIC_DESCRIPTIONS,
        "metric_explanations": [
            {
                "key": "rollout/ep_rew_mean",
                "label": "Reward mean",
                "first": 22.1,
                "latest": 231.7,
                "minimum": 22.1,
                "maximum": 231.7,
                "trend": "rising",
                "interpretation": "Reward is improving across logged rollout summaries.",
            },
            {
                "key": "train/value_loss",
                "label": "Value loss",
                "first": 96.0,
                "latest": 43.0,
                "minimum": 43.0,
                "maximum": 96.0,
                "trend": "falling",
                "interpretation": "Value loss is falling, which usually means critic fit improved.",
            },
        ],
        "notes": [
            {
                "severity": "low",
                "metric": "rollout/ep_rew_mean",
                "note": "Preview data: reward trend is positive.",
            }
        ],
        "evals": [
            {
                "file": "eval/manual_eval_001.json",
                "checkpoint": "latest.zip",
                "mean_reward": 238.4,
                "mean_episode_length": 238.4,
            },
            {
                "file": "eval/manual_eval_002.json",
                "checkpoint": "best.zip",
                "mean_reward": 251.2,
                "mean_episode_length": 251.2,
            },
        ],
        "config": {
            "run_name": "preview_cartpole",
            "seed": "42",
            "device": "cpu",
            "env.id": "CartPole-v1",
            "env.num_envs": "4",
            "algo.total_timesteps": "50000",
            "algo.learning_rate": "0.0003",
            "algo.gamma": "0.99",
            "policy.hidden_sizes": "[128, 128]",
            "eval.episodes": "10",
        },
        "artifacts": [],
    }


def _preview_runs() -> list[dict[str, Any]]:
    return [
        {
            "run_id": "preview_cartpole_001",
            "run_name": "preview_cartpole",
            "run_dir": "preview/runs/preview_cartpole_001",
            "tags": ["baseline", "preview"],
            "status": "completed",
            "environment": "CartPole-v1",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "total_timesteps": 50176,
            "final_rollout_reward": 231.7,
            "best_rollout_reward": 242.6,
            "latest_checkpoint": "checkpoints/latest.zip",
            "best_checkpoint": "checkpoints/best.zip",
            "latest_eval": {
                "source": "manual_eval_001.json",
                "mean_reward": 238.4,
                "mean_episode_length": 238.4,
            },
            "best_eval": {
                "source": "manual_eval_002.json",
                "mean_reward": 251.2,
                "mean_episode_length": 251.2,
            },
            "last_video_manifest": None,
        },
        {
            "run_id": "preview_cartpole_002",
            "run_name": "preview_cartpole",
            "run_dir": "preview/runs/preview_cartpole_002",
            "tags": ["advisor"],
            "status": "completed",
            "environment": "CartPole-v1",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "total_timesteps": 50176,
            "final_rollout_reward": 247.8,
            "best_rollout_reward": 259.1,
            "latest_checkpoint": "checkpoints/latest.zip",
            "best_checkpoint": "checkpoints/best.zip",
            "latest_eval": {
                "source": "manual_eval_001.json",
                "mean_reward": 266.8,
                "mean_episode_length": 266.8,
            },
            "best_eval": {
                "source": "manual_eval_001.json",
                "mean_reward": 266.8,
                "mean_episode_length": 266.8,
            },
            "last_video_manifest": None,
        },
        {
            "run_id": "preview_cartpole_003",
            "run_name": "preview_cartpole",
            "run_dir": "preview/runs/preview_cartpole_003",
            "tags": ["discarded"],
            "status": "completed",
            "environment": "CartPole-v1",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "total_timesteps": 50176,
            "final_rollout_reward": 190.2,
            "best_rollout_reward": 204.4,
            "latest_checkpoint": "checkpoints/latest.zip",
            "best_checkpoint": "checkpoints/best.zip",
            "latest_eval": {
                "source": "manual_eval_001.json",
                "mean_reward": 198.0,
                "mean_episode_length": 198.0,
            },
            "best_eval": {
                "source": "manual_eval_001.json",
                "mean_reward": 198.0,
                "mean_episode_length": 198.0,
            },
            "last_video_manifest": None,
        },
    ]


def _preview_research_bundles() -> list[dict[str, Any]]:
    return [
        {
            "bundle": "analysis/research/preview_cartpole_001_research_001",
            "initial_run_id": "preview_cartpole_001",
            "mode": "executed",
            "stop_reason": "completed requested rounds",
            "rounds": 3,
            "champion_run_id": "preview_cartpole_002",
            "champion_score": 266.8,
            "champion_score_source": "standalone eval",
        }
    ]


def _preview_research_manifest() -> dict[str, Any]:
    rounds = []
    champion = "preview_cartpole_001"
    champion_score = 238.4
    experiment = 2
    for round_index in range(1, 9):
        champion_before = champion
        champion_score_before = champion_score
        variants = []
        best_variant = None
        for variant_index in range(1, 7):
            score = 205 + ((round_index * 19 + variant_index * 11) % 88)
            if round_index in {2, 5, 7} and variant_index == 3:
                score = champion_score + 8 + round_index
            run_id = f"preview_cartpole_{experiment:03d}"
            promoted = score > champion_score
            if promoted:
                champion = run_id
                champion_score = float(score)
            variant = {
                "index": variant_index,
                "run_id": run_id,
                "status": "completed",
                "mutations": {
                    "algo.learning_rate": [0.0003, 0.0007, 0.001][variant_index % 3],
                    "algo.entropy_coef": [0.0, 0.005, 0.01][round_index % 3],
                },
                "score": float(score),
                "score_source": "preview eval",
                "promoted": promoted,
            }
            variants.append(variant)
            if best_variant is None or float(score) > float(best_variant["score"]):
                best_variant = variant
            experiment += 1
        assert best_variant is not None
        rounds.append(
            {
                "index": round_index,
                "baseline_run_id": champion_before,
                "advisor_bundle": f"analysis/advisor/preview_cartpole_001_advisor_{round_index:03d}",
                "advisor_manifest": (
                    f"analysis/advisor/preview_cartpole_001_advisor_{round_index:03d}/manifest.json"
                ),
                "champion_before": champion_before,
                "champion_score_before": champion_score_before,
                "candidate_run_id": best_variant["run_id"],
                "candidate_score": best_variant["score"],
                "candidate_score_source": best_variant["score_source"],
                "improvement": float(best_variant["score"]) - champion_score_before,
                "promoted": bool(best_variant["promoted"]),
                "variants": variants,
                "champion_after": champion,
                "champion_score_after": champion_score,
                "stop_reason": None,
            }
        )

    return {
        "kind": "research_bundle",
        "updated_at": _utc_now_iso(),
        "mode": "preview",
        "bundle": "analysis/research/preview_cartpole_001_research_001",
        "initial": {
            "run_id": "preview_cartpole_001",
            "score": 238.4,
            "score_source": "preview eval",
        },
        "initial_run_id": "preview_cartpole_001",
        "champion": {
            "run_id": champion,
            "score": champion_score,
            "score_source": "preview eval",
        },
        "settings": {"rounds": 8, "variants": 6},
        "protocol": {"version": 1, "planner": "preview"},
        "stop_reason": "preview data",
        "rounds": rounds,
        "artifacts": [],
    }


def _eval_payload(summary: Any) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "source": summary.source,
        "mean_reward": summary.mean_reward,
        "mean_episode_length": summary.mean_episode_length,
    }


def _load_metric_points(path: Path) -> dict[str, list[dict[str, float | int]]]:
    if not path.exists():
        return {}

    values: dict[str, list[dict[str, float | int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        step = payload.get("step")
        if not isinstance(step, int):
            continue
        for key, value in payload.items():
            if (
                key == "step"
                or key in _WEB_METRIC_EXCLUDED
                or not isinstance(value, int | float)
            ):
                continue
            values.setdefault(key, []).append({"step": step, "value": float(value)})
    values = {
        key: points
        for key, points in values.items()
        if len({point["step"] for point in points}) >= 2
    }
    return _sort_metric_series(values)


def _sort_metric_series(
    values: dict[str, list[dict[str, float | int]]],
) -> dict[str, list[dict[str, float | int]]]:
    ordered: dict[str, list[dict[str, float | int]]] = {}
    for key in _WEB_METRIC_ORDER:
        if key in values:
            ordered[key] = values[key]
    for key in sorted(values):
        if key not in ordered:
            ordered[key] = values[key]
    return ordered


def _available_metric_order(metrics: dict[str, list[dict[str, float | int]]]) -> list[str]:
    return [key for key in _WEB_METRIC_ORDER if key in metrics] + [
        key for key in sorted(metrics) if key not in _WEB_METRIC_ORDER
    ]


def _load_eval_rows(eval_dir: Path, *, project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("manual_eval_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = payload
        checkpoint = payload.get("checkpoint")
        checkpoint_name = "—"
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("name"), str):
            checkpoint_name = checkpoint["name"]
        rows.append(
            {
                "file": str(path.relative_to(project_root)),
                "checkpoint": checkpoint_name,
                "mean_reward": _maybe_float(summary.get("mean_reward")),
                "mean_episode_length": _maybe_float(summary.get("mean_episode_length")),
            }
        )

    training_eval = _training_eval_row(eval_dir / "evaluations.npz", project_root=project_root)
    if training_eval is not None:
        rows.append(training_eval)
    return rows


def _training_eval_row(path: Path, *, project_root: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        data = np.load(path)
        timesteps = data.get("timesteps")
        results = data.get("results")
        ep_lengths = data.get("ep_lengths")
    except Exception:
        return None
    if results is None or len(results) == 0:
        return None
    last_rewards = results[-1]
    mean_length = None
    if ep_lengths is not None and len(ep_lengths) > 0 and len(ep_lengths[-1]):
        mean_length = float(ep_lengths[-1].mean())
    step_label = "training eval"
    if timesteps is not None and len(timesteps) > 0:
        step_label = f"step {int(timesteps[-1])}"
    return {
        "file": str(path.relative_to(project_root)),
        "checkpoint": step_label,
        "mean_reward": float(last_rewards.mean()) if len(last_rewards) else None,
        "mean_episode_length": mean_length,
    }


def _run_artifacts(
    run: RunComparison,
    run_dir: Path,
    project_root: Path,
    output_dir: Path,
) -> list[dict[str, str]]:
    candidates = [
        ("Config snapshot", run_dir / CONFIG_SNAPSHOT_NAME),
        ("Metadata", run_dir / METADATA_NAME),
        ("Metrics JSONL", run_dir / METRICS_NAME),
    ]
    if run.latest_checkpoint:
        candidates.append(("Latest checkpoint", run_dir / run.latest_checkpoint))
    if run.best_checkpoint:
        candidates.append(("Best checkpoint", run_dir / run.best_checkpoint))
    if run.last_video_manifest:
        candidates.append(("Latest video manifest", run_dir / run.last_video_manifest))

    artifacts = []
    for label, path in candidates:
        if path.exists():
            artifacts.append(
                {
                    "label": label,
                    "path": str(path.relative_to(project_root)),
                    "href": _relative_url(output_dir, path),
                }
            )
    return artifacts


def _run_artifacts_for_app(
    run: RunComparison,
    run_dir: Path,
    project_root: Path,
) -> list[dict[str, str]]:
    candidates = [
        ("Config snapshot", run_dir / CONFIG_SNAPSHOT_NAME),
        ("Metadata", run_dir / METADATA_NAME),
        ("Metrics JSONL", run_dir / METRICS_NAME),
    ]
    if run.latest_checkpoint:
        candidates.append(("Latest checkpoint", run_dir / run.latest_checkpoint))
    if run.best_checkpoint:
        candidates.append(("Best checkpoint", run_dir / run.best_checkpoint))
    if run.last_video_manifest:
        candidates.append(("Latest video manifest", run_dir / run.last_video_manifest))

    artifacts = []
    for label, path in candidates:
        if path.exists():
            relative = path.relative_to(project_root).as_posix()
            artifacts.append(
                {
                    "label": label,
                    "path": relative,
                    "href": f"/files/{quote(relative)}",
                }
            )
    return artifacts


def _research_images(
    payload: dict[str, Any],
    project_root: Path,
    output_dir: Path,
) -> dict[str, str]:
    images: dict[str, str] = {}
    for artifact in payload.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        key = artifact.get("key")
        file_value = artifact.get("file")
        if not isinstance(key, str) or not isinstance(file_value, str):
            continue
        path = project_root / file_value
        if path.exists():
            images[key] = _relative_url(output_dir, path)
    return images


def _research_images_for_app(payload: dict[str, Any], project_root: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    for artifact in payload.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        key = artifact.get("key")
        file_value = artifact.get("file")
        if not isinstance(key, str) or not isinstance(file_value, str):
            continue
        path = project_root / file_value
        if path.exists():
            images[key] = f"/files/{quote(Path(file_value).as_posix())}"
    return images


def _research_score_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    initial = payload.get("initial", {})
    rows.append(
        {
            "experiment": 0,
            "run_id": initial.get("run_id") or payload.get("initial_run_id"),
            "score": initial.get("score"),
            "source": initial.get("score_source"),
            "promoted": False,
            "role": "baseline",
        }
    )
    experiment = 1
    for round_payload in payload.get("rounds", []):
        variants = round_payload.get("variants", [])
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                rows.append(
                    {
                        "experiment": experiment,
                        "run_id": variant.get("run_id"),
                        "score": variant.get("score"),
                        "source": variant.get("score_source"),
                        "promoted": bool(variant.get("promoted")),
                        "role": "variant",
                    }
                )
                experiment += 1
            continue
        rows.append(
            {
                "experiment": experiment,
                "run_id": round_payload.get("candidate_run_id"),
                "score": round_payload.get("candidate_score"),
                "source": round_payload.get("candidate_score_source"),
                "promoted": bool(round_payload.get("promoted")),
                "role": "candidate",
            }
        )
        experiment += 1
    return rows


def _load_research_bundles(project_root: Path) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for manifest_path in sorted((project_root / "analysis" / "research").glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("kind") != "research_bundle":
            continue
        champion = payload.get("champion")
        if not isinstance(champion, dict):
            champion = {}
        bundles.append(
            {
                "bundle": str(manifest_path.parent.relative_to(project_root)),
                "initial_run_id": payload.get("initial_run_id"),
                "mode": payload.get("mode"),
                "stop_reason": payload.get("stop_reason"),
                "rounds": len(payload.get("rounds", [])),
                "champion_run_id": champion.get("run_id"),
                "champion_score": champion.get("score"),
                "champion_score_source": champion.get("score_source"),
            }
        )
    bundles.sort(key=lambda item: str(item.get("bundle", "")), reverse=True)
    return bundles


def _resolve_research_manifest(target: str, cwd: Path) -> Path:
    candidate = Path(target).expanduser()
    candidates = []
    if candidate.exists():
        candidates.append(candidate)
    candidates.append((cwd / target).resolve())
    try:
        project_root = find_project_root(cwd)
        candidates.append((project_root / "analysis" / "research" / target).resolve())
    except ProjectLookupError:
        project_root = None

    for item in candidates:
        manifest = item / "manifest.json" if item.is_dir() else item
        if manifest.is_file():
            return manifest.resolve()

    if project_root is not None:
        matches = sorted((project_root / "analysis" / "research").glob(f"**/{target}/manifest.json"))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            raise ReportError(f"Research reference is ambiguous: {target}")

    raise ReportError(f"Target is not a run or research bundle: {target}")


def _infer_project_root(path: Path) -> Path:
    for parent in path.parents:
        try:
            return find_project_root(parent)
        except ProjectLookupError:
            continue
    raise ReportError(f"Could not infer project root for: {path}")


def _next_output_dir(root: Path, stem: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    slug = _slugify(stem)
    pattern = re.compile(rf"^{re.escape(slug)}_(\d{{3}})$")
    existing = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            existing.append(int(match.group(1)))
    return root / f"{slug}_{max(existing, default=0) + 1:03d}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "report"


def _relative_url(from_dir: Path, target: Path) -> str:
    relative = os.path.relpath(target.resolve(), start=from_dir.resolve())
    return quote(Path(relative).as_posix())


def _stat_card(label: str, value: Any) -> str:
    return f'<div class="stat-card"><span>{_escape(label)}</span><strong>{_escape(str(value))}</strong></div>'


def _artifact_links(items: list[dict[str, str]]) -> str:
    if not items:
        return '<div class="empty">No artifacts found.</div>'
    return "".join(
        f'<a class="artifact-link" href="{_escape(item["href"])}"><strong>{_escape(item["label"])}</strong><br><span class="muted">{_escape(item["path"])}</span></a>'
        for item in items
    )


def _eval_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">No eval artifacts found. Run `rlx eval --run ...` to add one.</div>'
    body = "".join(
        "<tr>"
        f"<td>{_escape(row['file'])}</td>"
        f"<td>{_escape(row['checkpoint'])}</td>"
        f"<td>{_escape(_fmt_optional(row.get('mean_reward')))}</td>"
        f"<td>{_escape(_fmt_optional(row.get('mean_episode_length')))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>File</th><th>Checkpoint</th><th>Mean Reward</th><th>Mean Length</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _config_table(config: dict[str, str]) -> str:
    priority = [
        "run_name",
        "seed",
        "device",
        "env.id",
        "env.num_envs",
        "algo.total_timesteps",
        "algo.learning_rate",
        "algo.gamma",
        "algo.gae_lambda",
        "algo.clip_range",
        "algo.entropy_coef",
        "algo.update_epochs",
        "policy.hidden_sizes",
        "checkpoint.save_every",
        "eval.every",
        "eval.episodes",
    ]
    rows = [(key, config[key]) for key in priority if key in config]
    rows.extend((key, value) for key, value in sorted(config.items()) if key not in priority)
    body = "".join(
        f"<tr><td>{_escape(key)}</td><td>{_escape(value)}</td></tr>"
        for key, value in rows
    )
    return f'<div class="table-wrap"><table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>{body}</tbody></table></div>'


def _image_card(title: str, src: str | None) -> str:
    if not src:
        return f'<div class="image-card"><h2>{_escape(title)}</h2><div class="empty">No image artifact found.</div></div>'
    return f'<div class="image-card"><h2>{_escape(title)}</h2><img src="{_escape(src)}" alt="{_escape(title)}" /></div>'


def _best_eval_label(runs: list[RunComparison]) -> str:
    scores = [
        run.best_eval.mean_reward
        for run in runs
        if run.best_eval is not None and run.best_eval.mean_reward is not None
    ]
    if not scores:
        return "—"
    return _fmt_optional(max(scores))


def _best_eval_label_from_payloads(runs: list[dict[str, Any]]) -> str:
    scores = []
    for run in runs:
        best_eval = run.get("best_eval")
        if isinstance(best_eval, dict):
            value = best_eval.get("mean_reward")
            if isinstance(value, int | float):
                scores.append(float(value))
    if not scores:
        return "—"
    return _fmt_optional(max(scores))


def _eval_label(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "—"
    return _fmt_optional(summary.get("mean_reward"))


def _fmt_optional(value: Any, *, digits: int = 2) -> str:
    if isinstance(value, int | float):
        return f"{float(value):,.{digits}f}" if digits > 0 else f"{int(value):,}"
    return "—"


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _escape_json(data: dict[str, Any]) -> str:
    return (
        json.dumps(data, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
