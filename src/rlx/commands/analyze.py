import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.analyze import AnalyzeError, RunAnalysis, analyze_run

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to analyze.",
)


def analyze_command(run_ref: str = RUN_REF_ARGUMENT) -> None:
    """Analyze one run using its local artifacts."""

    try:
        analysis = analyze_run(run_ref)
    except AnalyzeError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Run Analysis", _build_overview(analysis))
    print_panel("RLCLI Learning Signal", _build_learning(analysis))
    print_panel("RLCLI Findings", _build_findings(analysis))
    print_panel("RLCLI Next Moves", _build_suggestions(analysis))


def _build_overview(analysis: RunAnalysis):
    info = analysis.info
    rows = [
        ("[success]Run[/success]", f"[path]{info.run.run_dir}[/path]"),
        ("[muted]Run id[/muted]", f"[value]{info.run.run_id}[/value]"),
        ("[muted]Status[/muted]", _fmt_status(info.run.status)),
        ("[muted]Environment[/muted]", _fmt_text(info.run.environment)),
        ("[muted]Device[/muted]", _fmt_device(analysis)),
        ("[muted]Tags[/muted]", _fmt_tags(info.run.tags)),
        ("[muted]Sweep[/muted]", _fmt_text(info.sweep_name)),
        ("[muted]Resumed from[/muted]", _fmt_text(info.resumed_from_run)),
        ("[muted]Timesteps[/muted]", _fmt_int(info.run.total_timesteps)),
    ]
    return build_summary(rows)


def _build_learning(analysis: RunAnalysis):
    signal = analysis.learning
    rows = [
        ("[success]Trend[/success]", f"[value]{signal.trend}[/value]"),
        ("[muted]Points[/muted]", _fmt_int(len(signal.points))),
        ("[muted]First reward[/muted]", _fmt_number(signal.first_reward)),
        ("[muted]Final reward[/muted]", _fmt_number(signal.final_reward)),
        ("[muted]Best reward[/muted]", _fmt_best_reward(analysis)),
        ("[muted]Delta[/muted]", _fmt_delta(signal.delta)),
        ("[muted]Late mean[/muted]", _fmt_number(signal.late_mean)),
        ("[muted]Late std[/muted]", _fmt_number(signal.late_std)),
        ("[muted]Best eval[/muted]", _fmt_eval(analysis)),
    ]
    return build_summary(rows)


def _build_findings(analysis: RunAnalysis) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Area", style="accent", no_wrap=True)
    table.add_column("Signal", overflow="fold")
    table.add_column("Interpretation", overflow="fold")

    for finding in analysis.findings:
        table.add_row(
            finding.category,
            f"[value]{finding.signal}[/value]",
            finding.interpretation,
        )

    return table


def _build_suggestions(analysis: RunAnalysis) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Priority", style="accent", no_wrap=True)
    table.add_column("Action", overflow="fold")
    table.add_column("Why", overflow="fold")

    for suggestion in analysis.suggestions:
        table.add_row(
            _fmt_priority(suggestion.priority),
            f"[path]{suggestion.action}[/path]",
            suggestion.reason,
        )

    return table


def _fmt_status(status: str | None) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status in {"failed", "interrupted"}:
        return f"[error]{status}[/error]"
    if status:
        return f"[value]{status}[/value]"
    return "[muted]—[/muted]"


def _fmt_device(analysis: RunAnalysis) -> str:
    run = analysis.info.run
    if run.resolved_device:
        return f"[value]{run.resolved_device}[/value]"
    if run.requested_device:
        return f"[value]{run.requested_device}[/value]"
    return "[muted]—[/muted]"


def _fmt_text(value: str | None) -> str:
    if not value:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value:.2f}[/value]"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    style = "success" if value >= 0 else "error"
    return f"[{style}]{value:+.2f}[/{style}]"


def _fmt_best_reward(analysis: RunAnalysis) -> str:
    signal = analysis.learning
    if signal.best_reward is None:
        return "[muted]—[/muted]"
    if signal.best_step is None:
        return f"[value]{signal.best_reward:.2f}[/value]"
    return f"[value]{signal.best_reward:.2f}[/value] [muted]at step {signal.best_step}[/muted]"


def _fmt_eval(analysis: RunAnalysis) -> str:
    summary = analysis.info.run.best_eval
    if summary is None or summary.mean_reward is None:
        return "[muted]—[/muted]"
    if summary.mean_episode_length is None:
        return f"[value]{summary.mean_reward:.2f}[/value] [path]{summary.source}[/path]"
    return (
        f"[value]{summary.mean_reward:.2f}[/value] "
        f"[muted](len {summary.mean_episode_length:.2f})[/muted] "
        f"[path]{summary.source}[/path]"
    )


def _fmt_priority(priority: str) -> str:
    if priority == "high":
        return "[error]high[/error]"
    if priority == "medium":
        return "[warning]medium[/warning]"
    return "[muted]low[/muted]"


def _fmt_tags(tags: tuple[str, ...]) -> str:
    if not tags:
        return "[muted]—[/muted]"
    return f"[value]{', '.join(tags)}[/value]"
