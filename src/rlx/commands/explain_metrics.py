import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.explain_metrics import (
    ExplainMetricsError,
    MetricsExplanation,
    explain_metrics,
)

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id whose metrics should be explained.",
)


def explain_metrics_command(run_ref: str = RUN_REF_ARGUMENT) -> None:
    """Explain PPO metric columns for one run.

    Examples:

        rlx explain-metrics cartpole_ppo_001
        rlx explain-metrics runs/cartpole_ppo_001
    """

    try:
        explanation = explain_metrics(run_ref)
    except ExplainMetricsError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Metrics Overview", _build_overview(explanation))
    print_panel("RLCLI Metric Series", _build_series_table(explanation))
    print_panel("RLCLI Metric Notes", _build_notes_table(explanation))

    if explanation.missing:
        print_panel("RLCLI Missing Metrics", _build_missing_table(explanation))


def _build_overview(explanation: MetricsExplanation):
    info = explanation.info
    rows = [
        ("[success]Run[/success]", f"[path]{info.run.run_dir}[/path]"),
        ("[muted]Run id[/muted]", f"[value]{info.run.run_id}[/value]"),
        ("[muted]Environment[/muted]", _fmt_text(info.run.environment)),
        ("[muted]Timesteps[/muted]", _fmt_int(info.run.total_timesteps)),
        ("[muted]Records[/muted]", _fmt_int(explanation.total_records)),
        ("[muted]Metric points[/muted]", _fmt_int(explanation.numeric_records)),
        ("[muted]Series found[/muted]", _fmt_int(len(explanation.series))),
    ]
    return build_summary(rows)


def _build_series_table(explanation: MetricsExplanation) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Metric", style="accent", no_wrap=True)
    table.add_column("Min", justify="right", no_wrap=True)
    table.add_column("Max", justify="right", no_wrap=True)
    table.add_column("First", justify="right", no_wrap=True)
    table.add_column("Latest", justify="right", no_wrap=True)
    table.add_column("Trend", no_wrap=True)
    table.add_column("Explanation", overflow="fold")

    for series in explanation.series:
        table.add_row(
            series.label,
            _fmt_number(series.minimum),
            _fmt_number(series.maximum),
            _fmt_number(series.first),
            _fmt_number(series.latest),
            _fmt_trend(series.trend),
            series.interpretation,
        )

    return table


def _build_notes_table(explanation: MetricsExplanation) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Severity", style="accent", no_wrap=True)
    table.add_column("Metric", overflow="fold")
    table.add_column("Note", overflow="fold")

    for note in explanation.notes:
        table.add_row(
            _fmt_severity(note.severity),
            f"[value]{note.metric}[/value]",
            note.note,
        )

    return table


def _build_missing_table(explanation: MetricsExplanation) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Metric", style="accent", overflow="fold")
    table.add_column("Meaning", overflow="fold")

    for key in explanation.missing:
        table.add_row(
            key,
            "This metric was not present in metrics.jsonl for this run.",
        )

    return table


def _fmt_text(value: str | None) -> str:
    if not value:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value:.4g}[/value]"


def _fmt_trend(value: str) -> str:
    if value == "rising":
        return "[success]rising[/success]"
    if value == "falling":
        return "[error]falling[/error]"
    if value == "flat":
        return "[muted]flat[/muted]"
    return f"[value]{value}[/value]"


def _fmt_severity(value: str) -> str:
    if value == "high":
        return "[error]high[/error]"
    if value == "medium":
        return "[warning]medium[/warning]"
    return "[muted]low[/muted]"
