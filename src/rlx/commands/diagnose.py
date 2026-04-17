import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.diagnose import DiagnoseError, RunDiagnosis, diagnose_run

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to diagnose.",
)


def diagnose_command(run_ref: str = RUN_REF_ARGUMENT) -> None:
    """Find likely problems in one run.

    Examples:

        rlx diagnose cartpole_ppo_001
        rlx diagnose runs/cartpole_ppo_001
    """

    try:
        diagnosis = diagnose_run(run_ref)
    except DiagnoseError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Diagnosis Overview", _build_overview(diagnosis))
    print_panel("RLCLI Diagnostics", _build_diagnostics_table(diagnosis))


def _build_overview(diagnosis: RunDiagnosis):
    info = diagnosis.analysis.info
    rows = [
        ("[success]Run[/success]", f"[path]{info.run.run_dir}[/path]"),
        ("[muted]Run id[/muted]", f"[value]{info.run.run_id}[/value]"),
        ("[muted]Health[/muted]", _fmt_health(diagnosis.health)),
        ("[muted]Status[/muted]", _fmt_status(info.run.status)),
        ("[muted]Trend[/muted]", f"[value]{diagnosis.analysis.learning.trend}[/value]"),
        ("[muted]Issues[/muted]", f"[value]{len(diagnosis.diagnostics)}[/value]"),
        ("[muted]Best eval[/muted]", _fmt_eval(diagnosis)),
    ]
    return build_summary(rows)


def _build_diagnostics_table(diagnosis: RunDiagnosis) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Severity", style="accent", no_wrap=True)
    table.add_column("Area", no_wrap=True)
    table.add_column("Issue", overflow="fold")
    table.add_column("Evidence", overflow="fold")
    table.add_column("Recommendation", overflow="fold")

    for item in diagnosis.diagnostics:
        table.add_row(
            _fmt_severity(item.severity),
            item.area,
            item.issue,
            item.evidence,
            item.recommendation,
        )

    return table


def _fmt_health(value: str) -> str:
    if value == "needs attention":
        return "[error]needs attention[/error]"
    if value == "watch":
        return "[warning]watch[/warning]"
    return "[success]clean[/success]"


def _fmt_status(status: str | None) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status in {"failed", "interrupted"}:
        return f"[error]{status}[/error]"
    if status:
        return f"[value]{status}[/value]"
    return "[muted]—[/muted]"


def _fmt_severity(value: str) -> str:
    if value == "high":
        return "[error]high[/error]"
    if value == "medium":
        return "[warning]medium[/warning]"
    return "[muted]low[/muted]"


def _fmt_eval(diagnosis: RunDiagnosis) -> str:
    summary = diagnosis.analysis.info.run.best_eval
    if summary is None or summary.mean_reward is None:
        return "[muted]—[/muted]"
    return f"[value]{summary.mean_reward:.2f}[/value] [path]{summary.source}[/path]"
