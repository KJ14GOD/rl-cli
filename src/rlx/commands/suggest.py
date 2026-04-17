import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.suggest import RunSuggestions, SuggestError, suggest_next_steps

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to generate suggestions for.",
)


def suggest_command(run_ref: str = RUN_REF_ARGUMENT) -> None:
    """Recommend next actions for one run.

    Examples:

        rlx suggest cartpole_ppo_001
        rlx suggest runs/cartpole_ppo_001
    """

    try:
        suggestions = suggest_next_steps(run_ref)
    except SuggestError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Suggest Overview", _build_overview(suggestions))
    print_panel("RLCLI Suggested Actions", _build_actions_table(suggestions))


def _build_overview(suggestions: RunSuggestions):
    diagnosis = suggestions.diagnosis
    info = diagnosis.analysis.info
    rows = [
        ("[success]Run[/success]", f"[path]{info.run.run_dir}[/path]"),
        ("[muted]Run id[/muted]", f"[value]{info.run.run_id}[/value]"),
        ("[muted]Health[/muted]", _fmt_health(diagnosis.health)),
        ("[muted]Trend[/muted]", f"[value]{diagnosis.analysis.learning.trend}[/value]"),
        ("[muted]Actions[/muted]", f"[value]{len(suggestions.actions)}[/value]"),
    ]
    return build_summary(rows)


def _build_actions_table(suggestions: RunSuggestions) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Priority", style="accent", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Suggestion", overflow="fold")
    table.add_column("Reason", overflow="fold")

    for action in suggestions.actions:
        table.add_row(
            _fmt_priority(action.priority),
            action.kind,
            f"[path]{action.suggestion}[/path]",
            action.reason,
        )

    return table


def _fmt_health(value: str) -> str:
    if value == "needs attention":
        return "[error]needs attention[/error]"
    if value == "watch":
        return "[warning]watch[/warning]"
    return "[success]clean[/success]"


def _fmt_priority(value: str) -> str:
    if value == "high":
        return "[error]high[/error]"
    if value == "medium":
        return "[warning]medium[/warning]"
    return "[muted]low[/muted]"
