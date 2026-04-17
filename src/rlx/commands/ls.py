import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.list_runs import RunListError, list_runs


def ls_command() -> None:
    """List runs in the nearest RLCLI project.

    Examples:

        rlx ls
    """

    try:
        run_list = list_runs()
    except RunListError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    rows = [
        ("[success]Project[/success]", f"[path]{run_list.project_root}[/path]"),
        ("[muted]Runs[/muted]", f"[value]{len(run_list.runs)}[/value]"),
    ]
    if run_list.runs:
        rows.append(
            ("[muted]Newest[/muted]", f"[value]{run_list.runs[0].run_id}[/value]")
        )

    print_panel("RLCLI Runs", build_summary(rows))

    if not run_list.runs:
        print_panel(
            "RLCLI Run List",
            build_summary(
                [
                    (
                        "[muted]Status[/muted]",
                        "[value]No runs found yet. Start with `rlx train ...`.[/value]",
                    )
                ]
            ),
        )
        return

    table = Table(expand=True, padding=(0, 1))
    table.add_column("Run", style="accent", no_wrap=True)
    table.add_column("Tags", width=16, overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Env", no_wrap=True)
    table.add_column("Final Train", justify="right", no_wrap=True)
    table.add_column("Best Eval", justify="right", no_wrap=True)
    table.add_column("Timesteps", justify="right", no_wrap=True)

    for run in run_list.runs:
        table.add_row(
            run.run_id,
            _fmt_tags(run.tags),
            _fmt_status(run.status),
            run.environment or "[muted]—[/muted]",
            _fmt_number(run.final_rollout_reward),
            _fmt_number(run.best_eval.mean_reward if run.best_eval is not None else None),
            _fmt_int(run.total_timesteps),
        )

    print_panel("RLCLI Run List", table)


def _fmt_status(status: str | None) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status in {"failed", "interrupted"}:
        return f"[error]{status}[/error]"
    if status:
        return f"[value]{status}[/value]"
    return "[muted]—[/muted]"


def _fmt_device(run) -> str:
    if run.resolved_device:
        return f"[value]{run.resolved_device}[/value]"
    if run.requested_device:
        return f"[value]{run.requested_device}[/value]"
    return "[muted]—[/muted]"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value:.2f}[/value]"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_tags(tags: tuple[str, ...]) -> str:
    if not tags:
        return "[muted]—[/muted]"
    return f"[value]{', '.join(tags)}[/value]"
