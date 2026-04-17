import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.summarize import (
    SummarizeError,
    SummaryResult,
    best_eval_run,
    best_final_run,
    summarize_target,
)

TARGET_ARGUMENT = typer.Argument(
    ...,
    help="Project path, sweep bundle path, or run ref for a compact recap.",
)


def summarize_command(target: str = TARGET_ARGUMENT) -> None:
    """Summarize a project, sweep, or compact run recap.

    Examples:

        rlx summarize .
        rlx summarize cartpole_ppo_001
        rlx summarize analysis/sweeps/cartpole_seed_lr_entropy_001
    """

    try:
        summary = summarize_target(target)
    except SummarizeError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    if summary.kind == "run":
        _render_run_summary(summary)
        return
    if summary.kind == "project":
        _render_project_summary(summary)
        return
    if summary.kind == "sweep":
        _render_sweep_summary(summary)
        return

    console.print(f"[error]Unsupported summary kind: {summary.kind}[/error]")
    raise typer.Exit(code=1)


def _render_run_summary(summary: SummaryResult) -> None:
    analysis = summary.run
    if analysis is None:
        return
    info = analysis.info
    rows = [
        ("[success]Run[/success]", f"[path]{info.run.run_dir}[/path]"),
        ("[muted]Run id[/muted]", f"[value]{info.run.run_id}[/value]"),
        ("[muted]Status[/muted]", _fmt_status(info.run.status)),
        ("[muted]Trend[/muted]", f"[value]{analysis.learning.trend}[/value]"),
        ("[muted]Final train[/muted]", _fmt_number(info.run.final_rollout_reward)),
        ("[muted]Best eval[/muted]", _fmt_eval(info.run)),
        ("[muted]Suggested next[/muted]", analysis.suggestions[0].action),
    ]
    print_panel("RLCLI Run Summary", build_summary(rows))


def _render_project_summary(summary: SummaryResult) -> None:
    project = summary.project
    if project is None:
        return

    completed = [run for run in project.runs if run.status == "completed"]
    failed = [run for run in project.runs if run.status in {"failed", "interrupted"}]
    best_final = best_final_run(project.runs)
    best_eval = best_eval_run(project.runs)

    rows = [
        ("[success]Project[/success]", f"[path]{project.project_root}[/path]"),
        ("[muted]Runs[/muted]", f"[value]{len(project.runs)}[/value]"),
        ("[muted]Completed[/muted]", f"[value]{len(completed)}[/value]"),
        ("[muted]Problem runs[/muted]", f"[value]{len(failed)}[/value]"),
        ("[muted]Best final[/muted]", _fmt_run_metric(best_final, "final")),
        ("[muted]Best eval[/muted]", _fmt_run_metric(best_eval, "eval")),
    ]
    print_panel("RLCLI Project Summary", build_summary(rows))
    print_panel("RLCLI Project Runs", _runs_table(project.runs[:10]))


def _render_sweep_summary(summary: SummaryResult) -> None:
    sweep = summary.sweep
    if sweep is None:
        return

    runs = tuple(variant.run for variant in sweep.variants if variant.run is not None)
    completed = [variant for variant in sweep.variants if variant.status == "completed"]
    failed = [variant for variant in sweep.variants if variant.status != "completed"]
    best_final = best_final_run(runs)
    best_eval = best_eval_run(runs)

    rows = [
        ("[success]Sweep[/success]", f"[value]{sweep.name}[/value]"),
        ("[muted]Bundle[/muted]", f"[path]{sweep.bundle_dir}[/path]"),
        ("[muted]Variants[/muted]", f"[value]{len(sweep.variants)}[/value]"),
        ("[muted]Completed[/muted]", f"[value]{len(completed)}[/value]"),
        ("[muted]Failed[/muted]", f"[value]{len(failed)}[/value]"),
        ("[muted]Best final[/muted]", _fmt_run_metric(best_final, "final")),
        ("[muted]Best eval[/muted]", _fmt_run_metric(best_eval, "eval")),
    ]
    print_panel("RLCLI Sweep Summary", build_summary(rows))
    print_panel("RLCLI Sweep Variants", _sweep_table(sweep.variants))


def _runs_table(runs) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Run", style="accent", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Env", no_wrap=True)
    table.add_column("Final", justify="right", no_wrap=True)
    table.add_column("Best Eval", justify="right", no_wrap=True)

    for run in runs:
        table.add_row(
            run.run_id,
            _fmt_status(run.status),
            run.environment or "[muted]—[/muted]",
            _fmt_number(run.final_rollout_reward),
            _fmt_eval(run),
        )

    return table


def _sweep_table(variants) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Variant", style="accent", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Run", no_wrap=True)
    table.add_column("Best Eval", justify="right", no_wrap=True)
    table.add_column("Mutations", overflow="fold")

    for variant in variants:
        table.add_row(
            f"{variant.index:03d}",
            _fmt_status(variant.status),
            variant.run_id or "[muted]—[/muted]",
            _fmt_eval(variant.run) if variant.run is not None else "[muted]—[/muted]",
            _fmt_mutations(variant.mutations, variant.error),
        )

    return table


def _fmt_run_metric(run, field: str) -> str:
    if run is None:
        return "[muted]—[/muted]"
    if field == "eval":
        value = run.best_eval.mean_reward if run.best_eval is not None else None
    else:
        value = run.final_rollout_reward
    if value is None:
        return f"[value]{run.run_id}[/value] [muted](—)[/muted]"
    return f"[value]{run.run_id}[/value] [muted]({value:.2f})[/muted]"


def _fmt_status(status: str | None) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status in {"failed", "interrupted"}:
        return f"[error]{status}[/error]"
    if status:
        return f"[value]{status}[/value]"
    return "[muted]—[/muted]"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value:.2f}[/value]"


def _fmt_eval(run) -> str:
    if run is None or run.best_eval is None or run.best_eval.mean_reward is None:
        return "[muted]—[/muted]"
    return f"[value]{run.best_eval.mean_reward:.2f}[/value]"


def _fmt_mutations(mutations: dict[str, object], error: str | None) -> str:
    parts = [f"{key}={value}" for key, value in mutations.items()]
    if error:
        parts.append(f"error={error}")
    if not parts:
        return "[muted]—[/muted]"
    return f"[value]{'; '.join(parts)}[/value]"
