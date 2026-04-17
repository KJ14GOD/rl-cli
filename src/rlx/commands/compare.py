import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.compare import CompareError, diff_config_values, load_comparisons

RUN_REFS_ARGUMENT = typer.Argument(
    ...,
    help="Two or more run paths or run ids to compare.",
)


def compare_command(run_refs: list[str] = RUN_REFS_ARGUMENT) -> None:
    """Compare multiple runs side-by-side.

    Examples:

        rlx compare cartpole_ppo_001 cartpole_ppo_002
        rlx compare runs/cartpole_ppo_001 runs/cartpole_ppo_002 runs/cartpole_ppo_003
    """

    try:
        runs = load_comparisons(run_refs)
    except CompareError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    best_final_reward = _max_metric([run.final_rollout_reward for run in runs])
    best_eval_reward = _max_metric(
        [run.best_eval.mean_reward if run.best_eval is not None else None for run in runs]
    )

    highlight_summary = build_summary(
        [
            ("[muted]Runs[/muted]", f"[value]{len(runs)}[/value]"),
            (
                "[success]Best final train[/success]",
                _leader_text(runs, "final_rollout_reward", best_final_reward),
            ),
            (
                "[success]Best eval mean[/success]",
                _leader_text(
                    runs,
                    "best_eval.mean_reward",
                    best_eval_reward,
                    include_eval_source=True,
                ),
            ),
        ]
    )

    summary_table = Table(expand=True, padding=(0, 1))
    summary_table.add_column("Run", style="accent", no_wrap=True)
    summary_table.add_column("Status", no_wrap=True)
    summary_table.add_column("Env", no_wrap=True)
    summary_table.add_column("Device", no_wrap=True)
    summary_table.add_column("Final Train", justify="right", no_wrap=True)
    summary_table.add_column("Best Eval Mean", justify="right", no_wrap=True)
    summary_table.add_column("Timesteps", justify="right", no_wrap=True)

    for run in runs:
        summary_table.add_row(
            run.run_id,
            _style_status(run.status),
            run.environment or "[muted]—[/muted]",
            _display_device(run),
            _fmt_number(run.final_rollout_reward, best=best_final_reward),
            _fmt_number(
                run.best_eval.mean_reward if run.best_eval is not None else None,
                best=best_eval_reward,
            ),
            _fmt_int(run.total_timesteps),
        )

    artifact_table = Table(expand=True, padding=(0, 1))
    artifact_table.add_column("Run", style="accent", no_wrap=True)
    artifact_table.add_column("Best Ckpt", overflow="fold")
    artifact_table.add_column("Latest Ckpt", overflow="fold")
    artifact_table.add_column("Latest Eval", overflow="fold")
    artifact_table.add_column("Best Eval Src", overflow="fold")
    artifact_table.add_column("Video", overflow="fold")

    for run in runs:
        artifact_table.add_row(
            run.run_id,
            _fmt_path(run.best_checkpoint),
            _fmt_path(run.latest_checkpoint),
            _fmt_eval_source(run.latest_eval, label="latest"),
            _fmt_eval_source(run.best_eval, label="best"),
            _fmt_path(run.last_video_manifest),
        )

    print_panel("RLCLI Compare Highlights", highlight_summary)
    print_panel("RLCLI Run Comparison", summary_table)
    print_panel("RLCLI Run Artifacts", artifact_table)

    config_diffs = diff_config_values(runs)
    if config_diffs:
        diff_table = Table(expand=True, padding=(0, 1))
        diff_table.add_column("Config", style="accent", no_wrap=True)
        for run in runs:
            diff_table.add_column(run.run_id, overflow="fold")

        for key, values in config_diffs:
            diff_table.add_row(key, *[f"[value]{value}[/value]" for value in values])

        print_panel("RLCLI Config Differences", diff_table)
    else:
        summary = build_summary(
            [
                (
                    "[success]Configs[/success]",
                    "[value]No config differences across compared runs.[/value]",
                )
            ]
        )
        print_panel("RLCLI Config Differences", summary)


def _display_device(run) -> str:
    if run.resolved_device:
        return f"[value]{run.resolved_device}[/value]"
    if run.requested_device:
        return f"[value]{run.requested_device}[/value]"
    return "[muted]—[/muted]"


def _fmt_number(value: float | None, *, best: float | None = None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    if best is not None and value == best:
        return f"[success]{value:.2f}[/success]"
    return f"[value]{value:.2f}[/value]"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_path(value: str | None) -> str:
    if not value:
        return "[muted]—[/muted]"
    return f"[path]{value}[/path]"


def _fmt_eval_source(value, *, label: str) -> str:
    if value is None:
        return "[muted]—[/muted]"
    if value.mean_reward is None:
        return f"[path]{value.source}[/path]"
    return (
        f"[path]{value.source}[/path] "
        f"[muted]({label}: {value.mean_reward:.2f})[/muted]"
    )


def _style_status(status: str | None) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status in {"failed", "interrupted"}:
        return f"[error]{status}[/error]"
    if status:
        return f"[value]{status}[/value]"
    return "[muted]—[/muted]"


def _max_metric(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _leader_text(
    runs,
    attribute_path: str,
    best_value: float | None,
    *,
    include_eval_source: bool = False,
) -> str:
    if best_value is None:
        return "[muted]—[/muted]"

    leaders = []
    for run in runs:
        value = _nested_attr(run, attribute_path)
        if value == best_value:
            label = f"[success]{run.run_id}[/success] [muted]({best_value:.2f})[/muted]"
            if include_eval_source and run.best_eval is not None:
                label += f" [path]{run.best_eval.source}[/path]"
            leaders.append(label)

    return ", ".join(leaders) if leaders else "[muted]—[/muted]"


def _nested_attr(obj, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current
