import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.rl import EvaluationError, evaluate_checkpoints

CHECKPOINT_PATHS_ARGUMENT = typer.Argument(
    None,
    help="One or more PPO checkpoint paths to evaluate.",
)
RUN_OPTION = typer.Option(
    None,
    "--run",
    help=(
        "Path to a run directory. Uses latest.zip by default; "
        "pair with --all-checkpoints to sweep it."
    ),
)
ALL_CHECKPOINTS_OPTION = typer.Option(
    False,
    "--all-checkpoints",
    help="Evaluate all canonical checkpoints from the run passed to --run.",
)
EPISODES_OPTION = typer.Option(
    None,
    "--episodes",
    min=1,
    help="Override the number of evaluation episodes from the config snapshot.",
)
DEVICE_OPTION = typer.Option(
    None,
    "--device",
    help="Override the device used for standalone evaluation.",
)


def eval_command(
    checkpoint_paths: list[str] = CHECKPOINT_PATHS_ARGUMENT,
    run: str | None = RUN_OPTION,
    all_checkpoints: bool = ALL_CHECKPOINTS_OPTION,
    episodes: int | None = EPISODES_OPTION,
    device: str | None = DEVICE_OPTION,
) -> None:
    """Evaluate a saved PPO checkpoint using the run's snapped config."""

    try:
        results = evaluate_checkpoints(
            checkpoint_paths=checkpoint_paths,
            run_path=run,
            all_checkpoints=all_checkpoints,
            episodes=episodes,
            device=device,
        )
    except EvaluationError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    if len(results) > 1:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=22, no_wrap=True)
        table.add_column(width=18, no_wrap=True)
        table.add_column(width=16, no_wrap=True)
        table.add_column(ratio=1, overflow="fold")

        for result in results:
            table.add_row(
                f"[path]{result.checkpoint_name}[/path]",
                f"[value]{result.mean_reward:.2f} +/- {result.std_reward:.2f}[/value]",
                f"[value]{result.mean_episode_length:.2f}[/value]",
                f"[path]{result.result_path.relative_to(result.run_dir)}[/path]",
            )

        print_panel("RLCLI Batch Evaluation Complete", table)
        return

    result = results[0]
    summary = build_summary(
        [
            ("[success]Checkpoint[/success]", f"[path]{result.checkpoint_path}[/path]"),
            ("[muted]Run[/muted]", f"[path]{result.run_dir}[/path]"),
            ("[muted]Environment[/muted]", f"[value]{result.environment}[/value]"),
            ("[muted]Device[/muted]", f"[value]{result.resolved_device}[/value]"),
            ("[muted]Episodes[/muted]", f"[value]{result.episodes}[/value]"),
            (
                "[muted]Mean reward[/muted]",
                f"[value]{result.mean_reward:.2f} +/- {result.std_reward:.2f}[/value]",
            ),
            (
                "[muted]Reward range[/muted]",
                f"[value]{result.min_reward:.2f} .. {result.max_reward:.2f}[/value]",
            ),
            (
                "[muted]Median reward[/muted]",
                f"[value]{result.median_reward:.2f}[/value]",
            ),
            (
                "[muted]Mean ep length[/muted]",
                (
                    f"[value]{result.mean_episode_length:.2f}"
                    f" +/- {result.std_episode_length:.2f}[/value]"
                ),
            ),
            (
                "[muted]Length range[/muted]",
                f"[value]{result.min_episode_length} .. {result.max_episode_length}[/value]",
            ),
            (
                "[muted]Result[/muted]",
                f"[path]{result.result_path.relative_to(result.run_dir)}[/path]",
            ),
        ]
    )
    print_panel("RLCLI Evaluation Complete", summary)
