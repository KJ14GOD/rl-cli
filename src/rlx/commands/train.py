from pathlib import Path

import typer

from rlx.config import ConfigError, load_config
from rlx.console import build_summary, console, print_panel
from rlx.core.runs import RunPreparationError, prepare_run
from rlx.rl import TrainingError, train_ppo


def train_command(
    config_path: str = typer.Argument(..., help="Path to the YAML experiment config."),
) -> None:
    """Train a PPO experiment from a validated RLCLI config."""

    resolved_config_path = Path(config_path).expanduser().resolve()

    try:
        config = load_config(resolved_config_path)
        run = prepare_run(config, resolved_config_path)
        result = train_ppo(config, run)
    except (ConfigError, RunPreparationError, TrainingError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    rows = [
        ("[success]Completed run[/success]", f"[path]{result.run_dir}[/path]"),
        ("[muted]Device[/muted]", f"[value]{result.resolved_device}[/value]"),
        ("[muted]Timesteps[/muted]", f"[value]{result.total_timesteps}[/value]"),
        (
            "[muted]Latest checkpoint[/muted]",
            f"[path]{result.latest_checkpoint.relative_to(result.run_dir)}[/path]",
        ),
        (
            "[muted]Metrics[/muted]",
            f"[path]{result.metrics_path.relative_to(result.run_dir)}[/path]",
        ),
    ]
    if result.best_checkpoint is not None:
        rows.append(
            (
                "[muted]Best checkpoint[/muted]",
                f"[path]{result.best_checkpoint.relative_to(result.run_dir)}[/path]",
            )
        )
    if result.eval_log is not None:
        rows.append(
            (
                "[muted]Eval log[/muted]",
                f"[path]{result.eval_log.relative_to(result.run_dir)}[/path]",
            )
        )

    summary = build_summary(rows)
    print_panel("RLCLI Training Complete", summary)
