from pathlib import Path

import typer
from rich.panel import Panel

from rlx.config import ConfigError, load_config
from rlx.console import console
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

    lines = [
        [
            f"[success]Completed run[/success] [path]{result.run_dir}[/path]",
            f"[muted]Device[/muted] [value]{result.resolved_device}[/value]",
            f"[muted]Timesteps[/muted] [value]{result.total_timesteps}[/value]",
            f"[muted]Latest checkpoint[/muted] [path]{result.latest_checkpoint.relative_to(result.run_dir)}[/path]",
            f"[muted]Metrics[/muted] [path]{result.metrics_path.relative_to(result.run_dir)}[/path]",
        ]
    ][0]
    if result.best_checkpoint is not None:
        lines.append(
            f"[muted]Best checkpoint[/muted] [path]{result.best_checkpoint.relative_to(result.run_dir)}[/path]"
        )
    if result.eval_log is not None:
        lines.append(f"[muted]Eval log[/muted] [path]{result.eval_log.relative_to(result.run_dir)}[/path]")

    summary = "\n".join(lines)
    console.print(Panel.fit(summary, title="RLCLI Training Complete", border_style="accent"))
