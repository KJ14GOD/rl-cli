from pathlib import Path

import typer

from rlx.config import ConfigError, load_config
from rlx.console import build_summary, console, print_panel
from rlx.core.compare import CompareError, resolve_run_ref
from rlx.core.runs import RunPreparationError, prepare_run
from rlx.rl import TrainingError, train_ppo

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to resume from.",
)
CHECKPOINT_OPTION = typer.Option(
    "latest",
    "--checkpoint",
    help="Checkpoint selector: latest, best, or an explicit checkpoint path.",
)
TIMESTEPS_OPTION = typer.Option(
    None,
    "--timesteps",
    min=1,
    help="Additional timesteps to train after resuming. Defaults to the config total.",
)


def resume_command(
    run_ref: str = RUN_REF_ARGUMENT,
    checkpoint: str = CHECKPOINT_OPTION,
    timesteps: int | None = TIMESTEPS_OPTION,
) -> None:
    """Resume training from a prior checkpoint into a new tracked run.

    Examples:

        rlx resume cartpole_ppo_001
        rlx resume cartpole_ppo_001 --checkpoint best --timesteps 50000
        rlx resume cartpole_ppo_001 --checkpoint step_10000.zip
    """

    try:
        source_run_dir = resolve_run_ref(run_ref, Path.cwd().resolve())
        source_config_path = source_run_dir / "config_snapshot.yaml"
        config = load_config(source_config_path)
        resume_checkpoint = _resolve_resume_checkpoint(source_run_dir, checkpoint)
        run = prepare_run(config, source_config_path)
        result = train_ppo(
            config,
            run,
            resume_checkpoint=resume_checkpoint,
            additional_timesteps=timesteps,
            lineage_metadata={
                "resumed_from_run": source_run_dir.name,
                "resumed_from_checkpoint": str(_relative_to_run(resume_checkpoint, source_run_dir)),
            },
        )
    except (CompareError, ConfigError, RunPreparationError, TrainingError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    rows = [
        ("[success]Completed run[/success]", f"[path]{result.run_dir}[/path]"),
        ("[muted]Resumed from[/muted]", f"[value]{source_run_dir.name}[/value]"),
        (
            "[muted]Checkpoint[/muted]",
            f"[path]{_relative_to_run(resume_checkpoint, source_run_dir)}[/path]",
        ),
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

    print_panel("RLCLI Training Resumed", build_summary(rows))


def _resolve_resume_checkpoint(source_run_dir: Path, checkpoint: str) -> Path:
    checkpoints_dir = source_run_dir / "checkpoints"
    if checkpoint == "latest":
        path = checkpoints_dir / "latest.zip"
    elif checkpoint == "best":
        path = checkpoints_dir / "best.zip"
    else:
        explicit = Path(checkpoint).expanduser()
        if explicit.exists():
            path = explicit.resolve()
        else:
            path = (checkpoints_dir / checkpoint).resolve()

    if not path.exists():
        raise TrainingError(f"Resume checkpoint not found: {path}")
    if not path.is_file():
        raise TrainingError(f"Resume checkpoint path is not a file: {path}")
    return path


def _relative_to_run(path: Path, run_dir: Path) -> Path:
    try:
        return path.relative_to(run_dir)
    except ValueError:
        return path
