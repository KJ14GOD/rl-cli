from pathlib import Path

import typer

from rlx.console import build_summary, console, print_panel
from rlx.rl import VideoRenderError, render_checkpoint_video

CHECKPOINT_PATH_ARGUMENT = typer.Argument(..., help="Path to a saved PPO checkpoint.")
EPISODES_OPTION = typer.Option(
    1,
    "--episodes",
    min=1,
    help="Number of episodes to render into the output bundle.",
)
DEVICE_OPTION = typer.Option(
    None,
    "--device",
    help="Override the device used for standalone video rendering.",
)


def video_command(
    checkpoint_path: str = CHECKPOINT_PATH_ARGUMENT,
    episodes: int = EPISODES_OPTION,
    device: str | None = DEVICE_OPTION,
) -> None:
    """Render GIF episodes from a saved PPO checkpoint.

    Examples:

        rlx video runs/cartpole_ppo_001/checkpoints/best.zip
        rlx video runs/cartpole_ppo_001/checkpoints/latest.zip --episodes 3
    """

    try:
        result = render_checkpoint_video(Path(checkpoint_path), episodes=episodes, device=device)
    except VideoRenderError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    summary = build_summary(
        [
            ("[success]Checkpoint[/success]", f"[path]{result.checkpoint_path}[/path]"),
            ("[muted]Run[/muted]", f"[path]{result.run_dir}[/path]"),
            ("[muted]Environment[/muted]", f"[value]{result.environment}[/value]"),
            ("[muted]Device[/muted]", f"[value]{result.resolved_device}[/value]"),
            ("[muted]Episodes[/muted]", f"[value]{result.episodes}[/value]"),
            ("[muted]FPS[/muted]", f"[value]{result.fps}[/value]"),
            ("[muted]Mean reward[/muted]", f"[value]{result.mean_reward:.2f}[/value]"),
            (
                "[muted]Mean ep length[/muted]",
                f"[value]{result.mean_episode_length:.2f}[/value]",
            ),
            (
                "[muted]Output[/muted]",
                f"[path]{result.output_dir.relative_to(result.run_dir)}[/path]",
            ),
            (
                "[muted]Manifest[/muted]",
                f"[path]{result.manifest_path.relative_to(result.run_dir)}[/path]",
            ),
        ]
    )
    print_panel("RLCLI Video Rendered", summary)
