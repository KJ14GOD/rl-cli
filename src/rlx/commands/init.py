from pathlib import Path

import typer
from rich.panel import Panel

from rlx.console import console
from rlx.core.projects import ProjectInitError, init_project


def init_command(
    project_name: str = typer.Argument(..., help="Name of the new RL project directory."),
) -> None:
    """Create a new RLCLI project scaffold."""

    try:
        result = init_project(Path(project_name))
    except ProjectInitError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    summary = "\n".join(
        [
            f"[success]Created[/success] [path]{result.project_root}[/path]",
            f"[muted]Starter config[/muted] [path]{result.starter_config.relative_to(result.project_root)}[/path]",
            f"[muted]Directories[/muted] [value]{len(result.created_dirs)}[/value]",
            f"[muted]Next[/muted] cd {result.project_root.name}",
        ]
    )
    console.print(Panel.fit(summary, title="RLCLI Project Initialized", border_style="accent"))

