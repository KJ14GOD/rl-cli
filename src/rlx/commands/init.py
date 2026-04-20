from pathlib import Path

import typer

from rlx.console import build_summary, console, print_panel
from rlx.core.projects import ProjectInitError, init_project


def init_command(
    project_name: str = typer.Argument(..., help="Name of the new RL project directory."),
) -> None:
    """Create a new RLCLI project scaffold.

    Examples:

        rlx init bossfight
        rlx init experiments/cartpole_lab
    """

    try:
        result = init_project(Path(project_name))
    except ProjectInitError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    summary = build_summary(
        [
            ("[success]Created[/success]", f"[path]{result.project_root}[/path]"),
            (
                "[muted]Starter config[/muted]",
                f"[path]{result.starter_config.relative_to(result.project_root)}[/path]",
            ),
            ("[muted]Directories[/muted]", f"[value]{len(result.created_dirs)}[/value]"),
            (
                "[muted]LLM setup[/muted]",
                "[value]cp .env.example .env[/value]",
            ),
            ("[muted]Next[/muted]", f"[value]cd {result.project_root.name}[/value]"),
        ]
    )
    print_panel("RLCLI Project Initialized", summary)
