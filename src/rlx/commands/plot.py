import typer

from rlx.console import build_summary, console, print_panel
from rlx.core.plot import PlotError, plot_runs

RUN_REFS_ARGUMENT = typer.Argument(
    ...,
    help="One or more run paths or run ids to plot.",
)


def plot_command(run_refs: list[str] = RUN_REFS_ARGUMENT) -> None:
    """Generate training/eval plots for one or more runs.

    Examples:

        rlx plot cartpole_ppo_001
        rlx plot cartpole_ppo_001 cartpole_ppo_002
    """

    try:
        bundle = plot_runs(run_refs)
    except PlotError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    rows = [
        ("[success]Output[/success]", f"[path]{bundle.output_dir}[/path]"),
        ("[muted]Mode[/muted]", f"[value]{bundle.mode}[/value]"),
        ("[muted]Runs[/muted]", f"[value]{len(bundle.run_ids)}[/value]"),
        (
            "[muted]Manifest[/muted]",
            f"[path]{bundle.manifest_path.relative_to(bundle.project_root)}[/path]",
        ),
    ]
    for artifact in bundle.artifacts:
        rows.append(
            (
                f"[muted]{artifact.key.replace('_', ' ')}[/muted]",
                f"[path]{artifact.path.relative_to(bundle.project_root)}[/path]",
            )
        )

    print_panel("RLCLI Plot Bundle", build_summary(rows))
