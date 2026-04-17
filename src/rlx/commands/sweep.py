import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.sweep import SweepError, run_sweep

SWEEP_CONFIG_ARGUMENT = typer.Argument(
    ...,
    help="Path to the sweep YAML config.",
)


def sweep_command(sweep_config_path: str = SWEEP_CONFIG_ARGUMENT) -> None:
    """Train many config variants from one sweep file.

    Examples:

        rlx sweep configs/cartpole_sweep.yaml
    """

    try:
        result = run_sweep(sweep_config_path)
    except SweepError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    completed = [variant for variant in result.variants if variant.status == "completed"]
    failed = [variant for variant in result.variants if variant.status != "completed"]

    summary = build_summary(
        [
            ("[success]Sweep[/success]", f"[value]{result.sweep_name}[/value]"),
            ("[muted]Output[/muted]", f"[path]{result.bundle_dir}[/path]"),
            (
                "[muted]Manifest[/muted]",
                f"[path]{result.manifest_path.relative_to(result.project_root)}[/path]",
            ),
            ("[muted]Variants[/muted]", f"[value]{len(result.variants)}[/value]"),
            ("[muted]Completed[/muted]", f"[value]{len(completed)}[/value]"),
            ("[muted]Failed[/muted]", f"[value]{len(failed)}[/value]"),
        ]
    )
    print_panel("RLCLI Sweep Complete", summary)

    table = Table(expand=True, padding=(0, 1))
    table.add_column("Variant", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Run", no_wrap=True)
    table.add_column("Config", overflow="fold")
    table.add_column("Mutations", overflow="fold")

    for variant in result.variants:
        table.add_row(
            f"[value]{variant.index:03d}[/value]",
            _fmt_status(variant.status),
            f"[value]{variant.run_id}[/value]" if variant.run_id else "[muted]—[/muted]",
            f"[path]{variant.config_path.relative_to(result.project_root)}[/path]",
            _fmt_mutations(variant.mutations, variant.error),
        )

    print_panel("RLCLI Sweep Variants", table)

    if failed:
        raise typer.Exit(code=1)


def _fmt_status(status: str) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    return f"[error]{status}[/error]"


def _fmt_mutations(mutations: dict[str, object], error: str | None) -> str:
    parts = [f"{key}={value}" for key, value in mutations.items()]
    if error:
        parts.append(f"error={error}")
    if not parts:
        return "[muted]—[/muted]"
    return f"[value]{'; '.join(parts)}[/value]"
