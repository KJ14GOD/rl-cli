import typer

from rlx.console import build_summary, console, print_panel
from rlx.core.tag import TagError, add_tags

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to tag.",
)
TAGS_ARGUMENT = typer.Argument(
    ...,
    help="One or more tags to attach to the run.",
)


def tag_command(
    run_ref: str = RUN_REF_ARGUMENT,
    tags: list[str] = TAGS_ARGUMENT,
) -> None:
    """Attach one or more labels to a run."""

    try:
        result = add_tags(run_ref, tags)
    except TagError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    added = ", ".join(result.added_tags) if result.added_tags else "—"
    current = ", ".join(result.all_tags) if result.all_tags else "—"
    summary = build_summary(
        [
            ("[success]Run[/success]", f"[path]{result.run_dir}[/path]"),
            ("[muted]Run id[/muted]", f"[value]{result.run_id}[/value]"),
            ("[muted]Added[/muted]", f"[value]{added}[/value]"),
            ("[muted]Current[/muted]", f"[value]{current}[/value]"),
        ]
    )
    print_panel("RLCLI Run Tagged", summary)
