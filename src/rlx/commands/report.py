import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import typer

from rlx.console import build_summary, console, print_panel
from rlx.core.report import (
    ReportError,
    build_preview_report,
    build_preview_research_report,
    build_report,
)

TARGET_ARGUMENT = typer.Argument(
    None,
    help="Run path, run id, research bundle path, or research manifest to render.",
)
PREVIEW_OPTION = typer.Option(
    False,
    "--preview",
    help="Build a sample report with preview data instead of reading a run.",
)
PREVIEW_KIND_OPTION = typer.Option(
    "run",
    "--preview-kind",
    help="Preview type: run or research.",
)
SERVE_OPTION = typer.Option(False, "--serve", help="Serve the generated report locally.")
HOST_OPTION = typer.Option("127.0.0.1", "--host", help="Report preview host.")
PORT_OPTION = typer.Option(8766, "--port", help="Report preview port.")
OPEN_OPTION = typer.Option(False, "--open", help="Open the report in a browser.")


def report_command(
    target: str | None = TARGET_ARGUMENT,
    preview: bool = PREVIEW_OPTION,
    preview_kind: str = PREVIEW_KIND_OPTION,
    serve: bool = SERVE_OPTION,
    host: str = HOST_OPTION,
    port: int = PORT_OPTION,
    open_browser: bool = OPEN_OPTION,
) -> None:
    """Generate a local HTML report for one run or research bundle.

    Examples:

        rlx report --preview --serve
        rlx report --preview --preview-kind research --serve
        rlx report cartpole_ppo_001
        rlx report cartpole_ppo_001 --serve
        rlx report runs/cartpole_ppo_001
        rlx report analysis/research/cartpole_ppo_001_research_001
    """

    try:
        if preview:
            if preview_kind == "run":
                result = build_preview_report()
            elif preview_kind == "research":
                result = build_preview_research_report()
            else:
                raise ReportError("--preview-kind must be either run or research.")
        else:
            if target is None:
                raise ReportError("Pass a run/research target or use --preview.")
            result = build_report(target)
    except ReportError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    relative_index = result.index_path.relative_to(result.project_root)
    url = f"http://{host}:{port}/{relative_index.as_posix()}"
    print_panel(
        "RLX Web Report",
        build_summary(
            [
                ("[success]Report[/success]", f"[path]{result.index_path}[/path]"),
                ("[muted]Kind[/muted]", f"[value]{result.kind}[/value]"),
                ("[muted]Target[/muted]", f"[value]{result.target}[/value]"),
                (
                    "[muted]Output[/muted]",
                    f"[path]{result.output_dir.relative_to(result.project_root)}[/path]",
                ),
                (
                    "[muted]Preview URL[/muted]",
                    f"[value]{url}[/value]" if serve else "[muted]—[/muted]",
                ),
            ]
        ),
    )

    if serve:
        _serve_report(result.project_root, host=host, port=port, url=url, open_browser=open_browser)


def _serve_report(project_root, *, host: str, port: int, url: str, open_browser: bool) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(project_root))
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        console.print(f"[error]Could not start report server: {exc}[/error]")
        raise typer.Exit(code=1) from exc

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[muted]Report server stopped.[/muted]")
    finally:
        server.server_close()
