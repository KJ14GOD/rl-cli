import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import typer

from rlx.console import build_summary, console, print_panel
from rlx.core.report import (
    ReportError,
    WebAppProject,
    build_dashboard,
    build_preview_dashboard,
    resolve_web_app_project,
    web_app_html,
    web_file_path,
    web_project_payload,
    web_research_payload,
    web_run_logs_payload,
    web_run_payload,
)

PATH_ARGUMENT = typer.Argument(
    None,
    help="Project path. Defaults to the nearest RLX project.",
)

HOST_OPTION = typer.Option("127.0.0.1", "--host", help="Dashboard host.")
PORT_OPTION = typer.Option(8765, "--port", help="Dashboard port.")
OPEN_OPTION = typer.Option(False, "--open", help="Open the dashboard in a browser.")
EXPORT_OPTION = typer.Option(
    False,
    "--export",
    help="Write the dashboard HTML and exit without starting a server.",
)
DEMO_OPTION = typer.Option(
    False,
    "--demo",
    help="Serve a sample dashboard without requiring an initialized project.",
)


def dashboard_command(
    path: Path | None = PATH_ARGUMENT,
    host: str = HOST_OPTION,
    port: int = PORT_OPTION,
    open_browser: bool = OPEN_OPTION,
    export: bool = EXPORT_OPTION,
    demo: bool = DEMO_OPTION,
) -> None:
    """Launch or export a local visual dashboard for an RLX project.

    Examples:

        rlx dashboard
        rlx dashboard --demo
        rlx dashboard --port 9000
        rlx dashboard bossfight --export
    """

    try:
        if export:
            result = build_preview_dashboard(path) if demo else build_dashboard(path)
        else:
            result = None
        app_project = resolve_web_app_project(path, demo=demo)
    except ReportError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    url = (
        f"http://{host}:{port}/analysis/dashboard/index.html"
        if export
        else f"http://{host}:{port}/"
    )
    rows = [
        (
            "[success]Dashboard[/success]",
            f"[path]{result.index_path}[/path]" if result else "[value]connected app[/value]",
        ),
        ("[muted]Project[/muted]", f"[path]{app_project.project_root}[/path]"),
    ]
    if result:
        rows.extend(
            [
                ("[muted]Runs[/muted]", f"[value]{result.runs}[/value]"),
                (
                    "[muted]Research bundles[/muted]",
                    f"[value]{result.research_bundles}[/value]",
                ),
            ]
        )
    else:
        rows.append(("[muted]URL[/muted]", f"[value]{url}[/value]"))
    print_panel("RLX Local Dashboard", build_summary(rows))

    if export:
        return

    _serve_dashboard_app(app_project, host=host, port=port, url=url, open_browser=open_browser)


def _serve_dashboard_app(
    project: WebAppProject,
    *,
    host: str,
    port: int,
    url: str,
    open_browser: bool,
) -> None:
    handler = _make_dashboard_handler(project)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        console.print(f"[error]Could not start dashboard server: {exc}[/error]")
        raise typer.Exit(code=1) from exc

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[muted]Dashboard stopped.[/muted]")
    finally:
        server.server_close()


def _make_dashboard_handler(project: WebAppProject):
    class RLXDashboardHandler(BaseHTTPRequestHandler):
        server_version = "RLXDashboard/1.0"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)

            try:
                if route in {"/", "/app"}:
                    self._send_html(web_app_html())
                elif route == "/api/health":
                    self._send_json({"ok": True, "demo": project.demo})
                elif route == "/api/project":
                    self._send_json(web_project_payload(project))
                elif route == "/api/run":
                    ref = _single_query_value(query, "ref")
                    self._send_json(web_run_payload(project, ref))
                elif route == "/api/logs":
                    ref = _single_query_value(query, "ref")
                    tail = _int_query_value(query, "tail", default=80, minimum=1, maximum=500)
                    self._send_json(web_run_logs_payload(project, ref, tail=tail))
                elif route == "/api/research":
                    bundle = query.get("bundle", [None])[0]
                    self._send_json(web_research_payload(project, bundle))
                elif route.startswith("/files/"):
                    relative = unquote(route.removeprefix("/files/"))
                    self._send_file(web_file_path(project, relative))
                else:
                    self._send_json({"error": "Not found"}, status=404)
            except ReportError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._send_json({"error": f"Dashboard server error: {exc}"}, status=500)

        def _send_html(self, text: str) -> None:
            encoded = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict, *, status: int = 200) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, path: Path) -> None:
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return RLXDashboardHandler


def _single_query_value(query: dict[str, list[str]], key: str) -> str:
    value = query.get(key, [""])[0]
    if not value:
        raise ReportError(f"Missing query parameter: {key}")
    return value


def _int_query_value(
    query: dict[str, list[str]],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = query.get(key, [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"Query parameter must be an integer: {key}") from exc
    return min(max(value, minimum), maximum)
