from enum import StrEnum
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme


class OutputStyle(StrEnum):
    NEON = "neon"
    MINIMAL = "minimal"
    GRAPHITE = "graphite"
    ICE = "ice"
    FOREST = "forest"
    EMBER = "ember"


STYLE_DESCRIPTIONS = {
    OutputStyle.NEON: "Bright cyan default with clear green, amber, and red status accents.",
    OutputStyle.MINIMAL: "Quiet monochrome output for a restrained, low-noise terminal look.",
    OutputStyle.GRAPHITE: "Soft slate and steel tones with understated contrast.",
    OutputStyle.ICE: "Cool blue highlights with crisp, clean path emphasis.",
    OutputStyle.FOREST: "Emerald-led palette with softer green and gold accents.",
    OutputStyle.EMBER: "Warm orange and amber palette with strong readability.",
}


THEMES = {
    OutputStyle.NEON: Theme(
        {
            "accent": "bold cyan",
            "muted": "dim",
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "path": "cyan",
            "value": "white",
        }
    ),
    OutputStyle.MINIMAL: Theme(
        {
            "accent": "bold white",
            "muted": "grey62",
            "success": "bold white",
            "warning": "bold white",
            "error": "bold white",
            "path": "white",
            "value": "white",
        }
    ),
    OutputStyle.GRAPHITE: Theme(
        {
            "accent": "bold #a1a1aa",
            "muted": "#71717a",
            "success": "bold #e4e4e7",
            "warning": "bold #d4d4d8",
            "error": "bold #f87171",
            "path": "#cbd5e1",
            "value": "#f5f5f5",
        }
    ),
    OutputStyle.ICE: Theme(
        {
            "accent": "bold #7dd3fc",
            "muted": "#94a3b8",
            "success": "bold #a7f3d0",
            "warning": "bold #fde68a",
            "error": "bold #fca5a5",
            "path": "#bfdbfe",
            "value": "#eff6ff",
        }
    ),
    OutputStyle.FOREST: Theme(
        {
            "accent": "bold #34d399",
            "muted": "#6b7280",
            "success": "bold #86efac",
            "warning": "bold #facc15",
            "error": "bold #fb7185",
            "path": "#a7f3d0",
            "value": "#ecfdf5",
        }
    ),
    OutputStyle.EMBER: Theme(
        {
            "accent": "bold #fb923c",
            "muted": "#a3a3a3",
            "success": "bold #86efac",
            "warning": "bold #fbbf24",
            "error": "bold #f87171",
            "path": "#fdba74",
            "value": "#fff7ed",
        }
    ),
}

PANEL_BORDER_STYLES = {
    OutputStyle.NEON: "accent",
    OutputStyle.MINIMAL: "muted",
    OutputStyle.GRAPHITE: "accent",
    OutputStyle.ICE: "accent",
    OutputStyle.FOREST: "accent",
    OutputStyle.EMBER: "accent",
}


class StyledConsole:
    def __init__(self, style: OutputStyle = OutputStyle.NEON) -> None:
        self._style = style
        self._console = Console(theme=THEMES[style])

    @property
    def style(self) -> OutputStyle:
        return self._style

    def set_style(self, style: OutputStyle) -> None:
        self._style = style
        self._console = Console(theme=THEMES[style])

    def print(self, *args, **kwargs) -> None:
        self._console.print(*args, **kwargs)

    @property
    def width(self) -> int:
        return self._console.width


console = StyledConsole()


def set_output_style(style: OutputStyle) -> None:
    console.set_style(style)


def build_summary(rows: list[tuple[str, str]]) -> Table:
    label_width = 12 if console.width < 72 else 14 if console.width < 96 else 16
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=label_width, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")

    for label, value in rows:
        grid.add_row(label, value)

    return grid


def print_panel(title: str, body) -> None:
    console.print()
    console.print(
        Panel(
            body,
            title=title,
            border_style=PANEL_BORDER_STYLES[console.style],
            padding=(0, 1),
        )
    )
    console.print()
