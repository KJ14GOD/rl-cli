from rich.console import Console
from rich.theme import Theme


theme = Theme(
    {
        "accent": "bold cyan",
        "muted": "dim",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "path": "cyan",
        "value": "white",
    }
)

console = Console(theme=theme)

