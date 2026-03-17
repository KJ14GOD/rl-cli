from rich.table import Table

from rlx.console import STYLE_DESCRIPTIONS, OutputStyle, print_panel
from rlx.settings import load_settings


def styles_command() -> None:
    """Show the built-in RLCLI output styles."""

    saved_style = load_settings().style

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=10, no_wrap=True)
    table.add_column(width=26, no_wrap=True)
    table.add_column(ratio=1, overflow="fold")

    for style in OutputStyle:
        marker = "[success]saved[/success]" if style == saved_style else "[muted]available[/muted]"
        command = f"[value]rlx --style {style.value}[/value]"
        description = f"[muted]{STYLE_DESCRIPTIONS[style]}[/muted]"
        table.add_row(f"[accent]{style.value}[/accent]", marker, f"{command}  {description}")

    print_panel("RLCLI Styles", table)
