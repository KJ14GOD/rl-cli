from rich.table import Table

from rlx.console import print_panel
from rlx.core.env_catalog import list_env_catalog


def envs_command() -> None:
    """List bundled train-ready Gymnasium starter environments.

    Examples:

        rlx envs
    """

    table = Table(padding=(0, 1))
    table.add_column("Env ID", no_wrap=True)
    table.add_column("Train Command", overflow="fold")
    table.add_column("Notes", overflow="fold")

    for entry in list_env_catalog():
        table.add_row(
            entry.env_id,
            f"[path]rlx train {entry.config_path}[/path]",
            f"{entry.observation} -> {entry.action}; {entry.notes}",
        )

    print_panel("RLCLI Environment Catalog", table)
