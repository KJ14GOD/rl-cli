import typer
from rich.traceback import install as install_rich_traceback

from rlx.commands.init import init_command


install_rich_traceback(show_locals=False)

app = typer.Typer(
    name="rlx",
    help="Local-first CLI for reinforcement learning experiments.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main() -> None:
    """RLCLI keeps RL experiment workflows structured, reproducible, and local-first."""


app.command("init")(init_command)
