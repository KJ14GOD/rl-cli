import typer
from rich.traceback import install as install_rich_traceback

from rlx.commands.analyze import analyze_command
from rlx.commands.compare import compare_command
from rlx.commands.eval import eval_command
from rlx.commands.explain_metrics import explain_metrics_command
from rlx.commands.info import info_command
from rlx.commands.init import init_command
from rlx.commands.ls import ls_command
from rlx.commands.plot import plot_command
from rlx.commands.resume import resume_command
from rlx.commands.styles import styles_command
from rlx.commands.sweep import sweep_command
from rlx.commands.tag import tag_command
from rlx.commands.train import train_command
from rlx.commands.video import video_command
from rlx.console import OutputStyle, console, set_output_style
from rlx.settings import AppSettings, load_settings, save_settings

install_rich_traceback(show_locals=False)

STYLE_OPTION = typer.Option(
    None,
    "--style",
    help="Set or save the RLCLI output style. See `rlx styles`.",
    case_sensitive=False,
    show_choices=False,
)

app = typer.Typer(
    name="rlx",
    help="Local-first CLI for reinforcement learning experiments.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    style: OutputStyle | None = STYLE_OPTION,
) -> None:
    """RLCLI keeps RL experiment workflows structured, reproducible, and local-first."""

    settings = load_settings()
    resolved_style = style or settings.style
    set_output_style(resolved_style)

    if ctx.invoked_subcommand is None:
        if style is not None:
            path = save_settings(AppSettings(style=style))
            console.print()
            console.print(
                f"[success]Saved default style[/success] [value]{style.value}[/value] "
                f"[muted]at[/muted] [path]{path}[/path]"
            )
            console.print()
            raise typer.Exit()

        typer.echo(ctx.get_help())
        raise typer.Exit()


app.command("analyze")(analyze_command)
app.command("compare")(compare_command)
app.command("eval")(eval_command)
app.command("explain-metrics")(explain_metrics_command)
app.command("info")(info_command)
app.command("init")(init_command)
app.command("ls")(ls_command)
app.command("plot")(plot_command)
app.command("resume")(resume_command)
app.command("styles")(styles_command)
app.command("sweep")(sweep_command)
app.command("tag")(tag_command)
app.command("train")(train_command)
app.command("video")(video_command)
