import typer

from rlx.console import build_summary, console, print_panel
from rlx.core.info import RunInfo, RunInfoError, load_run_info

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to inspect.",
)


def info_command(run_ref: str = RUN_REF_ARGUMENT) -> None:
    """Inspect one RLCLI run and summarize its tracked state."""

    try:
        info = load_run_info(run_ref)
    except RunInfoError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Run Overview", _build_overview(info))
    print_panel("RLCLI Run Performance", _build_performance(info))
    print_panel("RLCLI Run Artifacts", _build_artifacts(info))
    print_panel("RLCLI Config Summary", build_summary(list(info.config_summary)))


def _build_overview(info: RunInfo):
    rows = [
        ("[success]Run[/success]", f"[path]{info.run.run_dir}[/path]"),
        ("[muted]Run id[/muted]", f"[value]{info.run.run_id}[/value]"),
        ("[muted]Run name[/muted]", _fmt_text(info.run.run_name)),
        ("[muted]Tags[/muted]", _fmt_tags(info.run.tags)),
        ("[muted]Status[/muted]", _fmt_status(info.run.status)),
        ("[muted]Environment[/muted]", _fmt_text(info.run.environment)),
        ("[muted]Device[/muted]", _fmt_device(info)),
        (
            "[muted]Timesteps[/muted]",
            _fmt_int(info.run.total_timesteps),
        ),
        ("[muted]Created[/muted]", _fmt_text(info.created_at)),
        ("[muted]Started[/muted]", _fmt_text(info.started_at)),
        ("[muted]Completed[/muted]", _fmt_text(info.completed_at)),
    ]

    if info.failed_at is not None:
        rows.append(("[muted]Failed[/muted]", _fmt_text(info.failed_at)))
    if info.interrupted_at is not None:
        rows.append(("[muted]Interrupted[/muted]", _fmt_text(info.interrupted_at)))
    if info.error is not None:
        rows.append(("[error]Error[/error]", f"[error]{info.error}[/error]"))

    return build_summary(rows)


def _build_performance(info: RunInfo):
    rows = [
        ("[success]Final train[/success]", _fmt_number(info.run.final_rollout_reward)),
        ("[muted]Best train[/muted]", _fmt_number(info.run.best_rollout_reward)),
        ("[muted]Latest eval[/muted]", _fmt_eval(info.run.latest_eval)),
        ("[muted]Best eval[/muted]", _fmt_eval(info.run.best_eval)),
        ("[muted]Last eval at[/muted]", _fmt_text(info.last_eval_at)),
        ("[muted]Last video at[/muted]", _fmt_text(info.last_video_at)),
        ("[muted]Last plot at[/muted]", _fmt_text(info.last_plot_at)),
    ]
    return build_summary(rows)


def _build_artifacts(info: RunInfo):
    rows = [
        ("[success]Config[/success]", f"[path]{info.config_snapshot}[/path]"),
        ("[muted]Metadata[/muted]", f"[path]{info.metadata_file}[/path]"),
        ("[muted]Metrics[/muted]", f"[path]{info.metrics_file}[/path]"),
        ("[muted]Latest ckpt[/muted]", _fmt_path(info.run.latest_checkpoint)),
        ("[muted]Best ckpt[/muted]", _fmt_path(info.run.best_checkpoint)),
        ("[muted]Latest eval[/muted]", _fmt_path(info.last_eval_result)),
        ("[muted]Eval log[/muted]", _fmt_path(info.eval_log)),
        ("[muted]Latest video[/muted]", _fmt_path(info.last_video_manifest)),
        ("[muted]Latest plot[/muted]", _fmt_path(info.last_plot_manifest)),
        ("[muted]Source config[/muted]", _fmt_path(info.source_config_path)),
        ("[muted]Git commit[/muted]", _fmt_text(info.git_commit)),
    ]
    return build_summary(rows)


def _fmt_status(status: str | None) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status in {"failed", "interrupted"}:
        return f"[error]{status}[/error]"
    if status:
        return f"[value]{status}[/value]"
    return "[muted]—[/muted]"


def _fmt_device(info: RunInfo) -> str:
    if info.run.resolved_device:
        return f"[value]{info.run.resolved_device}[/value]"
    if info.run.requested_device:
        return f"[value]{info.run.requested_device}[/value]"
    return "[muted]—[/muted]"


def _fmt_text(value: str | None) -> str:
    if not value:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_path(value: str | None) -> str:
    if not value:
        return "[muted]—[/muted]"
    return f"[path]{value}[/path]"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value:.2f}[/value]"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "[muted]—[/muted]"
    return f"[value]{value}[/value]"


def _fmt_eval(summary) -> str:
    if summary is None or summary.mean_reward is None:
        return "[muted]—[/muted]"

    value = f"{summary.mean_reward:.2f}"
    if summary.mean_episode_length is None:
        return f"[value]{value}[/value] [path]{summary.source}[/path]"

    return (
        f"[value]{value}[/value] "
        f"[muted](len {summary.mean_episode_length:.2f})[/muted] "
        f"[path]{summary.source}[/path]"
    )


def _fmt_tags(tags: tuple[str, ...]) -> str:
    if not tags:
        return "[muted]—[/muted]"
    return f"[value]{', '.join(tags)}[/value]"
