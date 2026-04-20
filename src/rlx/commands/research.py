import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.research import ResearchError, ResearchResult, resume_research, run_research
from rlx.llm.planner import LLM_DEFAULT_MODEL, LLM_DEFAULT_PROVIDER

RUN_REF_ARGUMENT = typer.Argument(
    None,
    help="Run path or run id to use as the initial research baseline.",
)

ROUNDS_OPTION = typer.Option(
    None,
    "--rounds",
    min=1,
    help="Maximum advisor rounds to run. New bundles default to 3; resume keeps prior setting.",
)

VARIANTS_OPTION = typer.Option(
    None,
    "--variants",
    min=1,
    help=(
        "Advisor variants to create per round. New bundles default to 4; "
        "resume keeps prior setting."
    ),
)

EXECUTE_OPTION = typer.Option(
    False,
    "--execute",
    help="Train advisor variants and promote the best-scoring run.",
)

TIMESTEPS_OPTION = typer.Option(
    None,
    "--timesteps",
    min=1,
    help="Lock total timesteps for every research-generated variant.",
)

MIN_IMPROVEMENT_OPTION = typer.Option(
    None,
    "--min-improvement",
    min=0.0,
    help="Minimum score gain required to promote a candidate.",
)

RESUME_OPTION = typer.Option(
    None,
    "--resume",
    help="Path to a research bundle directory or manifest.json to continue.",
)

PLANNER_OPTION = typer.Option(
    None,
    "--planner",
    help="Proposal planner for new bundles: rules or llm.",
)

LLM_PROVIDER_OPTION = typer.Option(
    None,
    "--llm-provider",
    help=f"LLM provider for --planner llm. Defaults to {LLM_DEFAULT_PROVIDER}.",
)

LLM_MODEL_OPTION = typer.Option(
    None,
    "--llm-model",
    help=f"LLM model for --planner llm. Defaults to {LLM_DEFAULT_MODEL}.",
)

LLM_STRICT_OPTION = typer.Option(
    False,
    "--llm-strict",
    help="Fail if the LLM cannot produce all requested valid variants.",
)


def research_command(
    run_ref: str | None = RUN_REF_ARGUMENT,
    rounds: int | None = ROUNDS_OPTION,
    variants: int | None = VARIANTS_OPTION,
    execute: bool = EXECUTE_OPTION,
    timesteps: int | None = TIMESTEPS_OPTION,
    min_improvement: float | None = MIN_IMPROVEMENT_OPTION,
    resume: str | None = RESUME_OPTION,
    planner: str | None = PLANNER_OPTION,
    llm_provider: str | None = LLM_PROVIDER_OPTION,
    llm_model: str | None = LLM_MODEL_OPTION,
    llm_strict: bool = LLM_STRICT_OPTION,
) -> None:
    """Run bounded advisor loops and promote the best-scoring variant.

    Examples:

        rlx research cartpole_ppo_001
        rlx research cartpole_ppo_001 --rounds 3 --variants 4
        rlx research cartpole_ppo_001 --execute --rounds 2 --timesteps 20000
        rlx research cartpole_ppo_001 --planner llm --execute --rounds 3
        rlx research --resume analysis/research/cartpole_ppo_001_research_001 --rounds 5
    """

    try:
        if resume is not None:
            if run_ref is not None:
                raise ResearchError("Use either a baseline run or --resume, not both.")
            result = resume_research(
                resume,
                rounds=rounds,
                variants=variants,
                execute=True if execute else None,
                timesteps=timesteps,
                min_improvement=min_improvement,
                planner=planner,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_strict=llm_strict if llm_strict else None,
            )
        else:
            if run_ref is None:
                raise ResearchError("Pass a baseline run, or use --resume with a research bundle.")
            result = run_research(
                run_ref,
                rounds=rounds or 3,
                variants=variants or 4,
                execute=execute,
                timesteps=timesteps,
                min_improvement=0.0 if min_improvement is None else min_improvement,
                planner=planner or "rules",
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_strict=llm_strict,
            )
    except ResearchError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Research Complete", _build_overview(result))
    print_panel("RLCLI Research Rounds", _build_rounds_table(result))


def _build_overview(result: ResearchResult):
    rows = [
        ("[success]Initial[/success]", f"[value]{result.initial_run_id}[/value]"),
        ("[muted]Mode[/muted]", f"[value]{result.mode}[/value]"),
        ("[muted]Champion[/muted]", f"[value]{result.champion_run_id}[/value]"),
        ("[muted]Champion score[/muted]", _fmt_score(result.champion_score)),
        ("[muted]Stop[/muted]", f"[value]{result.stop_reason}[/value]"),
        ("[muted]Output[/muted]", f"[path]{result.bundle_dir}[/path]"),
        (
            "[muted]Manifest[/muted]",
            f"[path]{result.manifest_path.relative_to(result.project_root)}[/path]",
        ),
        (
            "[muted]Report[/muted]",
            f"[path]{result.report_path.relative_to(result.project_root)}[/path]",
        ),
        ("[muted]Rounds[/muted]", f"[value]{len(result.rounds)}[/value]"),
    ]
    if result.score_plot_path is not None:
        rows.append(
            (
                "[muted]Score plot[/muted]",
                f"[path]{result.score_plot_path.relative_to(result.project_root)}[/path]",
            )
        )
    if result.progress_plot_path is not None:
        rows.append(
            (
                "[muted]Progress plot[/muted]",
                f"[path]{result.progress_plot_path.relative_to(result.project_root)}[/path]",
            )
        )
    if result.champion_score_source:
        rows.append(
            (
                "[muted]Score source[/muted]",
                f"[value]{result.champion_score_source}[/value]",
            )
        )
    return build_summary(rows)


def _build_rounds_table(result: ResearchResult) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Round", no_wrap=True)
    table.add_column("Baseline", no_wrap=True)
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Improve", justify="right", no_wrap=True)
    table.add_column("Promoted", no_wrap=True)
    table.add_column("Advisor", overflow="fold")

    for item in result.rounds:
        table.add_row(
            f"[value]{item.index:03d}[/value]",
            f"[value]{item.baseline_run_id}[/value]",
            (
                f"[value]{item.candidate_run_id}[/value]"
                if item.candidate_run_id
                else "[muted]—[/muted]"
            ),
            _fmt_score(item.candidate_score),
            _fmt_score(item.improvement),
            "[success]yes[/success]" if item.promoted else "[muted]no[/muted]",
            f"[path]{item.advisor_bundle.relative_to(result.project_root)}[/path]",
        )

    return table


def _fmt_score(score: float | None) -> str:
    if score is None:
        return "[muted]—[/muted]"
    return f"[value]{score:.2f}[/value]"
