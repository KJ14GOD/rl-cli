import typer
from rich.table import Table

from rlx.console import build_summary, console, print_panel
from rlx.core.advisor import AdvisorError, AdvisorResult, run_advisor
from rlx.llm.planner import LLM_DEFAULT_MODEL, LLM_DEFAULT_PROVIDER, LLM_SUPPORTED_PROVIDERS

RUN_REF_ARGUMENT = typer.Argument(
    ...,
    help="Run path or run id to use as the advisor baseline.",
)

VARIANTS_OPTION = typer.Option(
    4,
    "--variants",
    min=1,
    help="Number of advisor variants to create.",
)

EXECUTE_OPTION = typer.Option(
    False,
    "--execute",
    help="Train the generated variants after writing the advisor plan.",
)

TIMESTEPS_OPTION = typer.Option(
    None,
    "--timesteps",
    min=1,
    help="Override total timesteps for advisor-generated variants.",
)

PLANNER_OPTION = typer.Option(
    "rules",
    "--planner",
    help="Proposal planner to use: rules or llm.",
)

LLM_PROVIDER_OPTION = typer.Option(
    None,
    "--llm-provider",
    help=(
        f"LLM provider for --planner llm. Defaults to {LLM_DEFAULT_PROVIDER}. "
        f"Supported: {', '.join(LLM_SUPPORTED_PROVIDERS)}."
    ),
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


def advisor_command(
    run_ref: str = RUN_REF_ARGUMENT,
    variants: int = VARIANTS_OPTION,
    execute: bool = EXECUTE_OPTION,
    timesteps: int | None = TIMESTEPS_OPTION,
    planner: str = PLANNER_OPTION,
    llm_provider: str | None = LLM_PROVIDER_OPTION,
    llm_model: str | None = LLM_MODEL_OPTION,
    llm_strict: bool = LLM_STRICT_OPTION,
) -> None:
    """Create grounded next-experiment variants from one completed run.

    Examples:

        rlx advisor cartpole_ppo_001
        rlx advisor runs/cartpole_ppo_001 --variants 6
        rlx advisor cartpole_ppo_001 --execute --timesteps 20000
        rlx advisor cartpole_ppo_001 --planner llm --llm-model gpt-5.4-mini
    """

    try:
        result = run_advisor(
            run_ref,
            variants=variants,
            execute=execute,
            timesteps=timesteps,
            planner=planner,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_strict=llm_strict,
        )
    except AdvisorError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    print_panel("RLCLI Advisor Plan", _build_overview(result))
    print_panel("RLCLI Advisor Variants", _build_variants_table(result))
    if result.context_actions:
        print_panel("RLCLI Advisor Context", _build_context_table(result))


def _build_overview(result: AdvisorResult):
    rows = [
        ("[success]Baseline[/success]", f"[value]{result.baseline_run_id}[/value]"),
        ("[muted]Mode[/muted]", f"[value]{result.mode}[/value]"),
        ("[muted]Health[/muted]", f"[value]{result.diagnosis.health}[/value]"),
        ("[muted]Trend[/muted]", f"[value]{result.diagnosis.analysis.learning.trend}[/value]"),
        (
            "[muted]Baseline score[/muted]",
            _fmt_score(result.baseline_score, result.baseline_score_source),
        ),
        ("[muted]Output[/muted]", f"[path]{result.bundle_dir}[/path]"),
        (
            "[muted]Manifest[/muted]",
            f"[path]{result.manifest_path.relative_to(result.project_root)}[/path]",
        ),
        (
            "[muted]Plan[/muted]",
            f"[path]{result.plan_path.relative_to(result.project_root)}[/path]",
        ),
        ("[muted]Variants[/muted]", f"[value]{len(result.variants)}[/value]"),
    ]
    if result.best_variant is not None:
        rows.append(
            (
                "[muted]Best variant[/muted]",
                f"[value]{result.best_variant.index:03d}[/value]",
            )
        )
    return build_summary(rows)


def _build_variants_table(result: AdvisorResult) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Variant", no_wrap=True)
    table.add_column("Priority", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Run", no_wrap=True)
    table.add_column("Config", overflow="fold")
    table.add_column("Mutations", overflow="fold")
    table.add_column("Why", overflow="fold")

    for variant in result.variants:
        table.add_row(
            f"[value]{variant.index:03d}[/value]",
            _fmt_priority(variant.priority),
            _fmt_status(variant.status),
            f"[value]{variant.run_id}[/value]" if variant.run_id else "[muted]—[/muted]",
            f"[path]{variant.config_path.relative_to(result.project_root)}[/path]",
            _fmt_mutations(variant.mutations),
            (
                variant.rationale
                if not variant.error
                else f"{variant.rationale} Error: {variant.error}"
            ),
        )

    return table


def _build_context_table(result: AdvisorResult) -> Table:
    table = Table(expand=True, padding=(0, 1))
    table.add_column("Recommended command", overflow="fold")
    table.add_column("Reason", overflow="fold")

    reasons = {
        "rlx eval": "Create a stable eval artifact before trusting advisor scores.",
        "rlx plot": "Generate visual learning curves for manual inspection.",
        "rlx video": "Inspect checkpoint behavior, not just reward numbers.",
    }
    for action in result.context_actions:
        reason = next(
            (value for prefix, value in reasons.items() if action.startswith(prefix)),
            "Collect more context for the next advisor round.",
        )
        table.add_row(f"[path]{action}[/path]", reason)
    return table


def _fmt_score(score: float | None, source: str | None) -> str:
    if score is None:
        return "[muted]—[/muted]"
    if source is None:
        return f"[value]{score:.2f}[/value]"
    return f"[value]{score:.2f}[/value] [muted]{source}[/muted]"


def _fmt_priority(value: str) -> str:
    if value == "high":
        return "[error]high[/error]"
    if value == "medium":
        return "[warning]medium[/warning]"
    return "[muted]low[/muted]"


def _fmt_status(status: str) -> str:
    if status == "completed":
        return "[success]completed[/success]"
    if status == "failed":
        return "[error]failed[/error]"
    return f"[value]{status}[/value]"


def _fmt_mutations(mutations: dict[str, object]) -> str:
    if not mutations:
        return "[muted]—[/muted]"
    return f"[value]{'; '.join(f'{key}={value}' for key, value in mutations.items())}[/value]"
