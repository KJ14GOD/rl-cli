from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rlx.core.compare import CompareError, load_run_comparison, resolve_run_ref
from rlx.core.metadata import update_metadata
from rlx.paths import METADATA_NAME, METRICS_NAME
from rlx.rl.ppo import _prepare_matplotlib_cache


class PlotError(Exception):
    """Raised when RLCLI plots cannot be generated."""


@dataclass(frozen=True)
class PlotArtifact:
    key: str
    path: Path


@dataclass(frozen=True)
class PlotBundle:
    mode: str
    project_root: Path
    output_dir: Path
    manifest_path: Path
    run_ids: tuple[str, ...]
    artifacts: tuple[PlotArtifact, ...]


@dataclass(frozen=True)
class _RunSeries:
    run_dir: Path
    run_id: str
    label: str
    total_timesteps: int | None
    reward_steps: tuple[int, ...]
    reward_values: tuple[float, ...]
    length_steps: tuple[int, ...]
    length_values: tuple[float, ...]
    eval_steps: tuple[int, ...]
    eval_values: tuple[float, ...]


def plot_runs(run_refs: list[str], cwd: Path | None = None) -> PlotBundle:
    if not run_refs:
        raise PlotError("Pass at least one run to plot.")

    working_dir = (cwd or Path.cwd()).resolve()
    run_dirs = []
    for run_ref in run_refs:
        try:
            run_dirs.append(resolve_run_ref(run_ref, working_dir))
        except CompareError as exc:
            raise PlotError(str(exc)) from exc

    series = [_load_run_series(run_dir) for run_dir in run_dirs]
    if not any(run.reward_values or run.length_values or run.eval_values for run in series):
        raise PlotError("No plottable metrics or eval history found for the selected runs.")

    if len(series) == 1:
        project_root = series[0].run_dir.parent.parent
        output_dir = _next_bundle_dir(series[0].run_dir / "plots")
        mode = "single"
    else:
        project_roots = {run.run_dir.parent.parent.resolve() for run in series}
        if len(project_roots) != 1:
            raise PlotError("`rlx plot` currently requires compared runs to come from one project.")
        project_root = next(iter(project_roots))
        output_dir = _next_bundle_dir(project_root / "analysis" / "plots")
        mode = "compare"

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"

    artifacts = []
    reward_plot = _plot_training_reward(series, output_dir / "training_reward.png")
    if reward_plot is not None:
        artifacts.append(PlotArtifact("training_reward", reward_plot))

    length_plot = _plot_training_length(series, output_dir / "training_length.png")
    if length_plot is not None:
        artifacts.append(PlotArtifact("training_length", length_plot))

    eval_plot = _plot_eval_reward(series, output_dir / "eval_reward.png")
    if eval_plot is not None:
        artifacts.append(PlotArtifact("eval_reward", eval_plot))

    if not artifacts:
        raise PlotError("No plots were generated from the selected runs.")

    manifest = {
        "kind": "plot_bundle",
        "generated_at": _utc_now_iso(),
        "mode": mode,
        "run_ids": [run.run_id for run in series],
        "artifacts": [
            {
                "key": artifact.key,
                "file": str(artifact.path.relative_to(project_root)),
            }
            for artifact in artifacts
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if len(series) == 1:
        metadata_path = series[0].run_dir / METADATA_NAME
        if metadata_path.exists():
            update_metadata(
                metadata_path,
                last_plot_at=manifest["generated_at"],
                last_plot_dir=str(output_dir.relative_to(series[0].run_dir)),
                last_plot_manifest=str(manifest_path.relative_to(series[0].run_dir)),
            )

    return PlotBundle(
        mode=mode,
        project_root=project_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_ids=tuple(run.run_id for run in series),
        artifacts=tuple(artifacts),
    )


def _load_run_series(run_dir: Path) -> _RunSeries:
    comparison = load_run_comparison(run_dir)
    reward_steps: list[int] = []
    reward_values: list[float] = []
    length_steps: list[int] = []
    length_values: list[float] = []

    metrics_path = run_dir / METRICS_NAME
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            step = payload.get("step")
            if not isinstance(step, int):
                continue

            reward = payload.get("rollout/ep_rew_mean")
            if isinstance(reward, int | float):
                reward_steps.append(step)
                reward_values.append(float(reward))

            length = payload.get("rollout/ep_len_mean")
            if isinstance(length, int | float):
                length_steps.append(step)
                length_values.append(float(length))

    eval_steps, eval_values = _load_eval_points(run_dir, comparison.total_timesteps)

    return _RunSeries(
        run_dir=run_dir,
        run_id=comparison.run_id,
        label=comparison.run_id,
        total_timesteps=comparison.total_timesteps,
        reward_steps=tuple(reward_steps),
        reward_values=tuple(reward_values),
        length_steps=tuple(length_steps),
        length_values=tuple(length_values),
        eval_steps=tuple(eval_steps),
        eval_values=tuple(eval_values),
    )


def _load_eval_points(run_dir: Path, total_timesteps: int | None) -> tuple[list[int], list[float]]:
    eval_dir = run_dir / "eval"
    points: list[tuple[int, float]] = []

    training_eval = _load_training_eval_points(eval_dir / "evaluations.npz")
    points.extend(training_eval)

    for path in sorted(eval_dir.glob("manual_eval_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        mean_reward = summary.get("mean_reward")
        if not isinstance(mean_reward, int | float):
            continue

        checkpoint = payload.get("checkpoint")
        step = None
        if isinstance(checkpoint, dict):
            name = checkpoint.get("name")
            if isinstance(name, str):
                step = _checkpoint_step(name)
        if step is None:
            step = total_timesteps
        if step is None:
            continue
        points.append((int(step), float(mean_reward)))

    if not points:
        return [], []

    points.sort(key=lambda item: item[0])
    deduped: dict[int, float] = {}
    for step, value in points:
        deduped[step] = value
    ordered = sorted(deduped.items())
    return [step for step, _ in ordered], [value for _, value in ordered]


def _load_training_eval_points(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        return []

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - local environment dependent
        raise PlotError("NumPy is required for reading training eval history.") from exc

    try:
        data = np.load(path)
        timesteps = data.get("timesteps")
        results = data.get("results")
    except Exception:
        return []

    if timesteps is None or results is None:
        return []

    points = []
    for step, row in zip(timesteps, results, strict=False):
        try:
            value = float(row.mean())
            points.append((int(step), value))
        except Exception:
            continue
    return points


def _plot_training_reward(series: list[_RunSeries], output_path: Path) -> Path | None:
    candidates = [run for run in series if run.reward_values]
    if not candidates:
        return None
    return _plot_lines(
        series=candidates,
        output_path=output_path,
        x_getter=lambda run: run.reward_steps,
        y_getter=lambda run: run.reward_values,
        title="Training Reward",
        ylabel="Mean Episode Reward",
    )


def _plot_training_length(series: list[_RunSeries], output_path: Path) -> Path | None:
    candidates = [run for run in series if run.length_values]
    if not candidates:
        return None
    return _plot_lines(
        series=candidates,
        output_path=output_path,
        x_getter=lambda run: run.length_steps,
        y_getter=lambda run: run.length_values,
        title="Training Episode Length",
        ylabel="Mean Episode Length",
    )


def _plot_eval_reward(series: list[_RunSeries], output_path: Path) -> Path | None:
    candidates = [run for run in series if run.eval_values]
    if not candidates:
        return None
    return _plot_lines(
        series=candidates,
        output_path=output_path,
        x_getter=lambda run: run.eval_steps,
        y_getter=lambda run: run.eval_values,
        title="Evaluation Reward",
        ylabel="Mean Eval Reward",
    )


def _plot_lines(
    *,
    series: list[_RunSeries],
    output_path: Path,
    x_getter,
    y_getter,
    title: str,
    ylabel: str,
) -> Path:
    _prepare_matplotlib_cache()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - local environment dependent
        raise PlotError("matplotlib is required for `rlx plot`.") from exc

    colors = ["#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for index, run in enumerate(series):
        color = colors[index % len(colors)]
        ax.plot(
            x_getter(run),
            y_getter(run),
            label=run.label,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Timesteps", color="#334155")
    ax.set_ylabel(ylabel, color="#334155")
    ax.grid(True, linestyle="--", linewidth=0.7, color="#d4d4d8", alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#475569")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _next_bundle_dir(root: Path) -> Path:
    pattern = re.compile(r"^manual_plot_(\d{3})$")
    existing = []

    if root.exists():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                existing.append(int(match.group(1)))

    next_index = max(existing, default=0) + 1
    return root / f"manual_plot_{next_index:03d}"


def _checkpoint_step(name: str) -> int | None:
    match = re.search(r"step_(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
