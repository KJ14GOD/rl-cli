from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev

from rlx.config import ConfigError, load_config
from rlx.core.metadata import update_metadata
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME
from rlx.rl.device import DeviceResolutionError, resolve_device
from rlx.rl.envs import EnvironmentError, validate_environment
from rlx.rl.ppo import _prepare_matplotlib_cache


class EvaluationError(Exception):
    """Raised when a checkpoint evaluation cannot be completed."""


@dataclass(frozen=True)
class EvalResult:
    run_dir: Path
    checkpoint_path: Path
    checkpoint_name: str
    result_path: Path
    requested_device: str
    resolved_device: str
    run_id: str
    run_name: str
    environment: str
    episodes: int
    deterministic: bool
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float
    mean_episode_length: float
    std_episode_length: float
    min_episode_length: int
    max_episode_length: int


def collect_eval_targets(
    *,
    checkpoint_paths: list[str] | None = None,
    run_path: str | Path | None = None,
    all_checkpoints: bool = False,
) -> list[Path]:
    """Resolve one or many checkpoint targets for standalone evaluation."""

    explicit_targets = [Path(path).expanduser().resolve() for path in (checkpoint_paths or [])]
    if explicit_targets and run_path is not None:
        raise EvaluationError("Pass checkpoint paths or `--run`, not both.")
    if all_checkpoints and run_path is None:
        raise EvaluationError("`--all-checkpoints` requires `--run`.")

    if run_path is not None:
        run_dir = _resolve_run_dir(Path(run_path).expanduser().resolve())
        if all_checkpoints:
            checkpoints = _all_checkpoints_for_run(run_dir)
            if not checkpoints:
                raise EvaluationError(f"No checkpoints found in: {run_dir / 'checkpoints'}")
            return checkpoints

        latest_checkpoint = run_dir / "checkpoints" / "latest.zip"
        if not latest_checkpoint.exists():
            raise EvaluationError(f"Latest checkpoint not found: {latest_checkpoint}")
        return [latest_checkpoint]

    if not explicit_targets:
        raise EvaluationError("Pass at least one checkpoint path, or use `--run`.")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in explicit_targets:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def evaluate_checkpoints(
    checkpoint_paths: list[str] | None = None,
    *,
    run_path: str | Path | None = None,
    all_checkpoints: bool = False,
    episodes: int | None = None,
    device: str | None = None,
) -> list[EvalResult]:
    """Evaluate one or more checkpoints and return one result per target."""

    targets = collect_eval_targets(
        checkpoint_paths=checkpoint_paths,
        run_path=run_path,
        all_checkpoints=all_checkpoints,
    )
    return [evaluate_checkpoint(path, episodes=episodes, device=device) for path in targets]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    episodes: int | None = None,
    device: str | None = None,
) -> EvalResult:
    """Evaluate a saved PPO checkpoint against the run's snapped config."""

    _prepare_matplotlib_cache()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise EvaluationError(
            "Stable-Baselines3 is not installed. Install RL dependencies before running `rlx eval`."
        ) from exc

    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_checkpoint_path.exists():
        raise EvaluationError(f"Checkpoint file not found: {resolved_checkpoint_path}")
    if not resolved_checkpoint_path.is_file():
        raise EvaluationError(f"Checkpoint path is not a file: {resolved_checkpoint_path}")

    run_dir = _find_run_dir(resolved_checkpoint_path)
    config_path = run_dir / CONFIG_SNAPSHOT_NAME
    metadata_path = run_dir / METADATA_NAME
    eval_dir = run_dir / "eval"

    try:
        config = load_config(config_path)
        validate_environment(config)
        resolved_device = resolve_device(device or config.device)
    except (ConfigError, EnvironmentError, DeviceResolutionError) as exc:
        raise EvaluationError(str(exc)) from exc

    eval_episodes = episodes or config.eval.episodes
    if eval_episodes <= 0:
        raise EvaluationError("Evaluation episodes must be greater than zero.")
    deterministic = config.eval.deterministic

    env = None
    try:
        env = make_vec_env(config.env.id, n_envs=1, seed=config.seed + 20_000)
        model = PPO.load(str(resolved_checkpoint_path), device=resolved_device)

        rewards, lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
            warn=False,
        )

        mean_reward = float(mean(rewards))
        std_reward = float(pstdev(rewards)) if len(rewards) > 1 else 0.0
        min_reward = float(min(rewards))
        max_reward = float(max(rewards))
        median_reward = float(median(rewards))

        mean_episode_length = float(mean(lengths))
        std_episode_length = float(pstdev(lengths)) if len(lengths) > 1 else 0.0
        min_episode_length = int(min(lengths))
        max_episode_length = int(max(lengths))

        result_path = _next_eval_result_path(eval_dir)
        relative_checkpoint = _relative_to_run(resolved_checkpoint_path, run_dir)
        result_payload = {
            "kind": "standalone_eval",
            "evaluated_at": _utc_now_iso(),
            "run_id": run_dir.name,
            "run_name": config.run_name,
            "checkpoint": {
                "path": str(relative_checkpoint),
                "name": resolved_checkpoint_path.name,
            },
            "config": {
                "environment": config.env.id,
                "seed": config.seed,
                "episodes": eval_episodes,
                "deterministic": deterministic,
                "requested_device": device or config.device,
                "resolved_device": resolved_device,
            },
            "summary": {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "min_reward": min_reward,
                "max_reward": max_reward,
                "median_reward": median_reward,
                "mean_episode_length": mean_episode_length,
                "std_episode_length": std_episode_length,
                "min_episode_length": min_episode_length,
                "max_episode_length": max_episode_length,
            },
            "episodes": [
                {
                    "index": index,
                    "reward": float(reward),
                    "length": int(length),
                }
                for index, (reward, length) in enumerate(
                    zip(rewards, lengths, strict=True),
                    start=1,
                )
            ],
        }
        result_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")

        if metadata_path.exists():
            update_metadata(
                metadata_path,
                last_eval_at=result_payload["evaluated_at"],
                last_eval_result=str(result_path.relative_to(run_dir)),
                last_eval_checkpoint=str(relative_checkpoint),
            )

        return EvalResult(
            run_dir=run_dir,
            checkpoint_path=resolved_checkpoint_path,
            checkpoint_name=resolved_checkpoint_path.name,
            result_path=result_path,
            requested_device=device or config.device,
            resolved_device=resolved_device,
            run_id=run_dir.name,
            run_name=config.run_name,
            environment=config.env.id,
            episodes=eval_episodes,
            deterministic=deterministic,
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            median_reward=median_reward,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            min_episode_length=min_episode_length,
            max_episode_length=max_episode_length,
        )
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Checkpoint evaluation failed: {exc}") from exc
    finally:
        if env is not None:
            env.close()


def _find_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name != "checkpoints":
        raise EvaluationError(
            "Checkpoint path must point inside a run's checkpoints directory."
        )

    return _resolve_run_dir(checkpoint_path.parent.parent)


def _resolve_run_dir(run_dir: Path) -> Path:
    if not run_dir.is_dir():
        raise EvaluationError(f"Run directory not found: {run_dir}")
    if not (run_dir / CONFIG_SNAPSHOT_NAME).exists():
        raise EvaluationError(f"Run config snapshot not found: {run_dir / CONFIG_SNAPSHOT_NAME}")
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise EvaluationError(f"Run checkpoints directory not found: {checkpoints_dir}")
    return run_dir


def _all_checkpoints_for_run(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    step_checkpoints = sorted(
        checkpoints_dir.glob("step*.zip"),
        key=_checkpoint_sort_key,
    )

    ordered: list[Path] = []
    for path in step_checkpoints:
        if path.name == "best_model.zip":
            continue
        ordered.append(path)

    for name in ("best.zip", "latest.zip"):
        path = checkpoints_dir / name
        if path.exists():
            ordered.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"step_(\d+)", path.stem)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def _next_eval_result_path(eval_dir: Path) -> Path:
    pattern = re.compile(r"^manual_eval_(\d{3})\.json$")
    existing = []

    for path in eval_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            existing.append(int(match.group(1)))

    next_index = max(existing, default=0) + 1
    return eval_dir / f"manual_eval_{next_index:03d}.json"


def _relative_to_run(path: Path, run_dir: Path) -> Path:
    try:
        return path.relative_to(run_dir)
    except ValueError:
        return path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
