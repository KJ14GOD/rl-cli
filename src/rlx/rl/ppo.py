from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlx.config.schema import ExperimentConfig
from rlx.core.metadata import update_metadata
from rlx.core.metrics import JsonlMetricsWriter
from rlx.core.runs import RunPreparationResult
from rlx.rl.device import DeviceResolutionError, resolve_device
from rlx.rl.envs import EnvironmentError, validate_environment
from rlx.rl.runtime import resolve_runtime


class TrainingError(Exception):
    """Raised when PPO training cannot be started or completed."""


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    metrics_path: Path
    metadata_path: Path
    resolved_device: str
    total_timesteps: int
    latest_checkpoint: Path
    best_checkpoint: Path | None
    eval_log: Path | None


def train_ppo(
    config: ExperimentConfig,
    run: RunPreparationResult,
    *,
    resume_checkpoint: Path | None = None,
    additional_timesteps: int | None = None,
    lineage_metadata: dict[str, Any] | None = None,
) -> TrainResult:
    """Train or resume a PPO agent inside the prepared run scaffold."""

    _prepare_matplotlib_cache()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise TrainingError(
            "Stable-Baselines3 is not installed. Install RL dependencies before running "
            "`rlx train`."
        ) from exc

    try:
        runtime = resolve_runtime(config, project_root=run.project_root)
        validate_environment(config, project_root=run.project_root)
        resolved_device = resolve_device(config.device)
    except (EnvironmentError, DeviceResolutionError) as exc:
        raise TrainingError(str(exc)) from exc
    metadata_updates: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now_iso(),
        "requested_device": config.device,
        "resolved_device": resolved_device,
    }
    if lineage_metadata:
        metadata_updates.update(lineage_metadata)
    update_metadata(run.metadata_path, **metadata_updates)

    checkpoints_dir = run.run_dir / "checkpoints"
    eval_dir = run.run_dir / "eval"
    latest_checkpoint = checkpoints_dir / "latest.zip"
    best_checkpoint = checkpoints_dir / "best.zip"
    eval_log = eval_dir / "evaluations.npz"

    class MetricsCallback(BaseCallback):
        """Write periodic training metrics to JSONL during PPO learning."""

        def __init__(self, path: Path, total_timesteps: int) -> None:
            super().__init__(verbose=0)
            self._writer = JsonlMetricsWriter(path)
            self._total_timesteps = total_timesteps
            self._closed = False

        def _on_training_start(self) -> None:
            self._writer.write(
                {
                    "event": "training_start",
                    "step": int(self.model.num_timesteps),
                    "total_timesteps": self._total_timesteps,
                }
            )

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            record: dict[str, Any] = {
                "step": int(self.model.num_timesteps),
                "progress_remaining": float(self.model._current_progress_remaining),
            }
            rewards = [episode["r"] for episode in self.model.ep_info_buffer if "r" in episode]
            lengths = [episode["l"] for episode in self.model.ep_info_buffer if "l" in episode]
            if rewards:
                record["rollout/ep_rew_mean"] = float(sum(rewards) / len(rewards))
            if lengths:
                record["rollout/ep_len_mean"] = float(sum(lengths) / len(lengths))
            for key, value in self.model.logger.name_to_value.items():
                scalar = _to_scalar(value)
                if scalar is not None:
                    record[key] = scalar
            self._writer.write(record)

        def _on_training_end(self) -> None:
            self._writer.write(
                {
                    "event": "training_end",
                    "step": int(self.model.num_timesteps),
                }
            )
            self.close()

        def close(self) -> None:
            if not self._closed:
                self._writer.close()
                self._closed = True

    env = None
    eval_env = None
    metrics_callback: MetricsCallback | None = None
    learn_timesteps = additional_timesteps or config.algo.total_timesteps
    try:
        env = make_vec_env(runtime.env_id, n_envs=config.env.num_envs, seed=config.seed)
        eval_env = make_vec_env(runtime.env_id, n_envs=1, seed=config.seed + 10_000)

        if resume_checkpoint is None:
            policy_kwargs = (
                {"net_arch": list(config.policy.hidden_sizes)}
                if config.policy.hidden_sizes is not None
                else None
            )
            model = PPO(
                runtime.policy_spec,
                env,
                n_steps=config.algo.rollout_steps,
                batch_size=config.algo.batch_size,
                learning_rate=config.algo.learning_rate,
                gamma=config.algo.gamma,
                gae_lambda=config.algo.gae_lambda,
                clip_range=config.algo.clip_range,
                ent_coef=config.algo.entropy_coef,
                vf_coef=config.algo.value_coef,
                n_epochs=config.algo.update_epochs,
                policy_kwargs=policy_kwargs,
                seed=config.seed,
                device=resolved_device,
                verbose=0,
            )
        else:
            resolved_resume_checkpoint = resume_checkpoint.expanduser().resolve()
            if not resolved_resume_checkpoint.exists():
                raise TrainingError(f"Resume checkpoint not found: {resolved_resume_checkpoint}")
            if not resolved_resume_checkpoint.is_file():
                raise TrainingError(
                    f"Resume checkpoint path is not a file: {resolved_resume_checkpoint}"
                )
            model = PPO.load(str(resolved_resume_checkpoint), device=resolved_device)
            model.set_env(env)

        metrics_callback = MetricsCallback(
            run.metrics_path,
            int(model.num_timesteps + learn_timesteps),
        )

        callbacks = CallbackList(
            [
                metrics_callback,
                CheckpointCallback(
                    save_freq=max(config.checkpoint.save_every // config.env.num_envs, 1),
                    save_path=str(checkpoints_dir),
                    name_prefix="step",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(checkpoints_dir),
                    log_path=str(eval_dir),
                    eval_freq=max(config.eval.every // config.env.num_envs, 1),
                    n_eval_episodes=config.eval.episodes,
                    deterministic=config.eval.deterministic,
                    render=False,
                    verbose=0,
                ),
            ]
        )

        model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            log_interval=1,
            progress_bar=False,
            reset_num_timesteps=resume_checkpoint is None,
        )
        model.save(str(latest_checkpoint))

        saved_best = checkpoints_dir / "best_model.zip"
        copied_best: Path | None = None
        if saved_best.exists():
            shutil.copy2(saved_best, best_checkpoint)
            copied_best = best_checkpoint

        update_metadata(
            run.metadata_path,
            status="completed",
            completed_at=_utc_now_iso(),
            total_timesteps=model.num_timesteps,
            latest_checkpoint=str(latest_checkpoint.relative_to(run.run_dir)),
            best_checkpoint=(
                str(copied_best.relative_to(run.run_dir)) if copied_best is not None else None
            ),
            eval_log=str(eval_log.relative_to(run.run_dir)) if eval_log.exists() else None,
        )
        return TrainResult(
            run_dir=run.run_dir,
            metrics_path=run.metrics_path,
            metadata_path=run.metadata_path,
            resolved_device=resolved_device,
            total_timesteps=model.num_timesteps,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=copied_best,
            eval_log=eval_log if eval_log.exists() else None,
        )
    except KeyboardInterrupt as exc:
        update_metadata(
            run.metadata_path,
            status="interrupted",
            interrupted_at=_utc_now_iso(),
        )
        raise TrainingError("Training was interrupted.") from exc
    except (EnvironmentError, TrainingError):
        raise
    except Exception as exc:
        update_metadata(
            run.metadata_path,
            status="failed",
            failed_at=_utc_now_iso(),
            error=str(exc),
        )
        raise TrainingError(f"PPO training failed: {exc}") from exc
    finally:
        if metrics_callback is not None:
            metrics_callback.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


def _to_scalar(value: Any) -> int | float | bool | str | None:
    if isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, "item"):
        try:
            extracted = value.item()
        except Exception:
            return None
        if isinstance(extracted, bool | int | float | str):
            return extracted
    return None


def _prepare_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "rlx-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mpl"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
