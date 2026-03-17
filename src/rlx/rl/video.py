from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from rlx.config import ConfigError, load_config
from rlx.core.metadata import update_metadata
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME
from rlx.rl.device import DeviceResolutionError, resolve_device
from rlx.rl.ppo import _prepare_matplotlib_cache


class VideoRenderError(Exception):
    """Raised when a checkpoint video cannot be rendered."""


@dataclass(frozen=True)
class VideoResult:
    run_dir: Path
    checkpoint_path: Path
    checkpoint_name: str
    output_dir: Path
    manifest_path: Path
    requested_device: str
    resolved_device: str
    run_id: str
    run_name: str
    environment: str
    episodes: int
    deterministic: bool
    fps: int
    video_files: tuple[Path, ...]
    mean_reward: float
    mean_episode_length: float


def render_checkpoint_video(
    checkpoint_path: str | Path,
    *,
    episodes: int = 1,
    device: str | None = None,
) -> VideoResult:
    """Render one or more policy rollouts from a saved checkpoint."""

    _prepare_matplotlib_cache()

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise VideoRenderError(
            "Stable-Baselines3 and Gymnasium must be installed before running `rlx video`."
        ) from exc

    if episodes <= 0:
        raise VideoRenderError("Video episodes must be greater than zero.")

    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_checkpoint_path.exists():
        raise VideoRenderError(f"Checkpoint file not found: {resolved_checkpoint_path}")
    if not resolved_checkpoint_path.is_file():
        raise VideoRenderError(f"Checkpoint path is not a file: {resolved_checkpoint_path}")

    run_dir = _find_run_dir(resolved_checkpoint_path)
    config_path = run_dir / CONFIG_SNAPSHOT_NAME
    metadata_path = run_dir / METADATA_NAME

    try:
        config = load_config(config_path)
        resolved_device = resolve_device(device or config.device)
    except (ConfigError, DeviceResolutionError) as exc:
        raise VideoRenderError(str(exc)) from exc

    env = None
    try:
        try:
            env = gym.make(config.env.id, render_mode="rgb_array")
        except Exception as exc:
            raise VideoRenderError(
                f"Environment does not support rgb_array video rendering: {config.env.id}"
            ) from exc

        model = PPO.load(str(resolved_checkpoint_path), device=resolved_device)

        output_dir = _next_video_dir(run_dir / "videos")
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = output_dir / "manifest.json"

        fps = int(env.metadata.get("render_fps", 30) or 30)
        duration_ms = max(int(1000 / fps), 1)
        deterministic = config.eval.deterministic

        episode_payloads: list[dict[str, int | float | str]] = []
        video_files: list[Path] = []
        rewards: list[float] = []
        lengths: list[int] = []

        for episode_index in range(1, episodes + 1):
            obs, _info = env.reset(seed=config.seed + 30_000 + episode_index)
            frames = [_capture_frame(env, config.env.id)]
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                frames.append(_capture_frame(env, config.env.id))

            episode_file = output_dir / f"episode_{episode_index:03d}.gif"
            _save_gif(frames, episode_file, duration_ms)

            rewards.append(episode_reward)
            lengths.append(episode_length)
            video_files.append(episode_file)
            episode_payloads.append(
                {
                    "index": episode_index,
                    "reward": round(episode_reward, 6),
                    "length": episode_length,
                    "frames": len(frames),
                    "file": str(episode_file.relative_to(run_dir)),
                }
            )

        mean_reward = float(sum(rewards) / len(rewards))
        mean_episode_length = float(sum(lengths) / len(lengths))
        relative_checkpoint = _relative_to_run(resolved_checkpoint_path, run_dir)
        manifest = {
            "kind": "standalone_video",
            "rendered_at": _utc_now_iso(),
            "run_id": run_dir.name,
            "run_name": config.run_name,
            "checkpoint": {
                "path": str(relative_checkpoint),
                "name": resolved_checkpoint_path.name,
            },
            "config": {
                "environment": config.env.id,
                "seed": config.seed,
                "episodes": episodes,
                "deterministic": deterministic,
                "requested_device": device or config.device,
                "resolved_device": resolved_device,
                "fps": fps,
                "format": "gif",
            },
            "summary": {
                "mean_reward": mean_reward,
                "mean_episode_length": mean_episode_length,
                "video_count": len(video_files),
            },
            "episodes": episode_payloads,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if metadata_path.exists():
            update_metadata(
                metadata_path,
                last_video_at=manifest["rendered_at"],
                last_video_dir=str(output_dir.relative_to(run_dir)),
                last_video_manifest=str(manifest_path.relative_to(run_dir)),
                last_video_checkpoint=str(relative_checkpoint),
            )

        return VideoResult(
            run_dir=run_dir,
            checkpoint_path=resolved_checkpoint_path,
            checkpoint_name=resolved_checkpoint_path.name,
            output_dir=output_dir,
            manifest_path=manifest_path,
            requested_device=device or config.device,
            resolved_device=resolved_device,
            run_id=run_dir.name,
            run_name=config.run_name,
            environment=config.env.id,
            episodes=episodes,
            deterministic=deterministic,
            fps=fps,
            video_files=tuple(video_files),
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
        )
    except VideoRenderError:
        raise
    except Exception as exc:
        raise VideoRenderError(f"Video rendering failed: {exc}") from exc
    finally:
        if env is not None:
            env.close()


def _find_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name != "checkpoints":
        raise VideoRenderError(
            "Checkpoint path must point inside a run's checkpoints directory."
        )

    run_dir = checkpoint_path.parent.parent
    if not run_dir.is_dir():
        raise VideoRenderError(f"Run directory not found: {run_dir}")
    if not (run_dir / CONFIG_SNAPSHOT_NAME).exists():
        raise VideoRenderError(f"Run config snapshot not found: {run_dir / CONFIG_SNAPSHOT_NAME}")
    return run_dir


def _next_video_dir(videos_dir: Path) -> Path:
    pattern = re.compile(r"^manual_video_(\d{3})$")
    existing = []

    for path in videos_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            existing.append(int(match.group(1)))

    next_index = max(existing, default=0) + 1
    return videos_dir / f"manual_video_{next_index:03d}"


def _capture_frame(env, env_id: str):
    frame = env.render()
    if frame is None:
        raise VideoRenderError(f"Environment returned no frame while rendering: {env_id}")
    return Image.fromarray(frame)


def _save_gif(frames: list[Image.Image], output_path: Path, duration_ms: int) -> None:
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )


def _relative_to_run(path: Path, run_dir: Path) -> Path:
    try:
        return path.relative_to(run_dir)
    except ValueError:
        return path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
