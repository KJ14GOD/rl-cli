from __future__ import annotations

from rlx.config.schema import ExperimentConfig

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - handled at runtime in train flow
    gym = None  # type: ignore[assignment]


class EnvironmentError(Exception):
    """Raised when an RL environment cannot be created."""


def validate_environment(config: ExperimentConfig) -> None:
    """Validate that the configured environment id can be instantiated."""

    if gym is None:
        raise EnvironmentError(
            "Gymnasium is not installed. Install RL dependencies before running `rlx train`."
        )

    try:
        env = gym.make(config.env.id)
    except Exception as exc:
        raise EnvironmentError(
            f"Failed to create Gymnasium environment '{config.env.id}': {exc}"
        ) from exc
    finally:
        if "env" in locals():
            env.close()
