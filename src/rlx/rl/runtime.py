from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.config.schema import ExperimentConfig

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - handled in train/eval/video flows
    gym = None  # type: ignore[assignment]


class RuntimeResolutionError(Exception):
    """Raised when RLX cannot resolve project-local env or policy code."""


@dataclass(frozen=True)
class ResolvedRuntime:
    env_id: str
    policy_spec: str | type[Any]
    policy_label: str


def resolve_runtime(config: ExperimentConfig, *, project_root: Path) -> ResolvedRuntime:
    """Import any project-local env/policy modules and return resolved runtime handles."""

    _import_modules(config.env.import_modules, project_root=project_root, role="env")

    if config.policy.type == "mlp":
        return ResolvedRuntime(
            env_id=config.env.id,
            policy_spec="MlpPolicy",
            policy_label="MlpPolicy",
        )

    _ensure_project_path(project_root)
    assert config.policy.import_module is not None  # schema enforces this
    assert config.policy.class_name is not None  # schema enforces this

    module = _import_module(
        config.policy.import_module,
        project_root=project_root,
        role="policy",
    )
    policy_class = getattr(module, config.policy.class_name, None)
    if not isinstance(policy_class, type):
        raise RuntimeResolutionError(
            "Configured custom policy class was not found: "
            f"{config.policy.import_module}.{config.policy.class_name}"
        )

    try:
        from stable_baselines3.common.policies import ActorCriticPolicy
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeResolutionError(
            "Stable-Baselines3 is not installed. Install RL dependencies before using "
            "custom policies."
        ) from exc

    if not issubclass(policy_class, ActorCriticPolicy):
        raise RuntimeResolutionError(
            "Custom PPO policy must inherit from stable_baselines3.common.policies."
            f"ActorCriticPolicy: {config.policy.import_module}.{config.policy.class_name}"
        )

    return ResolvedRuntime(
        env_id=config.env.id,
        policy_spec=policy_class,
        policy_label=f"{config.policy.import_module}.{config.policy.class_name}",
    )


def validate_environment(
    config: ExperimentConfig,
    *,
    project_root: Path,
    render_mode: str | None = None,
) -> None:
    """Validate that the configured environment can be instantiated."""

    if gym is None:
        raise RuntimeResolutionError(
            "Gymnasium is not installed. Install RL dependencies before running RLX commands."
        )

    _import_modules(config.env.import_modules, project_root=project_root, role="env")
    kwargs = {"render_mode": render_mode} if render_mode is not None else {}

    try:
        env = gym.make(config.env.id, **kwargs)
    except Exception as exc:
        raise RuntimeResolutionError(
            f"Failed to create Gymnasium environment '{config.env.id}': {exc}"
        ) from exc
    finally:
        if "env" in locals():
            env.close()


def _import_modules(
    module_names: list[str],
    *,
    project_root: Path,
    role: str,
) -> None:
    for module_name in module_names:
        _import_module(module_name, project_root=project_root, role=role)


def _import_module(module_name: str, *, project_root: Path, role: str):
    _ensure_project_path(project_root)
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeResolutionError(
            f"Could not import {role} module '{module_name}' from project root "
            f"{project_root}: {exc}"
        ) from exc


def _ensure_project_path(project_root: Path) -> None:
    resolved = str(project_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        importlib.invalidate_caches()


def project_has_importable_module(project_root: Path, module_name: str) -> bool:
    """Return whether a project-local module path is importable from the given root."""

    _ensure_project_path(project_root)
    return importlib.util.find_spec(module_name) is not None
