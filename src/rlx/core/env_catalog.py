from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvCatalogEntry:
    key: str
    family: str
    env_id: str
    config_path: Path
    observation: str
    action: str
    notes: str


ENV_CATALOG: tuple[EnvCatalogEntry, ...] = (
    EnvCatalogEntry(
        key="cartpole",
        family="classic_control",
        env_id="CartPole-v1",
        config_path=Path("configs/ppo_cartpole.yaml"),
        observation="Box(4)",
        action="Discrete(2)",
        notes="Fast starter balancing task.",
    ),
    EnvCatalogEntry(
        key="acrobot",
        family="classic_control",
        env_id="Acrobot-v1",
        config_path=Path("configs/ppo_acrobot.yaml"),
        observation="Box(6)",
        action="Discrete(3)",
        notes="Swing-up control with negative rewards.",
    ),
    EnvCatalogEntry(
        key="mountain-car",
        family="classic_control",
        env_id="MountainCar-v0",
        config_path=Path("configs/ppo_mountain_car.yaml"),
        observation="Box(2)",
        action="Discrete(3)",
        notes="Sparse-reward hill climb; may need more timesteps.",
    ),
    EnvCatalogEntry(
        key="mountain-car-continuous",
        family="classic_control",
        env_id="MountainCarContinuous-v0",
        config_path=Path("configs/ppo_mountain_car_continuous.yaml"),
        observation="Box(2)",
        action="Box(1)",
        notes="Continuous-action hill climb.",
    ),
    EnvCatalogEntry(
        key="pendulum",
        family="classic_control",
        env_id="Pendulum-v1",
        config_path=Path("configs/ppo_pendulum.yaml"),
        observation="Box(3)",
        action="Box(1)",
        notes="Continuous-action torque control.",
    ),
    EnvCatalogEntry(
        key="taxi",
        family="toy_text",
        env_id="Taxi-v3",
        config_path=Path("configs/ppo_taxi.yaml"),
        observation="Discrete(500)",
        action="Discrete(6)",
        notes="Pickup/dropoff planning task.",
    ),
    EnvCatalogEntry(
        key="frozen-lake",
        family="toy_text",
        env_id="FrozenLake-v1",
        config_path=Path("configs/ppo_frozen_lake.yaml"),
        observation="Discrete(16)",
        action="Discrete(4)",
        notes="Sparse grid-world; stochastic by default.",
    ),
)


def list_env_catalog() -> tuple[EnvCatalogEntry, ...]:
    return ENV_CATALOG


def catalog_config_names() -> tuple[str, ...]:
    return tuple(entry.config_path.name for entry in ENV_CATALOG)
