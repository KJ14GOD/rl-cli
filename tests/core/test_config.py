from pathlib import Path
from textwrap import dedent

import pytest

from rlx.config.loader import ConfigFileNotFoundError, ConfigValidationError, load_config
from rlx.core.env_catalog import list_env_catalog


def test_load_config_accepts_starter_template() -> None:
    config = load_config(Path("src/rlx/templates/project/configs/ppo_cartpole.yaml"))

    assert config.run_name == "cartpole_ppo"
    assert config.algo.name == "ppo"
    assert config.env.id == "CartPole-v1"
    assert config.policy.hidden_sizes == [128, 128]


def test_load_config_accepts_all_classic_control_templates() -> None:
    for entry in list_env_catalog():
        config = load_config(Path("src/rlx/templates/project") / entry.config_path)

        assert config.env.id == entry.env_id
        assert config.policy.type == "mlp"
        assert config.policy.hidden_sizes == [128, 128]


def test_load_config_rejects_missing_required_field(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-field.yaml"
    config_path.write_text(
        dedent(
            """
            run_name: cartpole_ppo
            seed: 42
            device: cpu

            env:
              num_envs: 4

            algo:
              name: ppo
              total_timesteps: 50000
              rollout_steps: 128
              batch_size: 256
              learning_rate: 0.0003
              gamma: 0.99
              gae_lambda: 0.95
              clip_range: 0.2
              entropy_coef: 0.01
              value_coef: 0.5
              update_epochs: 4

            policy:
              type: mlp
              hidden_sizes: [128, 128]

            checkpoint:
              save_every: 10000

            eval:
              every: 10000
              episodes: 10
              deterministic: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match=r"env\.id: Field required"):
        load_config(config_path)


def test_load_config_rejects_wrong_type(tmp_path: Path) -> None:
    config_path = tmp_path / "wrong-type.yaml"
    config_path.write_text(
        dedent(
            """
            run_name: cartpole_ppo
            seed: forty-two
            device: cpu

            env:
              id: CartPole-v1
              num_envs: 4

            algo:
              name: ppo
              total_timesteps: 50000
              rollout_steps: 128
              batch_size: 256
              learning_rate: 0.0003
              gamma: 0.99
              gae_lambda: 0.95
              clip_range: 0.2
              entropy_coef: 0.01
              value_coef: 0.5
              update_epochs: 4

            policy:
              type: mlp
              hidden_sizes: [128, 128]

            checkpoint:
              save_every: 10000

            eval:
              every: 10000
              episodes: 10
              deterministic: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match=r"seed: Input should be a valid integer"):
        load_config(config_path)


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(ConfigFileNotFoundError, match="Config file not found"):
        load_config(config_path)
