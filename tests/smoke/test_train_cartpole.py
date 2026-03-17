import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app


runner = CliRunner()


def test_train_runs_tiny_cartpole_job() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/tiny_cartpole.yaml")
        config_path.write_text(
            dedent(
                """
                run_name: tiny_cartpole
                seed: 42
                device: cpu

                env:
                  id: CartPole-v1
                  num_envs: 1

                algo:
                  name: ppo
                  total_timesteps: 256
                  rollout_steps: 64
                  batch_size: 64
                  learning_rate: 0.0003
                  gamma: 0.99
                  gae_lambda: 0.95
                  clip_range: 0.2
                  entropy_coef: 0.01
                  value_coef: 0.5
                  update_epochs: 2

                policy:
                  type: mlp
                  hidden_sizes: [32, 32]

                checkpoint:
                  save_every: 128

                eval:
                  every: 128
                  episodes: 2
                  deterministic: true
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["train", str(config_path)])

        assert result.exit_code == 0
        assert "Completed run" in result.stdout

        run_dir = Path("bossfight/runs/tiny_cartpole_001")
        assert run_dir.is_dir()
        assert (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip() != ""
        assert (run_dir / "checkpoints/latest.zip").exists()

        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "completed"
        assert metadata["resolved_device"] == "cpu"
