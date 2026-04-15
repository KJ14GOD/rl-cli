import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_resume_creates_new_run_with_lineage() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/tiny_resume.yaml")
        config_path.write_text(
            dedent(
                """
                run_name: tiny_resume
                seed: 5
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

        train_result = runner.invoke(app, ["train", str(config_path)])
        assert train_result.exit_code == 0

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir("bossfight")
            resume_result = runner.invoke(app, ["resume", "tiny_resume_001", "--timesteps", "64"])
            info_result = runner.invoke(app, ["info", "tiny_resume_002"])
        finally:
            os.chdir(original_cwd)

        assert resume_result.exit_code == 0
        assert "RLCLI Training Resumed" in resume_result.stdout
        assert "tiny_resume_001" in resume_result.stdout

        resumed_run_dir = Path("bossfight/runs/tiny_resume_002")
        assert resumed_run_dir.is_dir()
        assert (resumed_run_dir / "checkpoints/latest.zip").exists()
        assert (resumed_run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip() != ""

        metadata = json.loads((resumed_run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["resumed_from_run"] == "tiny_resume_001"
        assert metadata["resumed_from_checkpoint"] == "checkpoints/latest.zip"
        assert metadata["status"] == "completed"

        assert info_result.exit_code == 0
        assert "tiny_resume_001" in info_result.stdout
        assert "checkpoints/latest.zip" in info_result.stdout
