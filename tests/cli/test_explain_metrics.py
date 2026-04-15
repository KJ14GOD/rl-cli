import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_explain_metrics_summarizes_logged_ppo_metrics() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_metrics_001")
        _write_fake_run(run_dir=run_dir)

        result = runner.invoke(app, ["explain-metrics", "tiny_metrics_001"])

        assert result.exit_code == 0
        assert "RLCLI Metrics Overview" in result.stdout
        assert "RLCLI Metric Series" in result.stdout
        assert "RLCLI Metric Notes" in result.stdout
        assert "Reward mean" in result.stdout
        assert "Approx KL" in result.stdout
        assert "Explained variance" in result.stdout
        assert "Reward trend is positive" in result.stdout
        assert "Critic fit is currently poor" in result.stdout


def _write_fake_run(*, run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for dirname in ("checkpoints", "eval", "videos", "plots", "logs"):
        (run_dir / dirname).mkdir()

    (run_dir / "config_snapshot.yaml").write_text(
        dedent(
            """
            run_name: tiny_metrics
            seed: 11
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

    metadata = {
        "run_id": run_dir.name,
        "run_name": "tiny_metrics",
        "status": "completed",
        "seed": 11,
        "device": "cpu",
        "resolved_device": "cpu",
        "environment": "CartPole-v1",
        "algorithm": "ppo",
        "project_root": str(Path("bossfight").resolve()),
        "run_dir": str(run_dir.resolve()),
        "source_config_path": "bossfight/configs/tiny_metrics.yaml",
        "total_timesteps": 256,
        "latest_checkpoint": "checkpoints/latest.zip",
        "best_checkpoint": "checkpoints/best.zip",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 64,
                "rollout/ep_rew_mean": 18.0,
                "rollout/ep_len_mean": 18.0,
                "train/approx_kl": 0.001,
                "train/clip_fraction": 0.01,
                "train/entropy_loss": -0.69,
                "train/explained_variance": 0.2,
                "train/value_loss": 42.0,
                "train/policy_gradient_loss": -0.002,
                "train/loss": 41.0,
                "train/learning_rate": 0.0003,
                "train/n_updates": 4,
                "progress_remaining": 0.75,
            }
        )
        + "\n"
        + json.dumps(
            {
                "step": 256,
                "rollout/ep_rew_mean": 48.0,
                "rollout/ep_len_mean": 48.0,
                "train/approx_kl": 0.004,
                "train/clip_fraction": 0.04,
                "train/entropy_loss": -0.62,
                "train/explained_variance": -0.1,
                "train/value_loss": 54.0,
                "train/policy_gradient_loss": -0.004,
                "train/loss": 51.0,
                "train/learning_rate": 0.0003,
                "train/n_updates": 8,
                "progress_remaining": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")
