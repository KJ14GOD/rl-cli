import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_analyze_summarizes_run_signals_and_next_moves() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_analyze_001")
        _write_fake_run(run_dir=run_dir)

        result = runner.invoke(app, ["analyze", "tiny_analyze_001"])

        assert result.exit_code == 0
        assert "RLCLI Run Analysis" in result.stdout
        assert "RLCLI Learning Signal" in result.stdout
        assert "RLCLI Findings" in result.stdout
        assert "RLCLI Next Moves" in result.stdout
        assert "improving" in result.stdout
        assert "Reward improved" in result.stdout
        assert "manual_eval_001.json" in result.stdout
        assert "rlx plot tiny_analyze_001" in result.stdout
        assert "rlx video" in result.stdout


def _write_fake_run(*, run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for dirname in ("checkpoints", "eval", "videos", "plots", "logs"):
        (run_dir / dirname).mkdir()

    (run_dir / "config_snapshot.yaml").write_text(
        dedent(
            """
            run_name: tiny_analyze
            seed: 7
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
        "run_name": "tiny_analyze",
        "status": "completed",
        "created_at": "2026-04-13T10:00:00Z",
        "started_at": "2026-04-13T10:00:01Z",
        "completed_at": "2026-04-13T10:00:05Z",
        "seed": 7,
        "device": "cpu",
        "resolved_device": "cpu",
        "environment": "CartPole-v1",
        "algorithm": "ppo",
        "project_root": str(Path("bossfight").resolve()),
        "run_dir": str(run_dir.resolve()),
        "source_config_path": "bossfight/configs/tiny_analyze.yaml",
        "total_timesteps": 256,
        "latest_checkpoint": "checkpoints/latest.zip",
        "best_checkpoint": "checkpoints/best.zip",
        "last_eval_at": "2026-04-13T10:00:06Z",
        "last_eval_result": "eval/manual_eval_001.json",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 64, "rollout/ep_rew_mean": 18.0}) + "\n"
        + json.dumps({"step": 128, "rollout/ep_rew_mean": 28.0}) + "\n"
        + json.dumps({"step": 192, "rollout/ep_rew_mean": 38.0}) + "\n"
        + json.dumps({"step": 256, "rollout/ep_rew_mean": 48.0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")
    (run_dir / "eval" / "manual_eval_001.json").write_text(
        json.dumps(
            {
                "kind": "standalone_eval",
                "summary": {
                    "mean_reward": 52.0,
                    "mean_episode_length": 110.0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
