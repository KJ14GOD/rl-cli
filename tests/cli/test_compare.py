import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_compare_summarizes_runs_and_config_differences() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_a = Path("bossfight/runs/tiny_a_001")
        run_b = Path("bossfight/runs/tiny_b_001")
        _write_fake_run(
            run_dir=run_a,
            run_name="tiny_a",
            seed=7,
            learning_rate=0.0003,
            final_reward=42.5,
            eval_reward=55.0,
        )
        _write_fake_run(
            run_dir=run_b,
            run_name="tiny_b",
            seed=11,
            learning_rate=0.0001,
            final_reward=61.25,
            eval_reward=74.5,
        )

        result = runner.invoke(app, ["compare", "tiny_a_001", "tiny_b_001"])

        assert result.exit_code == 0
        assert "RLCLI Run Comparison" in result.stdout
        assert "RLCLI Run Artifacts" in result.stdout
        assert "RLCLI Config Differences" in result.stdout
        assert "tiny_a_001" in result.stdout
        assert "tiny_b_001" in result.stdout
        assert "algo.learning_rate" in result.stdout
        assert "seed" in result.stdout


def _write_fake_run(
    *,
    run_dir: Path,
    run_name: str,
    seed: int,
    learning_rate: float,
    final_reward: float,
    eval_reward: float,
) -> None:
    run_dir.mkdir(parents=True)
    for dirname in ("checkpoints", "eval", "videos", "plots", "logs"):
        (run_dir / dirname).mkdir()

    (run_dir / "config_snapshot.yaml").write_text(
        dedent(
            f"""
            run_name: {run_name}
            seed: {seed}
            device: cpu

            env:
              id: CartPole-v1
              num_envs: 1

            algo:
              name: ppo
              total_timesteps: 256
              rollout_steps: 64
              batch_size: 64
              learning_rate: {learning_rate}
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
        "run_name": run_name,
        "status": "completed",
        "environment": "CartPole-v1",
        "device": "cpu",
        "resolved_device": "cpu",
        "total_timesteps": 256,
        "latest_checkpoint": "checkpoints/latest.zip",
        "best_checkpoint": "checkpoints/best.zip",
        "last_eval_result": "eval/manual_eval_001.json",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 64, "rollout/ep_rew_mean": final_reward - 10}) + "\n"
        + json.dumps({"step": 256, "rollout/ep_rew_mean": final_reward}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")
    (run_dir / "eval" / "manual_eval_001.json").write_text(
        json.dumps(
            {
                "kind": "standalone_eval",
                "summary": {
                    "mean_reward": eval_reward,
                    "mean_episode_length": 120.0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
