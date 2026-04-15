import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_ls_lists_runs_for_project() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(
            run_dir=Path("bossfight/runs/tiny_alpha_001"),
            run_name="tiny_alpha",
            final_reward=42.5,
            eval_reward=55.0,
        )
        _write_fake_run(
            run_dir=Path("bossfight/runs/tiny_beta_002"),
            run_name="tiny_beta",
            final_reward=61.25,
            eval_reward=74.5,
        )

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir("bossfight")
            result = runner.invoke(app, ["ls"], env={}, catch_exceptions=False)
        finally:
            os.chdir(original_cwd)

        assert result.exit_code == 0
        assert "RLCLI Runs" in result.stdout
        assert "RLCLI Run List" in result.stdout
        assert "tiny_alpha_001" in result.stdout
        assert "tiny_beta_002" in result.stdout
        assert "CartPole-v1" in result.stdout


def test_ls_reports_empty_project_cleanly() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir("bossfight")
            result = runner.invoke(app, ["ls"])
        finally:
            os.chdir(original_cwd)

        assert result.exit_code == 0
        assert "No runs found yet" in result.stdout


def _write_fake_run(
    *,
    run_dir: Path,
    run_name: str,
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
