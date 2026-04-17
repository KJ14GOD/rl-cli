import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_diagnose_reports_run_failure_modes() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(
            run_dir=Path("bossfight/runs/tiny_bad_001"),
            run_name="tiny_bad",
            rewards=(80.0, 60.0, 40.0),
            eval_reward=None,
        )

        result = runner.invoke(app, ["diagnose", "tiny_bad_001"])

        assert result.exit_code == 0
        assert "RLCLI Diagnosis Overview" in result.stdout
        assert "RLCLI Diagnostics" in result.stdout
        assert "regressed late" in result.stdout
        assert "Evaluation" in result.stdout
        assert "needs attention" in result.stdout


def test_suggest_proposes_next_actions() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(
            run_dir=Path("bossfight/runs/tiny_suggest_001"),
            run_name="tiny_suggest",
            rewards=(25.0, 30.0, 35.0),
            eval_reward=None,
        )

        result = runner.invoke(app, ["suggest", "tiny_suggest_001"])

        assert result.exit_code == 0
        assert "RLCLI Suggest Overview" in result.stdout
        assert "RLCLI Suggested Actions" in result.stdout
        assert "rlx eval --run" in result.stdout
        assert "rlx plot" in result.stdout
        assert "algo.learning_rate" in result.stdout


def test_summarize_project_reports_best_runs() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(
            run_dir=Path("bossfight/runs/tiny_a_001"),
            run_name="tiny_a",
            rewards=(20.0, 40.0),
            eval_reward=55.0,
        )
        _write_fake_run(
            run_dir=Path("bossfight/runs/tiny_b_001"),
            run_name="tiny_b",
            rewards=(30.0, 75.0),
            eval_reward=90.0,
        )

        result = runner.invoke(app, ["summarize", "bossfight"])

        assert result.exit_code == 0
        assert "RLCLI Project Summary" in result.stdout
        assert "RLCLI Project Runs" in result.stdout
        assert "tiny_b_001" in result.stdout
        assert "Best eval" in result.stdout


def _write_fake_run(
    *,
    run_dir: Path,
    run_name: str,
    rewards: tuple[float, ...],
    eval_reward: float | None,
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
    }
    if eval_reward is not None:
        metadata["last_eval_result"] = "eval/manual_eval_001.json"

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    records = []
    for index, reward in enumerate(rewards, start=1):
        records.append(
            {
                "step": index * 64,
                "rollout/ep_rew_mean": reward,
                "rollout/ep_len_mean": reward,
                "train/approx_kl": 0.001,
                "train/clip_fraction": 0.01,
                "train/explained_variance": -0.2,
                "train/value_loss": 40.0 + index * 10,
            }
        )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")

    if eval_reward is not None:
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
