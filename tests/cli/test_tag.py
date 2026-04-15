import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_tag_adds_tags_to_run_metadata() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_tag_001"), "tiny_tag")

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir("bossfight")
            result = runner.invoke(app, ["tag", "tiny_tag_001", "baseline", "seed-7"])
        finally:
            os.chdir(original_cwd)

        assert result.exit_code == 0
        assert "RLCLI Run Tagged" in result.stdout
        assert "baseline, seed-7" in result.stdout

        metadata = json.loads(
            Path("bossfight/runs/tiny_tag_001/metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["tags"] == ["baseline", "seed-7"]


def test_tag_shows_up_in_info_and_ls() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_tag_001"), "tiny_tag")

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir("bossfight")
            tag_result = runner.invoke(app, ["tag", "tiny_tag_001", "baseline"])
            info_result = runner.invoke(app, ["info", "tiny_tag_001"])
            ls_result = runner.invoke(app, ["ls"])
        finally:
            os.chdir(original_cwd)

        assert tag_result.exit_code == 0
        assert info_result.exit_code == 0
        assert ls_result.exit_code == 0
        assert "baseline" in info_result.stdout
        assert "baseline" in ls_result.stdout


def _write_fake_run(run_dir: Path, run_name: str) -> None:
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
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 64,
                "rollout/ep_rew_mean": 32.0,
                "rollout/ep_len_mean": 20.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "step": 256,
                "rollout/ep_rew_mean": 48.5,
                "rollout/ep_len_mean": 32.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")
