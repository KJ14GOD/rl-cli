import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_eval_runs_against_trained_checkpoint() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/tiny_eval.yaml")
        config_path.write_text(
            dedent(
                """
                run_name: tiny_eval
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

        train_result = runner.invoke(app, ["train", str(config_path)])
        assert train_result.exit_code == 0

        checkpoint_path = Path("bossfight/runs/tiny_eval_001/checkpoints/latest.zip")
        result = runner.invoke(app, ["eval", str(checkpoint_path)])

        assert result.exit_code == 0
        assert "RLCLI Evaluation Complete" in result.stdout

        eval_result_path = Path("bossfight/runs/tiny_eval_001/eval/manual_eval_001.json")
        assert eval_result_path.exists()

        payload = json.loads(eval_result_path.read_text(encoding="utf-8"))
        assert payload["kind"] == "standalone_eval"
        assert payload["run_id"] == "tiny_eval_001"
        assert payload["run_name"] == "tiny_eval"
        assert payload["checkpoint"]["name"] == "latest.zip"
        assert payload["config"]["environment"] == "CartPole-v1"
        assert payload["config"]["episodes"] == 2
        assert payload["config"]["deterministic"] is True
        assert payload["config"]["resolved_device"] == "cpu"
        assert "mean_reward" in payload["summary"]
        assert "median_reward" in payload["summary"]
        assert len(payload["episodes"]) == 2


def test_eval_supports_multiple_explicit_checkpoints() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/tiny_eval.yaml")
        config_path.write_text(
            dedent(
                """
                run_name: tiny_eval
                seed: 9
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

        run_dir = Path("bossfight/runs/tiny_eval_001")
        latest = run_dir / "checkpoints/latest.zip"
        best = run_dir / "checkpoints/best.zip"

        result = runner.invoke(app, ["eval", str(latest), str(best)])

        assert result.exit_code == 0
        assert "RLCLI Batch Evaluation Complete" in result.stdout
        assert (run_dir / "eval/manual_eval_001.json").exists()
        assert (run_dir / "eval/manual_eval_002.json").exists()


def test_eval_supports_all_checkpoints_for_run() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/tiny_eval.yaml")
        config_path.write_text(
            dedent(
                """
                run_name: tiny_eval
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

        train_result = runner.invoke(app, ["train", str(config_path)])
        assert train_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_eval_001")
        result = runner.invoke(app, ["eval", "--run", str(run_dir), "--all-checkpoints"])

        assert result.exit_code == 0
        assert "RLCLI Batch Evaluation Complete" in result.stdout

        result_files = sorted(path.name for path in (run_dir / "eval").glob("manual_eval_*.json"))
        assert len(result_files) >= 3
