import json
from pathlib import Path

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_custom_template_train_eval_and_video_work_end_to_end() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight", "--template", "custom"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/custom_ppo.yaml")
        config_text = config_path.read_text(encoding="utf-8")
        config_text = config_text.replace("total_timesteps: 50000", "total_timesteps: 256")
        config_text = config_text.replace("save_every: 10000", "save_every: 128")
        config_text = config_text.replace("every: 10000", "every: 128")
        config_text = config_text.replace("episodes: 10", "episodes: 2")
        config_text = config_text.replace("num_envs: 4", "num_envs: 1")
        config_path.write_text(config_text, encoding="utf-8")

        train_result = runner.invoke(app, ["train", str(config_path)])
        assert train_result.exit_code == 0
        assert "Completed run" in train_result.stdout

        run_dir = Path("bossfight/runs/custom_cartpole_ppo_001")
        assert run_dir.is_dir()
        assert (run_dir / "checkpoints/latest.zip").exists()

        eval_result = runner.invoke(app, ["eval", str(run_dir / "checkpoints/latest.zip")])
        assert eval_result.exit_code == 0
        assert (run_dir / "eval/manual_eval_001.json").exists()

        video_result = runner.invoke(
            app,
            ["video", str(run_dir / "checkpoints/latest.zip"), "--episodes", "1"],
        )
        assert video_result.exit_code == 0
        assert (run_dir / "videos/manual_video_001/manifest.json").exists()

        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "completed"
