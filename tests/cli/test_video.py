import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_video_renders_checkpoint_bundle() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        config_path = Path("bossfight/configs/tiny_video.yaml")
        config_path.write_text(
            dedent(
                """
                run_name: tiny_video
                seed: 13
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

        checkpoint_path = Path("bossfight/runs/tiny_video_001/checkpoints/latest.zip")
        result = runner.invoke(app, ["video", str(checkpoint_path), "--episodes", "1"])

        assert result.exit_code == 0
        assert "RLCLI Video Rendered" in result.stdout

        video_dir = Path("bossfight/runs/tiny_video_001/videos/manual_video_001")
        manifest_path = video_dir / "manifest.json"
        gif_path = video_dir / "episode_001.gif"

        assert manifest_path.exists()
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["kind"] == "standalone_video"
        assert payload["checkpoint"]["name"] == "latest.zip"
        assert payload["config"]["episodes"] == 1
        assert payload["config"]["format"] == "gif"
        assert payload["summary"]["video_count"] == 1
        assert payload["episodes"][0]["file"] == "videos/manual_video_001/episode_001.gif"
