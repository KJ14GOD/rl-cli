import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_sweep_runs_grid_variants_and_writes_bundle() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        base_config_path = Path("bossfight/configs/tiny_sweep.yaml")
        base_config_path.write_text(
            dedent(
                """
                run_name: tiny_sweep
                seed: 1
                device: cpu

                env:
                  id: CartPole-v1
                  num_envs: 1

                algo:
                  name: ppo
                  total_timesteps: 64
                  rollout_steps: 32
                  batch_size: 32
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
                  save_every: 64

                eval:
                  every: 64
                  episodes: 2
                  deterministic: true
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        sweep_config_path = Path("bossfight/configs/tiny_sweep_grid.yaml")
        sweep_config_path.write_text(
            dedent(
                """
                name: tiny_grid
                base_config: tiny_sweep.yaml
                tags: [grid]

                grid:
                  seed: [1, 2]
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir("bossfight")
            result = runner.invoke(app, ["sweep", "configs/tiny_sweep_grid.yaml"])
            info_result = runner.invoke(app, ["info", "tiny_sweep_001"])
        finally:
            os.chdir(original_cwd)

        assert result.exit_code == 0
        assert "RLCLI Sweep Complete" in result.stdout
        assert "RLCLI Sweep Variants" in result.stdout
        assert "tiny_sweep_001" in result.stdout
        assert "tiny_sweep_002" in result.stdout

        bundle_dir = Path("bossfight/analysis/sweeps/tiny_grid_001")
        manifest_path = bundle_dir / "manifest.json"
        assert manifest_path.exists()
        assert (bundle_dir / "configs/variant_001.yaml").exists()
        assert (bundle_dir / "configs/variant_002.yaml").exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["kind"] == "sweep_bundle"
        assert manifest["name"] == "tiny_grid"
        assert len(manifest["variants"]) == 2
        assert manifest["variants"][0]["status"] == "completed"
        assert manifest["variants"][1]["status"] == "completed"

        metadata = json.loads(
            Path("bossfight/runs/tiny_sweep_001/metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["sweep_name"] == "tiny_grid"
        assert metadata["sweep_variant_index"] == 1
        assert metadata["tags"] == ["sweep", "tiny_grid", "grid"]

        assert info_result.exit_code == 0
        assert "tiny_grid" in info_result.stdout
