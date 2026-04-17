import json
from pathlib import Path

from typer.testing import CliRunner

from rlx.cli import app
from rlx.paths import CONFIG_SNAPSHOT_NAME, METADATA_NAME, METRICS_NAME, RUN_ARTIFACT_DIRS

runner = CliRunner()


def test_train_prepares_run_storage_model() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        result = runner.invoke(app, ["train", "bossfight/configs/ppo_cartpole.yaml"])

        assert result.exit_code == 0
        assert "Completed run" in result.stdout

        run_dir = Path("bossfight/runs/cartpole_ppo_001")
        assert run_dir.is_dir()
        assert (run_dir / CONFIG_SNAPSHOT_NAME).exists()
        assert (run_dir / METADATA_NAME).exists()
        assert (run_dir / METRICS_NAME).exists()
        assert (run_dir / "checkpoints" / "latest.zip").exists()

        for dirname in RUN_ARTIFACT_DIRS:
            assert (run_dir / dirname).is_dir()

        metadata = json.loads((run_dir / METADATA_NAME).read_text(encoding="utf-8"))
        assert metadata["run_id"] == "cartpole_ppo_001"
        assert metadata["status"] == "completed"
