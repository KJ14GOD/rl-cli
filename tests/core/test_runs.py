import json
from pathlib import Path

from rlx.config import load_config
from rlx.core.projects import init_project
from rlx.core.runs import prepare_run
from rlx.paths import (
    CONFIG_SNAPSHOT_NAME,
    METADATA_NAME,
    METRICS_NAME,
    RUN_ARTIFACT_DIRS,
    STARTER_CONFIG,
)


def test_prepare_run_creates_tracked_run_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "bossfight"
    init_project(project_root)

    config_path = project_root / STARTER_CONFIG
    config = load_config(config_path)

    result = prepare_run(config, config_path)

    assert result.run_dir.name == "cartpole_ppo_001"
    assert result.config_snapshot.name == CONFIG_SNAPSHOT_NAME
    assert result.metadata_path.name == METADATA_NAME
    assert result.metrics_path.name == METRICS_NAME
    assert result.config_snapshot.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )
    assert result.metrics_path.read_text(encoding="utf-8") == ""

    for dirname in RUN_ARTIFACT_DIRS:
        assert (result.run_dir / dirname).is_dir()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "cartpole_ppo_001"
    assert metadata["run_name"] == "cartpole_ppo"
    assert metadata["status"] == "prepared"
    assert metadata["environment"] == "CartPole-v1"
    assert metadata["algorithm"] == "ppo"


def test_prepare_run_increments_run_suffix(tmp_path: Path) -> None:
    project_root = tmp_path / "bossfight"
    init_project(project_root)

    config_path = project_root / STARTER_CONFIG
    config = load_config(config_path)

    first = prepare_run(config, config_path)
    second = prepare_run(config, config_path)

    assert first.run_dir.name == "cartpole_ppo_001"
    assert second.run_dir.name == "cartpole_ppo_002"
