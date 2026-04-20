from pathlib import Path

from typer.testing import CliRunner

from rlx.cli import app
from rlx.core.env_catalog import catalog_config_names
from rlx.paths import PROJECT_DIRS, STARTER_CONFIG

runner = CliRunner()


def test_init_creates_project_scaffold() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "bossfight"])

        assert result.exit_code == 0

        project_root = Path("bossfight")
        assert project_root.is_dir()

        for dirname in PROJECT_DIRS:
            assert (project_root / dirname).is_dir()

        starter_config = project_root / STARTER_CONFIG
        assert starter_config.exists()
        assert "CartPole-v1" in starter_config.read_text(encoding="utf-8")
        assert (project_root / ".env.example").exists()
        assert "cp .env.example .env" in (project_root / ".env.example").read_text(
            encoding="utf-8"
        )
        assert (project_root / ".gitignore").exists()
        assert ".env" in (project_root / ".gitignore").read_text(encoding="utf-8")
        assert "cp .env.example .env" in result.stdout

        for filename in catalog_config_names():
            assert (project_root / "configs" / filename).exists()
