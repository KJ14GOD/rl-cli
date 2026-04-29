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
        protocol = project_root / "research.yaml"
        assert protocol.exists()
        protocol_text = protocol.read_text(encoding="utf-8")
        assert "objective: maximize eval reward" in protocol_text
        assert "program: program.md" in protocol_text
        assert "allowed_changes:" in protocol_text
        assert "locked:" in protocol_text
        assert (project_root / "program.md").exists()
        assert "cp .env.example .env" in result.stdout
        assert "research.yaml" in result.stdout
        assert "program.md" in result.stdout

        for filename in catalog_config_names():
            assert (project_root / "configs" / filename).exists()


def test_init_custom_template_scaffolds_editable_project_code() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "bossfight", "--template", "custom"])

        assert result.exit_code == 0

        project_root = Path("bossfight")
        assert (project_root / "configs/custom_ppo.yaml").exists()
        assert (project_root / "envs/__init__.py").exists()
        assert (project_root / "envs/custom_env.py").exists()
        assert (project_root / "policies/__init__.py").exists()
        assert (project_root / "policies/custom_policy.py").exists()
        assert (project_root / "program.md").exists()
        research_text = (project_root / "research.yaml").read_text(encoding="utf-8")
        assert "baseline: custom_cartpole_ppo_001" in research_text
        assert "editable_files:" in research_text
        assert "policies/custom_policy.py" in research_text
        assert "custom" in result.stdout
        assert "configs/custom_ppo.yaml" in result.stdout
