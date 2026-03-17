from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_help_shows_root_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Local-first CLI for reinforcement learning experiments." in result.stdout
    assert "--style" in result.stdout
    assert "compare" in result.stdout
    assert "eval" in result.stdout
    assert "init" in result.stdout
    assert "styles" in result.stdout
    assert "train" in result.stdout
    assert "video" in result.stdout


def test_style_option_without_command_persists_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RLX_CONFIG_HOME", str(tmp_path))

    result = runner.invoke(app, ["--style", "forest"])

    assert result.exit_code == 0
    assert "Saved default style" in result.stdout
    config_path = tmp_path / "rlx" / "config.toml"
    assert config_path.exists()
    assert 'style = "forest"' in config_path.read_text(encoding="utf-8")


def test_styles_command_lists_available_themes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RLX_CONFIG_HOME", str(tmp_path))
    runner.invoke(app, ["--style", "ice"])

    result = runner.invoke(app, ["styles"])

    assert result.exit_code == 0
    assert "RLCLI Styles" in result.stdout
    assert "neon" in result.stdout
    assert "graphite" in result.stdout
    assert "forest" in result.stdout
    assert "ice" in result.stdout
    assert "saved" in result.stdout
