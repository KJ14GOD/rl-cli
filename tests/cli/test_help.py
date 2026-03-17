from typer.testing import CliRunner

from rlx.cli import app


runner = CliRunner()


def test_help_shows_root_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Local-first CLI for reinforcement learning experiments." in result.stdout
    assert "init" in result.stdout
    assert "train" in result.stdout
