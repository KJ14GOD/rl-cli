from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_envs_lists_classic_control_catalog() -> None:
    result = runner.invoke(app, ["envs"])

    assert result.exit_code == 0
    assert "RLCLI Environment Catalog" in result.stdout
    assert "CartPole-v1" in result.stdout
    assert "Acrobot-v1" in result.stdout
    assert "MountainCarContinuous-v0" in result.stdout
    assert "Pendulum-v1" in result.stdout
