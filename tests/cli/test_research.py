import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app

runner = CliRunner()


def test_research_command_writes_journal_without_training() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))

        result = runner.invoke(app, ["research", "bossfight/runs/tiny_research_001"])

        assert result.exit_code == 0
        assert "RLCLI Research Complete" in result.stdout
        assert "RLCLI Research Rounds" in result.stdout
        assert "dry_run" in result.stdout

        bundle = Path("bossfight/analysis/research/tiny_research_001_research_001")
        assert (bundle / "manifest.json").exists()
        assert (bundle / "report.md").exists()
        assert (bundle / "scoreboard.png").exists()
        assert (bundle / "progress.png").exists()
        assert "research_bundle" in (bundle / "manifest.json").read_text(encoding="utf-8")
        assert "Score plot" in result.stdout
        assert "Progress plot" in result.stdout


def test_research_command_resumes_existing_journal() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))

        first = runner.invoke(
            app,
            ["research", "bossfight/runs/tiny_research_001", "--rounds", "1", "--variants", "2"],
        )
        assert first.exit_code == 0

        result = runner.invoke(
            app,
            [
                "research",
                "--resume",
                "bossfight/analysis/research/tiny_research_001_research_001",
                "--rounds",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "RLCLI Research Complete" in result.stdout
        assert "Rounds" in result.stdout

        manifest_path = Path(
            "bossfight/analysis/research/tiny_research_001_research_001/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["settings"]["rounds"] == 2
        assert len(manifest["rounds"]) == 2


def test_research_command_supports_mock_llm_planner(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))
        monkeypatch.setenv(
            "RLX_LLM_MOCK_RESPONSE",
            json.dumps(
                {
                    "proposals": [
                        {
                            "changes": [{"key": "algo.gamma", "value": 0.98}],
                            "signal": "mock llm signal",
                            "rationale": "Mock LLM proposes one safe PPO mutation.",
                            "priority": "high",
                        }
                    ]
                }
            ),
        )

        result = runner.invoke(
            app,
            [
                "research",
                "bossfight/runs/tiny_research_001",
                "--rounds",
                "1",
                "--planner",
                "llm",
                "--llm-provider",
                "mock",
                "--llm-model",
                "mock-model",
            ],
        )

        assert result.exit_code == 0
        manifest_path = Path(
            "bossfight/analysis/research/tiny_research_001_research_001/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["protocol"]["planner"] == "llm"
        assert manifest["protocol"]["llm_provider"] == "mock"
        assert manifest["protocol"]["llm_model"] == "mock-model"


def _write_fake_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for dirname in ("checkpoints", "eval", "videos", "plots", "logs"):
        (run_dir / dirname).mkdir()

    (run_dir / "config_snapshot.yaml").write_text(
        dedent(
            """
            run_name: tiny_research
            seed: 7
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
              entropy_coef: 0.0
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

    metadata = {
        "run_id": run_dir.name,
        "run_name": "tiny_research",
        "status": "completed",
        "environment": "CartPole-v1",
        "device": "cpu",
        "resolved_device": "cpu",
        "total_timesteps": 256,
        "latest_checkpoint": "checkpoints/latest.zip",
        "best_checkpoint": "checkpoints/best.zip",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    records = []
    for index, reward in enumerate((25.0, 26.0, 26.5), start=1):
        records.append(
            {
                "step": index * 64,
                "rollout/ep_rew_mean": reward,
                "rollout/ep_len_mean": reward,
                "train/approx_kl": 0.00001,
                "train/clip_fraction": 0.0,
                "train/explained_variance": -0.2,
                "train/value_loss": 40.0 + index * 10,
            }
        )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")
