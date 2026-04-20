import json
from pathlib import Path
from textwrap import dedent

import yaml
from typer.testing import CliRunner

from rlx.cli import app
from rlx.config import load_config
from rlx.core.advisor import run_advisor

runner = CliRunner()


def test_advisor_creates_grounded_variant_configs() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )

        result = run_advisor("tiny_advisor_001", variants=3, cwd=Path("bossfight"))

        assert result.mode == "dry_run"
        assert result.baseline_run_id == "tiny_advisor_001"
        assert result.manifest_path.exists()
        assert result.plan_path.exists()
        assert len(result.variants) == 3
        assert all(variant.status == "proposed" for variant in result.variants)
        assert any("algo.learning_rate" in variant.mutations for variant in result.variants)

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["kind"] == "advisor_bundle"
        assert manifest["mode"] == "dry_run"
        assert manifest["baseline"]["run_id"] == "tiny_advisor_001"
        assert len(manifest["variants"]) == 3

        generated = yaml.safe_load(result.variants[0].config_path.read_text(encoding="utf-8"))
        assert generated["run_name"] == "tiny_advisor"
        assert load_config(result.variants[0].config_path).run_name == "tiny_advisor"


def test_advisor_llm_planner_validates_allowed_config_mutations(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )
        Path("bossfight/.env").write_text(
            "RLX_LLM_PROVIDER=mock\nRLX_LLM_MODEL=mock-model\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(
            "RLX_LLM_MOCK_RESPONSE",
            json.dumps(
                {
                    "proposals": [
                        {
                            "changes": [{"key": "env.id", "value": "Acrobot-v1"}],
                            "signal": "illegal env change",
                            "rationale": "This should be rejected by RLX.",
                            "priority": "high",
                        },
                        {
                            "changes": [
                                {"key": "algo.gamma", "value": 0.98},
                                {"key": "policy.hidden_sizes", "value": [64, 64]},
                            ],
                            "signal": "critic and return horizon adjustment",
                            "rationale": "Test a shorter return horizon and wider MLP.",
                            "priority": "high",
                        },
                    ]
                }
            ),
        )

        result = run_advisor(
            "tiny_advisor_001",
            variants=1,
            planner="llm",
            cwd=Path("bossfight"),
        )

        assert len(result.variants) == 1
        assert result.variants[0].mutations == {
            "algo.gamma": 0.98,
            "policy.hidden_sizes": [64, 64],
        }

        generated = yaml.safe_load(result.variants[0].config_path.read_text(encoding="utf-8"))
        assert generated["env"]["id"] == "CartPole-v1"
        assert generated["algo"]["gamma"] == 0.98
        assert generated["policy"]["hidden_sizes"] == [64, 64]

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["protocol"]["planner"] == "llm"
        assert manifest["protocol"]["llm_provider"] == "mock"
        assert manifest["protocol"]["llm_model"] == "mock-model"
        assert "env.id" not in manifest["variants"][0]["mutations"]


def _write_fake_run(
    *,
    run_dir: Path,
    run_name: str,
    rewards: tuple[float, ...],
) -> None:
    run_dir.mkdir(parents=True)
    for dirname in ("checkpoints", "eval", "videos", "plots", "logs"):
        (run_dir / dirname).mkdir()

    (run_dir / "config_snapshot.yaml").write_text(
        dedent(
            f"""
            run_name: {run_name}
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
        "run_name": run_name,
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
    for index, reward in enumerate(rewards, start=1):
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
