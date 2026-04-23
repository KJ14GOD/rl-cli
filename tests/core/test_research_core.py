import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app
from rlx.core.research import resume_research, run_research

runner = CliRunner()


def test_research_dry_run_creates_research_journal() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))

        result = run_research(
            "tiny_research_001",
            rounds=3,
            variants=2,
            cwd=Path("bossfight"),
        )

        assert result.mode == "dry_run"
        assert result.initial_run_id == "tiny_research_001"
        assert result.champion_run_id == "tiny_research_001"
        assert result.stop_reason == "completed requested rounds"
        assert len(result.rounds) == 3
        assert result.manifest_path.exists()
        assert result.report_path.exists()
        assert result.score_plot_path is not None
        assert result.score_plot_path.exists()
        assert result.progress_plot_path is not None
        assert result.progress_plot_path.exists()

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["kind"] == "research_bundle"
        assert manifest["mode"] == "dry_run"
        assert manifest["initial_run_id"] == "tiny_research_001"
        assert manifest["artifacts"][0]["key"] == "scoreboard"
        assert any(artifact["key"] == "progress" for artifact in manifest["artifacts"])
        assert "patience" not in manifest["settings"]
        assert manifest["protocol"]["version"] == 2
        assert manifest["protocol"]["score_mode"] == "existing_eval_or_rollout_signal"
        assert manifest["protocol"]["objective"]["metric"] == "eval_mean_reward"
        assert manifest["protocol"]["objective"]["higher_is_better"] is True
        assert manifest["protocol"]["budget"]["timesteps_per_variant"] == 256
        assert manifest["protocol"]["locked_mutations"]["algo.total_timesteps"] == 256
        assert manifest["protocol"]["llm_strict"] is False
        assert "env.id" not in manifest["protocol"]["allowed_mutation_keys"]
        assert len(manifest["rounds"]) == 3
        assert len(manifest["rounds"][0]["variants"]) == 2
        assert manifest["rounds"][0]["advisor_bundle"].startswith("analysis/advisor/")

        signatures = []
        ignored = set(manifest["protocol"]["signature_ignored_keys"])
        for research_round in manifest["rounds"]:
            for variant in research_round["variants"]:
                effective = {
                    key: value
                    for key, value in variant.get("mutations", {}).items()
                    if key not in ignored
                }
                signatures.append(json.dumps(effective, sort_keys=True))
        assert len(signatures) == len(set(signatures))


def test_research_protocol_controls_budget_and_mutation_keys() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))
        Path("bossfight/research.yaml").write_text(
            dedent(
                """
                objective: maximize eval reward
                baseline: tiny_research_001

                budget:
                  max_rounds: 1
                  max_variants_per_round: 1
                  max_timesteps_per_variant: 128

                allowed_changes:
                  - algo.learning_rate

                locked:
                  - env.id
                  - algo.total_timesteps
                  - eval.episodes
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = run_research(None, protocol_path="research.yaml", cwd=Path("bossfight"))

        assert len(result.rounds) == 1
        assert len(result.rounds[0].variants) == 1
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        assert protocol["source_protocol_name"] == "research.yaml"
        assert protocol["budget"]["max_rounds"] == 1
        assert protocol["budget"]["max_variants_per_round"] == 1
        assert protocol["budget"]["timesteps_per_variant"] == 128
        assert protocol["allowed_mutation_keys"] == ["algo.learning_rate"]
        assert protocol["locked_mutations"]["env.id"] == "CartPole-v1"
        assert protocol["locked_mutations"]["algo.total_timesteps"] == 128
        assert result.rounds[0].variants[0].mutations["algo.total_timesteps"] == 128


def test_research_resume_extends_existing_bundle_without_repeats() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))

        first = run_research(
            "tiny_research_001",
            rounds=1,
            variants=2,
            cwd=Path("bossfight"),
        )
        resumed = resume_research(first.bundle_dir, rounds=3)

        assert resumed.bundle_dir == first.bundle_dir
        assert resumed.manifest_path == first.manifest_path
        assert len(resumed.rounds) == 3
        assert resumed.mode == "dry_run"
        assert resumed.champion_run_id == "tiny_research_001"

        manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
        assert manifest["settings"]["rounds"] == 3
        assert manifest["settings"]["variants"] == 2

        signatures = []
        ignored = set(manifest["protocol"]["signature_ignored_keys"])
        for research_round in manifest["rounds"]:
            for variant in research_round["variants"]:
                effective = {
                    key: value
                    for key, value in variant.get("mutations", {}).items()
                    if key not in ignored
                }
                signatures.append(json.dumps(effective, sort_keys=True))
        assert len(signatures) == len(set(signatures))


def test_research_stops_cleanly_when_proposal_space_is_exhausted() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_research_001"))

        result = run_research(
            "tiny_research_001",
            rounds=20,
            variants=3,
            cwd=Path("bossfight"),
        )

        assert result.mode == "dry_run"
        assert len(result.rounds) < 20
        assert "proposal space exhausted" in result.stop_reason

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["stop_reason"] == result.stop_reason

        exhausted_manifests = []
        for path in Path("bossfight/analysis/advisor").glob(
            "tiny_research_001_advisor_*/manifest.json"
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "exhausted":
                exhausted_manifests.append(payload)

        assert exhausted_manifests
        assert exhausted_manifests[-1]["stop_reason"] == "proposal space exhausted"
        assert exhausted_manifests[-1]["variants"] == []


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
