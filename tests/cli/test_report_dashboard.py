import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rlx.cli import app
from rlx.core.report import (
    resolve_web_app_project,
    web_app_html,
    web_project_payload,
    web_research_payload,
    web_run_payload,
)

runner = CliRunner()


def test_report_generates_run_html() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_web_001")
        _write_fake_run(run_dir)

        result = runner.invoke(app, ["report", "bossfight/runs/tiny_web_001"])

        assert result.exit_code == 0
        assert "RLX Web Report" in result.stdout

        report = Path("bossfight/analysis/reports/tiny_web_001_report_001/index.html")
        assert report.exists()
        html = report.read_text(encoding="utf-8")
        assert "RLX Run Report" in html
        assert "tiny_web_001" in html
        assert "Interactive Metrics" in html


def test_report_generates_research_html() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_research_bundle(Path("bossfight/analysis/research/tiny_web_research_001"))

        result = runner.invoke(
            app,
            ["report", "bossfight/analysis/research/tiny_web_research_001"],
        )

        assert result.exit_code == 0
        report = Path("bossfight/analysis/reports/tiny_web_001_research_report_001/index.html")
        assert report.exists()
        html = report.read_text(encoding="utf-8")
        assert "RLX Research Report" in html
        assert "tiny_web_001" in html
        assert "Research Score Over Experiments" in html


def test_dashboard_export_generates_project_html() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_web_001"))
        _write_fake_research_bundle(Path("bossfight/analysis/research/tiny_web_research_001"))

        result = runner.invoke(app, ["dashboard", "bossfight", "--export"])

        assert result.exit_code == 0
        assert "RLX Local Dashboard" in result.stdout

        dashboard = Path("bossfight/analysis/dashboard/index.html")
        assert dashboard.exists()
        html = dashboard.read_text(encoding="utf-8")
        assert "RLX Dashboard" in html
        assert "Experiment Dashboard" in html
        assert "tiny_web_001" in html
        assert "tiny_web_research_001" in html


def test_report_preview_generates_sample_html() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["report", "--preview"])

        assert result.exit_code == 0
        report = Path(".rlx_preview/analysis/reports/preview_run_report/index.html")
        assert report.exists()
        html = report.read_text(encoding="utf-8")
        assert "preview_cartpole_001" in html
        assert "Run report" in html


def test_report_preview_generates_research_sample_html() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["report", "--preview", "--preview-kind", "research"])

        assert result.exit_code == 0
        report = Path(".rlx_preview/analysis/reports/preview_research_report/index.html")
        assert report.exists()
        html = report.read_text(encoding="utf-8")
        assert "preview_cartpole_001" in html
        assert "Research report" in html
        assert "research-filter" in html


def test_dashboard_demo_export_generates_sample_html() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["dashboard", "--demo", "--export"])

        assert result.exit_code == 0
        dashboard = Path(".rlx_preview/analysis/dashboard/index.html")
        assert dashboard.exists()
        html = dashboard.read_text(encoding="utf-8")
        assert "preview_cartpole_001" in html
        assert "Experiment Dashboard" in html


def test_connected_dashboard_payloads_read_project_artifacts() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        _write_fake_run(Path("bossfight/runs/tiny_web_001"))
        _write_fake_research_bundle(Path("bossfight/analysis/research/tiny_web_research_001"))

        project = resolve_web_app_project(Path("bossfight"))
        shell = web_app_html()
        project_payload = web_project_payload(project)
        run_payload = web_run_payload(project, "tiny_web_001")
        research_payload = web_research_payload(project, "tiny_web_research_001")

        assert "/api/project" in shell
        assert "app-command-list" in shell
        assert "rlx eval --run" in shell
        assert project_payload["runs"][0]["run_id"] == "tiny_web_001"
        assert run_payload["metrics"]["rollout/ep_rew_mean"][-1]["value"] == 52.0
        assert run_payload["artifacts"][0]["href"].startswith("/files/")
        assert research_payload["score_rows"][-1]["run_id"] == "tiny_web_002"


def test_connected_dashboard_demo_payloads_are_available() -> None:
    with runner.isolated_filesystem():
        project = resolve_web_app_project(demo=True)
        payload = web_project_payload(project)
        run_payload = web_run_payload(project, "preview_cartpole_001")
        research_payload = web_research_payload(project)

        assert payload["demo"] is True
        assert payload["runs"][0]["run_id"] == "preview_cartpole_001"
        assert "rollout/ep_rew_mean" in run_payload["metrics"]
        assert research_payload["score_rows"][0]["role"] == "baseline"


def _write_fake_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for dirname in ("checkpoints", "eval", "videos", "plots", "logs"):
        (run_dir / dirname).mkdir()

    (run_dir / "config_snapshot.yaml").write_text(
        dedent(
            """
            run_name: tiny_web
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
              entropy_coef: 0.01
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
        "run_name": "tiny_web",
        "status": "completed",
        "environment": "CartPole-v1",
        "device": "cpu",
        "resolved_device": "cpu",
        "total_timesteps": 256,
        "latest_checkpoint": "checkpoints/latest.zip",
        "best_checkpoint": "checkpoints/best.zip",
        "last_eval_result": "eval/manual_eval_001.json",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    records = [
        {"step": 64, "rollout/ep_rew_mean": 20.0, "rollout/ep_len_mean": 20.0},
        {
            "step": 128,
            "rollout/ep_rew_mean": 35.0,
            "rollout/ep_len_mean": 35.0,
            "train/approx_kl": 0.0002,
            "train/value_loss": 40.0,
        },
        {
            "step": 256,
            "rollout/ep_rew_mean": 52.0,
            "rollout/ep_len_mean": 52.0,
            "train/approx_kl": 0.0004,
            "train/value_loss": 28.0,
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "latest.zip").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.zip").write_bytes(b"best")
    (run_dir / "eval" / "manual_eval_001.json").write_text(
        json.dumps(
            {
                "kind": "standalone_eval",
                "checkpoint": {"path": "checkpoints/latest.zip", "name": "latest.zip"},
                "summary": {"mean_reward": 61.0, "mean_episode_length": 120.0},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fake_research_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True)
    manifest = {
        "kind": "research_bundle",
        "updated_at": "2026-04-21T00:00:00Z",
        "mode": "dry_run",
        "bundle": str(bundle_dir),
        "initial": {
            "run_id": "tiny_web_001",
            "score": 61.0,
            "score_source": "manual eval",
        },
        "initial_run_id": "tiny_web_001",
        "champion": {
            "run_id": "tiny_web_002",
            "score": 74.0,
            "score_source": "manual eval",
        },
        "settings": {"rounds": 1, "variants": 2},
        "protocol": {"version": 1, "planner": "rules"},
        "stop_reason": "completed requested rounds",
        "rounds": [
            {
                "index": 1,
                "baseline_run_id": "tiny_web_001",
                "advisor_bundle": "analysis/advisor/tiny_web_001_advisor_001",
                "advisor_manifest": "analysis/advisor/tiny_web_001_advisor_001/manifest.json",
                "champion_before": "tiny_web_001",
                "champion_score_before": 61.0,
                "candidate_run_id": "tiny_web_002",
                "candidate_score": 74.0,
                "candidate_score_source": "manual eval",
                "improvement": 13.0,
                "promoted": True,
                "variants": [
                    {
                        "index": 1,
                        "run_id": "tiny_web_002",
                        "status": "completed",
                        "mutations": {"algo.learning_rate": 0.001},
                        "score": 74.0,
                        "score_source": "manual eval",
                        "promoted": True,
                    }
                ],
                "champion_after": "tiny_web_002",
                "champion_score_after": 74.0,
                "stop_reason": None,
            }
        ],
        "artifacts": [],
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
