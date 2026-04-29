import json
from pathlib import Path
from textwrap import dedent

import yaml
from typer.testing import CliRunner

from rlx.cli import app
from rlx.config import load_config
from rlx.core.advisor import AdvisorError, run_advisor

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
        assert manifest["protocol"]["llm_strict"] is False
        assert manifest["protocol"]["fallback"]["used"] is False
        response_audit = manifest["protocol"]["planner_audit"]["llm_response"]
        assert response_audit["raw_count"] == 2
        assert response_audit["accepted_count"] == 1
        assert response_audit["rejected_count"] == 1
        assert response_audit["rejected"][0]["rejected_changes"][0]["key"] == "env.id"
        assert manifest["variants"][0]["proposal_source"] == "llm"
        assert "env.id" not in manifest["variants"][0]["mutations"]


def test_advisor_llm_fills_missing_variants_with_rules(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )
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

        result = run_advisor(
            "tiny_advisor_001",
            variants=2,
            planner="llm",
            llm_provider="mock",
            llm_model="mock-model",
            cwd=Path("bossfight"),
        )

        assert len(result.variants) == 2
        assert [variant.proposal_source for variant in result.variants] == ["llm", "rules"]

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        fallback = manifest["protocol"]["fallback"]
        assert fallback["used"] is True
        assert fallback["mode"] == "fill_missing_with_rules"
        assert fallback["llm_valid"] == 1
        assert fallback["rules_filled"] == 1
        assert manifest["protocol"]["planner_audit"]["llm_response"]["accepted_count"] == 1
        assert manifest["protocol"]["planner_audit"]["rule_fill_count"] == 1
        assert [variant["proposal_source"] for variant in manifest["variants"]] == [
            "llm",
            "rules",
        ]


def test_advisor_llm_repairs_rejected_proposals_before_rules(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )
        monkeypatch.setenv("RLX_LLM_MOCK_RESPONSE_INDEX", "0")
        monkeypatch.setenv(
            "RLX_LLM_MOCK_RESPONSES",
            json.dumps(
                [
                    {
                        "proposals": [
                            {
                                "changes": [{"key": "env.id", "value": "Acrobot-v1"}],
                                "signal": "illegal env change",
                                "rationale": "This should be repaired.",
                                "priority": "high",
                            }
                        ]
                    },
                    {
                        "proposals": [
                            {
                                "changes": [
                                    {"key": "algo.learning_rate", "value": 0.0002},
                                    {"key": "algo.clip_range", "value": 0.15},
                                ],
                                "signal": "repair signal",
                                "rationale": "Use allowed PPO knobs for a replacement.",
                                "priority": "high",
                            }
                        ]
                    },
                ]
            ),
        )

        result = run_advisor(
            "tiny_advisor_001",
            variants=1,
            planner="llm",
            llm_provider="mock",
            llm_model="mock-model",
            cwd=Path("bossfight"),
        )

        assert len(result.variants) == 1
        assert result.variants[0].proposal_source == "llm"
        assert result.variants[0].mutations == {
            "algo.learning_rate": 0.0002,
            "algo.clip_range": 0.15,
        }

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        assert protocol["fallback"]["used"] is False
        assert protocol["fallback"]["llm_valid"] == 1
        audit = protocol["planner_audit"]
        assert audit["llm_response"]["accepted_count"] == 0
        assert audit["llm_response"]["rejected_count"] == 1
        assert audit["repair_attempts"] == 1
        assert audit["repairs"][0]["selected_count"] == 1
        assert audit["repairs"][0]["llm_response"]["accepted_count"] == 1
        assert any(row["source"] == "llm_repair" for row in audit["proposals"])


def test_advisor_ollama_provider_uses_local_chat_api(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )
        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "changes": [
                                                {
                                                    "key": "algo.learning_rate",
                                                    "value": 0.0002,
                                                }
                                            ],
                                            "signal": "local ollama signal",
                                            "rationale": (
                                                "Local model proposes a smaller PPO step."
                                            ),
                                            "priority": "high",
                                        }
                                    ]
                                }
                            ),
                        },
                        "done": True,
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        result = run_advisor(
            "tiny_advisor_001",
            variants=1,
            planner="llm",
            llm_provider="ollama",
            llm_model="qwen3:8b",
            cwd=Path("bossfight"),
        )

        assert len(result.variants) == 1
        assert result.variants[0].proposal_source == "llm"
        assert result.variants[0].mutations == {"algo.learning_rate": 0.0002}

        request, timeout = calls[0]
        assert request.full_url == "http://localhost:11434/api/chat"
        assert timeout == 300.0
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "qwen3:8b"
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert payload["think"] is False
        assert payload["options"]["num_predict"] == 1200
        assert "Allowed mutation keys" in payload["messages"][1]["content"]

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["protocol"]["planner"] == "llm"
        assert manifest["protocol"]["llm_provider"] == "ollama"
        assert manifest["protocol"]["llm_model"] == "qwen3:8b"
        assert manifest["variants"][0]["proposal_source"] == "llm"


def test_advisor_includes_workspace_context_in_llm_and_manifest(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )
        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "changes": [
                                                {
                                                    "key": "algo.learning_rate",
                                                    "value": 0.0002,
                                                }
                                            ],
                                            "signal": "workspace grounded",
                                            "rationale": "Use the scoped research context.",
                                            "priority": "high",
                                        }
                                    ]
                                }
                            ),
                        },
                        "done": True,
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        result = run_advisor(
            "tiny_advisor_001",
            variants=1,
            planner="llm",
            llm_provider="ollama",
            llm_model="qwen3:8b",
            planner_context={
                "research_program": {"path": "program.md", "text": "maximize reward"},
                "workspace_scope": {
                    "editable_files": [
                        {
                            "path": "policies/custom_policy.py",
                            "sha256": "abc",
                            "text": "class X: ...",
                        }
                    ],
                    "locked_files": ["envs/custom_env.py"],
                },
            },
            workspace_summary={
                "program": {
                    "path": "program.md",
                    "sha256": "abc",
                    "chars": 32,
                    "excerpt_chars": 32,
                },
                "editable_files": [
                    {
                        "path": "policies/custom_policy.py",
                        "sha256": "def",
                        "chars": 128,
                        "excerpt_chars": 128,
                    }
                ],
                "locked_files": ["envs/custom_env.py"],
            },
            cwd=Path("bossfight"),
        )

        payload = json.loads(calls[0][0].data.decode("utf-8"))
        assert "research_program" in payload["messages"][1]["content"]
        assert "policies/custom_policy.py" in payload["messages"][1]["content"]

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        workspace = manifest["protocol"]["workspace"]
        assert workspace["program"]["path"] == "program.md"
        assert workspace["editable_files"][0]["path"] == "policies/custom_policy.py"
        assert workspace["locked_files"] == ["envs/custom_env.py"]


def test_advisor_llm_strict_fails_when_variants_are_missing(monkeypatch) -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "bossfight"])
        assert init_result.exit_code == 0

        run_dir = Path("bossfight/runs/tiny_advisor_001")
        _write_fake_run(
            run_dir=run_dir,
            run_name="tiny_advisor",
            rewards=(25.0, 26.0, 26.5),
        )
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

        try:
            run_advisor(
                "tiny_advisor_001",
                variants=2,
                planner="llm",
                llm_provider="mock",
                llm_model="mock-model",
                llm_strict=True,
                cwd=Path("bossfight"),
            )
        except AdvisorError as exc:
            assert "strict mode" in str(exc)
        else:
            raise AssertionError("Expected strict LLM advisor to fail.")


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
