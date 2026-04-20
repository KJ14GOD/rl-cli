from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.core.diagnose import RunDiagnosis
from rlx.llm.env import get_env_value


class LLMPlannerError(Exception):
    """Raised when an LLM planner cannot produce usable proposals."""


LLM_DEFAULT_PROVIDER = "openai"
LLM_DEFAULT_MODEL = "gpt-5.4-mini"
LLM_ALLOWED_MUTATION_KEYS = (
    "seed",
    "device",
    "env.num_envs",
    "algo.rollout_steps",
    "algo.batch_size",
    "algo.learning_rate",
    "algo.gamma",
    "algo.gae_lambda",
    "algo.clip_range",
    "algo.entropy_coef",
    "algo.value_coef",
    "algo.update_epochs",
    "policy.hidden_sizes",
    "checkpoint.save_every",
)
LLM_FORBIDDEN_MUTATION_KEYS = (
    "env.id",
    "algo.total_timesteps",
    "eval.every",
    "eval.episodes",
    "eval.deterministic",
)


@dataclass(frozen=True)
class LLMProposal:
    mutations: dict[str, Any]
    signal: str
    rationale: str
    priority: str


def generate_llm_proposals(
    *,
    provider: str,
    model: str,
    project_root: Path,
    base_payload: dict[str, Any],
    diagnosis: RunDiagnosis,
    allowed_mutation_keys: tuple[str, ...],
    locked_mutations: dict[str, Any],
    excluded_mutation_signatures: set[str],
    signature_ignored_keys: tuple[str, ...],
    variants: int,
) -> list[LLMProposal]:
    context = _build_context(
        base_payload=base_payload,
        diagnosis=diagnosis,
        allowed_mutation_keys=allowed_mutation_keys,
        locked_mutations=locked_mutations,
        excluded_mutation_signatures=excluded_mutation_signatures,
        signature_ignored_keys=signature_ignored_keys,
        variants=variants,
    )
    payload = _call_provider(
        provider=provider,
        model=model,
        project_root=project_root,
        context=context,
        allowed_mutation_keys=allowed_mutation_keys,
    )
    return _parse_proposals(payload, allowed_mutation_keys=allowed_mutation_keys)


def _call_provider(
    *,
    provider: str,
    model: str,
    project_root: Path,
    context: dict[str, Any],
    allowed_mutation_keys: tuple[str, ...],
) -> dict[str, Any]:
    normalized = provider.lower()
    if normalized == "mock":
        return _mock_response()
    if normalized == "openai":
        return _openai_response(
            model=model,
            project_root=project_root,
            context=context,
            allowed_mutation_keys=allowed_mutation_keys,
        )
    raise LLMPlannerError(
        f"Unsupported LLM provider '{provider}'. Supported providers: openai, mock."
    )


def _openai_response(
    *,
    model: str,
    project_root: Path,
    context: dict[str, Any],
    allowed_mutation_keys: tuple[str, ...],
) -> dict[str, Any]:
    api_key = get_env_value("OPENAI_API_KEY", project_root=project_root)
    if not api_key:
        raise LLMPlannerError(
            "OPENAI_API_KEY was not found. In your RLX project, run "
            "`cp .env.example .env`, then add your OpenAI API key to `.env`."
        )

    base_url = (
        get_env_value("OPENAI_BASE_URL", project_root=project_root)
        or "https://api.openai.com/v1"
    ).rstrip("/")
    timeout_raw = get_env_value("RLX_LLM_TIMEOUT", project_root=project_root, default="60")
    timeout = _parse_timeout(timeout_raw)
    request_payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": _system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(context, indent=2, sort_keys=True),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rlx_llm_advisor_plan",
                "strict": True,
                "schema": _proposal_schema(allowed_mutation_keys),
            }
        },
    }
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMPlannerError(f"OpenAI planner request failed: HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMPlannerError(f"OpenAI planner request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMPlannerError("OpenAI planner response was not valid JSON.") from exc

    text = _extract_openai_text(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMPlannerError("OpenAI planner response text was not valid proposal JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMPlannerError("OpenAI planner response must be a JSON object.")
    return parsed


def _mock_response() -> dict[str, Any]:
    raw = os.environ.get("RLX_LLM_MOCK_RESPONSE")
    if not raw:
        raise LLMPlannerError("RLX_LLM_MOCK_RESPONSE is required when using mock provider.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMPlannerError("RLX_LLM_MOCK_RESPONSE is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise LLMPlannerError("RLX_LLM_MOCK_RESPONSE must be a JSON object.")
    return payload


def _parse_proposals(
    payload: dict[str, Any],
    *,
    allowed_mutation_keys: tuple[str, ...],
) -> list[LLMProposal]:
    raw_proposals = payload.get("proposals")
    if not isinstance(raw_proposals, list):
        raise LLMPlannerError("LLM planner response must contain a proposals list.")

    allowed = set(allowed_mutation_keys)
    proposals: list[LLMProposal] = []
    for item in raw_proposals:
        if not isinstance(item, dict):
            continue
        mutations = _changes_to_mutations(item.get("changes"), allowed=allowed)
        if not mutations:
            continue
        proposals.append(
            LLMProposal(
                mutations=mutations,
                signal=_clean_text(item.get("signal"), default="LLM planner signal"),
                rationale=_clean_text(
                    item.get("rationale"),
                    default="LLM planner proposed this variant from run context.",
                ),
                priority=_clean_priority(item.get("priority")),
            )
        )

    if not proposals:
        raise LLMPlannerError("LLM planner did not return any valid proposals.")
    return proposals


def _changes_to_mutations(value: Any, *, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    mutations: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str) or key not in allowed:
            continue
        mutation_value = item.get("value")
        if not _is_supported_value(mutation_value):
            continue
        mutations[key] = mutation_value
    return mutations


def _is_supported_value(value: Any) -> bool:
    if isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list) and value:
        return all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
    return False


def _build_context(
    *,
    base_payload: dict[str, Any],
    diagnosis: RunDiagnosis,
    allowed_mutation_keys: tuple[str, ...],
    locked_mutations: dict[str, Any],
    excluded_mutation_signatures: set[str],
    signature_ignored_keys: tuple[str, ...],
    variants: int,
) -> dict[str, Any]:
    analysis = diagnosis.analysis
    info = analysis.info
    run = info.run
    learning = analysis.learning
    return {
        "task": "Propose PPO YAML config mutations for the next RLX advisor variants.",
        "requested_variants": variants,
        "constraints": {
            "allowed_mutation_keys": list(allowed_mutation_keys),
            "forbidden_mutation_keys": list(LLM_FORBIDDEN_MUTATION_KEYS),
            "locked_mutations": locked_mutations,
            "signature_ignored_keys": list(signature_ignored_keys),
            "already_tried_signatures": sorted(excluded_mutation_signatures),
            "must_return_json_only": True,
        },
        "baseline_config": base_payload,
        "run_info": {
            "run_id": run.run_id,
            "status": run.status,
            "environment": run.environment,
            "requested_device": run.requested_device,
            "resolved_device": run.resolved_device,
            "timesteps": run.total_timesteps,
            "best_rollout_reward": run.best_rollout_reward,
            "final_rollout_reward": run.final_rollout_reward,
            "best_eval_reward": (
                run.best_eval.mean_reward
                if run.best_eval is not None and run.best_eval.mean_reward is not None
                else None
            ),
            "latest_eval_reward": (
                run.latest_eval.mean_reward
                if run.latest_eval is not None and run.latest_eval.mean_reward is not None
                else None
            ),
            "last_plot_manifest": info.last_plot_manifest,
            "last_video_manifest": info.last_video_manifest,
            "eval_log": info.eval_log,
        },
        "learning_signal": {
            "trend": learning.trend,
            "first_reward": learning.first_reward,
            "final_reward": learning.final_reward,
            "best_reward": learning.best_reward,
            "best_step": learning.best_step,
            "delta": learning.delta,
            "late_mean": learning.late_mean,
            "late_std": learning.late_std,
            "best_to_final_drop": learning.best_to_final_drop,
        },
        "findings": [
            {
                "category": item.category,
                "signal": item.signal,
                "interpretation": item.interpretation,
            }
            for item in analysis.findings
        ],
        "suggestions": [
            {
                "priority": item.priority,
                "action": item.action,
                "reason": item.reason,
            }
            for item in analysis.suggestions
        ],
        "diagnostics": [
            {
                "severity": item.severity,
                "area": item.area,
                "issue": item.issue,
                "evidence": item.evidence,
                "recommendation": item.recommendation,
            }
            for item in diagnosis.diagnostics
        ],
        "metric_series": [
            {
                "key": item.key,
                "label": item.label,
                "first": item.first,
                "latest": item.latest,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "mean": item.mean,
                "trend": item.trend,
                "interpretation": item.interpretation,
            }
            for item in diagnosis.metrics.series
        ],
        "metric_notes": [
            {
                "severity": item.severity,
                "metric": item.metric,
                "note": item.note,
            }
            for item in diagnosis.metrics.notes
        ],
    }


def _system_prompt() -> str:
    return (
        "You are the RLX LLM planner for PPO reinforcement-learning experiments. "
        "Use ML/RL knowledge plus the provided local run artifacts to propose compact, "
        "high-signal YAML config mutations. You must not propose code edits. You must not "
        "change forbidden or locked fields. Prefer diverse hypotheses that are comparable "
        "under the fixed environment, timestep budget, and eval settings. Return only JSON "
        "matching the provided schema."
    )


def _proposal_schema(allowed_mutation_keys: tuple[str, ...]) -> dict[str, Any]:
    value_schema = {
        "anyOf": [
            {"type": "number"},
            {"type": "integer"},
            {"type": "boolean"},
            {"type": "string"},
            {"type": "array", "items": {"type": "integer"}},
            {"type": "array", "items": {"type": "number"}},
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["changes", "signal", "rationale", "priority"],
                    "properties": {
                        "changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["key", "value"],
                                "properties": {
                                    "key": {
                                        "type": "string",
                                        "enum": list(allowed_mutation_keys),
                                    },
                                    "value": value_schema,
                                },
                            },
                        },
                        "signal": {"type": "string"},
                        "rationale": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                },
            }
        },
    }


def _extract_openai_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    text = "".join(chunks).strip()
    if not text:
        raise LLMPlannerError("OpenAI planner response did not contain output text.")
    return text


def _clean_text(value: Any, *, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _clean_priority(value: Any) -> str:
    if value in {"high", "medium", "low"}:
        return str(value)
    return "medium"


def _parse_timeout(value: str | None) -> float:
    try:
        parsed = float(value or "60")
    except ValueError:
        return 60.0
    return max(parsed, 1.0)
