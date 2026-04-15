from __future__ import annotations

import copy
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlx.config import ConfigError, load_config
from rlx.core.projects import ProjectLookupError, find_project_root
from rlx.core.runs import RunPreparationError, prepare_run
from rlx.rl import TrainingError, train_ppo


class SweepError(Exception):
    """Raised when an RLCLI sweep cannot be loaded or executed."""


@dataclass(frozen=True)
class SweepVariantResult:
    index: int
    config_path: Path
    mutations: dict[str, Any]
    status: str
    run_id: str | None
    run_dir: Path | None
    error: str | None


@dataclass(frozen=True)
class SweepResult:
    project_root: Path
    bundle_dir: Path
    manifest_path: Path
    sweep_name: str
    source_path: Path
    variants: tuple[SweepVariantResult, ...]


def run_sweep(sweep_config_path: str | Path) -> SweepResult:
    resolved_sweep_path = Path(sweep_config_path).expanduser().resolve()
    payload = _load_sweep_payload(resolved_sweep_path)

    base_config_value = payload.get("base_config")
    if not isinstance(base_config_value, str) or not base_config_value.strip():
        raise SweepError("Sweep config must define a non-empty `base_config` path.")

    base_config_path = (resolved_sweep_path.parent / base_config_value).resolve()
    if not base_config_path.exists() or not base_config_path.is_file():
        raise SweepError(f"Base config not found: {base_config_path}")

    try:
        project_root = find_project_root(base_config_path)
    except ProjectLookupError as exc:
        raise SweepError(str(exc)) from exc

    fixed = _normalize_override_mapping(payload.get("fixed"), field_name="fixed")
    grid = _normalize_grid_mapping(payload.get("grid"))
    if not fixed and not grid:
        raise SweepError("Sweep config must define `fixed`, `grid`, or both.")

    sweep_name = _resolve_sweep_name(payload.get("name"))
    bundle_dir = _next_sweep_bundle_dir(project_root / "analysis" / "sweeps", sweep_name)
    configs_dir = bundle_dir / "configs"
    bundle_dir.mkdir(parents=True, exist_ok=False)
    configs_dir.mkdir()

    source_snapshot = bundle_dir / resolved_sweep_path.name
    source_snapshot.write_text(resolved_sweep_path.read_text(encoding="utf-8"), encoding="utf-8")

    base_raw = _load_yaml_mapping(base_config_path, "base config")

    tags = _normalize_tag_list(payload.get("tags"))
    sweep_tags = ["sweep", sweep_name, *tags]
    variants = []
    for index, mutations in enumerate(_expand_grid(grid), start=1):
        variant_payload = copy.deepcopy(base_raw)
        _apply_overrides(variant_payload, fixed)
        _apply_overrides(variant_payload, mutations)

        variant_config_path = configs_dir / f"variant_{index:03d}.yaml"
        variant_config_path.write_text(
            yaml.safe_dump(variant_payload, sort_keys=False),
            encoding="utf-8",
        )

        try:
            config = load_config(variant_config_path)
            run = prepare_run(config, variant_config_path)
            result = train_ppo(
                config,
                run,
                lineage_metadata={
                    "sweep_name": sweep_name,
                    "sweep_variant_index": index,
                    "sweep_source": str(_relative_to(project_root, source_snapshot)),
                    "sweep_mutations": mutations,
                    "tags": sweep_tags,
                },
            )
            variants.append(
                SweepVariantResult(
                    index=index,
                    config_path=variant_config_path,
                    mutations=mutations,
                    status="completed",
                    run_id=result.run_dir.name,
                    run_dir=result.run_dir,
                    error=None,
                )
            )
        except (ConfigError, RunPreparationError, TrainingError) as exc:
            variants.append(
                SweepVariantResult(
                    index=index,
                    config_path=variant_config_path,
                    mutations=mutations,
                    status="failed",
                    run_id=None,
                    run_dir=None,
                    error=str(exc),
                )
            )

    manifest_path = bundle_dir / "manifest.json"
    manifest = {
        "kind": "sweep_bundle",
        "name": sweep_name,
        "source": str(_relative_to(project_root, source_snapshot)),
        "base_config": str(_relative_to(project_root, base_config_path)),
        "fixed": fixed,
        "grid": grid,
        "tags": sweep_tags,
        "variants": [
            {
                "index": variant.index,
                "config": str(_relative_to(project_root, variant.config_path)),
                "mutations": variant.mutations,
                "status": variant.status,
                "run_id": variant.run_id,
                "run_dir": (
                    str(_relative_to(project_root, variant.run_dir))
                    if variant.run_dir is not None
                    else None
                ),
                "error": variant.error,
            }
            for variant in variants
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return SweepResult(
        project_root=project_root,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        sweep_name=sweep_name,
        source_path=source_snapshot,
        variants=tuple(variants),
    )


def _load_sweep_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SweepError(f"Sweep config not found: {path}")
    if not path.is_file():
        raise SweepError(f"Sweep config path is not a file: {path}")
    return _load_yaml_mapping(path, "sweep config")


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SweepError(f"Failed to parse {label} {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SweepError(f"{label.capitalize()} must be a top-level mapping: {path}")
    return raw


def _normalize_override_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SweepError(f"`{field_name}` must be a mapping of dotted config keys to values.")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise SweepError(f"`{field_name}` keys must be non-empty strings.")
        normalized[key.strip()] = item
    return normalized


def _normalize_grid_mapping(value: Any) -> dict[str, list[Any]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SweepError("`grid` must be a mapping of dotted config keys to value lists.")

    normalized: dict[str, list[Any]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise SweepError("`grid` keys must be non-empty strings.")
        if not isinstance(item, list) or not item:
            raise SweepError(f"`grid.{key}` must be a non-empty list of candidate values.")
        normalized[key.strip()] = item
    return normalized


def _normalize_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SweepError("`tags` must be a list of strings.")
    tags = []
    for item in value:
        if not isinstance(item, str):
            raise SweepError("`tags` entries must be strings.")
        stripped = item.strip()
        if stripped and stripped not in tags:
            tags.append(stripped)
    return tags


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]

    keys = list(grid.keys())
    variants = []
    for combo in itertools.product(*(grid[key] for key in keys)):
        variants.append({key: value for key, value in zip(keys, combo, strict=True)})
    return variants


def _apply_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        _set_dotted_value(payload, dotted_key, value)


def _set_dotted_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _next_sweep_bundle_dir(root: Path, name: str) -> Path:
    slug = _slugify(name) or "manual_sweep"
    pattern = re.compile(rf"^{re.escape(slug)}_(\d{{3}})$")
    existing = []

    if root.exists():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                existing.append(int(match.group(1)))

    next_index = max(existing, default=0) + 1
    return root / f"{slug}_{next_index:03d}"


def _resolve_sweep_name(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "manual_sweep"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path
