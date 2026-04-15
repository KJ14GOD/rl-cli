from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rlx.core.compare import CompareError, resolve_run_ref
from rlx.paths import METADATA_NAME


class TagError(Exception):
    """Raised when tags cannot be applied to a run."""


@dataclass(frozen=True)
class TagResult:
    run_dir: Path
    run_id: str
    added_tags: tuple[str, ...]
    all_tags: tuple[str, ...]


def add_tags(run_ref: str, tags: list[str], cwd: Path | None = None) -> TagResult:
    normalized_tags = _normalize_tags(tags)
    if not normalized_tags:
        raise TagError("Pass at least one non-empty tag.")

    working_dir = (cwd or Path.cwd()).resolve()
    try:
        run_dir = resolve_run_ref(run_ref, working_dir)
    except CompareError as exc:
        raise TagError(str(exc)) from exc

    metadata_path = run_dir / METADATA_NAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TagError(f"Could not read run metadata: {metadata_path}") from exc
    except json.JSONDecodeError as exc:
        raise TagError(f"Run metadata is not valid JSON: {metadata_path}") from exc

    existing = []
    raw_existing = metadata.get("tags")
    if isinstance(raw_existing, list):
        for item in raw_existing:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped and stripped not in existing:
                    existing.append(stripped)

    added = []
    for tag in normalized_tags:
        if tag not in existing:
            existing.append(tag)
            added.append(tag)

    metadata["tags"] = existing
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return TagResult(
        run_dir=run_dir,
        run_id=str(metadata.get("run_id", run_dir.name)),
        added_tags=tuple(added),
        all_tags=tuple(existing),
    )


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized = []
    for tag in tags:
        stripped = tag.strip()
        if stripped and stripped not in normalized:
            normalized.append(stripped)
    return normalized
