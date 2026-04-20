from __future__ import annotations

import os
from pathlib import Path


def get_env_value(
    name: str,
    *,
    project_root: Path,
    default: str | None = None,
) -> str | None:
    """Read an environment value, preferring project-local .env over the shell."""

    dotenv = _read_dotenv(project_root / ".env")
    value = dotenv.get(name)
    if value:
        return value
    shell_value = os.environ.get(name)
    if shell_value:
        return shell_value
    return default


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _clean_value(value.strip())
    return values


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
