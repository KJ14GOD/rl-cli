import json
from pathlib import Path
from typing import Any


class JsonlMetricsWriter:
    """Append scalar training records to a JSONL metrics file."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
