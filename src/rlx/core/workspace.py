from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class WorkspaceContextError(Exception):
    """Raised when research workspace context cannot be resolved safely."""


MAX_PROGRAM_CHARS = 12_000
MAX_FILE_CHARS = 8_000


@dataclass(frozen=True)
class WorkspaceFileContext:
    path: Path
    sha256: str
    chars: int
    excerpt: str


@dataclass(frozen=True)
class ResearchWorkspaceContext:
    program: WorkspaceFileContext | None
    editable_files: tuple[WorkspaceFileContext, ...]
    locked_files: tuple[Path, ...]

    def summary(self, *, project_root: Path) -> dict[str, object]:
        return {
            "program": _file_summary(self.program, project_root=project_root)
            if self.program is not None
            else None,
            "editable_files": [
                _file_summary(item, project_root=project_root)
                for item in self.editable_files
            ],
            "locked_files": [
                str(path.relative_to(project_root))
                for path in self.locked_files
            ],
        }

    def planner_context(self, *, project_root: Path) -> dict[str, object]:
        return {
            "research_program": {
                "path": str(self.program.path.relative_to(project_root)),
                "text": self.program.excerpt,
            }
            if self.program is not None
            else None,
            "workspace_scope": {
                "editable_files": [
                    {
                        "path": str(item.path.relative_to(project_root)),
                        "sha256": item.sha256,
                        "text": item.excerpt,
                    }
                    for item in self.editable_files
                ],
                "locked_files": [
                    str(path.relative_to(project_root))
                    for path in self.locked_files
                ],
            },
        }


def resolve_workspace_context(
    *,
    project_root: Path,
    program_path: str | None,
    editable_files: tuple[str, ...],
    locked_files: tuple[str, ...],
) -> ResearchWorkspaceContext:
    resolved_program = _resolve_optional_file(
        project_root=project_root,
        path_value=(
            program_path
            or ("program.md" if (project_root / "program.md").exists() else None)
        ),
        field_name="program",
        max_chars=MAX_PROGRAM_CHARS,
    )
    resolved_editable = tuple(
        _resolve_required_file(
            project_root=project_root,
            path_value=value,
            field_name="workspace.editable_files",
            max_chars=MAX_FILE_CHARS,
        )
        for value in editable_files
    )
    resolved_locked = tuple(
        _resolve_required_path(
            project_root=project_root,
            path_value=value,
            field_name="workspace.locked_files",
            require_file=True,
        )
        for value in locked_files
    )

    editable_set = {item.path for item in resolved_editable}
    locked_set = set(resolved_locked)
    overlap = sorted(str(path.relative_to(project_root)) for path in editable_set & locked_set)
    if overlap:
        raise WorkspaceContextError(
            "Research workspace cannot mark the same file as editable and locked: "
            + ", ".join(overlap)
        )

    return ResearchWorkspaceContext(
        program=resolved_program,
        editable_files=resolved_editable,
        locked_files=resolved_locked,
    )


def resolve_workspace_context_from_protocol(
    *,
    project_root: Path,
    protocol: dict[str, object],
) -> ResearchWorkspaceContext:
    workspace = protocol.get("workspace")
    if not isinstance(workspace, dict):
        return ResearchWorkspaceContext(program=None, editable_files=(), locked_files=())

    program_summary = workspace.get("program")
    program_path = None
    if isinstance(program_summary, dict):
        raw_path = program_summary.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            program_path = raw_path.strip()

    editable_files = _path_tuple_from_workspace(workspace.get("editable_files"))
    locked_files = _path_tuple_from_workspace(workspace.get("locked_files"), item_key=None)
    return resolve_workspace_context(
        project_root=project_root,
        program_path=program_path,
        editable_files=editable_files,
        locked_files=locked_files,
    )


def _path_tuple_from_workspace(value: object, *, item_key: str | None = "path") -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if item_key is None:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            continue
        if isinstance(item, dict):
            raw_path = item.get(item_key)
            if isinstance(raw_path, str) and raw_path.strip():
                result.append(raw_path.strip())
    return tuple(result)


def _resolve_optional_file(
    *,
    project_root: Path,
    path_value: str | None,
    field_name: str,
    max_chars: int,
) -> WorkspaceFileContext | None:
    if path_value is None:
        return None
    path = _resolve_required_path(
        project_root=project_root,
        path_value=path_value,
        field_name=field_name,
        require_file=True,
    )
    return _read_file_context(path, max_chars=max_chars)


def _resolve_required_file(
    *,
    project_root: Path,
    path_value: str,
    field_name: str,
    max_chars: int,
) -> WorkspaceFileContext:
    path = _resolve_required_path(
        project_root=project_root,
        path_value=path_value,
        field_name=field_name,
        require_file=True,
    )
    return _read_file_context(path, max_chars=max_chars)


def _resolve_required_path(
    *,
    project_root: Path,
    path_value: str,
    field_name: str,
    require_file: bool,
) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise WorkspaceContextError(
            f"Research {field_name} must stay inside the project root: {path_value}"
        )
    if require_file and not path.is_file():
        raise WorkspaceContextError(
            f"Research {field_name} entry is not a file: {path_value}"
        )
    return path


def _read_file_context(path: Path, *, max_chars: int) -> WorkspaceFileContext:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkspaceContextError(f"Could not read workspace file: {path}") from exc

    excerpt = text[:max_chars]
    return WorkspaceFileContext(
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chars=len(text),
        excerpt=excerpt,
    )


def _file_summary(item: WorkspaceFileContext, *, project_root: Path) -> dict[str, object]:
    return {
        "path": str(item.path.relative_to(project_root)),
        "sha256": item.sha256,
        "chars": item.chars,
        "excerpt_chars": len(item.excerpt),
    }
