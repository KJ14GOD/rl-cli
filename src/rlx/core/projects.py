from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path

from rlx.paths import CUSTOM_STARTER_CONFIG, PROJECT_DIRS, STARTER_CONFIG


class ProjectInitError(Exception):
    """Raised when a project scaffold cannot be created."""


class ProjectLookupError(Exception):
    """Raised when an RLCLI project root cannot be located."""


class ProjectTemplate(str, Enum):
    starter = "starter"
    custom = "custom"


@dataclass(frozen=True)
class ProjectInitResult:
    project_root: Path
    created_dirs: tuple[Path, ...]
    starter_config: Path
    created_files: tuple[Path, ...]
    template: ProjectTemplate


def init_project(
    project_root: Path,
    *,
    template: ProjectTemplate = ProjectTemplate.starter,
) -> ProjectInitResult:
    """Create the standard RLCLI project layout in a new directory."""

    destination = project_root.resolve()
    if destination.exists():
        raise ProjectInitError(f"Project directory already exists: {destination}")

    destination.mkdir(parents=True)

    created_dirs = []
    for dirname in PROJECT_DIRS:
        path = destination / dirname
        path.mkdir()
        created_dirs.append(path)

    created_files = _copy_template_tree("project", destination)
    starter_config = destination / STARTER_CONFIG
    if template is ProjectTemplate.custom:
        created_files.extend(_copy_template_tree("project_custom", destination))
        starter_config = destination / CUSTOM_STARTER_CONFIG

    return ProjectInitResult(
        project_root=destination,
        created_dirs=tuple(created_dirs),
        starter_config=starter_config,
        created_files=tuple(created_files),
        template=template,
    )


def find_project_root(path: Path) -> Path:
    """Locate the nearest ancestor that matches the standard RLCLI project layout."""

    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if all((current / dirname).is_dir() for dirname in PROJECT_DIRS):
            return current

    raise ProjectLookupError(
        "No RLCLI project root found. Run this command inside an initialized project or use "
        "`rlx init` first."
    )


def _copy_template_tree(template_name: str, destination: Path) -> list[Path]:
    template_root = files("rlx.templates").joinpath(template_name)
    created_files: list[Path] = []

    def visit(source, relative: Path) -> None:
        if source.name == "__pycache__" or source.name.endswith(".pyc"):
            return
        if relative == Path("__init__.py"):
            return
        if source.is_dir():
            for child in source.iterdir():
                visit(child, relative / child.name)
            return

        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        created_files.append(target)

    visit(template_root, Path())
    return created_files
