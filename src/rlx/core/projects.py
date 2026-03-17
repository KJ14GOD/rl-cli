from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from rlx.paths import PROJECT_DIRS, STARTER_CONFIG


class ProjectInitError(Exception):
    """Raised when a project scaffold cannot be created."""


class ProjectLookupError(Exception):
    """Raised when an RLCLI project root cannot be located."""


@dataclass(frozen=True)
class ProjectInitResult:
    project_root: Path
    created_dirs: tuple[Path, ...]
    starter_config: Path


def init_project(project_root: Path) -> ProjectInitResult:
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

    starter_config = destination / STARTER_CONFIG
    template = files("rlx.templates").joinpath("project", "configs", "ppo_cartpole.yaml")
    starter_config.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")

    return ProjectInitResult(
        project_root=destination,
        created_dirs=tuple(created_dirs),
        starter_config=starter_config,
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
