from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
LICENSE = ROOT / "LICENSE"
STARTER_CONFIG = "rlx/templates/project/configs/ppo_cartpole.yaml"
STARTER_CONFIGS = (
    "rlx/templates/project/configs/ppo_acrobot.yaml",
    STARTER_CONFIG,
    "rlx/templates/project/configs/ppo_mountain_car.yaml",
    "rlx/templates/project/configs/ppo_mountain_car_continuous.yaml",
    "rlx/templates/project/configs/ppo_pendulum.yaml",
    "rlx/templates/project/configs/ppo_taxi.yaml",
    "rlx/templates/project/configs/ppo_frozen_lake.yaml",
)
ENTRYPOINT = "rlx = rlx.cli:app"


class ReleaseCheckError(Exception):
    """Raised when release validation fails."""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RLCLI packaging before a PyPI upload.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building and inspect existing dist artifacts instead.",
    )
    args = parser.parse_args()

    try:
        metadata = _check_static_metadata()
        artifacts_dir = ROOT / "dist"
        if not args.skip_build:
            artifacts_dir = _build_to_temp_dir()
        _check_dist_artifacts(artifacts_dir, metadata)
    except ReleaseCheckError as exc:
        print(f"release check failed: {exc}", file=sys.stderr)
        return 1

    print("release check passed")
    return 0


def _check_static_metadata() -> dict[str, str]:
    if not PYPROJECT.exists():
        raise ReleaseCheckError("pyproject.toml is missing.")
    if not README.exists():
        raise ReleaseCheckError("README.md is missing.")
    if not LICENSE.exists():
        raise ReleaseCheckError("LICENSE is missing.")

    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ReleaseCheckError("[project] metadata is missing from pyproject.toml.")

    name = _required_string(project, "name")
    version = _required_string(project, "version")
    description = _required_string(project, "description")
    if not description:
        raise ReleaseCheckError("Project description is empty.")
    if name in {"rlcli", "rl-cli"}:
        raise ReleaseCheckError("Package name is too close to the rejected rlcli namespace.")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.-]+)?", version):
        raise ReleaseCheckError(f"Version does not look release-ready: {version}")

    scripts = project.get("scripts")
    if not isinstance(scripts, dict) or scripts.get("rlx") != "rlx.cli:app":
        raise ReleaseCheckError("[project.scripts] must expose `rlx = rlx.cli:app`.")

    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ReleaseCheckError("Project dependencies are missing.")

    return {"name": name, "version": version}


def _build_to_temp_dir() -> Path:
    if shutil.which(sys.executable) is None:
        raise ReleaseCheckError(f"Python executable not found: {sys.executable}")

    temp_root = Path(tempfile.mkdtemp(prefix="rlx-release-"))
    dist_dir = temp_root / "dist"
    _run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist_dir)])
    return dist_dir


def _check_dist_artifacts(artifacts_dir: Path, metadata: dict[str, str]) -> None:
    if not artifacts_dir.exists():
        raise ReleaseCheckError(f"dist directory not found: {artifacts_dir}")

    wheels = sorted(artifacts_dir.glob("*.whl"))
    sdists = sorted(artifacts_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        raise ReleaseCheckError(f"Expected exactly one wheel, found {len(wheels)}.")
    if len(sdists) != 1:
        raise ReleaseCheckError(f"Expected exactly one sdist, found {len(sdists)}.")

    _run([sys.executable, "-m", "twine", "check", str(wheels[0]), str(sdists[0])])
    _check_wheel(wheels[0], metadata)
    _check_sdist(sdists[0])


def _check_wheel(path: Path, metadata: dict[str, str]) -> None:
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        if any("__pycache__" in name for name in names):
            raise ReleaseCheckError("Wheel contains __pycache__ files.")
        _require_member(names, "rlx/cli.py", path)
        for config_path in STARTER_CONFIGS:
            _require_member(names, config_path, path)

        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        entrypoint_files = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_files) != 1:
            raise ReleaseCheckError("Wheel must contain exactly one METADATA file.")
        if len(entrypoint_files) != 1:
            raise ReleaseCheckError("Wheel must contain exactly one entry_points.txt file.")

        wheel_metadata = wheel.read(metadata_files[0]).decode("utf-8")
        if f"Name: {metadata['name']}" not in wheel_metadata:
            raise ReleaseCheckError("Wheel METADATA has the wrong package name.")
        if f"Version: {metadata['version']}" not in wheel_metadata:
            raise ReleaseCheckError("Wheel METADATA has the wrong package version.")

        entrypoints = wheel.read(entrypoint_files[0]).decode("utf-8")
        if ENTRYPOINT not in entrypoints:
            raise ReleaseCheckError("Wheel entry_points.txt does not expose the rlx command.")


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as sdist:
        names = set(sdist.getnames())
        if not any(name.endswith("/README.md") for name in names):
            raise ReleaseCheckError("sdist is missing README.md.")
        if not any(name.endswith("/pyproject.toml") for name in names):
            raise ReleaseCheckError("sdist is missing pyproject.toml.")
        for config_path in STARTER_CONFIGS:
            if not any(name.endswith(f"/{config_path}") for name in names):
                raise ReleaseCheckError(f"sdist is missing {config_path}.")


def _required_string(project: dict[str, object], key: str) -> str:
    value = project.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseCheckError(f"project.{key} must be a non-empty string.")
    return value.strip()


def _require_member(names: set[str], expected: str, artifact: Path) -> None:
    if expected not in names:
        raise ReleaseCheckError(f"{artifact.name} is missing {expected}.")


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        output = result.stdout.strip()
        raise ReleaseCheckError(f"`{' '.join(command)}` failed\n{output}")


if __name__ == "__main__":
    raise SystemExit(main())
