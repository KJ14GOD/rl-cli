from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rlx.console import OutputStyle


@dataclass(frozen=True)
class AppSettings:
    style: OutputStyle = OutputStyle.NEON


def load_settings() -> AppSettings:
    """Load persisted RLCLI settings, falling back to defaults when absent or invalid."""

    path = get_settings_path()
    if not path.exists():
        return AppSettings()

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return AppSettings()

    raw_style = data.get("style")
    if not isinstance(raw_style, str):
        return AppSettings()

    try:
        style = OutputStyle(raw_style.lower())
    except ValueError:
        return AppSettings()
    return AppSettings(style=style)


def save_settings(settings: AppSettings) -> Path:
    """Persist RLCLI settings to the user config path."""

    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'style = "{settings.style.value}"\n', encoding="utf-8")
    return path


def get_settings_path() -> Path:
    """Return the settings file path, honoring overrides when present."""

    root_override = os.environ.get("RLX_CONFIG_HOME")
    if root_override:
        return Path(root_override).expanduser() / "rlx" / "config.toml"

    xdg_root = os.environ.get("XDG_CONFIG_HOME")
    if xdg_root:
        return Path(xdg_root).expanduser() / "rlx" / "config.toml"

    return Path.home() / ".config" / "rlx" / "config.toml"
