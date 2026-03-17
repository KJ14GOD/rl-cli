from pathlib import Path

import yaml
from pydantic import ValidationError

from rlx.config.schema import ExperimentConfig


class ConfigError(Exception):
    """Base exception for config loading failures."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when a config path does not exist or is not a file."""


class ConfigParseError(ConfigError):
    """Raised when a config file cannot be parsed as YAML."""


class ConfigValidationError(ConfigError):
    """Raised when parsed config data fails schema validation."""


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config and validate it against the RLCLI schema."""

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise ConfigFileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigFileNotFoundError(f"Config path is not a file: {config_path}")

    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"Failed to parse YAML config {config_path}: {exc}") from exc

    if raw_data is None:
        raise ConfigParseError(f"Config file is empty: {config_path}")
    if not isinstance(raw_data, dict):
        raise ConfigParseError(f"Top-level config must be a mapping: {config_path}")

    try:
        return ExperimentConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error(config_path, exc)) from exc


def _format_validation_error(path: Path, error: ValidationError) -> str:
    lines = [f"Config validation failed for {path}:"]
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"]
        lines.append(f"- {location}: {message}")
    return "\n".join(lines)

