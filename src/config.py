from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a YAML config is missing or malformed."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config_file_not_found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"invalid_config_schema: root must be a mapping in {config_path}")
    return data


def require_keys(config: dict[str, Any], keys: list[str], source: str = "config") -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"invalid_config_schema: missing key(s) in {source}: {joined}")


def config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def copy_config(config_path: str | Path, output_dir: str | Path, name: str = "training_config.yaml") -> Path:
    destination = ensure_dir(output_dir) / name
    shutil.copyfile(Path(config_path), destination)
    return destination


def get_path(config: dict[str, Any], section: str, key: str, default: str | None = None) -> Path:
    value = config.get(section, {}).get(key, default)
    if value is None:
        raise ConfigError(f"invalid_config_schema: missing path {section}.{key}")
    return Path(value)
