"""Configuration loading utilities.

Configs are plain YAML files under ``configs/`` and are loaded into
nested dictionaries. Environment variables referenced as ``${VAR_NAME}``
inside string values are expanded automatically (useful for credentials
that must never be hard-coded into YAML files).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: "re.Match[str]") -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand ``${ENV_VAR}`` placeholders."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env_vars(raw)


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return project_root() / "configs"


def data_dir() -> Path:
    return project_root() / "data"
