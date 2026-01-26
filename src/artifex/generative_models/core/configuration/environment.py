"""Environment-specific configuration utilities for artifex.generative_models.core."""

import os
from pathlib import Path
from typing import Any

from artifex.configs.utils.config_loader import load_yaml_config


def get_env_name() -> str:
    """Get the current environment name from ENV_NAME environment variable."""
    return os.environ.get("ENV_NAME", "local")


def detect_environment() -> str:
    """
    Detect the current environment from ARTIFEX_ENV environment variable.

    Returns:
        Environment name (development, production, staging, etc.)
        Default is 'development' if not specified.
    """
    return os.environ.get("ARTIFEX_ENV", "development")


def get_env_config_path(config_dir: str | Path, env_name: str | None = None) -> Path | None:
    """
    Get the path to an environment-specific configuration file.

    Args:
        config_dir: Base directory for configuration files
        env_name: Environment name (default: from ENV_NAME environment variable)

    Returns:
        Path to environment configuration file or None if not found
    """
    if env_name is None:
        env_name = get_env_name()

    # Ensure config_dir is a Path object
    config_dir = Path(config_dir) if isinstance(config_dir, str) else config_dir

    # Look for environment config in the env subdirectory
    env_config_path = config_dir / "env" / f"{env_name}.yaml"
    if env_config_path.exists():
        return env_config_path

    return None


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively update a dictionary with values from another dictionary.

    Args:
        base: Base dictionary to update
        update: dictionary with values to apply

    Returns:
        Updated dictionary
    """
    result = base.copy()

    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively update nested dictionaries
            result[key] = deep_update(result[key], value)
        else:
            # Replace or add value
            result[key] = value

    return result


def load_env_config(config_dir: str | Path, env_name: str | None = None) -> dict[str, Any]:
    """
    Load environment-specific configuration.

    Args:
        config_dir: Base directory for configuration files
        env_name: Environment name (default: from ENV_NAME environment variable)

    Returns:
        Environment configuration dictionary
    """
    env_config_path = get_env_config_path(config_dir, env_name)
    if env_config_path and env_config_path.exists():
        return load_yaml_config(env_config_path)

    return {}


def apply_env_overrides(
    config_dict: dict[str, Any],
    config_dir: str | Path,
    env_name: str | None = None,
) -> dict[str, Any]:
    """
    Apply environment-specific overrides to a configuration dictionary.

    Args:
        config_dict: Configuration dictionary to modify
        config_dir: Base directory for configuration files
        env_name: Environment name (default: from ARTIFEX_ENV environment variable)

    Returns:
        Updated configuration dictionary
    """
    if env_name is None:
        env_name = detect_environment()

    env_config = load_env_config(config_dir, env_name)
    return deep_update(config_dict, env_config)
