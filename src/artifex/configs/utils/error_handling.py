"""Utilities for handling configuration errors with clear context and user-friendly messages."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from dacite.exceptions import DaciteError


_CONFIG_VALIDATION_ERRORS = (DaciteError, KeyError, TypeError, ValueError)


class ConfigError(Exception):
    """Base exception for configuration errors with enhanced context."""

    def __init__(
        self,
        message: str,
        config_path: str | Path | None = None,
        field: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize a configuration error with rich context.

        Args:
            message: The error message
            config_path: Path to the configuration file
            field: The field name causing the error
            context: Additional context information
        """
        self.message = message
        self.config_path = str(config_path) if config_path else None
        self.field = field
        self.context = context or {}

        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Build a detailed error message with context."""
        parts: list[str] = [self.message]

        if self.config_path:
            file_name = Path(self.config_path).name
            parts.append(f"File: {file_name}")

        if self.field:
            parts.append(f"Field: {self.field}")

        for key, value in self.context.items():
            parts.append(f"{key}: {value}")

        return "\n".join(parts)


class ConfigNotFoundError(ConfigError):
    """Error raised when a configuration file cannot be found."""

    def __init__(self, config_name: str, search_paths: list[str] | None = None):
        """Initialize a configuration not found error.

        Args:
            config_name: Name or path of the configuration that was not found
            search_paths: Paths that were searched
        """
        context: dict[str, Any] = {}
        if search_paths:
            context["Search paths"] = "\n  - " + "\n  - ".join(search_paths)

        super().__init__(
            f"Configuration '{config_name}' could not be found",
            field=None,
            context=context,
        )


class ConfigLoadError(ConfigError):
    """Error raised when a configuration file cannot be loaded."""

    def __init__(
        self,
        config_path: str | Path,
        original_error: Exception,
    ):
        """Initialize a configuration load error.

        Args:
            config_path: Path to the configuration file
            original_error: The original exception that was raised
        """
        error_type = type(original_error).__name__
        super().__init__(
            f"Failed to load configuration: {original_error!s}",
            config_path=config_path,
            context={"Error type": error_type},
        )


class ConfigValidationError(ConfigError):
    """Error raised when a configuration fails validation."""

    def __init__(
        self,
        config_path: str | Path,
        validation_error: Exception,
    ):
        """Initialize a configuration validation error.

        Args:
            config_path: Path to the configuration file
            validation_error: The validation error (ValueError or similar)
        """
        super().__init__(
            f"Configuration validation failed: {validation_error!s}",
            config_path=config_path,
            context={"Error type": type(validation_error).__name__},
        )


def safe_load_config(
    load_func: Callable[[str | Path], Any],
    config_path: str | Path,
) -> Any:
    """Safely load a configuration with error handling.

    Args:
        load_func: Function to load the configuration
        config_path: Path to the configuration file

    Returns:
        The loaded configuration

    Raises:
        ConfigError: If the configuration cannot be loaded
    """
    try:
        return load_func(config_path)
    except ConfigError:
        raise
    except FileNotFoundError as e:
        not_found_error = ConfigNotFoundError(str(config_path), None)
        raise not_found_error from e
    except _CONFIG_VALIDATION_ERRORS as e:
        validation_error = ConfigValidationError(config_path, e)
        raise validation_error from e
    except Exception as e:  # noqa: BLE001 — boundary handler wrapping arbitrary load_func errors
        load_error = ConfigLoadError(config_path, e)
        raise load_error from e
