"""Utilities for handling configuration errors with clear context and user-friendly messages."""

import os
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError


class ConfigError(Exception):
    """Base exception for configuration errors with enhanced context."""

    def __init__(
        self,
        message: str,
        config_path: str | Path | None = None,
        field: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize a configuration error with rich context.

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

        # Build full message with context
        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Build a detailed error message with context."""
        parts: list[str] = []

        # Include main error message
        parts.append(self.message)

        # Add file context if available
        if self.config_path:
            file_name = os.path.basename(self.config_path)
            parts.append(f"File: {file_name}")

        # Add field information if available
        if self.field:
            parts.append(f"Field: {self.field}")

        # Add any additional context
        for key, value in self.context.items():
            parts.append(f"{key}: {value}")

        return "\n".join(parts)


class ConfigNotFoundError(ConfigError):
    """Error raised when a configuration file cannot be found."""

    def __init__(self, config_name: str, search_paths: list[str] | None = None):
        """
        Initialize a configuration not found error.

        Args:
            config_name: Name or path of the configuration that was not found
            search_paths: Paths that were searched
        """
        context = {}
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
        """
        Initialize a configuration load error.

        Args:
            config_path: Path to the configuration file
            original_error: The original exception that was raised
        """
        error_type = type(original_error).__name__
        super().__init__(
            f"Failed to load configuration: {str(original_error)}",
            config_path=config_path,
            context={"Error type": error_type},
        )


class ConfigValidationError(ConfigError):
    """Error raised when a configuration fails validation."""

    def __init__(
        self,
        config_path: str | Path,
        validation_error: ValidationError,
    ):
        """
        Initialize a configuration validation error.

        Args:
            config_path: Path to the configuration file
            validation_error: The pydantic validation error
        """
        # Extract error details from ValidationError
        error_details = self._extract_validation_details(validation_error)

        # Get the first field with an error for the main message
        first_field, first_error = (
            error_details[0] if error_details else (None, "Unknown validation error")
        )

        # Format all errors for additional context
        all_errors: dict[str, Any] = {}  # type: ignore
        for field, error in error_details:
            if field in all_errors:
                if isinstance(all_errors[field], list):
                    all_errors[field].append(error)
                else:
                    all_errors[field] = [all_errors[field], error]
            else:
                all_errors[field] = error  # type: ignore

        context = {"All validation errors": self._format_validation_errors(all_errors)}

        super().__init__(
            f"Configuration validation failed: {first_error}",
            config_path=config_path,
            field=first_field,
            context=context,
        )

    def _extract_validation_details(self, error: ValidationError) -> list[tuple[str, str]]:
        """Extract field names and error messages from a ValidationError."""
        details: list[tuple[str, str]] = []

        for error_item in error.errors():
            # Get the field location as a string
            location = ".".join(str(loc) for loc in error_item["loc"])

            # Get the error message
            message = error_item["msg"]

            details.append((location, message))

        return details

    def _format_validation_errors(self, all_errors: dict[str, Any]) -> str:
        """Format validation errors for display."""
        parts: list[str] = []

        for field, errors in sorted(all_errors.items()):
            if isinstance(errors, list):
                parts.append(f"  {field}:")
                for error in errors:
                    parts.append(f"    - {error}")
            else:
                parts.append(f"  {field}: {errors}")

        return "\n" + "\n".join(parts) if parts else "None"


def format_validation_error(
    error: ValidationError,
    config_path: str | Path | None = None,
) -> str:
    """
    Format a validation error into a user-friendly message.

    Args:
        error: The validation error
        config_path: Path to the configuration file

    Returns:
        A formatted error message
    """
    try:
        # Try to create a rich error with our custom class
        custom_error = ConfigValidationError(config_path or "unknown", error)
        return str(custom_error)
    except Exception:
        # Fallback to basic formatting if our custom handling fails
        return f"Validation error in {config_path or 'config'}:\n{error}"


def format_config_error(
    error: Exception,
    config_path: str | Path | None = None,
    config_type: str | None = None,
) -> str:
    """
    Format any configuration-related error into a user-friendly message.

    Args:
        error: The exception
        config_path: Path to the configuration file
        config_type: Type of configuration (models, data, etc.)

    Returns:
        A formatted error message
    """
    if isinstance(error, ValidationError):
        return format_validation_error(error, config_path)
    elif isinstance(error, ConfigError):
        return str(error)
    else:
        # Generic error formatting
        error_type = type(error).__name__
        config_desc = f"{config_type} configuration" if config_type else "configuration"
        file_desc = f" in {os.path.basename(str(config_path))}" if config_path else ""

        return f"{error_type} while processing {config_desc}{file_desc}: {str(error)}"


def safe_load_config(
    load_func: Callable[[str | Path], Any],
    config_path: str | Path,
    config_type: str | None = None,
    error_handler: Callable[[ConfigError], Any] | None = None,
) -> Any:
    """
    Safely load a configuration with comprehensive error handling.

    Args:
        load_func: Function to load the configuration
        config_path: Path to the configuration file
        config_type: Type of configuration (models, data, etc.)
        error_handler: Optional function to handle errors (takes exception as arg)

    Returns:
        The loaded configuration

    Raises:
        ConfigError: If the configuration cannot be loaded
    """
    try:
        return load_func(config_path)
    except FileNotFoundError as e:
        not_found_error = ConfigNotFoundError(str(config_path), None)
        if error_handler:
            return error_handler(not_found_error)
        raise not_found_error from e
    except ValidationError as e:
        validation_error = ConfigValidationError(config_path, e)
        if error_handler:
            return error_handler(validation_error)
        raise validation_error from e
    except Exception as e:
        load_error = ConfigLoadError(config_path, e)
        if error_handler:
            return error_handler(load_error)
        raise load_error from e
