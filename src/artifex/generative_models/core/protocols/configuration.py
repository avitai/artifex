"""Configuration protocol definitions for artifex.generative_models.core."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    name: str = Field(..., description="Unique name identifier for this configuration")
    description: str | None = Field(None, description="Human-readable description")

    model_config = {
        "extra": "forbid",  # Prevent extra fields not defined in the model
        "validate_assignment": True,  # Validate values when assigned
    }

    def update(self, config_dict: dict[str, Any]) -> "BaseConfig":
        """Update configuration with values from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BaseConfig":
        """Create a configuration instance from a dictionary."""
        return cls(**config_dict)


class ConfigTemplate:
    """Template for generating configurations with validation."""

    def __init__(
        self,
        name: str,
        base_config: dict[str, Any],
        required_params: list[str],
        optional_params: dict[str, Any] | None = None,
        config_class: type[BaseConfig] | None = None,
    ):
        """
        Initialize configuration template.

        Args:
            name: Template name
            base_config: Base configuration dictionary
            required_params: list of required parameter names
            optional_params: Optional parameters with default values
            config_class: Configuration class for validation
        """
        self.name = name
        self.base_config = base_config
        self.required_params = required_params
        self.optional_params = optional_params or {}
        self.config_class = config_class

    def generate(self, **params) -> dict[str, Any]:
        """
        Generate configuration from template.

        Args:
            **params: Parameters to fill in template

        Returns:
            Generated configuration dictionary

        Raises:
            ValueError: If required parameters are missing
        """
        # Validate required parameters
        missing = set(self.required_params) - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Start with base config
        config = self.base_config.copy()

        # Apply optional defaults for missing params
        for key, default_value in self.optional_params.items():
            if key not in params:
                params[key] = default_value

        # Deep merge parameters into config
        config = self._deep_merge(config, params)

        # Validate if config class provided
        if self.config_class:
            try:
                validated_config = self.config_class(**config)
                return validated_config.to_dict()
            except Exception as e:
                raise ValueError(f"Configuration validation failed: {e}") from e

        return config

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


@dataclass
class BaseModalityConfig:
    """Base configuration class for all modalities.

    This provides common configuration parameters that most modalities need.
    Specific modalities should inherit from this and add their own parameters.
    """

    # Core configuration
    name: str = "base"

    # Data preprocessing
    normalize: bool = True
    augmentation: bool = False

    # Training parameters
    batch_size: int = 32

    # Validation parameters
    validate_config: bool = True

    def __post_init__(self):
        """Base validation that can be extended by subclasses."""
        if self.validate_config:
            self.validate()

    def validate(self):
        """Validate configuration parameters.

        Subclasses should override this to add their own validation.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
