"""Modality configuration using frozen dataclasses.

This module provides the typed runtime configuration for modality-specific
settings.
"""

import dataclasses
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig, ConfigDocument


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BaseModalityConfig(ConfigDocument):
    """Shared runtime configuration foundation for modality implementations.

    This lightweight config base backs modality-processing components such as
    audio/image/timeseries/tabular modality adapters. It intentionally stays
    smaller than the higher-level ``ModalityConfig`` runtime document below.
    """

    name: str = "base"
    normalize: bool = True
    augmentation: bool = False
    batch_size: int = 32
    validate_config: bool = True

    def __post_init__(self) -> None:
        """Validate shared modality settings and delegate to subclass hooks."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.validate_config:
            self.validate()

    def validate(self) -> None:
        """Validate modality-specific settings in subclasses."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModalityConfig(BaseConfig):
    """Configuration for modality-specific settings.

    This dataclass provides a type-safe, immutable configuration for
    modality-specific settings in generative models.

    Attributes:
        modality_name: Name of the modality (e.g., "image", "text", "audio")
        supported_models: Tuple of supported model types for this modality
        preprocessing_steps: Tuple of preprocessing step configurations
        default_metrics: Tuple of default evaluation metrics
        extensions: Additional modality-specific configuration options
    """

    # Required field with dummy default (validated in __post_init__)
    modality_name: str = ""

    # Modality configuration
    supported_models: tuple[str, ...] = ()
    preprocessing_steps: tuple[dict[str, Any], ...] = ()
    default_metrics: tuple[str, ...] = ()
    extensions: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super(ModalityConfig, self).__post_init__()

        # Validate modality_name (required field with dummy default)
        if not self.modality_name or not self.modality_name.strip():
            raise ValueError("modality_name cannot be empty")
