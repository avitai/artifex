"""Modality configuration using frozen dataclasses.

This module provides a frozen dataclass-based configuration for modality-specific
settings, replacing the Pydantic-based ModalityConfig.
"""

import dataclasses
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig


@dataclasses.dataclass(frozen=True)
class ModalityConfig(BaseConfig):
    """Configuration for modality-specific settings.

    This dataclass provides a type-safe, immutable configuration for
    modality-specific settings in generative models. It replaces the
    Pydantic ModalityConfig.

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
        super().__post_init__()

        # Validate modality_name (required field with dummy default)
        if not self.modality_name or not self.modality_name.strip():
            raise ValueError("modality_name cannot be empty")
