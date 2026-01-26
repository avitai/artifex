"""Base network configuration for DRY (Don't Repeat Yourself).

This module provides BaseNetworkConfig that contains common fields
shared by all network configurations (Generator, Discriminator, Encoder, Decoder, etc.).

This eliminates duplication across network configs.
"""

import dataclasses

from .base_dataclass import BaseConfig
from .validation import (
    validate_activation,
    validate_dropout_rate,
    validate_positive_tuple,
)


@dataclasses.dataclass(frozen=True)
class BaseNetworkConfig(BaseConfig):
    """Base configuration for neural network architectures.

    This class contains common fields shared by all network configs:
    - Generator, Discriminator (GANs)
    - Encoder, Decoder (VAEs)
    - Denoiser (Diffusion models)
    - Flow layers (Normalizing flows)

    By using this base class, we eliminate duplication of validation logic
    and field definitions across all network configs (DRY principle).

    Attributes:
        hidden_dims: Tuple of hidden layer dimensions (immutable!)
        activation: Activation function name
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0.0 = no dropout, 1.0 = full dropout)
    """

    # Required fields (dummy defaults for dataclass field ordering)
    hidden_dims: tuple[int, ...] = ()
    activation: str = ""

    # Optional fields with defaults
    batch_norm: bool = False
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate network configuration.

        Uses DRY validation utilities to avoid duplication.
        This runs automatically after __init__.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation first (validates name field)
        super().__post_init__()

        # Validate required fields (check for dummy defaults)
        if not self.hidden_dims:
            raise ValueError("hidden_dims is required and cannot be empty")
        if not self.activation:
            raise ValueError("activation is required and cannot be empty")

        # Validate network-specific fields using DRY utilities
        validate_positive_tuple(self.hidden_dims, "hidden_dims")
        validate_activation(self.activation)
        validate_dropout_rate(self.dropout_rate)
