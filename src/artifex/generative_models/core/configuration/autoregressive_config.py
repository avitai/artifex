"""Autoregressive model configuration classes.

This module provides dataclass-based configuration for autoregressive models
(Transformer, PixelCNN, WaveNet), following the established pattern from
GAN, VAE, Diffusion, and Flow configs.

Autoregressive models generate sequences token by token, where each token
depends only on previously generated tokens (via the chain rule of probability).
"""

import dataclasses
from typing import Any

from .base_dataclass import BaseConfig
from .base_network import BaseNetworkConfig
from .validation import (
    validate_dropout_rate,
    validate_positive_int,
)


# Valid options for validation
VALID_POSITIONAL_ENCODING_TYPES = ("sinusoidal", "learned", "rotary", "none")


@dataclasses.dataclass(frozen=True)
class TransformerNetworkConfig(BaseNetworkConfig):
    """Configuration for transformer network architecture.

    This configures the transformer blocks used in autoregressive models.
    It extends BaseNetworkConfig to inherit hidden_dims, activation, etc.

    Attributes:
        embed_dim: Embedding dimension for token representations
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        positional_encoding: Type of positional encoding
        use_bias: Whether to use bias in linear layers
        attention_dropout_rate: Dropout rate for attention weights
        layer_norm_eps: Epsilon for layer normalization
    """

    # Transformer-specific fields
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    positional_encoding: str = "sinusoidal"
    use_bias: bool = True
    attention_dropout_rate: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate transformer network configuration."""
        super().__post_init__()

        # Validate embed_dim
        validate_positive_int(self.embed_dim, "embed_dim")

        # Validate num_heads
        validate_positive_int(self.num_heads, "num_heads")

        # Validate embed_dim is divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        # Validate mlp_ratio
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")

        # Validate positional_encoding
        if self.positional_encoding not in VALID_POSITIONAL_ENCODING_TYPES:
            raise ValueError(
                f"positional_encoding must be one of {VALID_POSITIONAL_ENCODING_TYPES}, "
                f"got '{self.positional_encoding}'"
            )

        # Validate attention_dropout_rate
        validate_dropout_rate(self.attention_dropout_rate)

        # Validate layer_norm_eps
        if self.layer_norm_eps <= 0.0:
            raise ValueError(f"layer_norm_eps must be positive, got {self.layer_norm_eps}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerNetworkConfig":
        """Create config from dictionary.

        Handles conversion of lists to tuples for hidden_dims.
        """
        data = data.copy()

        # Convert lists to tuples for immutability
        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class AutoregressiveConfig(BaseConfig):
    """Base configuration for autoregressive models.

    This is the base class for all autoregressive model configurations.
    Autoregressive models generate sequences where each element depends
    only on previously generated elements.

    Attributes:
        vocab_size: Size of the vocabulary/output space
        sequence_length: Maximum sequence length
        dropout_rate: Global dropout rate for regularization
    """

    # Required autoregressive fields
    vocab_size: int = 0
    sequence_length: int = 0

    # Common optional fields
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate autoregressive configuration."""
        super().__post_init__()

        # Validate vocab_size
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        # Validate sequence_length
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")

        # Validate dropout_rate
        validate_dropout_rate(self.dropout_rate)


@dataclasses.dataclass(frozen=True)
class TransformerConfig(AutoregressiveConfig):
    """Configuration for Transformer autoregressive models.

    Transformer models use self-attention mechanisms with causal masking
    to maintain the autoregressive property while enabling parallel training.

    Attributes:
        network: Configuration for the transformer network architecture
        num_layers: Number of transformer layers/blocks
        use_cache: Whether to use KV cache for fast generation
    """

    # Nested network configuration (required)
    network: TransformerNetworkConfig | None = None

    # Transformer-specific fields
    num_layers: int = 6
    use_cache: bool = True

    def __post_init__(self) -> None:
        """Validate Transformer configuration."""
        super().__post_init__()

        # Validate network is provided
        if self.network is None:
            raise ValueError("network is required and cannot be None")

        # Validate network type
        if not isinstance(self.network, TransformerNetworkConfig):
            raise TypeError(
                f"network must be TransformerNetworkConfig, got {type(self.network).__name__}"
            )

        # Validate num_layers
        validate_positive_int(self.num_layers, "num_layers")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        # Convert nested config to dict
        if self.network is not None:
            data["network"] = self.network.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerConfig":
        """Create config from dictionary with nested config handling."""
        data = data.copy()

        # Convert nested dict to TransformerNetworkConfig
        if "network" in data and isinstance(data["network"], dict):
            data["network"] = TransformerNetworkConfig.from_dict(data["network"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class PixelCNNConfig(BaseConfig):
    """Configuration for PixelCNN autoregressive image models.

    PixelCNN generates images pixel by pixel using masked convolutions
    to maintain the autoregressive property in 2D spatial coordinates.

    Note: This inherits from BaseConfig directly (not AutoregressiveConfig)
    because vocab_size and sequence_length are derived from image_shape.

    Attributes:
        image_shape: Shape of images (height, width, channels)
        hidden_channels: Number of hidden channels in convolutions
        num_layers: Number of masked convolution layers
        num_residual_blocks: Number of residual blocks
        kernel_size: Size of convolution kernels
        dropout_rate: Dropout rate for regularization
    """

    # Image-specific fields
    image_shape: tuple[int, int, int] | None = None

    # Architecture fields
    hidden_channels: int = 128
    num_layers: int = 7
    num_residual_blocks: int = 5
    kernel_size: int = 3
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate PixelCNN configuration."""
        super().__post_init__()

        # Validate image_shape is provided
        if self.image_shape is None:
            raise ValueError("image_shape is required for PixelCNNConfig")

        # Validate image_shape dimensions
        if len(self.image_shape) != 3:
            raise ValueError(
                f"image_shape must have 3 dimensions (H, W, C), got {len(self.image_shape)}"
            )

        height, width, channels = self.image_shape
        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError(f"All image_shape dimensions must be positive, got {self.image_shape}")

        # Validate other fields
        validate_positive_int(self.hidden_channels, "hidden_channels")
        validate_positive_int(self.num_layers, "num_layers")

        if self.num_residual_blocks < 0:
            raise ValueError(
                f"num_residual_blocks must be non-negative, got {self.num_residual_blocks}"
            )

        validate_positive_int(self.kernel_size, "kernel_size")

        # Validate dropout_rate
        validate_dropout_rate(self.dropout_rate)

    @property
    def derived_vocab_size(self) -> int:
        """Get vocab size (256 for 8-bit pixel values)."""
        return 256

    @property
    def derived_sequence_length(self) -> int:
        """Get sequence length (H * W * C for images)."""
        if self.image_shape is None:
            return 0
        h, w, c = self.image_shape
        return h * w * c

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PixelCNNConfig":
        """Create config from dictionary."""
        data = data.copy()

        # Convert image_shape list to tuple
        if "image_shape" in data and isinstance(data["image_shape"], list):
            data["image_shape"] = tuple(data["image_shape"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class WaveNetConfig(AutoregressiveConfig):
    """Configuration for WaveNet autoregressive audio models.

    WaveNet uses dilated causal convolutions to capture long-range
    dependencies while maintaining the autoregressive property.

    Attributes:
        residual_channels: Number of residual channels
        skip_channels: Number of skip connection channels
        num_blocks: Number of dilation blocks (each block cycles through dilations)
        layers_per_block: Number of layers per dilation block
        kernel_size: Convolution kernel size
        dilation_base: Base for exponential dilation (typically 2)
        use_gated_activation: Whether to use gated activation units
    """

    # WaveNet architecture fields
    residual_channels: int = 32
    skip_channels: int = 256
    num_blocks: int = 3
    layers_per_block: int = 10
    kernel_size: int = 2
    dilation_base: int = 2
    use_gated_activation: bool = True

    def __post_init__(self) -> None:
        """Validate WaveNet configuration."""
        super().__post_init__()

        # Validate channel counts
        validate_positive_int(self.residual_channels, "residual_channels")
        validate_positive_int(self.skip_channels, "skip_channels")

        # Validate block/layer counts
        validate_positive_int(self.num_blocks, "num_blocks")
        validate_positive_int(self.layers_per_block, "layers_per_block")

        # Validate kernel_size
        validate_positive_int(self.kernel_size, "kernel_size")

        # Validate dilation_base
        if self.dilation_base < 2:
            raise ValueError(f"dilation_base must be at least 2, got {self.dilation_base}")

    @property
    def receptive_field(self) -> int:
        """Calculate the receptive field of the WaveNet model.

        Returns:
            Size of the receptive field in timesteps
        """
        receptive_field = 1
        for _ in range(self.num_blocks):
            for layer in range(self.layers_per_block):
                dilation = self.dilation_base**layer
                receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaveNetConfig":
        """Create config from dictionary."""
        return cls(**data)
