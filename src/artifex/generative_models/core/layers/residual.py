"""Residual block implementations for generative models.

This module provides flexible residual block implementations that can be used
across different model architectures, supporting both 1D and 2D convolutions.
"""

from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class BaseResidualBlock(nnx.Module, ABC):
    """Base class for residual blocks.

    Defines the common interface for all residual block implementations.
    Subclasses should implement the specific convolution operations.
    """

    def __init__(
        self,
        channels: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize base residual block.

        Args:
            channels: Number of channels
            rngs: Random number generators
        """
        super().__init__()
        self.channels = channels

    @abstractmethod
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply residual block.

        Args:
            x: Input tensor
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor(s) - may be single output or tuple for skip connections
        """
        pass


class Conv1DResidualBlock(BaseResidualBlock):
    """Residual block with 1D convolutions for sequence models like WaveNet.

    Supports skip connections and gated activations commonly used in WaveNet.
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int | None = None,
        kernel_size: int = 2,
        dilation: int = 1,
        use_gated_activation: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize 1D residual block.

        Args:
            residual_channels: Number of residual channels
            skip_channels: Number of skip channels (if None, no skip connection)
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            use_gated_activation: Whether to use gated activation (tanh * sigmoid)
            rngs: Random number generators
        """
        super().__init__(residual_channels, rngs=rngs)

        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_gated_activation = use_gated_activation

        if use_gated_activation:
            # Gated activation: separate convolutions for tanh and sigmoid
            self.tanh_conv = self._create_causal_conv(
                residual_channels, residual_channels, kernel_size, dilation, rngs
            )
            self.sigmoid_conv = self._create_causal_conv(
                residual_channels, residual_channels, kernel_size, dilation, rngs
            )
        else:
            # Standard convolution
            self.conv = self._create_causal_conv(
                residual_channels, residual_channels, kernel_size, dilation, rngs
            )

        # 1x1 convolution for residual connection
        self.residual_conv = nnx.Conv(
            in_features=residual_channels,
            out_features=residual_channels,
            kernel_size=(1,),
            rngs=rngs,
        )

        # 1x1 convolution for skip connection (if needed)
        if skip_channels is not None:
            self.skip_conv = nnx.Conv(
                in_features=residual_channels,
                out_features=skip_channels,
                kernel_size=(1,),
                rngs=rngs,
            )

    def _create_causal_conv(
        self, in_features: int, out_features: int, kernel_size: int, dilation: int, rngs: nnx.Rngs
    ) -> nnx.Conv:
        """Create causal 1D convolution with appropriate padding."""
        return nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size,),
            padding="CAUSAL",
            kernel_dilation=(dilation,),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply 1D residual block.

        Args:
            x: Input tensor [batch, length, channels]
            **kwargs: Additional keyword arguments

        Returns:
            If skip_channels is None: residual output tensor
            If skip_channels is not None: tuple of (residual_output, skip_output)
        """
        if self.use_gated_activation:
            # Gated activation: tanh(conv1(x)) * sigmoid(conv2(x))
            tanh_out = jnp.tanh(self.tanh_conv(x))
            sigmoid_out = jax.nn.sigmoid(self.sigmoid_conv(x))
            activated = tanh_out * sigmoid_out
        else:
            # Standard ReLU activation
            activated = nnx.relu(self.conv(x))

        # Residual connection
        residual_out = self.residual_conv(activated)
        residual_out = x + residual_out

        if self.skip_channels is not None:
            # Skip connection
            skip_out = self.skip_conv(activated)
            return residual_out, skip_out
        else:
            return residual_out


class Conv2DResidualBlock(BaseResidualBlock):
    """Residual block with 2D convolutions for image models like PixelCNN.

    Supports masked convolutions for maintaining autoregressive property in 2D.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int] = 3,
        mask_type: str | None = None,
        activation: Callable = nnx.relu,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize 2D residual block.

        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            mask_type: Type of mask for masked convolutions ("A", "B", or None)
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__(channels, rngs=rngs)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.activation = activation

        if mask_type is not None:
            # Use masked convolutions (implementation would need MaskedConv2D)
            # For now, use regular convolutions as placeholder
            self.conv1 = nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=kernel_size,
                padding="SAME",
                rngs=rngs,
            )

            self.conv2 = nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=kernel_size,
                padding="SAME",
                rngs=rngs,
            )
        else:
            # Standard 2D convolutions
            self.conv1 = nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=kernel_size,
                padding="SAME",
                rngs=rngs,
            )

            self.conv2 = nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=kernel_size,
                padding="SAME",
                rngs=rngs,
            )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Apply 2D residual block.

        Args:
            x: Input tensor [batch, height, width, channels]
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor with residual connection
        """
        residual = x

        # First convolution + activation
        x = self.conv1(x)
        x = self.activation(x)

        # Second convolution
        x = self.conv2(x)

        # Residual connection
        return x + residual


class MaskedConv2DResidualBlock(Conv2DResidualBlock):
    """Residual block with masked 2D convolutions for PixelCNN.

    This requires the MaskedConv2D layer to be properly implemented.
    For now, this is a placeholder that inherits from Conv2DResidualBlock.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int] = 3,
        mask_type: str = "B",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize masked 2D residual block.

        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            mask_type: Type of mask ("A" or "B")
            rngs: Random number generators
        """
        super().__init__(
            channels=channels,
            kernel_size=kernel_size,
            mask_type=mask_type,
            rngs=rngs,
        )


# Factory function for creating appropriate residual blocks
def create_residual_block(block_type: str, **kwargs) -> BaseResidualBlock:
    """Factory function for creating residual blocks.

    Args:
        block_type: Type of block ("conv1d", "conv2d", "masked_conv2d")
        **kwargs: Arguments passed to the block constructor

    Returns:
        Appropriate residual block instance

    Raises:
        ValueError: If block_type is not recognized
    """
    if block_type == "conv1d":
        return Conv1DResidualBlock(**kwargs)
    elif block_type == "conv2d":
        return Conv2DResidualBlock(**kwargs)
    elif block_type == "masked_conv2d":
        return MaskedConv2DResidualBlock(**kwargs)
    else:
        raise ValueError(f"Unknown block type: {block_type}")


# Backward compatibility aliases
ResidualBlock = Conv2DResidualBlock  # Default to 2D for backward compatibility
WaveNetResidualBlock = Conv1DResidualBlock
PixelCNNResidualBlock = MaskedConv2DResidualBlock
