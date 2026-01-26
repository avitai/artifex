"""ResNet block implementations for JAX/Flax models.

This module provides ResNet and Bottleneck block implementations using the latest Flax NNX API.
"""

from typing import Callable, Sequence

import jax
from flax import nnx

from artifex.generative_models.core.layers._utils import (
    apply_norm,
    create_norm_layer,
    normalize_size_param,
)


class ResNetBlock(nnx.Module):
    """Basic ResNet block with optional normalization.

    This implements the standard ResNet block with two convolutions and a skip connection.
    Supports different normalization types and configurable activation functions.

    Attributes:
        in_features: Number of input features.
        features: Number of output features.
        kernel_size_tuple: Normalized kernel size as a 2-tuple.
        strides_tuple: Normalized stride as a 2-tuple.
        padding: Padding type ('SAME' or 'VALID').
        use_bias: Whether to use bias in convolutions.
        use_norm: Whether to use normalization.
        norm_type: Type of normalization ('batch', 'layer', or 'group').
        activation: Activation function.
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        group_norm_num_groups: Number of groups for GroupNormalization.
    """

    def __init__(
        self,
        in_features: int,
        features: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        use_norm: bool = True,
        norm_type: str = "batch",  # 'batch', 'layer', 'group'
        activation: Callable[[jax.Array], jax.Array] = nnx.relu,
        kernel_init: Callable = nnx.initializers.glorot_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        group_norm_num_groups: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the ResNet block.

        Args:
            in_features: Number of input features.
            features: Number of output features.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution.
            padding: Padding type ('SAME' or 'VALID').
            use_bias: Whether to use bias in convolutions.
            use_norm: Whether to use normalization.
            norm_type: Type of normalization ('batch', 'layer', or 'group').
            activation: Activation function.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            group_norm_num_groups: Number of groups for GroupNormalization.
            rngs: Random number generators.
        """
        super().__init__()

        # Validate inputs
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}")
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if norm_type not in ["batch", "layer", "group"]:
            raise ValueError(
                f"norm_type must be one of ['batch', 'layer', 'group'], got {norm_type}"
            )
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

        self.in_features = in_features
        self.features = features
        self.padding = padding
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.norm_type = norm_type
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.group_norm_num_groups = group_norm_num_groups

        # Process kernel_size and stride
        self.kernel_size_tuple = normalize_size_param(kernel_size, 2, "kernel_size")
        self.strides_tuple = normalize_size_param(stride, 2, "stride")

        # Define convolution layers
        # First conv: in_features -> features, stride (1,1)
        self.conv1 = nnx.Conv(
            in_features=self.in_features,
            out_features=self.features,
            kernel_size=self.kernel_size_tuple,
            strides=(1, 1),  # First conv always uses stride 1
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        # Second conv: features -> features, with specified stride
        self.conv2 = nnx.Conv(
            in_features=self.features,
            out_features=self.features,
            kernel_size=self.kernel_size_tuple,
            strides=self.strides_tuple,  # Apply stride in the second conv
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        # Skip connection: projects input if dimensions or stride mismatch
        # Only create skip_projection if needed (don't initialize to None)
        needs_projection = any(s > 1 for s in self.strides_tuple) or (
            self.in_features != self.features
        )
        if needs_projection:
            self.skip_projection = nnx.Conv(
                in_features=self.in_features,
                out_features=self.features,
                kernel_size=(1, 1),  # 1x1 convolution for projection
                strides=self.strides_tuple,  # Apply the same stride
                padding=self.padding,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                rngs=rngs,
            )

        # Create normalization layers
        if self.use_norm:
            self.norm1 = create_norm_layer(
                self.norm_type,
                self.features,
                group_norm_num_groups=self.group_norm_num_groups,
                rngs=rngs,
            )
            self.norm2 = create_norm_layer(
                self.norm_type,
                self.features,
                group_norm_num_groups=self.group_norm_num_groups,
                rngs=rngs,
            )
            self.norm_skip = (
                create_norm_layer(
                    self.norm_type,
                    self.features,
                    group_norm_num_groups=self.group_norm_num_groups,
                    rngs=rngs,
                )
                if hasattr(self, "skip_projection")
                else None
            )
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm_skip = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply ResNet block to the input.

        Args:
            x: Input tensor of shape [batch, height, width, channels].
            deterministic: Whether to use deterministic behavior for
                normalization (e.g., use running averages in BatchNorm).
            rngs: Optional RNGs for stateful operations.

        Returns:
            Output tensor of the same spatial dimensions but possibly different channels.
        """
        identity = x

        # Main path
        # First conv block
        out = self.conv1(x)
        out = apply_norm(out, self.norm1, self.norm_type, deterministic=deterministic)
        out = self.activation(out)

        # Second conv block
        out = self.conv2(out)
        out = apply_norm(out, self.norm2, self.norm_type, deterministic=deterministic)
        # Note: No activation here - it's applied after residual addition

        # Skip connection
        if hasattr(self, "skip_projection"):
            identity = self.skip_projection(identity)
            identity = apply_norm(
                identity, self.norm_skip, self.norm_type, deterministic=deterministic
            )

        # Residual connection and final activation
        out = out + identity
        out = self.activation(out)

        return out


class BottleneckBlock(nnx.Module):
    """Bottleneck ResNet block with 1x1 -> 3x3 -> 1x1 convolutions.

    This implements the ResNet bottleneck block which reduces computation by first
    reducing the number of channels, applying 3x3 convolution at reduced dimensions,
    then expanding back to the output dimensions.

    The ``out_features`` argument corresponds to the number of output features
    of the block (after the final 1x1 expansion). The bottleneck
    channel count is ``out_features // bottleneck_expansion_ratio``.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features for the block.
        bottleneck_channels: Number of channels in the bottleneck.
        bottleneck_expansion_ratio: Factor by which out_features is divided.
        kernel_size_tuple: Normalized kernel size as a 2-tuple.
        strides_tuple: Normalized stride as a 2-tuple.
        padding: Padding type ('SAME' or 'VALID').
        use_bias: Whether to use bias in convolutions.
        use_norm: Whether to use normalization.
        norm_type: Type of normalization ('batch', 'layer', or 'group').
        activation: Activation function.
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        group_norm_num_groups: Number of groups for GroupNormalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bottleneck_expansion_ratio: float = 4.0,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        use_norm: bool = True,
        norm_type: str = "batch",
        activation: Callable[[jax.Array], jax.Array] = nnx.relu,
        kernel_init: Callable = nnx.initializers.glorot_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        group_norm_num_groups: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the bottleneck block.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bottleneck_expansion_ratio: Expansion ratio for bottleneck.
            kernel_size: Size of the middle convolution kernel.
            stride: Stride for the middle convolution.
            padding: Padding type ('SAME' or 'VALID').
            use_bias: Whether to use bias in convolutions.
            use_norm: Whether to use normalization.
            norm_type: Type of normalization ('batch', 'layer', or 'group').
            activation: Activation function.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            group_norm_num_groups: Number of groups for GroupNormalization.
            rngs: Random number generators.
        """
        super().__init__()

        # Validate inputs
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if bottleneck_expansion_ratio <= 0:
            raise ValueError(
                f"bottleneck_expansion_ratio must be positive, got {bottleneck_expansion_ratio}"
            )
        if norm_type not in ["batch", "layer", "group"]:
            raise ValueError(
                f"norm_type must be one of ['batch', 'layer', 'group'], got {norm_type}"
            )
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck_channels = max(1, int(self.out_features // bottleneck_expansion_ratio))
        self.padding = padding
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.norm_type = norm_type
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.group_norm_num_groups = group_norm_num_groups

        # Process parameters
        self.kernel_size_tuple = normalize_size_param(kernel_size, 2, "kernel_size")
        self.strides_tuple = normalize_size_param(stride, 2, "stride")

        # Validate group norm divisibility before creating norm layers
        if self.use_norm and self.norm_type == "group":
            if self.bottleneck_channels % self.group_norm_num_groups != 0:
                raise ValueError(
                    f"Bottleneck channels ({self.bottleneck_channels}) must be divisible by "
                    f"group_norm_num_groups ({self.group_norm_num_groups}) for GroupNorm."
                )
            if self.out_features % self.group_norm_num_groups != 0:
                raise ValueError(
                    f"Output features ({self.out_features}) must be divisible by "
                    f"group_norm_num_groups ({self.group_norm_num_groups}) for GroupNorm."
                )

        # Bottleneck layers
        # Conv 1: 1x1, in_features -> bottleneck_channels, stride (1,1)
        self.conv1 = nnx.Conv(
            in_features=self.in_features,
            out_features=self.bottleneck_channels,
            kernel_size=(1, 1),
            strides=(1, 1),  # No stride in first bottleneck conv
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        # Conv 2: 3x3 (or kernel_size), bottleneck_channels -> bottleneck_channels, applies stride
        self.conv2 = nnx.Conv(
            in_features=self.bottleneck_channels,
            out_features=self.bottleneck_channels,
            kernel_size=self.kernel_size_tuple,
            strides=self.strides_tuple,  # Apply stride to middle conv
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        # Conv 3: 1x1, bottleneck_channels -> out_features, stride (1,1)
        self.conv3 = nnx.Conv(
            in_features=self.bottleneck_channels,
            out_features=self.out_features,
            kernel_size=(1, 1),
            strides=(1, 1),  # No stride in final bottleneck conv
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        # Skip connection - only create if needed (don't initialize to None)
        needs_projection = any(s > 1 for s in self.strides_tuple) or (
            self.in_features != self.out_features
        )
        if needs_projection:
            self.skip_projection = nnx.Conv(
                in_features=self.in_features,
                out_features=self.out_features,
                kernel_size=(1, 1),
                strides=self.strides_tuple,
                padding=self.padding,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                rngs=rngs,
            )

        # Create normalization layers
        if self.use_norm:
            self.norm1 = create_norm_layer(
                self.norm_type,
                self.bottleneck_channels,
                group_norm_num_groups=self.group_norm_num_groups,
                rngs=rngs,
            )
            self.norm2 = create_norm_layer(
                self.norm_type,
                self.bottleneck_channels,
                group_norm_num_groups=self.group_norm_num_groups,
                rngs=rngs,
            )
            self.norm3 = create_norm_layer(
                self.norm_type,
                self.out_features,
                group_norm_num_groups=self.group_norm_num_groups,
                rngs=rngs,
            )
            self.norm_skip = (
                create_norm_layer(
                    self.norm_type,
                    self.out_features,
                    group_norm_num_groups=self.group_norm_num_groups,
                    rngs=rngs,
                )
                if hasattr(self, "skip_projection")
                else None
            )
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None
            self.norm_skip = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply bottleneck block to the input.

        Args:
            x: Input tensor of shape [batch, height, width, channels].
            deterministic: Whether to use deterministic behavior for
                normalization (e.g., use running averages in BatchNorm).
            rngs: Optional RNGs for stateful operations.

        Returns:
            Output tensor of the same spatial dimensions but possibly different channels.
        """
        identity = x

        # Main path
        # First 1x1 conv (reduce dimensions)
        out = self.conv1(x)
        out = apply_norm(out, self.norm1, self.norm_type, deterministic=deterministic)
        out = self.activation(out)

        # 3x3 conv (at reduced dimensions, applies stride)
        out = self.conv2(out)
        out = apply_norm(out, self.norm2, self.norm_type, deterministic=deterministic)
        out = self.activation(out)

        # Last 1x1 conv (restore dimensions)
        out = self.conv3(out)
        out = apply_norm(out, self.norm3, self.norm_type, deterministic=deterministic)
        # Note: No activation here - it's applied after residual addition

        # Skip connection
        if hasattr(self, "skip_projection"):
            identity = self.skip_projection(identity)
            identity = apply_norm(
                identity, self.norm_skip, self.norm_type, deterministic=deterministic
            )

        # Residual connection and final activation
        out = out + identity
        out = self.activation(out)

        return out


def create_resnet_block(
    block_type: str,
    in_features: int,
    out_features: int,
    stride: int | Sequence[int] = 1,
    norm_type: str = "batch",
    activation: Callable[[jax.Array], jax.Array] = nnx.relu,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> ResNetBlock | BottleneckBlock:
    """Factory function to create ResNet blocks.

    Args:
        block_type: Type of block ('basic' or 'bottleneck').
        in_features: Number of input features.
        out_features: Number of output features.
        stride: Stride for the block.
        norm_type: Type of normalization.
        activation: Activation function.
        rngs: Random number generators.
        **kwargs: Additional arguments for the block.

    Returns:
        ResNet block instance.

    Raises:
        ValueError: If block_type is not supported.
    """
    if block_type == "basic":
        return ResNetBlock(
            in_features=in_features,
            features=out_features,
            stride=stride,
            norm_type=norm_type,
            activation=activation,
            rngs=rngs,
            **kwargs,
        )
    elif block_type == "bottleneck":
        return BottleneckBlock(
            in_features=in_features,
            out_features=out_features,
            stride=stride,
            norm_type=norm_type,
            activation=activation,
            rngs=rngs,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported block_type: {block_type}. Must be 'basic' or 'bottleneck'.")


def create_resnet_stage(
    block_type: str,
    num_blocks: int,
    in_features: int,
    out_features: int,
    stride: int | Sequence[int] = 1,
    norm_type: str = "batch",
    activation: Callable[[jax.Array], jax.Array] = nnx.relu,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> list[ResNetBlock | BottleneckBlock]:
    """Create a stage of ResNet blocks.

    Args:
        block_type: Type of block ('basic' or 'bottleneck').
        num_blocks: Number of blocks in the stage.
        in_features: Number of input features for the first block.
        out_features: Number of output features for all blocks.
        stride: Stride for the first block (others use stride=1).
        norm_type: Type of normalization.
        activation: Activation function.
        rngs: Random number generators.
        **kwargs: Additional arguments for the blocks.

    Returns:
        List of ResNet blocks forming a stage.
    """
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")

    blocks = []

    # First block may have stride > 1 and dimension change
    blocks.append(
        create_resnet_block(
            block_type=block_type,
            in_features=in_features,
            out_features=out_features,
            stride=stride,
            norm_type=norm_type,
            activation=activation,
            rngs=rngs,
            **kwargs,
        )
    )

    # Remaining blocks have stride=1 and same dimensions
    for _ in range(num_blocks - 1):
        blocks.append(
            create_resnet_block(
                block_type=block_type,
                in_features=out_features,
                out_features=out_features,
                stride=1,
                norm_type=norm_type,
                activation=activation,
                rngs=rngs,
                **kwargs,
            )
        )

    return blocks
