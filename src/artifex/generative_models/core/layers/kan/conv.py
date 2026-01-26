"""Convolutional KAN layer.

Applies a KAN layer over spatial patches extracted from 1D, 2D, or 3D
inputs, analogous to how a standard convolution applies a linear
transformation over local patches.

This layer is new to Artifex (not in jaxKAN).
"""

from typing import Callable

import jax
from flax import nnx

from artifex.generative_models.core.layers.kan.spline import EfficientKANLayer


class ConvKANLayer(nnx.Module):
    """Convolutional KAN layer for 1D, 2D, or 3D spatial inputs.

    Extracts local patches using ``jax.lax.conv_general_dilated_patches``,
    then applies an ``EfficientKANLayer`` to each spatial position.

    Supports the same kernel_size / stride / padding conventions as
    standard convolutions.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size (tuple).
        stride: Spatial stride (tuple).
        padding: Padding mode string (e.g. ``"SAME"``, ``"VALID"``).
        spatial_ndim: Number of spatial dimensions (1, 2, or 3).
        kan: The underlying KAN layer applied per position.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        padding: str = "SAME",
        spatial_ndim: int = 2,
        k: int = 3,
        grid_intervals: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        residual: Callable | None = nnx.silu,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize convolutional KAN layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size per spatial dimension (scalar or tuple).
            stride: Stride per spatial dimension (scalar or tuple).
            padding: Padding mode (``"SAME"`` or ``"VALID"``).
            spatial_ndim: Number of spatial dimensions (1, 2, or 3).
            k: Spline order for the underlying KAN layer.
            grid_intervals: Number of grid intervals.
            grid_range: Initial grid range.
            residual: Residual activation, or None.
            add_bias: Whether to include bias.
            rngs: Random number generators.

        Raises:
            ValueError: If spatial_ndim is not 1, 2, or 3.
        """
        super().__init__()

        if spatial_ndim not in (1, 2, 3):
            raise ValueError(f"spatial_ndim must be 1, 2, or 3, got {spatial_ndim}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_ndim = spatial_ndim
        self.padding = padding

        # Normalise kernel_size and stride to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * spatial_ndim
        if isinstance(stride, int):
            stride = (stride,) * spatial_ndim

        self.kernel_size = kernel_size
        self.stride = stride

        # Patch dimension = in_channels * prod(kernel_size)
        patch_dim = in_channels
        for ks in kernel_size:
            patch_dim *= ks

        self.kan = EfficientKANLayer(
            n_in=patch_dim,
            n_out=out_channels,
            k=k,
            grid_intervals=grid_intervals,
            grid_range=grid_range,
            residual=residual,
            add_bias=add_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass: extract patches, apply KAN, reshape.

        Args:
            x: Input tensor. Expected layout:
                - 1D: (batch, length, channels)
                - 2D: (batch, height, width, channels)
                - 3D: (batch, d1, d2, d3, channels)
            deterministic: Unused (KAN forward is deterministic).

        Returns:
            Output tensor with same spatial layout but ``out_channels``.
        """
        batch = x.shape[0]

        # Build dimension_numbers strings for channels-last layout.
        # e.g. 2D -> ("NHWC", "HWIO", "NHWC")
        spatial_chars = "DHW"[-self.spatial_ndim :]
        lhs_spec = "N" + spatial_chars + "C"
        rhs_spec = spatial_chars + "IO"
        out_spec = lhs_spec
        dn = (lhs_spec, rhs_spec, out_spec)

        # Extract patches: shape (batch, *output_spatial, patch_dim)
        # channels-last layout means patch dim is last
        patches = jax.lax.conv_general_dilated_patches(
            lhs=x,
            filter_shape=self.kernel_size,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # patches shape: (batch, *out_spatial, patch_dim)
        out_spatial = patches.shape[1:-1]
        patch_dim = patches.shape[-1]
        n_positions = 1
        for s in out_spatial:
            n_positions *= s

        # Reshape to (batch * n_positions, patch_dim)
        patches_flat = patches.reshape(batch * n_positions, patch_dim)

        # Apply KAN
        y = self.kan(patches_flat)  # (batch * n_positions, out_channels)

        # Reshape back to (batch, *out_spatial, out_channels)
        y = y.reshape(batch, *out_spatial, self.out_channels)

        return y
