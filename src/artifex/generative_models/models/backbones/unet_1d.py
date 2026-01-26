"""1D U-Net backbone for diffusion models on 1D signals.

This module provides a 1D U-Net architecture for diffusion models operating
on 1D sequences such as audio waveforms, time series, and other 1D signals.

Uses the (config, *, rngs) signature pattern following Artifex conventions.

Reuses TimeEmbedding from unet.py (DRY principle).
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.backbone_config import UNet1DBackboneConfig
from artifex.generative_models.models.backbones.unet import TimeEmbedding


def _get_num_groups(channels: int) -> int:
    """Get number of groups for GroupNorm ensuring divisibility.

    Args:
        channels: Number of channels

    Returns:
        Number of groups for GroupNorm
    """
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConvBlock1D(nnx.Module):
    """1D convolution block with time embedding conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embedding_dim: Dimension of time embeddings (None to disable)
            rngs: Random number generators
        """
        super().__init__()

        self.norm = nnx.GroupNorm(
            num_features=in_channels, num_groups=_get_num_groups(in_channels), rngs=rngs
        )
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3,),
            padding="SAME",
            rngs=rngs,
        )

        # Time embedding projection if provided
        self.time_proj: nnx.Linear | None
        if time_embedding_dim is not None:
            self.time_proj = nnx.Linear(time_embedding_dim, out_channels, rngs=rngs)
        else:
            self.time_proj = None

    def __call__(self, x: jnp.ndarray, time_emb: jnp.ndarray | None = None) -> jnp.ndarray:
        """Apply conv block.

        Args:
            x: Input tensor of shape (batch, length, channels)
            time_emb: Time embeddings of shape (batch, time_dim)

        Returns:
            Output tensor of shape (batch, length, out_channels)
        """
        h = self.norm(x)
        h = nnx.silu(h)
        h = self.conv(h)

        # Add time embedding if provided
        if self.time_proj is not None and time_emb is not None:
            time_proj = self.time_proj(time_emb)
            h = h + time_proj[:, None, :]  # Broadcast over time dimension

        return h


class DownBlock1D(nnx.Module):
    """1D downsampling block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embedding_dim: Dimension of time embeddings
            rngs: Random number generators
        """
        super().__init__()

        # Two conv blocks
        self.block1 = ConvBlock1D(in_channels, out_channels, time_embedding_dim, rngs=rngs)
        self.block2 = ConvBlock1D(out_channels, out_channels, time_embedding_dim, rngs=rngs)

        # Downsampling
        self.downsample = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(
        self, x: jnp.ndarray, time_emb: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Input tensor
            time_emb: Time embeddings

        Returns:
            Tuple of (downsampled output, skip features)
        """
        # Apply conv blocks
        h = self.block1(x, time_emb)
        h = self.block2(h, time_emb)

        # Save for skip connection BEFORE downsampling
        skip_features = h

        # Downsample
        h = self.downsample(h)

        return h, skip_features


class UpBlock1D(nnx.Module):
    """1D upsampling block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embedding_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            skip_channels: Number of channels from skip connection
            out_channels: Number of output channels
            time_embedding_dim: Dimension of time embeddings
            rngs: Random number generators
        """
        super().__init__()

        # Upsampling (preserves channels)
        self.upsample = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(4,),
            strides=(2,),
            padding=((1, 1),),
            rngs=rngs,
        )

        # Conv blocks after concatenation
        self.block1 = ConvBlock1D(
            in_channels + skip_channels, out_channels, time_embedding_dim, rngs=rngs
        )
        self.block2 = ConvBlock1D(out_channels, out_channels, time_embedding_dim, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        skip_features: jnp.ndarray,
        time_emb: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor
            skip_features: Skip connection features
            time_emb: Time embeddings

        Returns:
            Output tensor
        """
        # Upsample
        h = self.upsample(x)

        # Handle potential size mismatch due to stride/padding
        if h.shape[1] != skip_features.shape[1]:
            # Trim or pad to match skip features size
            if h.shape[1] > skip_features.shape[1]:
                h = h[:, : skip_features.shape[1], :]
            else:
                # Pad with zeros
                pad_size = skip_features.shape[1] - h.shape[1]
                h = jnp.pad(h, ((0, 0), (0, pad_size), (0, 0)))

        # Concatenate with skip features
        h = jnp.concatenate([h, skip_features], axis=-1)

        # Apply conv blocks
        h = self.block1(h, time_emb)
        h = self.block2(h, time_emb)

        return h


class UNet1D(nnx.Module):
    """1D U-Net for diffusion models on 1D signals.

    This backbone uses 1D convolutions and is suitable for audio waveforms,
    time series, and other 1D sequential data.

    Uses the (config, *, rngs) signature pattern.
    Reuses TimeEmbedding from the 2D UNet module (DRY principle).

    Attributes:
        config: UNet1DBackboneConfig
        hidden_dims: Channel dimensions per level
        in_channels: Number of input channels
    """

    def __init__(
        self,
        config: UNet1DBackboneConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize 1D U-Net from config.

        Args:
            config: UNet1DBackboneConfig containing all parameters
            rngs: Random number generators

        Example:
            config = UNet1DBackboneConfig(
                name="audio_unet",
                hidden_dims=(32, 64, 128, 256),
                activation="gelu",
                in_channels=1,
                time_embedding_dim=128,
            )
            unet = UNet1D(config, rngs=rngs)
        """
        super().__init__()

        # Store config
        self.config = config

        # Extract parameters from config
        hidden_dims = list(config.hidden_dims)
        time_embedding_dim = config.time_embedding_dim
        in_channels = config.in_channels

        self.hidden_dims = hidden_dims
        self.in_channels = in_channels

        # Time embedding (reused from 2D UNet - same sinusoidal + MLP pattern)
        self.time_embedding = TimeEmbedding(time_embedding_dim, rngs=rngs)

        # Initial projection
        self.in_conv = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_dims[0],
            kernel_size=(3,),
            padding="SAME",
            rngs=rngs,
        )

        # Encoder (downsampling path)
        self.down_blocks = nnx.List([])
        prev_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.down_blocks.append(DownBlock1D(prev_dim, dim, time_embedding_dim, rngs=rngs))
            prev_dim = dim

        # Middle blocks
        self.mid_block1 = ConvBlock1D(
            hidden_dims[-1], hidden_dims[-1], time_embedding_dim, rngs=rngs
        )
        self.mid_block2 = ConvBlock1D(
            hidden_dims[-1], hidden_dims[-1], time_embedding_dim, rngs=rngs
        )

        # Decoder (upsampling path)
        self.up_blocks = nnx.List([])
        prev_dim = hidden_dims[-1]

        # Skip channels come from encoder in reverse order
        skip_dims = list(reversed(hidden_dims[1:]))
        out_dims = list(reversed(hidden_dims[:-1]))

        for skip_dim, out_dim in zip(skip_dims, out_dims):
            self.up_blocks.append(
                UpBlock1D(prev_dim, skip_dim, out_dim, time_embedding_dim, rngs=rngs)
            )
            prev_dim = out_dim

        # Final projection
        self.out_norm = nnx.GroupNorm(
            num_features=hidden_dims[0], num_groups=_get_num_groups(hidden_dims[0]), rngs=rngs
        )
        self.out_conv = nnx.Conv(
            in_features=hidden_dims[0],
            out_features=in_channels,
            kernel_size=(3,),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        timesteps: jnp.ndarray,
        *,
        conditioning: Any | None = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length) or (batch, length, channels)
            timesteps: Timestep tensor of shape (batch,)
            conditioning: Optional conditioning (not used, for API compatibility)
            deterministic: Whether to use deterministic mode (unused for this model)
            **kwargs: Additional arguments (ignored)

        Returns:
            Output tensor matching input shape
        """
        # Ensure input has channel dimension
        squeeze_output = False
        if x.ndim == 2:
            x = x[..., None]
            squeeze_output = True

        # Time embedding
        time_emb = self.time_embedding(timesteps)

        # Initial projection
        h = self.in_conv(x)

        # Encoder with skip collection
        skip_features = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skip_features.append(skip)

        # Middle blocks
        h = self.mid_block1(h, time_emb)
        h = self.mid_block2(h, time_emb)

        # Decoder with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skip_features)):
            h = up_block(h, skip, time_emb)

        # Final projection
        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)

        # Remove channel dimension to match input if needed
        if squeeze_output and h.shape[-1] == 1:
            h = h.squeeze(-1)

        return h
