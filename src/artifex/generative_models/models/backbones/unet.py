"""UNet backbone for diffusion models using Flax NNX.

This module implements a UNet architecture for diffusion models with:
- Time embedding via sinusoidal positional encoding
- Skip connections between encoder and decoder
- GroupNorm for normalization
- Configurable via UNetBackboneConfig for nested config architecture
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModule
from artifex.generative_models.core.configuration import UNetBackboneConfig


class TimeEmbedding(GenerativeModule):
    """Time embedding module for diffusion timesteps."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize time embedding.

        Args:
            embedding_dim: Dimension of the embedding
            rngs: Random number generators
        """
        super().__init__(
            rngs=rngs,
        )
        self.embedding_dim = embedding_dim
        self.dense1 = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)
        self.dense2 = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Embed timestep t into a higher dimensional space.

        Args:
            t: Timestep indices [batch_size]

        Returns:
            Time embeddings [batch_size, embedding_dim]
        """
        # Ensure t is a jax array
        t = jnp.asarray(t)

        # Create sinusoidal position embeddings
        half_dim = self.embedding_dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)

        # Calculate embeddings
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)

        # If odd embedding dimension, add padding
        if self.embedding_dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))

        # Pass through MLP
        emb = self.dense1(emb)
        emb = nnx.silu(emb)
        emb = self.dense2(emb)

        return emb


class ConvBlock(GenerativeModule):
    """Convolutional block with skip connections and time embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embedding_dim: Optional time embedding dimension
            rngs: Random number generators
        """
        super().__init__(
            rngs=rngs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate number of groups for GroupNorm
        def get_num_groups(channels: int) -> int:
            """Get number of groups ensuring divisibility."""
            # Start with a reasonable default and work down
            for g in [32, 16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1

        n_groups_in = get_num_groups(in_channels)
        n_groups_out = get_num_groups(out_channels)

        # GroupNorm and convolutions
        self.norm1 = nnx.GroupNorm(num_features=in_channels, num_groups=n_groups_in, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

        self.norm2 = nnx.GroupNorm(num_features=out_channels, num_groups=n_groups_out, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

        # Time embedding projection (if provided)
        # Use static boolean flag for JIT compatibility
        self.use_time_emb = time_embedding_dim is not None
        if self.use_time_emb:
            self.time_emb = nnx.Linear(
                in_features=time_embedding_dim, out_features=out_channels, rngs=rngs
            )

        # Skip connection if channel dimensions don't match
        # Use static boolean flag for JIT compatibility
        self.use_skip_connection = in_channels != out_channels
        if self.use_skip_connection:
            self.skip_connection = nnx.Conv(
                in_features=in_channels, out_features=out_channels, kernel_size=(1, 1), rngs=rngs
            )

    def __call__(self, x: jax.Array, time_emb: jax.Array | None = None) -> jax.Array:
        """Forward pass through the convolutional block.

        Args:
            x: Input tensor [batch, height, width, channels]
            time_emb: Optional time embedding [batch, embedding_dim]

        Returns:
            Output tensor [batch, height, width, channels]
        """
        # Identity to use for residual connection
        identity = x

        # First convolution block
        h = self.norm1(x)
        h = nnx.silu(h)
        h = self.conv1(h)

        # Add time embedding if module supports it
        # NOTE: If use_time_emb is True, time_emb MUST be provided by caller
        if self.use_time_emb:
            # Project and reshape time embedding
            time_signal = self.time_emb(nnx.silu(time_emb))
            h = h + time_signal[:, None, None, :]

        # Second convolution block
        h = self.norm2(h)
        h = nnx.silu(h)
        h = self.conv2(h)

        # Skip connection
        if self.use_skip_connection:
            identity = self.skip_connection(x)

        return h + identity


class DownBlock(GenerativeModule):
    """Downsampling block for UNet."""

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
            time_embedding_dim: Optional time embedding dimension
            rngs: Random number generators
        """
        super().__init__(
            rngs=rngs,
        )
        # Two convolution blocks
        self.block1 = ConvBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)
        self.block2 = ConvBlock(out_channels, out_channels, time_embedding_dim, rngs=rngs)

        # Downsampling via strided convolution
        self.downsample = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, time_emb: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass through the downsampling block.

        Args:
            x: Input tensor [batch, height, width, channels]
            time_emb: Optional time embedding [batch, embedding_dim]

        Returns:
            Tuple of (downsampled output, skip connection features)
        """
        # Apply convolution blocks
        h = self.block1(x, time_emb)
        h = self.block2(h, time_emb)

        # Save features for skip connection
        skip_features = h

        # Downsample
        h = self.downsample(h)

        return h, skip_features


class UpBlock(GenerativeModule):
    """Upsampling block for UNet."""

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
            in_channels: Number of input channels from previous layer
            skip_channels: Number of channels in skip connection features
            out_channels: Number of output channels
            time_embedding_dim: Optional time embedding dimension
            rngs: Random number generators
        """
        super().__init__(
            rngs=rngs,
        )
        # After upsampling and concatenation: in_channels + skip_channels
        self.block1 = ConvBlock(
            in_channels + skip_channels, out_channels, time_embedding_dim, rngs=rngs
        )
        self.block2 = ConvBlock(out_channels, out_channels, time_embedding_dim, rngs=rngs)

        # Upsampling preserves input channels - we don't change channels here
        self.upsample = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,  # Keep same channels after upsampling
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        skip_features: jax.Array,
        time_emb: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass through the upsampling block.

        Args:
            x: Input tensor [batch, height, width, in_channels]
            skip_features: Skip connection features from the encoder
            time_emb: Optional time embedding [batch, embedding_dim]

        Returns:
            Upsampled output [batch, height, width, out_channels]
        """
        # Get target height and width from skip features
        target_shape = (
            x.shape[0],  # batch
            skip_features.shape[1],  # height
            skip_features.shape[2],  # width
            x.shape[3],  # channels
        )

        # Simple resize upsampling - safer for shape matching
        h = jax.image.resize(x, shape=target_shape, method="nearest")

        # Convolution after resizing
        h = self.upsample(h)

        # Concatenate with skip connection features
        h = jnp.concatenate([h, skip_features], axis=-1)

        # Apply convolution blocks
        h = self.block1(h, time_emb)
        h = self.block2(h, time_emb)

        return h


class UNet(GenerativeModule):
    """UNet architecture for diffusion models.

    This implementation supports configuration via UNetBackboneConfig for the nested
    config architecture used in diffusion models.

    Attributes:
        hidden_dims: Channel dimensions for each UNet level
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_embedding_dim: Dimension of time embeddings
    """

    def __init__(
        self,
        config: UNetBackboneConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize UNet model from config.

        Args:
            config: UNetBackboneConfig containing all UNet parameters
            rngs: Random number generators

        Example:
            config = UNetBackboneConfig(
                name="unet",
                hidden_dims=(32, 64, 128, 256),
                activation="gelu",
                in_channels=3,
                out_channels=3,
                time_embedding_dim=128,
            )
            unet = UNet(config=config, rngs=rngs)
        """
        super().__init__(
            rngs=rngs,
        )
        # Extract parameters from config
        hidden_dims = list(config.hidden_dims)
        time_embedding_dim = config.time_embedding_dim
        in_channels = config.in_channels
        out_channels = config.out_channels

        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embedding_dim, rngs=rngs)

        # Initial projection
        self.in_conv = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_dims[0],
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

        # Encoder (downsampling path)
        self.down_blocks = nnx.List([])
        prev_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.down_blocks.append(DownBlock(prev_dim, dim, time_embedding_dim, rngs=rngs))
            prev_dim = dim

        # Middle block
        self.mid_block1 = ConvBlock(hidden_dims[-1], hidden_dims[-1], time_embedding_dim, rngs=rngs)
        self.mid_block2 = ConvBlock(hidden_dims[-1], hidden_dims[-1], time_embedding_dim, rngs=rngs)

        # Decoder (upsampling path)
        # FIXED: Calculate correct skip channel dimensions
        self.up_blocks = nnx.List([])
        prev_dim = hidden_dims[-1]  # Start with the last hidden dimension

        # Skip connection channels come from the encoder in reverse order
        # For hidden_dims = [32, 64, 128, 256]:
        # Down blocks produce skip features: [64, 128, 256]
        # Up blocks need them in reverse: [256, 128, 64]
        skip_dims = list(reversed(hidden_dims[1:]))  # [256, 128, 64]
        out_dims = list(reversed(hidden_dims[:-1]))  # [128, 64, 32]

        for skip_dim, out_dim in zip(skip_dims, out_dims):
            self.up_blocks.append(
                UpBlock(prev_dim, skip_dim, out_dim, time_embedding_dim, rngs=rngs)
            )
            prev_dim = out_dim

        # Final projection
        def get_num_groups(channels: int) -> int:
            """Get number of groups ensuring divisibility."""
            if channels >= 32:
                return 32
            for g in [16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1

        n_groups_final = get_num_groups(hidden_dims[0])
        self.out_norm = nnx.GroupNorm(
            num_features=hidden_dims[0], num_groups=n_groups_final, rngs=rngs
        )
        self.out_conv = nnx.Conv(
            in_features=hidden_dims[0],
            out_features=out_channels,  # Use configured output channels
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        *,
        conditioning: jax.Array | None = None,
        **kwargs,
    ) -> jax.Array:
        """Forward pass through the UNet.

        Args:
            x: Input tensor [batch, height, width, channels]
            t: Timestep indices [batch]
            conditioning: Optional conditioning info (ignored in base UNet)
            **kwargs: Additional arguments (ignored in base UNet)

        Returns:
            Output tensor [batch, height, width, channels]

        Note:
            Train/eval mode controlled via model.train()/model.eval().
            RNGs stored at init time per NNX best practices.
            Base UNet ignores conditioning - use UNet2DCondition for conditional models.
        """
        # Note: conditioning is ignored in base UNet
        # For conditional UNet, see UNet2DCondition
        del conditioning, kwargs  # Explicitly mark as unused
        # Embed time
        time_emb = self.time_embedding(t)

        # Initial convolution
        h = self.in_conv(x)

        # Downsampling and collect skip features
        skip_features = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skip_features.append(skip)

        # Middle blocks
        h = self.mid_block1(h, time_emb)
        h = self.mid_block2(h, time_emb)

        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skip_features)):
            h = up_block(h, skip, time_emb)

        # Final convolution
        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)

        return h
