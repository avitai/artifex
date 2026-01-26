"""UNet with Cross-Attention for text-conditioned diffusion models.

This module implements UNet2DCondition, which extends the base UNet with
cross-attention mechanisms for text conditioning, as used in Stable Diffusion.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModule
from artifex.generative_models.models.backbones.unet import ConvBlock, TimeEmbedding


class CrossAttentionBlock(GenerativeModule):
    """Cross-attention block for conditioning on text embeddings.

    This block performs cross-attention where the queries come from the
    spatial features and keys/values come from text embeddings.
    """

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize cross-attention block.

        Args:
            channels: Number of channels in spatial features
            cross_attention_dim: Dimension of text embeddings
            num_heads: Number of attention heads
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)

        self.channels = channels
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads

        # Layer norm before attention
        self.norm = nnx.LayerNorm(num_features=channels, rngs=rngs)

        # Cross-attention
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=channels,
            qkv_features=channels,
            out_features=channels,
            decode=False,
            rngs=rngs,
        )

        # Text projection (project text embeddings to query/key/value space)
        self.text_proj = nnx.Linear(
            in_features=cross_attention_dim,
            out_features=channels,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        text_embeddings: jax.Array,
    ) -> jax.Array:
        """Apply cross-attention to spatial features.

        Args:
            x: Spatial features [batch, height, width, channels]
            text_embeddings: Text embeddings [batch, seq_len, cross_attention_dim]

        Returns:
            Output features [batch, height, width, channels]
        """
        batch_size, height, width, channels = x.shape

        # Reshape spatial features to sequence: [batch, height*width, channels]
        x_flat = x.reshape(batch_size, height * width, channels)

        # Normalize
        x_norm = self.norm(x_flat)

        # Project text embeddings: [batch, seq_len, channels]
        text_proj = self.text_proj(text_embeddings)

        # Cross-attention: queries from spatial features, keys/values from text
        attn_out = self.attention(
            inputs_q=x_norm,  # queries from spatial features
            inputs_k=text_proj,  # keys from text
            inputs_v=text_proj,  # values from text
        )

        # Residual connection
        x_flat = x_flat + attn_out

        # Reshape back to spatial: [batch, height, width, channels]
        x_out = x_flat.reshape(batch_size, height, width, channels)

        return x_out


class DownBlockWithCrossAttention(GenerativeModule):
    """Downsampling block with cross-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        cross_attention_dim: int,
        num_heads: int,
        num_res_blocks: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize downsampling block with cross-attention.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embedding_dim: Time embedding dimension
            cross_attention_dim: Text embedding dimension
            num_heads: Number of attention heads
            num_res_blocks: Number of residual blocks
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)

        # Residual blocks
        self.res_blocks = nnx.List([])
        self.attn_blocks = nnx.List([])

        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ConvBlock(in_ch, out_channels, time_embedding_dim, rngs=rngs))
            self.attn_blocks.append(
                CrossAttentionBlock(out_channels, cross_attention_dim, num_heads, rngs=rngs)
            )

        # Downsampling
        self.downsample = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        time_emb: jax.Array,
        text_embeddings: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass through downsampling block with cross-attention.

        Args:
            x: Input tensor [batch, height, width, channels]
            time_emb: Time embedding [batch, time_dim]
            text_embeddings: Text embeddings [batch, seq_len, text_dim]

        Returns:
            Tuple of (downsampled output, skip connection features)
        """
        h = x

        # Apply residual blocks with cross-attention
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, time_emb)
            h = attn_block(h, text_embeddings)

        # Save for skip connection
        skip_features = h

        # Downsample
        h = self.downsample(h)

        return h, skip_features


class UpBlockWithCrossAttention(GenerativeModule):
    """Upsampling block with cross-attention."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        cross_attention_dim: int,
        num_heads: int,
        num_res_blocks: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize upsampling block with cross-attention.

        Args:
            in_channels: Number of input channels from previous layer
            skip_channels: Number of channels in skip connection
            out_channels: Number of output channels
            time_embedding_dim: Time embedding dimension
            cross_attention_dim: Text embedding dimension
            num_heads: Number of attention heads
            num_res_blocks: Number of residual blocks
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)

        # Residual blocks
        self.res_blocks = nnx.List([])
        self.attn_blocks = nnx.List([])

        for i in range(num_res_blocks):
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            self.res_blocks.append(ConvBlock(in_ch, out_channels, time_embedding_dim, rngs=rngs))
            self.attn_blocks.append(
                CrossAttentionBlock(out_channels, cross_attention_dim, num_heads, rngs=rngs)
            )

        # Upsampling
        self.upsample = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        skip_features: jax.Array,
        time_emb: jax.Array,
        text_embeddings: jax.Array,
    ) -> jax.Array:
        """Forward pass through upsampling block with cross-attention.

        Args:
            x: Input tensor [batch, height, width, in_channels]
            skip_features: Skip connection features
            time_emb: Time embedding [batch, time_dim]
            text_embeddings: Text embeddings [batch, seq_len, text_dim]

        Returns:
            Upsampled output [batch, height, width, out_channels]
        """
        # Upsample to match skip features size
        target_shape = (
            x.shape[0],
            skip_features.shape[1],
            skip_features.shape[2],
            x.shape[3],
        )
        h = jax.image.resize(x, shape=target_shape, method="nearest")
        h = self.upsample(h)

        # Concatenate with skip connection
        h = jnp.concatenate([h, skip_features], axis=-1)

        # Apply residual blocks with cross-attention
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, time_emb)
            h = attn_block(h, text_embeddings)

        return h


class UNet2DCondition(GenerativeModule):
    """UNet with cross-attention for text-conditioned generation.

    This implements a UNet architecture with cross-attention layers that
    condition on text embeddings, as used in Stable Diffusion.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        hidden_dims: List of channel dimensions for each level
        num_res_blocks: Number of residual blocks per level
        attention_levels: Which levels to add cross-attention (e.g., [0, 1, 2])
        cross_attention_dim: Dimension of text embeddings
        num_heads: Number of attention heads
        time_embedding_dim: Dimension for time embeddings
        rngs: Random number generators

    Example:
        >>> unet = UNet2DCondition(
        ...     in_channels=4,
        ...     out_channels=4,
        ...     hidden_dims=[320, 640, 1280],
        ...     num_res_blocks=2,
        ...     attention_levels=[0, 1, 2],
        ...     cross_attention_dim=768,
        ...     num_heads=8,
        ...     rngs=rngs,
        ... )
        >>> output = unet(x, timesteps, conditioning=text_embeddings)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: list[int],
        num_res_blocks: int,
        attention_levels: list[int],
        cross_attention_dim: int,
        num_heads: int,
        time_embedding_dim: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize UNet2DCondition."""
        super().__init__(rngs=rngs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embedding_dim, rngs=rngs)

        # Initial convolution
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

        for i, dim in enumerate(hidden_dims[1:]):
            level = i
            if level in attention_levels:
                # Use cross-attention block
                self.down_blocks.append(
                    DownBlockWithCrossAttention(
                        prev_dim,
                        dim,
                        time_embedding_dim,
                        cross_attention_dim,
                        num_heads,
                        num_res_blocks,
                        rngs=rngs,
                    )
                )
            else:
                # Use regular down block (import from base unet)
                from artifex.generative_models.models.backbones.unet import DownBlock

                self.down_blocks.append(DownBlock(prev_dim, dim, time_embedding_dim, rngs=rngs))

            prev_dim = dim

        # Middle blocks (always with cross-attention)
        self.mid_block1 = ConvBlock(hidden_dims[-1], hidden_dims[-1], time_embedding_dim, rngs=rngs)
        self.mid_attn = CrossAttentionBlock(
            hidden_dims[-1], cross_attention_dim, num_heads, rngs=rngs
        )
        self.mid_block2 = ConvBlock(hidden_dims[-1], hidden_dims[-1], time_embedding_dim, rngs=rngs)

        # Decoder (upsampling path)
        self.up_blocks = nnx.List([])
        prev_dim = hidden_dims[-1]

        skip_dims = list(reversed(hidden_dims[1:]))
        out_dims = list(reversed(hidden_dims[:-1]))

        for i, (skip_dim, out_dim) in enumerate(zip(skip_dims, out_dims)):
            # Level in reverse (for matching with attention_levels)
            level = len(hidden_dims) - 2 - i

            if level in attention_levels:
                # Use cross-attention block
                self.up_blocks.append(
                    UpBlockWithCrossAttention(
                        prev_dim,
                        skip_dim,
                        out_dim,
                        time_embedding_dim,
                        cross_attention_dim,
                        num_heads,
                        num_res_blocks,
                        rngs=rngs,
                    )
                )
            else:
                # Use regular up block (import from base unet)
                from artifex.generative_models.models.backbones.unet import UpBlock

                self.up_blocks.append(
                    UpBlock(prev_dim, skip_dim, out_dim, time_embedding_dim, rngs=rngs)
                )

            prev_dim = out_dim

        # Final output
        def get_num_groups(channels: int) -> int:
            """Get number of groups ensuring divisibility."""
            for g in [32, 16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1

        self.out_norm = nnx.GroupNorm(
            num_features=hidden_dims[0],
            num_groups=get_num_groups(hidden_dims[0]),
            rngs=rngs,
        )
        self.out_conv = nnx.Conv(
            in_features=hidden_dims[0],
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        *,
        conditioning: jax.Array | None = None,
        **kwargs,
    ) -> jax.Array:
        """Forward pass through UNet with cross-attention.

        Args:
            x: Input latents [batch, height, width, channels]
            timesteps: Timestep indices [batch]
            conditioning: Text embeddings [batch, seq_len, text_dim]
                         (aliased from encoder_hidden_states for consistency)
            **kwargs: Additional arguments (ignored)

        Returns:
            Predicted noise [batch, height, width, channels]

        Note:
            The conditioning parameter is the text embeddings used for cross-attention.
            Unlike the base UNet which ignores conditioning, this backbone uses it
            for classifier-free guidance and text-to-image generation.
        """
        del kwargs  # Unused

        # Handle None conditioning for classifier-free guidance unconditional pass
        if conditioning is None:
            # Use zeros for unconditional generation (null text embedding)
            # Shape: [batch, 1, cross_attention_dim]
            batch_size = x.shape[0]
            encoder_hidden_states = jnp.zeros((batch_size, 1, self.cross_attention_dim))
        else:
            encoder_hidden_states = conditioning

        # Time embedding
        time_emb = self.time_embedding(timesteps)

        # Initial convolution
        h = self.in_conv(x)

        # Downsampling with skip connections
        skip_features = []
        for down_block in self.down_blocks:
            if isinstance(down_block, DownBlockWithCrossAttention):
                h, skip = down_block(h, time_emb, encoder_hidden_states)
            else:
                # Regular DownBlock
                h, skip = down_block(h, time_emb)
            skip_features.append(skip)

        # Middle blocks with cross-attention
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h, encoder_hidden_states)
        h = self.mid_block2(h, time_emb)

        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skip_features)):
            if isinstance(up_block, UpBlockWithCrossAttention):
                h = up_block(h, skip, time_emb, encoder_hidden_states)
            else:
                # Regular UpBlock
                h = up_block(h, skip, time_emb)

        # Final output
        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)

        return h
