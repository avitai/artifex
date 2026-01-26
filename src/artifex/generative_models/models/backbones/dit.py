"""Diffusion Transformer (DiT) backbone implementation using Flax NNX.

Based on "Scalable Diffusion Models with Transformers" by Peebles & Xie (2023).
Replaces U-Net backbone with Vision Transformers for improved scalability.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModule


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    """Apply adaptive layer norm modulation (adaLN).

    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter

    Returns:
        Modulated tensor
    """
    return x * (1 + scale) + shift


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int | tuple[int, int], add_cls_token: bool = False
) -> jax.Array:
    """Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (H, W) or single int for square grid
        add_cls_token: Whether to add class token

    Returns:
        Positional embeddings [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    grid_h_embed = get_1d_sincos_pos_embed(embed_dim // 2, grid_h)
    grid_w_embed = get_1d_sincos_pos_embed(embed_dim // 2, grid_w)

    # Create 2D positions
    pos_embed = jnp.concatenate(
        [
            jnp.repeat(grid_h_embed[:, None, :], grid_w, axis=1).reshape(-1, embed_dim // 2),
            jnp.tile(grid_w_embed[None, :, :], (grid_h, 1, 1)).reshape(-1, embed_dim // 2),
        ],
        axis=1,
    )

    if add_cls_token:
        cls_token_embed = jnp.zeros((1, embed_dim))
        pos_embed = jnp.concatenate([cls_token_embed, pos_embed], axis=0)

    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, pos: int) -> jax.Array:
    """Generate 1D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension
        pos: Number of positions

    Returns:
        Positional embeddings [pos, embed_dim]
    """
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos_range = jnp.arange(pos, dtype=jnp.float32)
    out = pos_range[:, None] * omega[None, :]

    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)

    return jnp.concatenate([emb_sin, emb_cos], axis=1)


class TimestepEmbedder(nnx.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs):
        """Initialize timestep embedder.

        Args:
            hidden_size: Hidden dimension
            frequency_embedding_size: Size of frequency embedding
            rngs: Random number generators
        """
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nnx.Sequential(
            nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )

    def timestep_embedding(self, t: jax.Array) -> jax.Array:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timesteps [batch_size]

        Returns:
            Embeddings [batch_size, frequency_embedding_size]
        """
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb

    def __call__(self, t: jax.Array) -> jax.Array:
        """Embed timesteps.

        Args:
            t: Timesteps [batch_size]

        Returns:
            Timestep embeddings [batch_size, hidden_size]
        """
        t_emb = self.timestep_embedding(t)
        return self.mlp(t_emb)


class LabelEmbedder(nnx.Module):
    """Embeds class labels into vector representations."""

    def __init__(
        self, num_classes: int, hidden_size: int, dropout_rate: float = 0.0, *, rngs: nnx.Rngs
    ):
        """Initialize label embedder.

        Args:
            num_classes: Number of classes
            hidden_size: Hidden dimension
            dropout_rate: Dropout rate for label dropout
            rngs: Random number generators
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Add extra embedding for unconditional generation (class dropout)
        self.embedding_table = nnx.Embed(
            num_embeddings=num_classes + 1, features=hidden_size, rngs=rngs
        )

        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        labels: jax.Array,
        *,
        deterministic: bool = False,
        force_drop_ids: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Embed class labels with optional dropout.

        Args:
            labels: Class labels [batch_size]
            deterministic: Whether to apply dropout
            force_drop_ids: Force specific samples to be dropped (for CFG)

        Returns:
            Label embeddings [batch_size, hidden_size]
        """
        if force_drop_ids is not None:
            # Force specific samples to use unconditional embedding
            labels = jnp.where(force_drop_ids, self.num_classes, labels)
        elif self.dropout is not None and not deterministic:
            # Random dropout during training
            drop_mask = self.dropout(jnp.ones(labels.shape), deterministic=deterministic) < 0.5
            labels = jnp.where(drop_mask, self.num_classes, labels)

        return self.embedding_table(labels)


class DiTBlock(nnx.Module):
    """Transformer block with adaptive layer norm (adaLN) for DiT."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize DiT block.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Attention components
        self.norm1 = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,  # Explicitly set qkv features
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )

        # MLP components
        self.norm2 = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_hidden_dim, hidden_size, rngs=rngs),
        )

        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        # Adaptive layer norm parameters
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(hidden_size, 6 * hidden_size, rngs=rngs),
        )

    def __call__(self, x: jax.Array, c: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Apply DiT block.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            c: Conditioning tensor [batch, hidden_size]
            deterministic: Whether to apply dropout

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Get adaptive layer norm parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaLN_modulation(c), 6, axis=-1
        )

        # Expand for sequence dimension
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]

        # Multi-head self-attention with adaLN
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # For self-attention, inputs_q is the only required argument
        # The MultiHeadAttention will use it for key and value as well
        attn_out = self.attn(inputs_q=norm_x, deterministic=deterministic)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out, deterministic=deterministic)
        x = x + gate_msa * attn_out

        # MLP with adaLN
        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x)
        if self.dropout is not None:
            mlp_out = self.dropout(mlp_out, deterministic=deterministic)
        x = x + gate_mlp * mlp_out

        return x


class FinalLayer(nnx.Module):
    """Final layer for DiT model."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, *, rngs: nnx.Rngs):
        """Initialize final layer.

        Args:
            hidden_size: Hidden dimension
            patch_size: Patch size
            out_channels: Output channels
            rngs: Random number generators
        """
        super().__init__()
        self.norm_final = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, patch_size * patch_size * out_channels, rngs=rngs)
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs),
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        """Apply final layer.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            c: Conditioning tensor [batch, hidden_size]

        Returns:
            Output tensor [batch, seq_len, patch_size^2 * out_channels]
        """
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=-1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiffusionTransformer(GenerativeModule):
    """Diffusion Transformer (DiT) for diffusion models.

    A transformer backbone that replaces U-Net in diffusion models,
    offering better scalability and performance.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.0,
        learn_sigma: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Diffusion Transformer.

        Args:
            img_size: Input image size
            patch_size: Patch size for patchification
            in_channels: Number of input channels
            hidden_size: Hidden dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            num_classes: Number of classes for conditional generation
            dropout_rate: Dropout rate
            learn_sigma: Whether to learn variance
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)  # Pass rngs to GenerativeModule

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma

        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            padding=0,
            rngs=rngs,
        )

        # Positional embedding
        self.pos_embed = nnx.Param(
            get_2d_sincos_pos_embed(hidden_size, img_size // patch_size), trainable=False
        )

        # Time embedding
        self.time_embed = TimestepEmbedder(hidden_size, rngs=rngs)

        # Optional label embedding
        if num_classes is not None:
            self.label_embed = LabelEmbedder(num_classes, hidden_size, dropout_rate, rngs=rngs)
        else:
            self.label_embed = None

        # Transformer blocks
        self.blocks = nnx.List(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, rngs=rngs)

    def unpatchify(self, x: jax.Array) -> jax.Array:
        """Convert patches back to images.

        Args:
            x: Patches [batch, num_patches, patch_size^2 * channels]

        Returns:
            Images [batch, height, width, channels]
        """
        batch_size = x.shape[0]
        patches_per_dim = int(math.sqrt(self.num_patches))

        # Reshape to [batch, h_patches, w_patches, patch_size, patch_size, channels]
        x = x.reshape(
            batch_size,
            patches_per_dim,
            patches_per_dim,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )

        # Transpose to [batch, h_patches, patch_size, w_patches, patch_size, channels]
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))

        # Reshape to [batch, height, width, channels]
        x = x.reshape(
            batch_size,
            patches_per_dim * self.patch_size,
            patches_per_dim * self.patch_size,
            self.out_channels,
        )

        return x

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        y: Optional[jax.Array] = None,
        *,
        deterministic: bool = False,
        cfg_scale: float = 1.0,
    ) -> jax.Array:
        """Forward pass of Diffusion Transformer.

        Args:
            x: Input images [batch, height, width, channels]
            t: Timesteps [batch]
            y: Optional class labels [batch]
            deterministic: Whether to apply dropout
            cfg_scale: Classifier-free guidance scale

        Returns:
            Predicted noise or learned mean/variance [batch, height, width, channels]
        """
        batch_size = x.shape[0]

        # Patchify: [batch, height, width, channels] -> [batch, num_patches, hidden_size]
        x = self.patch_embed(x)
        x = x.reshape(batch_size, -1, self.hidden_size)

        # Add positional embedding
        x = x + self.pos_embed.value

        # Time and optional label embeddings
        t_emb = self.time_embed(t)

        if self.label_embed is not None and y is not None:
            y_emb = self.label_embed(y, deterministic=deterministic)
            c = t_emb + y_emb
        else:
            c = t_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c, deterministic=deterministic)

        # Final layer
        x = self.final_layer(x, c)

        # Unpatchify:
        # [batch, num_patches, patch_size^2 * channels]
        # -> [batch, height, width, channels]
        x = self.unpatchify(x)

        return x

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs, **kwargs):
        """Generate samples (placeholder for GenerativeModule interface)."""
        raise NotImplementedError("Use with DiffusionModel wrapper for generation")

    def loss_fn(self, batch, model_outputs, **kwargs):
        """Compute loss (placeholder for GenerativeModule interface)."""
        raise NotImplementedError("Use with DiffusionModel wrapper for loss computation")
