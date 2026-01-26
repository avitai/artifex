"""Transformer implementation with flexible positional encodings - FINAL CORRECTED VERSION.

This module provides transformer encoder and decoder implementations that
support various types of positional encodings using the latest Flax NNX API.
"""

import logging
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

# Import positional encodings
from artifex.generative_models.core.layers.positional import (
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)


# Configure logger
logger = logging.getLogger(__name__)


class FeedForwardNetwork(nnx.Module):
    """Feed-forward network (MLP) for transformer blocks.

    This implements the standard two-layer MLP used in transformer architectures.

    Attributes:
        in_features: Input dimension
        hidden_features: Hidden dimension
        out_features: Output dimension
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the feed-forward network.

        Args:
            in_features: Input dimension.
            hidden_features: Hidden dimension. If None, defaults to 4*in_features.
            out_features: Output dimension. If None, defaults to in_features.
            dropout_rate: Dropout probability.
            use_bias: Whether to use bias in linear projections.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            rngs: Random number generators.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features * 4
        self.out_features = out_features or in_features
        self.dropout_rate = dropout_rate

        # Initialize layers
        self.fc1 = nnx.Linear(
            in_features=in_features,
            out_features=self.hidden_features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

        self.fc2 = nnx.Linear(
            in_features=self.hidden_features,
            out_features=self.out_features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

        # Only create dropout if rate > 0 and rngs has 'dropout' stream
        if dropout_rate > 0:
            # Check if rngs has 'dropout' stream
            # This allows creating the module even if dropout stream is missing
            # (useful for deterministic-only usage)
            if rngs is not None and "dropout" in rngs:
                self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
            else:
                self.dropout: nnx.Dropout | None = None
        else:
            self.dropout: nnx.Dropout | None = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply feed-forward network to input.

        Args:
            x: Input tensor of shape [batch, ..., in_features].
            deterministic: Whether to apply dropout.
            rngs: Optional RNGs (kept for API compatibility, not used as
                Dropout uses internal state).

        Returns:
            Output tensor of shape [batch, ..., out_features].
        """
        # First layer with activation
        x = self.fc1(x)
        x = nnx.gelu(x)  # Use nnx.gelu instead of nnx.gelu

        # Apply dropout after first layer if needed
        if self.dropout is not None and not deterministic:
            x = self.dropout(x)

        # Second layer
        x = self.fc2(x)

        # Apply dropout after second layer if needed
        if self.dropout is not None and not deterministic:
            x = self.dropout(x)

        return x


def create_attention_mask(
    mask: jax.Array,
    num_heads: int,
    is_decoder: bool = False,
) -> jax.Array:
    """Create proper attention mask for multi-head attention.

    Args:
        mask: Input mask of shape [batch, length] or [batch, query_len, key_len].
        num_heads: Number of attention heads.
        is_decoder: Whether the mask is for a decoder (causal).

    Returns:
        Attention mask of shape [batch, num_heads, query_len, key_len].
    """
    # Handle 2D mask [batch, length]
    if mask.ndim == 2:
        batch_size, seq_length = mask.shape

        # For encoder: Create symmetric mask [batch, 1, length, length]
        # For decoder: Create causal mask [batch, 1, length, length]
        if is_decoder:
            # Create causal mask
            causal_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
            # Combine with padding mask
            mask = mask[:, None, None, :] * causal_mask[None, None, :, :]
        else:
            # Create symmetric mask for encoder
            mask = mask[:, None, None, :] * mask[:, None, :, None]

    # Handle 3D mask [batch, query_len, key_len]
    elif mask.ndim == 3:
        # Add head dimension
        mask = mask[:, None, :, :]

    # Make sure we have the right number of heads
    if mask.shape[1] == 1:
        # Broadcast to all heads
        batch, _, query_len, key_len = mask.shape
        mask = jnp.broadcast_to(mask, (batch, num_heads, query_len, key_len))
    elif mask.shape[1] != num_heads:
        logger.warning(f"Mask shape {mask.shape} incompatible with {num_heads} attention heads")

    return mask


class TransformerEncoderBlock(nnx.Module):
    """Transformer encoder block.

    This implements a single layer of the transformer encoder with self-attention.

    Attributes:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to hidden_dim
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        use_bias: bool = True,
        normalize_qk: bool = False,
        broadcast_dropout: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer encoder block.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to hidden_dim.
            dropout_rate: Dropout probability.
            attention_dropout_rate: Dropout probability for attention.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            use_bias: Whether to use bias in linear projections.
            normalize_qk: Whether to apply QK normalization (improves training stability).
            broadcast_dropout: Whether to use broadcasted dropout along batch dims.
            rngs: Random number generators.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Self-attention layer
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            broadcast_dropout=broadcast_dropout,
            dropout_rate=attention_dropout_rate,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            decode=False,  # For encoder, decode is always False
            normalize_qk=normalize_qk,
            rngs=rngs,
        )

        # Feed-forward network
        self.mlp = FeedForwardNetwork(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

        # Layer normalization
        self.norm1 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

        # Dropout (only create if rate > 0 and rngs has 'dropout' stream)
        if dropout_rate > 0:
            if rngs is not None and "dropout" in rngs:
                self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
            else:
                self.dropout: nnx.Dropout | None = None
        else:
            self.dropout: nnx.Dropout | None = None

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply transformer encoder block to input.

        Args:
            x: Input tensor of shape [batch, length, hidden_dim].
            mask: Attention mask.
            deterministic: Whether to apply dropout.
            rngs: Random number generators.

        Returns:
            Output tensor of shape [batch, length, hidden_dim].
        """
        # Process mask if provided
        if mask is not None:
            mask = create_attention_mask(mask, self.num_heads, is_decoder=False)

        # Self-attention with pre-normalization
        residual = x
        x = self.norm1(x)
        x = self.attention(
            inputs_q=x,  # Explicit parameter name
            mask=mask,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Apply dropout if needed
        if self.dropout is not None and not deterministic:
            x = self.dropout(x)

        # Add residual connection
        x = residual + x

        # Feed-forward network with pre-normalization
        residual = x
        x = self.norm2(x)
        x = self.mlp(x, deterministic=deterministic, rngs=rngs)

        # Add residual connection
        x = residual + x

        return x


class TransformerDecoderBlock(nnx.Module):
    """Transformer decoder block.

    This implements a single layer of the transformer decoder with self-attention
    and cross-attention.

    Attributes:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to hidden_dim
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        use_bias: bool = True,
        normalize_qk: bool = False,
        broadcast_dropout: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer decoder block.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to hidden_dim.
            dropout_rate: Dropout probability.
            attention_dropout_rate: Dropout probability for attention.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            use_bias: Whether to use bias in linear projections.
            normalize_qk: Whether to apply QK normalization (improves training stability).
            broadcast_dropout: Whether to use broadcasted dropout along batch dims.
            rngs: Random number generators.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Self-attention layer
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            broadcast_dropout=broadcast_dropout,
            dropout_rate=attention_dropout_rate,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            decode=True,  # For decoder, decode can be True
            normalize_qk=normalize_qk,
            rngs=rngs,
        )

        # Cross-attention layer
        self.cross_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            broadcast_dropout=broadcast_dropout,
            dropout_rate=attention_dropout_rate,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            decode=False,  # Cross-attention doesn't use autoregressive cache
            normalize_qk=normalize_qk,
            rngs=rngs,
        )

        # Feed-forward network
        self.mlp = FeedForwardNetwork(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

        # Layer normalization
        self.norm1 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

        # Dropout (only create if rate > 0 and rngs has 'dropout' stream)
        if dropout_rate > 0:
            if rngs is not None and "dropout" in rngs:
                self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
            else:
                self.dropout: nnx.Dropout | None = None
        else:
            self.dropout: nnx.Dropout | None = None

    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array,
        self_attention_mask: jax.Array | None = None,
        cross_attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
        decode: bool = False,
    ) -> jax.Array:
        """Apply transformer decoder block to input.

        Args:
            x: Input tensor of shape [batch, length, hidden_dim].
            encoder_output: Encoder output of shape [batch, enc_length, hidden_dim].
            self_attention_mask: Mask for self-attention.
            cross_attention_mask: Mask for cross-attention.
            deterministic: Whether to apply dropout.
            rngs: Random number generators.
            decode: Whether to use autoregressive decoding.

        Returns:
            Output tensor of shape [batch, length, hidden_dim].
        """
        # Process masks if provided
        if self_attention_mask is not None:
            self_attention_mask = create_attention_mask(
                self_attention_mask, self.num_heads, is_decoder=True
            )

        if cross_attention_mask is not None:
            cross_attention_mask = create_attention_mask(
                cross_attention_mask, self.num_heads, is_decoder=False
            )

        # Self-attention with pre-normalization
        residual = x
        x = self.norm1(x)
        x = self.self_attention(
            inputs_q=x,  # Explicit parameter name for self-attention
            mask=self_attention_mask,
            deterministic=deterministic,
            rngs=rngs,
            decode=decode,
        )

        # Apply dropout if needed
        if self.dropout is not None and not deterministic:
            x = self.dropout(x)

        # Add residual connection
        x = residual + x

        # Cross-attention with pre-normalization
        residual = x
        x = self.norm2(x)
        x = self.cross_attention(
            inputs_q=x,  # query from decoder
            inputs_k=encoder_output,  # key from encoder
            inputs_v=encoder_output,  # value from encoder
            mask=cross_attention_mask,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Apply dropout if needed
        if self.dropout is not None and not deterministic:
            x = self.dropout(x)

        # Add residual connection
        x = residual + x

        # Feed-forward network with pre-normalization
        residual = x
        x = self.norm3(x)
        x = self.mlp(x, deterministic=deterministic, rngs=rngs)

        # Add residual connection
        x = residual + x

        return x


class TransformerEncoder(nnx.Module):
    """Transformer encoder.

    This implements a stack of transformer encoder blocks with positional encoding.

    Attributes:
        num_layers: Number of encoder layers
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to hidden_dim
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        use_bias: bool = True,
        normalize_qk: bool = False,
        broadcast_dropout: bool = True,
        max_len: int = 1024,
        pos_encoding_type: Literal["sinusoidal", "learned", "rotary", "none"] = "sinusoidal",
        pos_encoding_base: int = 10000,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer encoder.

        Args:
            num_layers: Number of encoder layers.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to hidden_dim.
            dropout_rate: Dropout probability.
            attention_dropout_rate: Dropout probability for attention.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            use_bias: Whether to use bias in linear projections.
            normalize_qk: Whether to apply QK normalization (improves training stability).
            broadcast_dropout: Whether to use broadcasted dropout along batch dims.
            max_len: Maximum sequence length for positional encoding.
            pos_encoding_type: Type of positional encoding to use.
                One of "sinusoidal", "learned", "rotary", or "none".
            pos_encoding_base: Base for frequency computation in rotary encoding.
            rngs: Random number generators.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_encoding_type = pos_encoding_type

        # Create positional encoding if needed (don't initialize to None)
        if pos_encoding_type != "none":
            if pos_encoding_type == "sinusoidal":
                # Only pass rngs if dropout_rate > 0
                pe_rngs = rngs if dropout_rate > 0 else None
                self.pos_encoding: (
                    SinusoidalPositionalEncoding
                    | LearnedPositionalEncoding
                    | RotaryPositionalEncoding
                ) = SinusoidalPositionalEncoding(
                    dim=hidden_dim,
                    max_len=max_len,
                    dropout_rate=dropout_rate,
                    rngs=pe_rngs,
                )
            elif pos_encoding_type == "learned":
                # LearnedPositionalEncoding always requires rngs
                self.pos_encoding: (
                    SinusoidalPositionalEncoding
                    | LearnedPositionalEncoding
                    | RotaryPositionalEncoding
                ) = LearnedPositionalEncoding(
                    dim=hidden_dim,
                    max_len=max_len,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
            elif pos_encoding_type == "rotary":
                if hidden_dim % 2 != 0:
                    raise ValueError(
                        f"Hidden dimension must be even for rotary encoding, got {hidden_dim}"
                    )
                # Only pass rngs if dropout_rate > 0
                pe_rngs = rngs if dropout_rate > 0 else None
                self.pos_encoding: (
                    SinusoidalPositionalEncoding
                    | LearnedPositionalEncoding
                    | RotaryPositionalEncoding
                ) = RotaryPositionalEncoding(
                    dim=hidden_dim,
                    max_len=max_len,
                    base=pos_encoding_base,
                    dropout_rate=dropout_rate,
                    rngs=pe_rngs,
                )
            else:
                raise ValueError(f"Unsupported positional encoding type: {pos_encoding_type}")

        # Create encoder layers
        self.layers = nnx.List([])
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    use_bias=use_bias,
                    normalize_qk=normalize_qk,
                    broadcast_dropout=broadcast_dropout,
                    rngs=rngs,
                )
            )

        # Final layer normalization
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply transformer encoder to input.

        Args:
            x: Input tensor of shape [batch, length, hidden_dim].
            mask: Attention mask.
            deterministic: Whether to apply dropout.
            rngs: Random number generators.

        Returns:
            Output tensor of shape [batch, length, hidden_dim].
        """
        # Apply positional encoding if enabled
        if hasattr(self, "pos_encoding"):
            x = self.pos_encoding(x, deterministic=deterministic, rngs=rngs)

        # Apply encoder layers
        for layer in self.layers:
            x = layer(
                x,
                mask=mask,
                deterministic=deterministic,
                rngs=rngs,
            )

        # Apply final layer normalization
        x = self.norm(x)

        return x


class TransformerDecoder(nnx.Module):
    """Transformer decoder.

    This implements a stack of transformer decoder blocks with positional encoding.

    Attributes:
        num_layers: Number of decoder layers
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to hidden_dim
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
        use_bias: bool = True,
        normalize_qk: bool = False,
        broadcast_dropout: bool = True,
        max_len: int = 1024,
        pos_encoding_type: Literal["sinusoidal", "learned", "rotary", "none"] = "sinusoidal",
        pos_encoding_base: int = 10000,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer decoder.

        Args:
            num_layers: Number of decoder layers.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to hidden_dim.
            dropout_rate: Dropout probability.
            attention_dropout_rate: Dropout probability for attention.
            kernel_init: Kernel initialization function.
            bias_init: Bias initialization function.
            use_bias: Whether to use bias in linear projections.
            normalize_qk: Whether to apply QK normalization (improves training stability).
            broadcast_dropout: Whether to use broadcasted dropout along batch dims.
            max_len: Maximum sequence length for positional encoding.
            pos_encoding_type: Type of positional encoding to use.
                One of "sinusoidal", "learned", "rotary", or "none".
            pos_encoding_base: Base for frequency computation in rotary encoding.
            rngs: Random number generators.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_encoding_type = pos_encoding_type

        # Create positional encoding if needed (don't initialize to None)
        if pos_encoding_type != "none":
            if pos_encoding_type == "sinusoidal":
                # Only pass rngs if dropout_rate > 0
                pe_rngs = rngs if dropout_rate > 0 else None
                self.pos_encoding: (
                    SinusoidalPositionalEncoding
                    | LearnedPositionalEncoding
                    | RotaryPositionalEncoding
                ) = SinusoidalPositionalEncoding(
                    dim=hidden_dim,
                    max_len=max_len,
                    dropout_rate=dropout_rate,
                    rngs=pe_rngs,
                )
            elif pos_encoding_type == "learned":
                # LearnedPositionalEncoding always requires rngs
                self.pos_encoding: (
                    SinusoidalPositionalEncoding
                    | LearnedPositionalEncoding
                    | RotaryPositionalEncoding
                ) = LearnedPositionalEncoding(
                    dim=hidden_dim,
                    max_len=max_len,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
            elif pos_encoding_type == "rotary":
                if hidden_dim % 2 != 0:
                    raise ValueError(
                        f"Hidden dimension must be even for rotary encoding, got {hidden_dim}"
                    )
                # Only pass rngs if dropout_rate > 0
                pe_rngs = rngs if dropout_rate > 0 else None
                self.pos_encoding: (
                    SinusoidalPositionalEncoding
                    | LearnedPositionalEncoding
                    | RotaryPositionalEncoding
                ) = RotaryPositionalEncoding(
                    dim=hidden_dim,
                    max_len=max_len,
                    base=pos_encoding_base,
                    dropout_rate=dropout_rate,
                    rngs=pe_rngs,
                )
            else:
                raise ValueError(f"Unsupported positional encoding type: {pos_encoding_type}")

        # Create decoder layers
        self.layers = nnx.List([])
        for _ in range(num_layers):
            self.layers.append(
                TransformerDecoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    use_bias=use_bias,
                    normalize_qk=normalize_qk,
                    broadcast_dropout=broadcast_dropout,
                    rngs=rngs,
                )
            )

        # Final layer normalization
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array,
        self_attention_mask: jax.Array | None = None,
        cross_attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
        decode: bool = False,
    ) -> jax.Array:
        """Apply transformer decoder to input.

        Args:
            x: Input tensor of shape [batch, length, hidden_dim].
            encoder_output: Encoder output of shape [batch, enc_length, hidden_dim].
            self_attention_mask: Mask for self-attention.
            cross_attention_mask: Mask for cross-attention.
            deterministic: Whether to apply dropout.
            rngs: Random number generators.
            decode: Whether to use autoregressive decoding.

        Returns:
            Output tensor of shape [batch, length, hidden_dim].
        """
        # Apply positional encoding if enabled
        if hasattr(self, "pos_encoding"):
            x = self.pos_encoding(x, deterministic=deterministic, rngs=rngs)

        # Initialize caches if decoding
        if decode:
            for layer in self.layers:
                if hasattr(layer.self_attention, "init_cache"):
                    layer.self_attention.init_cache(x.shape)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                deterministic=deterministic,
                rngs=rngs,
                decode=decode,
            )

        # Apply final layer normalization
        x = self.norm(x)

        return x


def create_transformer(
    num_encoder_layers: int,
    num_decoder_layers: int,
    hidden_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    normalize_qk: bool = False,
    broadcast_dropout: bool = True,
    max_len: int = 1024,
    pos_encoding_type: Literal["sinusoidal", "learned", "rotary", "none"] = "sinusoidal",
    pos_encoding_base: int = 10000,
    use_different_encoder_decoder_pos: bool = False,
    encoder_pos_encoding_type: Literal["sinusoidal", "learned", "rotary", "none"] | None = None,
    decoder_pos_encoding_type: Literal["sinusoidal", "learned", "rotary", "none"] | None = None,
    *,
    rngs: nnx.Rngs,
) -> tuple[TransformerEncoder, TransformerDecoder]:
    """Create a transformer encoder and decoder.

    Args:
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to hidden_dim.
        dropout_rate: Dropout probability.
        attention_dropout_rate: Dropout probability for attention.
        normalize_qk: Whether to apply QK normalization (improves training stability).
        broadcast_dropout: Whether to use broadcasted dropout along batch dims.
        max_len: Maximum sequence length for positional encoding.
        pos_encoding_type: Type of positional encoding to use for both encoder and decoder.
            One of "sinusoidal", "learned", "rotary", or "none".
        pos_encoding_base: Base for frequency computation in rotary encoding.
        use_different_encoder_decoder_pos: Whether to use different positional encoding types
            for encoder and decoder.
        encoder_pos_encoding_type: Type of positional encoding to use for encoder.
            Only used if use_different_encoder_decoder_pos is True.
        decoder_pos_encoding_type: Type of positional encoding to use for decoder.
            Only used if use_different_encoder_decoder_pos is True.
        rngs: Random number generators.


    Returns:
        Tuple of (encoder, decoder)
    """
    if use_different_encoder_decoder_pos:
        if encoder_pos_encoding_type is None or decoder_pos_encoding_type is None:
            raise ValueError(
                "encoder_pos_encoding_type and decoder_pos_encoding_type must be provided "
                "when use_different_encoder_decoder_pos is True"
            )
        encoder_pos_type = encoder_pos_encoding_type
        decoder_pos_type = decoder_pos_encoding_type
    else:
        encoder_pos_type = pos_encoding_type
        decoder_pos_type = pos_encoding_type

    encoder = TransformerEncoder(
        num_layers=num_encoder_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        normalize_qk=normalize_qk,
        broadcast_dropout=broadcast_dropout,
        max_len=max_len,
        pos_encoding_type=encoder_pos_type,
        pos_encoding_base=pos_encoding_base,
        rngs=rngs,
    )

    decoder = TransformerDecoder(
        num_layers=num_decoder_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        normalize_qk=normalize_qk,
        broadcast_dropout=broadcast_dropout,
        max_len=max_len,
        pos_encoding_type=decoder_pos_type,
        pos_encoding_base=pos_encoding_base,
        rngs=rngs,
    )

    return encoder, decoder
