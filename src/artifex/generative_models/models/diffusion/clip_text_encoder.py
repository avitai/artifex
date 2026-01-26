"""CLIP Text Encoder for Stable Diffusion.

This module implements a simplified CLIP-like text encoder using Flax NNX,
compatible with the Artifex framework.
"""

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.transformers import TransformerEncoder


class CLIPTextEncoder(nnx.Module):
    """CLIP-like text encoder for Stable Diffusion.

    This encoder converts token IDs to contextualized text embeddings using
    a transformer architecture similar to CLIP.

    Architecture:
        1. Token embedding layer
        2. Learned positional embeddings
        3. Stack of transformer encoder blocks
        4. Final layer normalization

    Args:
        vocab_size: Size of the vocabulary
        max_length: Maximum sequence length
        embedding_dim: Dimension of token and positional embeddings
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads in each transformer layer
        rngs: Random number generators

    Example:
        >>> import jax
        >>> from flax import nnx
        >>> rngs = nnx.Rngs(0)
        >>> encoder = CLIPTextEncoder(
        ...     vocab_size=49408,
        ...     max_length=77,
        ...     embedding_dim=768,
        ...     num_layers=12,
        ...     num_heads=12,
        ...     rngs=rngs,
        ... )
        >>> token_ids = jax.random.randint(jax.random.key(0), (2, 77), 0, 49408)
        >>> embeddings = encoder(token_ids)
        >>> embeddings.shape
        (2, 77, 768)
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        d_ff: int | None = None,
    ):
        """Initialize the CLIP text encoder.

        Args:
            vocab_size: Size of the vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            rngs: Random number generators
            dropout_rate: Dropout rate (default: 0.0)
            d_ff: Feed-forward dimension (default: 4 * embedding_dim)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Feed-forward ratio (default is 4x for transformers)
        mlp_ratio = d_ff / embedding_dim if d_ff is not None else 4.0

        # Token embedding layer
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embedding_dim,
            rngs=rngs,
        )

        # Learned positional embeddings (CLIP uses learned, not sinusoidal)
        self.positional_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (max_length, embedding_dim))
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            hidden_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=dropout_rate,
            use_bias=True,
            normalize_qk=False,
            broadcast_dropout=True,
            max_len=max_length,
            pos_encoding_type="none",  # We add positional embeddings manually
            rngs=rngs,
        )

        # Final layer normalization
        self.ln_final = nnx.LayerNorm(
            num_features=embedding_dim,
            epsilon=1e-5,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        *,
        attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Encode text tokens to embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
                           1 for real tokens, 0 for padding

        Returns:
            Text embeddings [batch_size, seq_len, embedding_dim]
        """
        seq_len = input_ids.shape[1]

        # Get token embeddings: [batch_size, seq_len, embedding_dim]
        token_emb = self.token_embedding(input_ids)

        # Add positional embeddings: [seq_len, embedding_dim]
        pos_emb = self.positional_embedding.value[:seq_len, :]

        # Combine: [batch_size, seq_len, embedding_dim]
        x = token_emb + pos_emb[None, :, :]

        # Create attention mask for transformer if provided
        if attention_mask is not None:
            # Convert to the format expected by transformer
            # [batch_size, 1, 1, seq_len] for broadcasting
            attn_mask = attention_mask[:, None, None, :]
            # Convert 0/1 mask to attention bias
            # 0 (padding) -> -1e9 (ignore), 1 (real) -> 0 (attend)
            attn_mask = (1.0 - attn_mask) * -1e9
        else:
            attn_mask = None

        # Apply transformer encoder
        x = self.transformer(x, mask=attn_mask)

        # Final layer normalization
        x = self.ln_final(x)

        # Apply attention mask to zero out padding positions
        if attention_mask is not None:
            # Expand mask to match embedding dimension
            mask = attention_mask[:, :, None]  # [batch_size, seq_len, 1]
            x = x * mask

        return x
