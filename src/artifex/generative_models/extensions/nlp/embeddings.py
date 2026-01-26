"""Text embedding utilities for generation tasks.

This module provides text embedding and representation learning utilities
for generative models working with textual data, including modern positional
encoding methods like RoPE (Rotary Position Embeddings) and sinusoidal encodings.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import ModelExtension


# =============================================================================
# Rotary Position Embeddings (RoPE) Utility Functions
# =============================================================================


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Precompute rotary position embedding frequencies.

    RoPE applies rotation to embedding vectors based on their position,
    enabling the model to learn relative position information.

    Args:
        dim: Dimension of the embeddings (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base for the exponential frequency scaling.
        dtype: Data type for the output arrays.

    Returns:
        Tuple of (freqs_sin, freqs_cos) arrays of shape [max_seq_len, dim//2].
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for RoPE, got {dim}")

    # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim//2)
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))

    # Create position indices
    positions = jnp.arange(max_seq_len, dtype=dtype)

    # Compute outer product: [max_seq_len, dim//2]
    freqs = jnp.outer(positions, inv_freq)

    # Precompute sin and cos
    freqs_sin = jnp.sin(freqs)
    freqs_cos = jnp.cos(freqs)

    return freqs_sin, freqs_cos


def apply_rope(
    x: jax.Array,
    freqs_sin: jax.Array,
    freqs_cos: jax.Array,
) -> jax.Array:
    """Apply rotary position embeddings to input tensor.

    The rotation is applied by splitting the embedding dimension in half
    and applying 2D rotation matrices based on position.

    Args:
        x: Input tensor of shape [..., seq_len, dim].
        freqs_sin: Sine frequencies [seq_len, dim//2] or broadcastable.
        freqs_cos: Cosine frequencies [seq_len, dim//2] or broadcastable.

    Returns:
        Rotated tensor of same shape as input.
    """
    # Split x into two halves along the last dimension
    x_shape = x.shape
    dim = x_shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]

    # Get sequence length from input
    seq_len = x_shape[-2]

    # Slice freqs to match sequence length
    freqs_sin = freqs_sin[:seq_len]
    freqs_cos = freqs_cos[:seq_len]

    # Broadcast freqs to match x dimensions
    # x has shape [..., seq_len, dim//2], freqs has shape [seq_len, dim//2]
    # We need to add dimensions for batch if present
    while freqs_sin.ndim < x1.ndim:
        freqs_sin = freqs_sin[None, ...]
        freqs_cos = freqs_cos[None, ...]

    # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    rotated_x1 = x1 * freqs_cos - x2 * freqs_sin
    rotated_x2 = x1 * freqs_sin + x2 * freqs_cos

    # Concatenate back
    return jnp.concatenate([rotated_x1, rotated_x2], axis=-1)


# =============================================================================
# Sinusoidal Positional Encoding Utility Functions
# =============================================================================


def create_sinusoidal_positions(
    max_seq_len: int,
    dim: int,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Create sinusoidal positional encodings as in the original Transformer paper.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_seq_len: Maximum sequence length.
        dim: Embedding dimension.
        base: Base for the exponential frequency scaling.
        dtype: Data type for the output array.

    Returns:
        Positional encodings of shape [max_seq_len, dim].
    """
    positions = jnp.arange(max_seq_len, dtype=dtype)[:, None]
    dims = jnp.arange(dim, dtype=dtype)[None, :]

    # Compute the scaling factors
    angles = positions / jnp.power(base, (2 * (dims // 2)) / dim)

    # Apply sin to even indices, cos to odd indices
    sin_mask = (dims % 2 == 0).astype(dtype)
    cos_mask = 1 - sin_mask

    encodings = jnp.sin(angles) * sin_mask + jnp.cos(angles) * cos_mask

    return encodings


class TextEmbeddings(ModelExtension):
    """Text embedding utilities for generation tasks."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize text embeddings module.

        Args:
            config: Extension configuration with embedding parameters:
                - weight: Weight for the extension (default: 1.0)
                - enabled: Whether the extension is enabled (default: True)
                - extensions.embeddings.embedding_dim: Dimension of embeddings
                - extensions.embeddings.vocab_size: Size of vocabulary
                - extensions.embeddings.max_position_embeddings: Maximum number of positions
                - extensions.embeddings.dropout_rate: Dropout rate for embeddings
                - extensions.embeddings.use_position_embeddings: Whether to use positional
                  embeddings
            rngs: Random number generator keys
        """
        # Handle configuration
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get embedding parameters from extensions field
        embedding_params = getattr(config, "extensions", {}).get("embeddings", {})

        self.embedding_dim = embedding_params.get("embedding_dim", 512)
        self.vocab_size = embedding_params.get("vocab_size", 50000)
        self.max_position_embeddings = embedding_params.get("max_position_embeddings", 512)
        self.dropout_rate = embedding_params.get("dropout_rate", 0.1)
        self.use_position_embeddings = embedding_params.get("use_position_embeddings", True)
        self.rngs = rngs

        # Token embeddings
        self.token_embedding = nnx.Embed(
            num_embeddings=self.vocab_size, features=self.embedding_dim, rngs=self.rngs
        )

        # Position embeddings
        if self.use_position_embeddings:
            self.position_embedding = nnx.Embed(
                num_embeddings=self.max_position_embeddings,
                features=self.embedding_dim,
                rngs=self.rngs,
            )

        # Dropout
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=self.rngs)
        else:
            self.dropout = None

    def embed(
        self,
        tokens: jax.Array,
        position_ids: jax.Array | None = None,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Embed token sequences.

        Args:
            tokens: Token sequences [batch_size, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            deterministic: Whether to disable dropout

        Returns:
            Embedded sequences [batch_size, seq_length, embedding_dim]
        """
        # Token embeddings
        embeddings = self.token_embedding(tokens)

        # Add position embeddings
        if self.use_position_embeddings:
            if position_ids is None:
                seq_length = tokens.shape[-1]
                position_ids = jnp.arange(seq_length)[None, :]  # [1, seq_length]
                position_ids = jnp.broadcast_to(position_ids, tokens.shape)

            position_embeddings = self.position_embedding(position_ids)
            embeddings = embeddings + position_embeddings

        # Apply dropout
        if self.dropout is not None:
            embeddings = self.dropout(embeddings, deterministic=deterministic)

        return embeddings

    def get_token_embeddings(self, token_ids: jax.Array) -> jax.Array:
        """Get embeddings for specific tokens.

        Args:
            token_ids: Token IDs [num_tokens]

        Returns:
            Token embeddings [num_tokens, embedding_dim]
        """
        return self.token_embedding(token_ids)

    def compute_similarity(
        self, embeddings_a: jax.Array, embeddings_b: jax.Array, similarity_type: str = "cosine"
    ) -> jax.Array:
        """Compute similarity between embeddings.

        Args:
            embeddings_a: First set of embeddings [..., embedding_dim]
            embeddings_b: Second set of embeddings [..., embedding_dim]
            similarity_type: Type of similarity ("cosine", "dot", "euclidean")

        Returns:
            Similarity scores
        """
        if similarity_type == "cosine":
            # Cosine similarity
            norm_a = jnp.linalg.norm(embeddings_a, axis=-1, keepdims=True)
            norm_b = jnp.linalg.norm(embeddings_b, axis=-1, keepdims=True)

            normalized_a = embeddings_a / (norm_a + 1e-8)
            normalized_b = embeddings_b / (norm_b + 1e-8)

            similarity = jnp.sum(normalized_a * normalized_b, axis=-1)

        elif similarity_type == "dot":
            # Dot product similarity
            similarity = jnp.sum(embeddings_a * embeddings_b, axis=-1)

        elif similarity_type == "euclidean":
            # Negative Euclidean distance (higher = more similar)
            distance = jnp.linalg.norm(embeddings_a - embeddings_b, axis=-1)
            similarity = -distance

        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")

        return similarity

    def create_contextual_embeddings(self, tokens: jax.Array, context_window: int = 5) -> jax.Array:
        """Create context-aware embeddings using local averaging.

        Args:
            tokens: Token sequences [batch_size, seq_length]
            context_window: Size of context window for averaging

        Returns:
            Contextual embeddings [batch_size, seq_length, embedding_dim]
        """
        # Get base token embeddings
        token_embeddings = self.token_embedding(tokens)

        # Vectorized sliding window average using 1D convolution
        # Pad with reflect to handle boundaries like the original loop
        pad_left = context_window // 2
        pad_right = context_window // 2
        # Pad along sequence dimension
        padded = jnp.pad(token_embeddings, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge")

        # 1D average pooling via depthwise convolution
        # Reshape for lax.conv: (batch, embedding_dim, seq_length)
        embed_dim = token_embeddings.shape[-1]
        padded_t = jnp.transpose(padded, (0, 2, 1))  # (batch, embed_dim, padded_seq)
        kernel = jnp.ones((embed_dim, 1, context_window)) / context_window
        contextual_t = jax.lax.conv_general_dilated(
            padded_t,
            kernel,
            window_strides=(1,),
            padding="VALID",
            feature_group_count=embed_dim,
        )
        contextual = jnp.transpose(contextual_t, (0, 2, 1))  # (batch, seq, embed_dim)

        return contextual

    def project_to_vocabulary(self, embeddings: jax.Array) -> jax.Array:
        """Project embeddings back to vocabulary space.

        Args:
            embeddings: Embeddings [..., embedding_dim]

        Returns:
            Vocabulary logits [..., vocab_size]
        """
        # Use token embedding weights as projection matrix
        vocab_weights = self.token_embedding.embedding  # [vocab_size, embedding_dim]

        # Compute dot product with vocabulary
        logits = jnp.dot(embeddings, vocab_weights.T)

        return logits

    def extract_sentence_embedding(
        self,
        token_embeddings: jax.Array,
        attention_mask: jax.Array | None = None,
        pooling_method: str = "mean",
    ) -> jax.Array:
        """Extract sentence-level embeddings from token embeddings.

        Args:
            token_embeddings: Token embeddings [batch_size, seq_length, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_length]
            pooling_method: Pooling method ("mean", "max", "cls", "last")

        Returns:
            Sentence embeddings [batch_size, embedding_dim]
        """
        if pooling_method == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                masked_embeddings = token_embeddings * attention_mask[..., None]
                sum_embeddings = jnp.sum(masked_embeddings, axis=1)
                valid_tokens = jnp.sum(attention_mask, axis=1, keepdims=True)
                sentence_embedding = sum_embeddings / (valid_tokens + 1e-8)
            else:
                # Simple mean pooling
                sentence_embedding = jnp.mean(token_embeddings, axis=1)

        elif pooling_method == "max":
            # Max pooling
            if attention_mask is not None:
                # Set masked positions to large negative value before max
                masked_embeddings = token_embeddings + (1 - attention_mask[..., None]) * (-1e9)
                sentence_embedding = jnp.max(masked_embeddings, axis=1)
            else:
                sentence_embedding = jnp.max(token_embeddings, axis=1)

        elif pooling_method == "cls":
            # Use first token (CLS token) embedding
            sentence_embedding = token_embeddings[:, 0, :]

        elif pooling_method == "last":
            # Use last valid token embedding
            if attention_mask is not None:
                # Find last valid position for each sequence
                last_positions = jnp.sum(attention_mask, axis=1) - 1
                last_positions = jnp.clip(last_positions, 0, token_embeddings.shape[1] - 1)
                last_positions = last_positions.astype(jnp.int32)

                batch_indices = jnp.arange(token_embeddings.shape[0])
                sentence_embedding = token_embeddings[batch_indices, last_positions]
            else:
                # Use last token
                sentence_embedding = token_embeddings[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        return sentence_embedding

    def compute_attention_weights(
        self, query_embeddings: jax.Array, key_embeddings: jax.Array, temperature: float = 1.0
    ) -> jax.Array:
        """Compute attention weights between embeddings.

        Args:
            query_embeddings: Query embeddings [..., embedding_dim]
            key_embeddings: Key embeddings [..., embedding_dim]
            temperature: Temperature for softmax

        Returns:
            Attention weights [..., num_keys]
        """
        # Compute attention scores
        scores = jnp.dot(query_embeddings, key_embeddings.T) / temperature

        # Apply softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)

        return attention_weights

    def interpolate_embeddings(
        self, embeddings_a: jax.Array, embeddings_b: jax.Array, alpha: float = 0.5
    ) -> jax.Array:
        """Interpolate between two embeddings.

        Args:
            embeddings_a: First embeddings
            embeddings_b: Second embeddings
            alpha: Interpolation factor (0 = all A, 1 = all B)

        Returns:
            Interpolated embeddings
        """
        return (1 - alpha) * embeddings_a + alpha * embeddings_b

    def apply_rope_embeddings(
        self,
        embeddings: jax.Array,
        base: float = 10000.0,
    ) -> jax.Array:
        """Apply Rotary Position Embeddings (RoPE) to input embeddings.

        RoPE encodes position information through rotation of embedding vectors,
        enabling models to learn relative position attention patterns. This is
        the de facto standard for modern LLMs like Llama 2.

        Args:
            embeddings: Input embeddings of shape [..., seq_len, dim].
                        The dim must be even.
            base: Base for the exponential frequency scaling (default: 10000.0).

        Returns:
            Embeddings with rotary position encoding applied, same shape as input.
        """
        seq_len = embeddings.shape[-2]
        dim = embeddings.shape[-1]

        # Precompute frequencies for this sequence
        freqs_sin, freqs_cos = precompute_rope_freqs(
            dim=dim,
            max_seq_len=seq_len,
            base=base,
            dtype=embeddings.dtype,
        )

        # Apply rotary embeddings
        return apply_rope(embeddings, freqs_sin, freqs_cos)

    def get_sinusoidal_embeddings(
        self,
        seq_len: int,
        dim: int | None = None,
        base: float = 10000.0,
    ) -> jax.Array:
        """Get sinusoidal positional embeddings.

        Creates fixed positional encodings using sine and cosine functions
        as described in the original Transformer paper "Attention is All You Need".

        Args:
            seq_len: Sequence length.
            dim: Embedding dimension (default: self.embedding_dim).
            base: Base for the exponential frequency scaling (default: 10000.0).

        Returns:
            Positional encodings of shape [seq_len, dim].
        """
        if dim is None:
            dim = self.embedding_dim

        return create_sinusoidal_positions(
            max_seq_len=seq_len,
            dim=dim,
            base=base,
        )

    def embed_with_sinusoidal_positions(
        self,
        tokens: jax.Array,
        *,
        deterministic: bool = False,
        base: float = 10000.0,
    ) -> jax.Array:
        """Embed tokens with fixed sinusoidal positional encodings.

        This is an alternative to learned position embeddings that uses
        the fixed sinusoidal encodings from the original Transformer.

        Args:
            tokens: Token sequences [batch_size, seq_length].
            deterministic: Whether to disable dropout.
            base: Base for the exponential frequency scaling.

        Returns:
            Embedded sequences [batch_size, seq_length, embedding_dim].
        """
        # Get token embeddings
        embeddings = self.token_embedding(tokens)

        # Get sinusoidal position embeddings
        seq_length = tokens.shape[-1]
        position_embeddings = self.get_sinusoidal_embeddings(
            seq_len=seq_length,
            dim=self.embedding_dim,
            base=base,
        )

        # Add position embeddings (broadcast over batch)
        embeddings = embeddings + position_embeddings

        # Apply dropout
        if self.dropout is not None:
            embeddings = self.dropout(embeddings, deterministic=deterministic)

        return embeddings

    def embed_with_rope(
        self,
        tokens: jax.Array,
        *,
        deterministic: bool = False,
        base: float = 10000.0,
    ) -> jax.Array:
        """Embed tokens with Rotary Position Embeddings (RoPE).

        RoPE applies position-dependent rotations to embedding vectors,
        enabling relative position awareness. This is the modern standard
        for LLMs like Llama 2.

        Args:
            tokens: Token sequences [batch_size, seq_length].
            deterministic: Whether to disable dropout.
            base: Base for the exponential frequency scaling.

        Returns:
            Embedded sequences with RoPE applied [batch_size, seq_length, embedding_dim].
        """
        # Get token embeddings
        embeddings = self.token_embedding(tokens)

        # Apply rotary position embeddings
        embeddings = self.apply_rope_embeddings(embeddings, base=base)

        # Apply dropout
        if self.dropout is not None:
            embeddings = self.dropout(embeddings, deterministic=deterministic)

        return embeddings

    def get_embedding_statistics(self, embeddings: jax.Array) -> dict[str, jax.Array]:
        """Compute statistics of embeddings.

        Args:
            embeddings: Input embeddings [..., embedding_dim]

        Returns:
            Dictionary of embedding statistics
        """
        return {
            "mean": jnp.mean(embeddings, axis=-1),
            "std": jnp.std(embeddings, axis=-1),
            "norm": jnp.linalg.norm(embeddings, axis=-1),
            "min": jnp.min(embeddings, axis=-1),
            "max": jnp.max(embeddings, axis=-1),
        }

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension outputs including embedding features.
        """
        if not self.enabled:
            return {"extension_type": "text_embeddings"}

        results = {"extension_type": "text_embeddings"}

        # Process input tokens if available
        if isinstance(inputs, dict) and "tokens" in inputs:
            tokens = inputs["tokens"]
            position_ids = inputs.get("position_ids")
            deterministic = kwargs.get("deterministic", False)

            # Embed tokens
            embeddings = self.embed(tokens, position_ids, deterministic=deterministic)
            results["input_embeddings"] = embeddings

            # Extract sentence embeddings if requested
            if kwargs.get("extract_sentence_embedding", False):
                attention_mask = inputs.get("attention_mask")
                pooling_method = kwargs.get("pooling_method", "mean")
                sentence_emb = self.extract_sentence_embedding(
                    embeddings, attention_mask, pooling_method
                )
                results["sentence_embeddings"] = sentence_emb

            # Compute embedding statistics
            if kwargs.get("compute_statistics", False):
                results["embedding_statistics"] = self.get_embedding_statistics(embeddings)

        # Process model output embeddings if available
        if isinstance(model_outputs, dict):
            if "embeddings" in model_outputs:
                output_embeddings = model_outputs["embeddings"]

                # Project to vocabulary if requested
                if kwargs.get("project_to_vocab", False):
                    vocab_logits = self.project_to_vocabulary(output_embeddings)
                    results["vocabulary_logits"] = vocab_logits

                # Compute attention weights if query embeddings provided
                if "query_embeddings" in kwargs:
                    query_embeddings = kwargs["query_embeddings"]
                    temperature = kwargs.get("temperature", 1.0)
                    attention_weights = self.compute_attention_weights(
                        query_embeddings, output_embeddings, temperature
                    )
                    results["attention_weights"] = attention_weights

            elif "hidden_states" in model_outputs:
                # Handle hidden states as embeddings
                hidden_states = model_outputs["hidden_states"]

                # Extract sentence embeddings from hidden states
                if kwargs.get("extract_sentence_embedding", False):
                    attention_mask = model_outputs.get("attention_mask")
                    pooling_method = kwargs.get("pooling_method", "mean")
                    sentence_emb = self.extract_sentence_embedding(
                        hidden_states, attention_mask, pooling_method
                    )
                    results["sentence_embeddings"] = sentence_emb

        return results
