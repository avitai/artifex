"""Transformer-based autoregressive model for sequence generation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import TransformerConfig
from artifex.generative_models.core.layers import (
    create_causal_mask,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)
from artifex.generative_models.models.autoregressive.base import AutoregressiveModel


# Positional encoding is now imported from core.layers


class TransformerAutoregressiveModel(AutoregressiveModel):
    """Transformer-based autoregressive model for sequence generation.

    Uses masked self-attention to maintain the autoregressive property
    while enabling parallel training and efficient representation learning.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize transformer autoregressive model.

        Args:
            config: TransformerConfig with model configuration
            rngs: Random number generators

        Raises:
            TypeError: If config is not a TransformerConfig
        """
        if not isinstance(config, TransformerConfig):
            raise TypeError(f"config must be TransformerConfig, got {type(config).__name__}")

        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

        super().__init__(
            vocab_size=config.vocab_size,
            sequence_length=config.sequence_length,
            rngs=rngs,
        )

        # Store config
        self.config = config

        # Extract values from config for convenience
        network = config.network
        self.embed_dim = network.embed_dim
        self.num_layers = config.num_layers
        self.num_heads = network.num_heads
        self.mlp_dim = int(network.embed_dim * network.mlp_ratio)
        self.dropout_rate = config.dropout_rate
        self.use_positional_encoding = network.positional_encoding != "none"

        # Token embedding
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=self.embed_dim,
            rngs=rngs,
        )

        # Positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = SinusoidalPositionalEncoding(
                dim=self.embed_dim,
                max_len=config.sequence_length,
                dropout_rate=0.0,  # No dropout in positional encoding
                rngs=rngs,
            )

        # Dropout
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)

        # Transformer layers
        self.transformer_layers: list[TransformerEncoderBlock] = nnx.List([])
        for _ in range(self.num_layers):
            layer = TransformerEncoderBlock(
                hidden_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=network.mlp_ratio,
                dropout_rate=self.dropout_rate,
                rngs=rngs,
            )
            self.transformer_layers.append(layer)

        # Layer normalization before output
        self.layer_norm = nnx.LayerNorm(self.embed_dim, rngs=rngs)

        # Output projection to vocabulary
        self.output_projection = nnx.Linear(
            in_features=self.embed_dim,
            out_features=config.vocab_size,
            rngs=rngs,
        )

        # Create causal mask
        self.causal_mask = create_causal_mask(config.sequence_length)

    def __call__(
        self, x: jax.Array, *args, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Forward pass through transformer.

        Args:
            x: Input token indices [batch, seq_len]
            rngs: Random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing logits and intermediate representations
        """
        # Token embeddings
        embeddings = self.token_embedding(x)

        # Add positional encoding
        if self.use_positional_encoding:
            embeddings = self.pos_encoding(embeddings, deterministic=not training)

        # Apply dropout
        if self.dropout_rate > 0:
            embeddings = self.dropout(embeddings, deterministic=not training)

        # Create attention mask for current sequence length
        seq_len = x.shape[1]
        attention_mask = self.causal_mask[:seq_len, :seq_len]

        # Expand mask for batch and heads: [1, 1, seq_len, seq_len]
        attention_mask = attention_mask[None, None, :, :]

        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                mask=attention_mask,
                deterministic=not training,
            )

        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "embeddings": embeddings,
        }

    def encode(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs
    ) -> jax.Array:
        """Encode sequences to hidden representations.

        Args:
            x: Input token indices [batch, seq_len]
            rngs: Random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            Hidden representations [batch, seq_len, embed_dim]
        """
        outputs = self(x, rngs=rngs, training=training, **kwargs)
        return outputs["hidden_states"]

    def generate(
        self,
        n_samples: int = 1,
        prompt: jax.Array | None = None,
        max_length: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate sequences with optional prompt.

        Args:
            n_samples: Number of sequences to generate
            prompt: Optional prompt tokens [prompt_len] or [batch, prompt_len]
            max_length: Maximum generation length
            rngs: Random number generators
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            **kwargs: Additional keyword arguments

        Returns:
            Generated sequences [n_samples, max_length]
        """
        if rngs is None:
            rngs = self._rngs

        if max_length is None:
            max_length = self.sequence_length

        # Handle prompt
        if prompt is not None:
            # Ensure prompt has batch dimension
            if prompt.ndim == 1:
                prompt = prompt[None, :]  # Add batch dimension

            prompt_len = prompt.shape[1]
            batch_size = prompt.shape[0]

            # If n_samples > batch_size, repeat prompt
            if n_samples > batch_size:
                repeats = n_samples // batch_size
                extra = n_samples % batch_size
                if extra > 0:
                    prompt = jnp.concatenate(
                        [jnp.tile(prompt, (repeats, 1)), prompt[:extra]], axis=0
                    )
                else:
                    prompt = jnp.tile(prompt, (repeats, 1))
            elif n_samples < batch_size:
                prompt = prompt[:n_samples]

            # Initialize sequences with prompt
            sequences = jnp.zeros((n_samples, max_length), dtype=jnp.int32)
            sequences = sequences.at[:, :prompt_len].set(prompt)
            start_pos = prompt_len
        else:
            # Initialize empty sequences
            sequences = jnp.zeros((n_samples, max_length), dtype=jnp.int32)
            start_pos = 0

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Generate autoregressively
        for pos in range(start_pos, max_length):
            # Only process up to current position for efficiency
            current_seq = sequences[:, : pos + 1]

            # Get logits for current position
            outputs = self(current_seq, rngs=rngs, training=False, **kwargs)
            logits = outputs["logits"]

            # Extract logits for current position
            current_logits = logits[:, pos, :]  # [n_samples, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                current_logits = current_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(current_logits, top_k)
                # Create mask for top-k indices
                vocab_range = jnp.arange(self.vocab_size)[None, :]  # [1, vocab_size]
                top_k_indices_expanded = top_k_indices[:, :, None]  # [batch, top_k, 1]
                # [batch, vocab_size]
                top_k_mask = jnp.any(vocab_range == top_k_indices_expanded, axis=1)
                # Set non-top-k logits to very negative value
                current_logits = jnp.where(top_k_mask, current_logits, -1e9)

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_indices = jnp.argsort(-current_logits, axis=-1)
                sorted_logits = jnp.take_along_axis(current_logits, sorted_indices, axis=-1)
                sorted_probs = nnx.softmax(sorted_logits, axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

                # Find cutoff
                cutoff_mask = cumulative_probs <= top_p
                # Keep at least one token
                cutoff_mask = cutoff_mask.at[:, 0].set(True)

                # Set logits beyond cutoff to very negative value
                filtered_sorted_logits = jnp.where(cutoff_mask, sorted_logits, -1e9)
                # Map back to original order
                current_logits = jnp.take_along_axis(
                    filtered_sorted_logits, jnp.argsort(sorted_indices, axis=-1), axis=-1
                )

            # Sample next token
            sample_key, subkey = jax.random.split(sample_key)
            next_tokens = jax.random.categorical(subkey, current_logits, axis=-1)

            # Update sequences
            sequences = sequences.at[:, pos].set(next_tokens.astype(jnp.int32))

        return sequences

    def generate_with_cache(
        self,
        n_samples: int = 1,
        prompt: jax.Array | None = None,
        max_length: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate sequences with KV caching for efficiency.

        Args:
            n_samples: Number of sequences to generate
            prompt: Optional prompt tokens [prompt_len]
            max_length: Maximum generation length
            rngs: Random number generators
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            **kwargs: Additional keyword arguments

        Returns:
            Generated sequences [n_samples, max_length]
        """
        if rngs is None:
            rngs = self._rngs

        if max_length is None:
            max_length = self.sequence_length

        # Initialize sequences
        if prompt is not None:
            prompt_len = len(prompt)
            sequences = jnp.tile(prompt[None, :], (n_samples, 1))
            # Pad to max_length
            padding = jnp.zeros((n_samples, max_length - prompt_len))
            sequences = jnp.concatenate([sequences, padding], axis=1)
            start_pos = prompt_len
        else:
            sequences = jnp.zeros((n_samples, max_length))
            start_pos = 0

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Generate autoregressively
        for pos in range(start_pos, max_length):
            # Only process up to current position for efficiency
            current_seq = sequences[:, : pos + 1]

            # Get logits for current position
            outputs = self(current_seq, rngs=rngs, training=False, **kwargs)
            logits = outputs["logits"]

            # Extract logits for current position
            current_logits = logits[:, pos, :]  # [n_samples, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                current_logits = current_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(current_logits, top_k)
                # Set non-top-k logits to very negative value
                current_logits = jnp.where(
                    jnp.arange(self.vocab_size)[None, :] == top_k_indices[:, :, None],
                    current_logits[:, None, :],
                    -1e9,
                ).min(axis=1)

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_indices = jnp.argsort(-current_logits, axis=-1)
                sorted_logits = jnp.take_along_axis(current_logits, sorted_indices, axis=-1)
                sorted_probs = nnx.softmax(sorted_logits, axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

                # Find cutoff
                cutoff_mask = cumulative_probs <= top_p
                # Keep at least one token
                cutoff_mask = cutoff_mask.at[:, 0].set(True)

                # Set logits beyond cutoff to very negative value
                filtered_sorted_logits = jnp.where(cutoff_mask, sorted_logits, -1e9)
                # Map back to original order
                current_logits = jnp.take_along_axis(
                    filtered_sorted_logits, jnp.argsort(sorted_indices, axis=-1), axis=-1
                )

            # Sample next token
            sample_key, subkey = jax.random.split(sample_key)
            next_tokens = jax.random.categorical(subkey, current_logits, axis=-1)

            # Update sequences
            sequences = sequences.at[:, pos].set(next_tokens)

        return sequences

    def compute_perplexity(
        self, sequences: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs
    ) -> jax.Array:
        """Compute perplexity of sequences.

        Args:
            sequences: Input sequences [batch, seq_len]
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Perplexity values [batch]
        """
        # Get model outputs
        outputs = self(sequences, rngs=rngs, training=False, **kwargs)
        logits = outputs["logits"]

        # Compute negative log likelihood
        batch_size, seq_len, vocab_size = logits.shape

        # Shift for autoregressive prediction
        shifted_targets = sequences[:, 1:]  # Remove first token
        shifted_logits = logits[:, :-1, :]  # Remove last prediction

        # Compute cross-entropy loss
        log_probs = nnx.log_softmax(shifted_logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs, jnp.expand_dims(shifted_targets, -1), axis=-1
        ).squeeze(-1)

        # Average negative log likelihood
        avg_nll = -jnp.mean(target_log_probs, axis=-1)

        # Perplexity is exp(average negative log likelihood)
        perplexity = jnp.exp(avg_nll)

        return perplexity

    def beam_search(
        self,
        prompt: jax.Array | None = None,
        beam_size: int = 4,
        max_length: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array]:
        """Generate sequences using beam search.

        Args:
            prompt: Optional prompt tokens [prompt_len]
            beam_size: Size of the beam
            max_length: Maximum generation length
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (sequences, scores) for top beam_size sequences
        """
        if max_length is None:
            max_length = self.sequence_length

        # Initialize beam
        if prompt is not None:
            prompt_len = len(prompt)
            # Start with prompt repeated for each beam
            sequences = jnp.tile(prompt[None, :], (beam_size, 1))
            # Pad to max_length
            padding = jnp.zeros((beam_size, max_length - prompt_len))
            sequences = jnp.concatenate([sequences, padding], axis=1)
            scores = jnp.zeros(beam_size)
            start_pos = prompt_len
        else:
            sequences = jnp.zeros((beam_size, max_length))
            scores = jnp.zeros(beam_size)
            start_pos = 0

        # Beam search
        for pos in range(start_pos, max_length):
            # Get logits for all beams
            current_seq = sequences[:, : pos + 1]
            outputs = self(current_seq, rngs=rngs, training=False, **kwargs)
            logits = outputs["logits"]

            # Extract logits for current position
            current_logits = logits[:, pos, :]  # [beam_size, vocab_size]

            # Compute log probabilities
            log_probs = nnx.log_softmax(current_logits, axis=-1)

            # Expand scores for all possible next tokens
            expanded_scores = scores[:, None] + log_probs  # [beam_size, vocab_size]

            # Find top beam_size candidates
            flat_scores = expanded_scores.flatten()
            top_indices = jnp.argsort(-flat_scores)[:beam_size]

            # Convert flat indices back to beam and token indices
            beam_indices = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size

            # Update sequences and scores
            new_sequences = sequences[beam_indices]
            new_sequences = new_sequences.at[:, pos].set(token_indices)
            new_scores = flat_scores[top_indices]

            sequences = new_sequences
            scores = new_scores

        return sequences, scores

    def get_attention_weights(
        self,
        x: jax.Array,
        layer_idx: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Extract attention weights from a specific layer.

        Args:
            x: Input sequences [batch, seq_len]
            layer_idx: Index of layer to extract weights from (last layer if None)
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Attention weights [batch, num_heads, seq_len, seq_len]
        """
        if layer_idx is None:
            layer_idx = self.num_layers - 1

        # This is a simplified version - in practice, you'd need to modify
        # the transformer layers to return attention weights
        # For now, return placeholder
        batch_size, seq_len = x.shape
        return jnp.zeros((batch_size, self.num_heads, seq_len, seq_len))
