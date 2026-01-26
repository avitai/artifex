"""Base class for autoregressive generative models."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


class AutoregressiveModel(GenerativeModel):
    """Base class for autoregressive generative models.

    Autoregressive models decompose the joint distribution using the chain rule:
    p(x1, x2, ..., xn) = ∏ p(xi | x1, ..., xi-1)

    This enables sequential generation while maintaining tractable likelihood computation.
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ) -> None:
        """Initialize autoregressive model.

        Args:
            vocab_size: Size of the vocabulary/output space
            sequence_length: Maximum sequence length
            rngs: Random number generators

            precision: Numerical precision for computations
        """
        super().__init__(
            rngs=rngs,
            precision=precision,
        )

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # Store RNGs for later use
        if rngs is not None:
            self._rngs = rngs
        else:
            self._rngs = nnx.Rngs(params=jax.random.key(0))

    def __call__(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Forward pass through the autoregressive model.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input sequence [batch_size, sequence_length] or [..., sequence_length]
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing model outputs including 'logits'
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples autoregressively.

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators
            max_length: Maximum generation length (uses sequence_length if None)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (only consider k most likely tokens)
            top_p: Top-p (nucleus) sampling (cumulative probability threshold)
            **kwargs: Additional keyword arguments

        Returns:
            Generated sequences [n_samples, max_length]
        """
        if rngs is None:
            rngs = self._rngs

        if max_length is None:
            max_length = self.sequence_length

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Initialize sequences (start with zeros or special tokens)
        sequences = jnp.zeros((n_samples, max_length), dtype=jnp.int32)

        # Generate autoregressively
        for pos in range(max_length):
            # Get logits for current position
            outputs = self(sequences, rngs=rngs, **kwargs)
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
                batch_size = current_logits.shape[0]
                vocab_range = jnp.arange(self.vocab_size)
                vocab_mask = jnp.broadcast_to(
                    vocab_range[None, None, :], (batch_size, top_k, self.vocab_size)
                )
                indices_mask = jnp.broadcast_to(
                    top_k_indices[:, :, None], (batch_size, top_k, self.vocab_size)
                )
                top_k_mask = jnp.any(vocab_mask == indices_mask, axis=1)
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

            # Update sequences - ensure integer type
            sequences = sequences.at[:, pos].set(next_tokens.astype(jnp.int32))

        return sequences

    def compute_loss(
        self, logits: jax.Array, targets: jax.Array, mask: jax.Array | None = None
    ) -> jax.Array:
        """Compute autoregressive loss (negative log likelihood).

        Args:
            logits: Model predictions [batch_size, sequence_length, vocab_size]
            targets: Target tokens [batch_size, sequence_length]
            mask: Optional mask [batch_size, sequence_length]

        Returns:
            Loss value
        """
        # Shift targets for autoregressive prediction
        # Predict token i given tokens 0..i-1
        shifted_targets = targets[:, 1:]  # Remove first token
        shifted_logits = logits[:, :-1, :]  # Remove last prediction

        # Compute cross-entropy loss
        loss = jnp.sum(
            -nnx.log_softmax(shifted_logits) * nnx.one_hot(shifted_targets, self.vocab_size),
            axis=-1,
        )  # [batch_size, sequence_length-1]

        # Apply mask if provided
        if mask is not None:
            shifted_mask = mask[:, 1:]  # Remove first mask position
            loss = loss * shifted_mask
            # Normalize by valid positions
            loss = jnp.sum(loss) / (jnp.sum(shifted_mask) + 1e-8)
        else:
            loss = jnp.mean(loss)

        return loss

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute loss for autoregressive model.

        Args:
            batch: Input batch containing sequences
            model_outputs: Model outputs from forward pass
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing loss and metrics
        """
        # Extract data from batch
        if isinstance(batch, dict):
            if "inputs" in batch:
                x = batch["inputs"]
            elif "x" in batch:
                x = batch["x"]
            elif "sequences" in batch:
                x = batch["sequences"]
            else:
                x = batch.get("targets", batch)
            mask = batch.get("mask", None)
        else:
            x = batch
            mask = None

        # Extract logits from model outputs
        logits = model_outputs["logits"]

        # Compute autoregressive loss
        loss = self.compute_loss(logits, x, mask)

        # Compute accuracy for the shifted prediction task
        shifted_targets = x[:, 1:]
        shifted_logits = logits[:, :-1, :]
        predictions = jnp.argmax(shifted_logits, axis=-1)

        if mask is not None:
            shifted_mask = mask[:, 1:]
            correct = (predictions == shifted_targets) * shifted_mask
            accuracy = jnp.sum(correct) / (jnp.sum(shifted_mask) + 1e-8)
        else:
            correct = predictions == shifted_targets
            accuracy = jnp.mean(correct)

        # Compute perplexity
        perplexity = jnp.exp(loss)

        return {
            "loss": loss,
            "nll_loss": loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
        }

    def log_prob(self, x: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs: Any) -> jax.Array:
        """Compute log probability of sequences.

        Args:
            x: Input sequences [batch_size, sequence_length]
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Log probabilities [batch_size]
        """
        # Get model outputs
        outputs = self(x, rngs=rngs, **kwargs)
        logits = outputs["logits"]

        # Compute log probabilities
        log_probs = nnx.log_softmax(logits, axis=-1)

        # Extract probabilities for actual tokens (shifted for autoregressive)
        shifted_targets = x[:, 1:]  # Remove first token
        shifted_log_probs = log_probs[:, :-1, :]  # Remove last prediction

        # Gather log probabilities for actual tokens
        token_log_probs = jnp.take_along_axis(
            shifted_log_probs, jnp.expand_dims(shifted_targets, -1), axis=-1
        ).squeeze(-1)  # [batch_size, sequence_length-1]

        # Sum over sequence length to get total log probability
        sequence_log_probs = jnp.sum(token_log_probs, axis=-1)

        return sequence_log_probs

    def sample_with_conditioning(
        self,
        conditioning: jax.Array,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples conditioned on a prefix.

        Args:
            conditioning: Conditioning prefix [batch_size, prefix_length]
            n_samples: Number of samples per conditioning
            rngs: Random number generators
            temperature: Sampling temperature
            **kwargs: Additional keyword arguments

        Returns:
            Generated sequences [batch_size * n_samples, total_length]
        """
        if rngs is None:
            rngs = self._rngs

        batch_size, prefix_length = conditioning.shape
        remaining_length = self.sequence_length - prefix_length

        if remaining_length <= 0:
            # If conditioning is already full length, just repeat it
            return jnp.tile(conditioning, (n_samples, 1))

        # Expand conditioning for multiple samples
        expanded_conditioning = jnp.tile(conditioning, (n_samples, 1))

        # Initialize full sequences with conditioning - ensure integer dtype
        sequences = jnp.zeros((batch_size * n_samples, self.sequence_length), dtype=jnp.int32)
        sequences = sequences.at[:, :prefix_length].set(expanded_conditioning.astype(jnp.int32))

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Generate remaining tokens
        for pos in range(prefix_length, self.sequence_length):
            # Get logits for current position
            outputs = self(sequences, rngs=rngs, **kwargs)
            logits = outputs["logits"]

            # Extract and temperature-scale logits for current position
            current_logits = logits[:, pos, :] / temperature

            # Sample next token
            sample_key, subkey = jax.random.split(sample_key)
            next_tokens = jax.random.categorical(subkey, current_logits, axis=-1)

            # Update sequences - ensure integer type
            sequences = sequences.at[:, pos].set(next_tokens.astype(jnp.int32))

        return sequences

    def _get_rng_key(
        self, rngs: nnx.Rngs | None, key_name: str, default_seed: int = 0
    ) -> jax.Array:
        """Get RNG key from rngs object.

        Args:
            rngs: Random number generators
            key_name: Name of the key to extract
            default_seed: Default seed if rngs is None

        Returns:
            JAX random key
        """
        if rngs is not None:
            # Try to get the specific key, then fallback to sample, then params
            try:
                # Split from the available RNG stream
                if hasattr(rngs, key_name):
                    return rngs.fork(key_name)
                elif hasattr(rngs, "sample"):
                    return rngs.fork("sample")
                elif hasattr(rngs, "params"):
                    return rngs.fork("params")
            except Exception:
                # Fallback to direct JAX key generation
                pass

        # Ensure we return a proper JAX PRNG key with correct shape
        return jax.random.key(default_seed)
