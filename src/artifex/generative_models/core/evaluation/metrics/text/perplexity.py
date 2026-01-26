"""Perplexity metric implementation using JAX and NNX.

Perplexity is a common evaluation metric for language models that measures
how well a probability model predicts a sample. Lower perplexity indicates
better prediction performance.
"""

from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from ..base import SequenceMetric


class Perplexity(SequenceMetric):
    """Computes the perplexity of a language model on a given dataset.

    Perplexity is defined as the exponentiated average negative
    log-likelihood of a sequence. It measures how well a probability
    model predicts a sample.

    Attributes:
        model: Function that returns log probabilities for the input
        batch_size: Batch size for processing samples
    """

    def __init__(
        self,
        model: Callable[[jax.Array], jax.Array] | None = None,
        batch_size: int = 32,
        name: str = "perplexity",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the Perplexity metric.

        Args:
            model: Function that returns log probabilities for the input
                If None, the metric expects log probabilities to be provided
                directly in the compute call
            batch_size: Batch size for processing samples
            name: Name of the metric
            rngs: Optional RNG state
        """
        super().__init__(name=name, batch_size=batch_size, rngs=rngs)
        self.model = model

    def compute_log_probs(self, inputs: jax.Array) -> jax.Array:
        """Compute log probabilities for the inputs using the model.

        Args:
            inputs: Input sequence tokens

        Returns:
            Log probabilities for each token in the sequences
        """
        if self.model is None:
            raise ValueError(
                "No model provided during initialization. "
                "Either provide a model or pass log_probs directly to compute."
            )

        batch_size = self.batch_size
        n_samples = inputs.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        log_probs = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = inputs[start_idx:end_idx]
            batch_log_probs = self.model(batch)
            log_probs.append(batch_log_probs)

        return jnp.concatenate(log_probs, axis=0)

    @staticmethod
    def calculate_perplexity(log_probs: jax.Array, mask: jax.Array | None = None) -> float:
        """Calculate perplexity from log probabilities.

        Args:
            log_probs: Log probabilities for each token
            mask: Optional mask to apply to log_probs
                If provided, only masked positions contribute to the perplexity

        Returns:
            Perplexity value
        """
        # If no mask provided, use all tokens
        if mask is None:
            mask = jnp.ones_like(log_probs)

        # Apply mask to log probs
        masked_log_probs = log_probs * mask

        # Calculate total log prob and number of tokens
        total_log_prob = jnp.sum(masked_log_probs)
        n_tokens = jnp.sum(mask)

        # Calculate average negative log likelihood
        nll = -total_log_prob / n_tokens

        # Perplexity is exp(nll)
        return float(jnp.exp(nll))

    def compute(
        self,
        inputs: jax.Array | None = None,
        log_probs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> dict[str, float]:
        """Compute perplexity for the given inputs or log probabilities.

        Args:
            inputs: Input sequence tokens
                Required if log_probs is not provided
            log_probs: Log probabilities for each token
                Required if inputs is not provided
            mask: Optional mask to apply to log_probs
                If provided, only masked positions contribute to the perplexity

        Returns:
            Dictionary containing the perplexity
        """
        # Check if either inputs or log_probs is provided
        if inputs is None and log_probs is None:
            raise ValueError("Either inputs or log_probs must be provided.")

        # If log_probs not provided, compute them using the model
        if log_probs is None:
            if inputs is not None:
                log_probs = self.compute_log_probs(inputs)
            else:
                raise ValueError("If log_probs is not provided, inputs must be provided.")

        # Calculate perplexity
        ppl = self.calculate_perplexity(log_probs, mask)

        return {self.name: ppl}
