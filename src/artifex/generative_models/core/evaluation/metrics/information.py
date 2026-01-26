"""Information-theoretic metrics for evaluation across modalities."""

import jax.numpy as jnp


def compute_perplexity(total_log_prob: float, total_tokens: int) -> float:
    """Compute perplexity from total log probability and token count.

    Args:
        total_log_prob: Total log probability across all tokens
        total_tokens: Total number of tokens

    Returns:
        Perplexity score
    """
    if total_tokens == 0:
        return float("inf")

    avg_log_prob = total_log_prob / total_tokens
    perplexity = jnp.exp(-avg_log_prob)

    return float(perplexity)


def compute_average_log_probability(total_log_prob: float, total_tokens: int) -> float:
    """Compute average log probability from total log probability and token count.

    Args:
        total_log_prob: Total log probability across all tokens
        total_tokens: Total number of tokens

    Returns:
        Average log probability
    """
    if total_tokens == 0:
        return float("-inf")

    avg_log_prob = total_log_prob / total_tokens
    return float(avg_log_prob)
