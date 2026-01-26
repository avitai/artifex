"""Causal utilities for autoregressive models.

This module provides utility functions commonly used in autoregressive models,
particularly for creating causal masks and shifting sequences.
"""

import jax
import jax.numpy as jnp


def create_causal_mask(sequence_length: int) -> jax.Array:
    """Create causal (autoregressive) attention mask.

    Creates a lower triangular mask that prevents attention to future positions,
    maintaining the autoregressive property during training and inference.

    Args:
        sequence_length: Length of the sequence

    Returns:
        Causal mask [sequence_length, sequence_length] where True = attend, False = mask

    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        [[ True False False False]
         [ True  True False False]
         [ True  True  True False]
         [ True  True  True  True]]
    """
    return jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))


def shift_right(x: jax.Array, axis: int = -1, pad_value: int = 0) -> jax.Array:
    """Shift sequence right by one position for autoregressive training.

    This function shifts sequences to the right, adding padding at the beginning.
    This is commonly used to create teacher-forcing inputs for autoregressive training
    where the model predicts the next token given previous tokens.

    Args:
        x: Input sequence tensor
        axis: Axis to shift along (default: -1, last axis)
        pad_value: Value to use for padding (default: 0)

    Returns:
        Right-shifted sequence with padding at the beginning

    Example:
        >>> x = jnp.array([1, 2, 3, 4])
        >>> shifted = shift_right(x)
        >>> print(shifted)
        [0 1 2 3]
    """
    # Create padding specification
    padding = [(0, 0)] * x.ndim
    padding[axis] = (1, 0)

    # Pad the array
    shifted = jnp.pad(x, padding, mode="constant", constant_values=pad_value)

    # Remove the last element along the shifted axis to maintain original shape
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, -1)

    return shifted[tuple(slices)]


def create_attention_mask(
    query_length: int, key_length: int, causal: bool = True, mask_value: float = -1e9
) -> jax.Array:
    """Create attention mask with optional causal masking.

    Args:
        query_length: Length of query sequence
        key_length: Length of key sequence
        causal: Whether to apply causal masking (default: True)
        mask_value: Value to use for masked positions (default: -1e9)

    Returns:
        Attention mask [query_length, key_length] with mask_value for masked positions
        and 0.0 for valid positions
    """
    if causal:
        # Create causal mask - can only attend to positions <= current position
        if query_length != key_length:
            raise ValueError(
                f"For causal masking, query_length ({query_length}) must equal "
                f"key_length ({key_length})"
            )

        mask = create_causal_mask(query_length)
        # Convert boolean mask to float with mask_value for False positions
        return jnp.where(mask, 0.0, mask_value)
    else:
        # No masking - all positions are valid
        return jnp.zeros((query_length, key_length))


def apply_causal_mask(logits: jax.Array, mask_value: float = -1e9) -> jax.Array:
    """Apply causal mask to logits tensor.

    Args:
        logits: Logits tensor [..., sequence_length, sequence_length]
        mask_value: Value to use for masked positions

    Returns:
        Masked logits with causal mask applied
    """
    *batch_dims, seq_len_q, seq_len_k = logits.shape

    if seq_len_q != seq_len_k:
        raise ValueError(
            f"For causal masking, sequence lengths must match. "
            f"Got query length {seq_len_q} and key length {seq_len_k}"
        )

    # Create causal mask
    causal_mask = create_causal_mask(seq_len_q)

    # Apply mask to logits
    return jnp.where(causal_mask, logits, mask_value)
