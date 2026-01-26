"""Utility functions for test fixtures.

This module provides utility functions for creating test fixtures
that can be used consistently across the test suite.
"""

import jax


def get_rng_key(seed: int = 42) -> jax.Array:
    """Create a fixed RNG key for tests.

    Args:
        seed: Random seed to use

    Returns:
        JAX random key
    """
    return jax.random.key(seed)


def get_rng_keys(seed: int = 42) -> dict[str, jax.Array]:
    """Create a dictionary of RNG keys for different purposes.

    Args:
        seed: Random seed to use

    Returns:
        Dictionary of JAX random keys for different purposes
    """
    main_key = jax.random.key(seed)
    keys = jax.random.split(main_key, 5)
    return {
        "init": keys[0],
        "params": keys[1],
        "dropout": keys[2],
        "sampling": keys[3],
        "data": keys[4],
    }


def get_standard_dims() -> dict[str, int]:
    """Get standard dimensions for tests.

    Returns:
        Dictionary of standard dimensions
    """
    return {
        "batch_size": 4,
        "seq_length": 16,
        "embed_dim": 32,
        "hidden_dim": 64,
        "num_heads": 4,
        "height": 32,
        "width": 32,
        "channels": 3,
    }


def get_image_sample(
    batch_size: int | None = None,
    height: int | None = None,
    width: int | None = None,
    channels: int | None = None,
    key: jax.Array | None = None,
) -> jax.Array:
    """Get a sample image tensor for tests.

    Args:
        batch_size: Batch size (default from standard_dims)
        height: Image height (default from standard_dims)
        width: Image width (default from standard_dims)
        channels: Number of channels (default from standard_dims)
        key: JAX random key (default: new key from seed 42)

    Returns:
        Sample image tensor with shape (batch_size, height, width, channels)
    """
    dims = get_standard_dims()
    batch_size = batch_size if batch_size is not None else dims["batch_size"]
    height = height if height is not None else dims["height"]
    width = width if width is not None else dims["width"]
    channels = channels if channels is not None else dims["channels"]

    if key is None:
        key = get_rng_key()

    return jax.random.normal(key, (batch_size, height, width, channels))


def get_sequence_sample(
    batch_size: int | None = None,
    seq_length: int | None = None,
    embed_dim: int | None = None,
    key: jax.Array | None = None,
) -> jax.Array:
    """Get a sample sequence tensor for tests.

    Args:
        batch_size: Batch size (default from standard_dims)
        seq_length: Sequence length (default from standard_dims)
        embed_dim: Embedding dimension (default from standard_dims)
        key: JAX random key (default: new key from seed 42)

    Returns:
        Sample sequence tensor with shape (batch_size, seq_length, embed_dim)
    """
    dims = get_standard_dims()
    batch_size = batch_size if batch_size is not None else dims["batch_size"]
    seq_length = seq_length if seq_length is not None else dims["seq_length"]
    embed_dim = embed_dim if embed_dim is not None else dims["embed_dim"]

    if key is None:
        key = get_rng_key()

    return jax.random.normal(key, (batch_size, seq_length, embed_dim))
