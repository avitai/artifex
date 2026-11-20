"""Inputs and builders the family modules share."""

from __future__ import annotations

from collections.abc import Callable

import jax


BATCH = 4


def normal(*shape: int) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Inputs of one standard normal of ``shape``, with a leading batch axis of ``BATCH``."""
    return lambda key: (jax.random.normal(key, (BATCH, *shape)),)


def arrays(*shapes: tuple[int, ...]) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Inputs of standard normals, one per shape, each with a leading batch axis of ``BATCH``."""

    def inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        keys = jax.random.split(key, len(shapes))
        return tuple(
            jax.random.normal(k, (BATCH, *shape)) for k, shape in zip(keys, shapes, strict=True)
        )

    return inputs


def labels(key: jax.Array, num_classes: int) -> jax.Array:
    """Integer class labels in ``[0, num_classes)``, one per batch row."""
    return jax.random.randint(key, (BATCH,), 0, num_classes)


def timesteps(*shapes: tuple[int, ...]) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Normal inputs of ``shapes``, then timesteps in [0, 1000)."""

    def inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        k1, k2 = jax.random.split(key)
        return (*arrays(*shapes)(k1), jax.random.uniform(k2, (BATCH,), maxval=1000.0))

    return inputs
