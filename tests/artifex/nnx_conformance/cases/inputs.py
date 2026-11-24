"""Inputs and builders the family modules share."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.typing import PyTree


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


def integers(low: int, high: int, *shape: int) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Inputs of one integer array of ``shape`` in ``[low, high)``, with a batch axis of ``BATCH``."""
    return lambda key: (jax.random.randint(key, (BATCH, *shape), low, high),)


def uniform(
    *shapes: tuple[int, ...], minval: float = 0.0, maxval: float = 1.0
) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Inputs of uniforms in ``[minval, maxval)``, one per shape, with a batch axis of ``BATCH``."""

    def inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        keys = jax.random.split(key, len(shapes))
        return tuple(
            jax.random.uniform(k, (BATCH, *shape), minval=minval, maxval=maxval)
            for k, shape in zip(keys, shapes, strict=True)
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


def each_row(call: Callable[..., PyTree]) -> Callable[..., PyTree]:
    """Apply a call that takes one sample to every row, stacking the outputs on a new batch axis.

    For a module whose callers pass one sample at a time. A leaf that is not an array (a string,
    a host float) is kept as the list of its per-row values, so the checks still report it.
    Keywords such as ``rngs`` go to every row's call; one ``nnx.Rngs`` advances from row to row,
    so each row draws its own keys, as separate calls would.
    """

    def rows(module: nnx.Module, *inputs: jax.Array, **keywords: object) -> PyTree:
        outputs = [
            call(module, *(x[i] for x in inputs), **keywords) for i in range(inputs[0].shape[0])
        ]

        def stack(*leaves: object) -> object:
            if all(isinstance(leaf, jax.Array) for leaf in leaves):
                return jnp.stack(leaves)
            return list(leaves)

        return jax.tree.map(stack, *outputs)

    return rows
