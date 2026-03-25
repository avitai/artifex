"""Shared RNG helpers for explicit runtime ownership."""

from __future__ import annotations

import jax
from flax import nnx


def _stream_phrase(streams: tuple[str, ...]) -> str:
    if len(streams) == 1:
        return streams[0]
    if len(streams) == 2:
        return f"{streams[0]} or {streams[1]}"
    return ", ".join(streams)


def extract_rng_key(
    rng: jax.Array | nnx.Rngs | None,
    *,
    streams: tuple[str, ...] = ("sample", "default"),
    context: str = "sampling",
) -> jax.Array:
    """Return a concrete JAX key from an explicit RNG owner."""
    if rng is None:
        raise ValueError("rngs must be provided for sampling")

    if isinstance(rng, nnx.Rngs):
        for stream in streams:
            if stream in rng:
                return getattr(rng, stream)()

        phrase = _stream_phrase(streams)
        raise ValueError(f"{context} requires an nnx.Rngs object with a {phrase} stream")

    return rng
