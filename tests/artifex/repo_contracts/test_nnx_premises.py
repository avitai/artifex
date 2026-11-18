"""The Flax and JAX behaviour artifex's code and contracts rely on, measured on what is installed.

Each test states one fact about the installed framework that a rule, a floor or a fix in this
repository depends on. ``upstream-compat.yml`` runs these against the newest jax and flax the
bounds admit, so a release that changes one fails here, naming the fact, before the code that
relies on it misbehaves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def test_causal_decode_attends_to_every_cached_token() -> None:
    """Causal decode one token at a time equals one causal pass over the whole sequence.

    flax 0.12.9 forwarded ``is_causal`` during decode, building ``tril((1, max_length))``, so
    every step after the first attended to cached position 0 alone; flax c3721a25 (0.12.11)
    stopped it, and ``pyproject.toml``'s flax floor rests on that release. Both paths compute
    the same products on the same values and differ only in how a five-term softmax sum is
    associated, so they agree to a few float32 roundings of the output scale; the defect is of
    the order of the output itself (an error of 3.08 on flax 0.12.9).
    """
    length, features = 5, 8
    x = jax.random.normal(jax.random.key(0), (1, length, features))
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))[None, None]

    def attention(*, decode: bool) -> nnx.MultiHeadAttention:
        return nnx.MultiHeadAttention(
            num_heads=2, in_features=features, decode=decode, deterministic=True, rngs=nnx.Rngs(0)
        )

    full = attention(decode=False)(x, mask=causal)
    stepper = attention(decode=True)
    stepper.init_cache((1, length, features))
    steps = jnp.concatenate(
        [stepper(x[:, t : t + 1], is_causal=True) for t in range(length)], axis=1
    )

    tolerance = 16 * jnp.finfo(full.dtype).eps * float(jnp.max(jnp.abs(full)))
    assert float(jnp.max(jnp.abs(steps - full))) <= tolerance
