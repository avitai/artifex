"""Cases for the distributions, ``artifex.generative_models.core.distributions``.

A case calls ``log_prob`` on a batch and, where the distribution has an analytic entropy,
``entropy()`` too, which runs the base class's entropy cache. The entropy has no batch axis, so
the case repeats it once per row: every output then has the leading batch axis the row check
indexes, and a row's copy is the entropy itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal, uniform


_DISTRIBUTIONS = "artifex.generative_models.core.distributions"
_EVENT, _CLASSES, _COMPONENTS = 3, 4, 2


def _density_and_entropy(module: nnx.Module, x: jax.Array) -> PyTree:
    entropy = module.entropy()
    return module.log_prob(x), jnp.broadcast_to(entropy, (x.shape[0], *entropy.shape))


def _unit_interval(key: jax.Array) -> tuple[jax.Array, ...]:
    return (jax.random.uniform(key, (BATCH, _EVENT), minval=0.05, maxval=0.95),)


def _binary(key: jax.Array) -> tuple[jax.Array, ...]:
    return (jax.random.bernoulli(key, shape=(BATCH, _EVENT)).astype(jnp.float32),)


def _classes(key: jax.Array) -> tuple[jax.Array, ...]:
    return (jax.random.randint(key, (BATCH,), 0, _CLASSES),)


def _one_hot(key: jax.Array) -> tuple[jax.Array, ...]:
    return (jax.nn.one_hot(jax.random.randint(key, (BATCH,), 0, _CLASSES), _CLASSES),)


def _vector(*values: float) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.float32)


def cases() -> dict[str, Case]:
    """Return the distribution cases, by the qualified name of the class each builds."""

    import distrax  # noqa: PLC0415

    from artifex.generative_models.core.distributions import (  # noqa: PLC0415
        continuous,
        discrete,
        mixture,
        transformations,
    )

    def gaussian(r: nnx.Rngs) -> nnx.Module:
        return continuous.Normal(_vector(0.1, -0.2, 0.3), _vector(1.0, 0.7, 1.3), rngs=r)

    logits = _vector(0.2, -0.1, 0.4, 0.0)
    return {
        f"{_DISTRIBUTIONS}.continuous.Normal": Case(
            build=gaussian,
            inputs=normal(_EVENT),
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.continuous.Beta": Case(
            build=lambda r: continuous.Beta(_vector(2.0, 1.5, 3.0), _vector(1.2, 2.5, 2.0), rngs=r),
            inputs=_unit_interval,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.discrete.Bernoulli": Case(
            build=lambda r: discrete.Bernoulli(logits=_vector(0.3, -0.5, 1.0), rngs=r),
            inputs=_binary,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.discrete.Categorical": Case(
            build=lambda r: discrete.Categorical(logits=logits, rngs=r),
            inputs=_classes,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.discrete.OneHotCategorical": Case(
            build=lambda r: discrete.OneHotCategorical(logits=logits, rngs=r),
            inputs=_one_hot,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.mixture.MixtureOfGaussians": Case(
            build=lambda r: mixture.MixtureOfGaussians(
                jnp.array([[0.0, 1.0, -1.0], [2.0, -0.5, 0.5]]),
                jnp.array([[1.0, 0.8, 1.2], [0.6, 1.0, 0.9]]),
                _vector(0.3, 0.7),
                rngs=r,
            ),
            inputs=normal(_EVENT),
            call=lambda module, x: module.log_prob(x),
        ),
        f"{_DISTRIBUTIONS}.mixture.Mixture": Case(
            build=lambda r: mixture.Mixture(
                [
                    continuous.Normal(_vector(0.0), _vector(1.0), rngs=r),
                    continuous.Normal(_vector(1.5), _vector(0.5), rngs=r),
                ],
                _vector(0.4, 0.6),
                rngs=r,
            ),
            inputs=normal(1),
            call=lambda module, x: module.log_prob(x),
        ),
        f"{_DISTRIBUTIONS}.transformations.AffineTransform": Case(
            build=lambda r: transformations.AffineTransform(
                gaussian(r), _vector(0.5, 0.0, -0.5), _vector(2.0, 1.0, 0.5), rngs=r
            ),
            inputs=normal(_EVENT),
            call=lambda module, x: module.log_prob(x),
        ),
        f"{_DISTRIBUTIONS}.transformations.TransformedDistribution": Case(
            build=lambda r: transformations.TransformedDistribution(
                gaussian(r), distrax.Tanh(), rngs=r
            ),
            inputs=uniform((_EVENT,), minval=-0.9, maxval=0.9),
            call=lambda module, x: module.log_prob(x),
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_DISTRIBUTIONS}.base.Distribution": (
        "base class: it has no distribution of its own (_dist is None), so every method raises"
        " 'Distribution not initialized'; each subclass's case runs it"
    ),
}
