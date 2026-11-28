"""Cases for the distributions, ``artifex.generative_models.core.distributions``.

A case calls ``log_prob`` on a batch and, where the distribution has an analytic entropy,
``entropy()`` too, which runs the base class's entropy cache. The entropy has no batch axis, so
the case repeats it once per row: every output then has the leading batch axis the row check
indexes, and a row's copy is the entropy itself.

Each class has a second case, keyed ``<class>#sample``, that draws ``BATCH`` samples with the key
the checks choose: it owes ``output_follows_key`` in place of ``eval_deterministic``. A draw of
integers or one-hot vectors has no gradient in the parameters, so those cases do not claim one.
A case is one call, and ``log_prob`` and ``sample`` owe different checks, so one case cannot hold
both. The sampling case runs what the distributions' own tests do not: a draw under ``nnx.jit``
against eager, the tree-mode split, ``nnx.view`` against the modes, float64 finite differences
of a draw, and the same key giving the same draw after every key the module holds is changed.
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
        return continuous.Normal(_vector(0.1, -0.2, 0.3), _vector(1.0, 0.7, 1.3))

    logits = _vector(0.2, -0.1, 0.4, 0.0)
    return {
        f"{_DISTRIBUTIONS}.continuous.Normal": Case(
            build=gaussian,
            inputs=normal(_EVENT),
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.continuous.Beta": Case(
            build=lambda r: continuous.Beta(_vector(2.0, 1.5, 3.0), _vector(1.2, 2.5, 2.0)),
            inputs=_unit_interval,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.discrete.Bernoulli": Case(
            build=lambda r: discrete.Bernoulli(logits=_vector(0.3, -0.5, 1.0)),
            inputs=_binary,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.discrete.Categorical": Case(
            build=lambda r: discrete.Categorical(logits=logits),
            inputs=_classes,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.discrete.OneHotCategorical": Case(
            build=lambda r: discrete.OneHotCategorical(logits=logits),
            inputs=_one_hot,
            call=_density_and_entropy,
        ),
        f"{_DISTRIBUTIONS}.mixture.MixtureOfGaussians": Case(
            build=lambda r: mixture.MixtureOfGaussians(
                jnp.array([[0.0, 1.0, -1.0], [2.0, -0.5, 0.5]]),
                jnp.array([[1.0, 0.8, 1.2], [0.6, 1.0, 0.9]]),
                _vector(0.3, 0.7),
            ),
            inputs=normal(_EVENT),
            call=lambda module, x: module.log_prob(x),
        ),
        f"{_DISTRIBUTIONS}.mixture.Mixture": Case(
            build=lambda r: mixture.Mixture(
                [
                    continuous.Normal(_vector(0.0), _vector(1.0)),
                    continuous.Normal(_vector(1.5), _vector(0.5)),
                ],
                _vector(0.4, 0.6),
            ),
            inputs=normal(1),
            call=lambda module, x: module.log_prob(x),
        ),
        f"{_DISTRIBUTIONS}.transformations.AffineTransform": Case(
            build=lambda r: transformations.AffineTransform(
                gaussian(r), _vector(0.5, 0.0, -0.5), _vector(2.0, 1.0, 0.5)
            ),
            inputs=normal(_EVENT),
            call=lambda module, x: module.log_prob(x),
        ),
        f"{_DISTRIBUTIONS}.transformations.TransformedDistribution": Case(
            build=lambda r: transformations.TransformedDistribution(gaussian(r), distrax.Tanh()),
            inputs=uniform((_EVENT,), minval=-0.9, maxval=0.9),
            call=lambda module, x: module.log_prob(x),
        ),
    }


def _draw(module: nnx.Module, *, rngs: nnx.Rngs) -> PyTree:
    return module.sample((BATCH,), rngs=rngs)


def _no_inputs(key: jax.Array) -> tuple[jax.Array, ...]:  # noqa: ARG001  # a draw takes no array
    return ()


_INTEGER_DRAWS = ("Bernoulli", "Categorical", "OneHotCategorical")


def draw_cases() -> dict[str, Case]:
    """Return one sampling case per distribution, keyed ``<class>#sample``."""
    return {
        f"{name}#sample": Case(
            build=case.build,
            inputs=_no_inputs,
            call=_draw,
            is_rowwise=False,
            is_differentiable=not name.endswith(_INTEGER_DRAWS),
            rng_streams=("sample",),
            draws=True,
        )
        for name, case in cases().items()
    }


NOT_APPLICABLE: dict[tuple[str, str], str] = {
    (f"{_DISTRIBUTIONS}.continuous.Beta#sample", "gradient_matches_finite_differences"): (
        "a Beta draw is built from Gamma draws (distrax beta.py:90, jax.random.beta), whose"
        " gradient in the concentration is the implicit reparameterization gradient"
        " -(dF/da)/pdf (Figurnov et al. 2018), not the derivative of the sampler's arithmetic at"
        " a fixed key: in float64 jax's gradient of jax.random.gamma equals that formula to 8"
        " digits (0.88626226 at a=2) while finite differences at the same key give 0.89910824."
        " numpyro's finite-difference test excludes Beta's concentrations for the same reason"
        " (test_distributions.py, gamma_derived_params); the gradient is checked against the"
        " exact moment gradient in test_continuous_distributions.py::TestBetaSampleGradient"
    ),
}
"""Checks a sampling case does not owe, each with the reason."""

EXEMPT: dict[str, str] = {
    f"{_DISTRIBUTIONS}.base.Distribution": (
        "base class: it has no distribution of its own (_dist is None), so every method raises"
        " 'Distribution not initialized'; each subclass's case runs it"
    ),
}
