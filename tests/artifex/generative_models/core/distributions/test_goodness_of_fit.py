"""A distribution's samples follow its density, and the distribution it is meant to be.

Two tests from two references, of different strength:

* **Samples against the distribution's own density**, as numpyro's ``test/test_gof.py`` does with
  ``numpyro.distributions.gof`` (a port of the ``goftests`` library): sorted samples' gaps, scaled
  by the density, are Exponential(1) in one dimension, and so is the density times the volume of
  each sample's nearest-neighbour ball in several (Bickel and Breiman 1983). Either is mapped to
  Uniform(0, 1) and binned for Pearson's chi-squared test, ``scipy.stats.chisquare`` here. It
  needs only ``sample`` and ``log_prob``, and it is coarse: with 10,000 draws of a unit normal it
  rejects a density 1.5 times too wide or shifted by 0.3, and passes one 1.2 times too wide or
  shifted by 0.1 (measured 2026-11-27).
* **Scalar samples against a reference CDF from scipy**, with ``scipy.stats.kstest`` as jax's
  tests call it (``tests/random_lax_test.py``, ``_CheckKolmogorovSmirnovCDF``). With the same
  draws it rejects a scale 1.2 times off.

A discrete distribution's counts are tested against its probabilities with Pearson's test.

A p-value below ``FAILURE_RATE`` fails. The keys are fixed, so a run is deterministic; the rate is
the chance that a correct sampler would fail one comparison with another key, numpyro's
``TEST_FAILURE_RATE``; over this file's 23 comparisons that is about 5e-4 for the file.
Each test has a control it must reject, chosen inside its measured resolution.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy import stats
from scipy.spatial import cKDTree

from artifex.generative_models.core.distributions import (
    AffineTransform,
    Bernoulli,
    Beta,
    Categorical,
    Distribution,
    Mixture,
    MixtureOfGaussians,
    Normal,
    OneHotCategorical,
    TransformedDistribution,
)


FAILURE_RATE = 2e-5
DRAWS = 10_000
MINIMUM_BINS = 7


def _uniform_p_value(uniform: np.ndarray) -> float:
    """Pearson's chi-squared p-value of samples that should be Uniform(0, 1), binned."""
    bins = round(len(uniform) ** (1 / 3))
    assert bins >= MINIMUM_BINS, "too few samples for the binned test"
    binned = np.clip((uniform * bins).astype(int), 0, bins - 1)
    return float(stats.chisquare(np.bincount(binned, minlength=bins)).pvalue)


def _scalar_p_value(samples: np.ndarray, density: np.ndarray) -> float:
    """Goodness of fit of scalar samples to a density (numpyro ``density_goodness_of_fit``)."""
    order = np.argsort(samples, kind="stable")
    samples, density = samples[order], density[order]
    sparsity = 0.5 * (1 / density[1:] + 1 / density[:-1])
    exponential = len(samples) * np.diff(samples) / sparsity
    return _uniform_p_value(np.exp(-exponential))


def _vector_p_value(samples: np.ndarray, density: np.ndarray) -> float:
    """The same for vectors (numpyro ``vector_density_goodness_of_fit``)."""
    dim = samples.shape[-1]
    assert len(samples) > 1000 * dim, "too few samples for the nearest-neighbour test"
    radii = cKDTree(samples).query(samples, k=2)[0][:, 1]
    volume = radii**dim * math.pi ** (0.5 * dim) / math.gamma(0.5 * dim + 1)
    return _uniform_p_value(np.exp(-len(samples) * density * volume))


def _draws_and_density(
    sampler: Distribution, density: Distribution, key: int
) -> tuple[np.ndarray, np.ndarray]:
    samples = sampler.sample((DRAWS,), rngs=nnx.Rngs(sample=jax.random.key(key)))
    probabilities = np.exp(np.asarray(density.log_prob(samples), dtype=np.float64))
    return np.asarray(samples, dtype=np.float64), probabilities


def _elementwise_p_values(sampler: Distribution, density: Distribution) -> list[float]:
    """One p-value per batch element of a distribution of scalars, each tested on its own."""
    samples, probabilities = _draws_and_density(sampler, density, key=0)
    samples, probabilities = samples.reshape(DRAWS, -1), probabilities.reshape(DRAWS, -1)
    return [
        _scalar_p_value(samples[:, column], probabilities[:, column])
        for column in range(samples.shape[1])
    ]


def _normal(scale: float = 1.0) -> Normal:
    return Normal(jnp.array([0.1, -0.2]), scale * jnp.array([1.0, 0.7]))


def _mixture(*weights: float) -> Mixture:
    components = [
        Normal(jnp.array([0.0]), jnp.array([1.0])),
        Normal(jnp.array([3.0]), jnp.array([0.5])),
    ]
    return Mixture(components, jnp.array(weights))


SCALAR_VALUED: dict[str, Callable[[], Distribution]] = {
    "Normal": _normal,
    "Beta": lambda: Beta(jnp.array([2.0, 1.5]), jnp.array([1.2, 2.5])),
    "Mixture": lambda: _mixture(0.3, 0.7),
    "AffineTransform": lambda: AffineTransform(_normal(), jnp.array([0.5, 0.0]), 2.0),
    "TransformedDistribution": lambda: TransformedDistribution(_normal(), distrax.Tanh()),
}


@pytest.mark.parametrize("name", SCALAR_VALUED)
def test_samples_follow_the_density(name: str) -> None:
    distribution = SCALAR_VALUED[name]()
    assert min(_elementwise_p_values(distribution, distribution)) > FAILURE_RATE


def test_vector_samples_follow_the_density() -> None:
    mixture = MixtureOfGaussians(
        jnp.array([[0.0, 1.0], [2.0, -0.5]]),
        jnp.array([[1.0, 0.8], [0.6, 1.0]]),
        jnp.array([0.3, 0.7]),
    )
    assert _vector_p_value(*_draws_and_density(mixture, mixture, key=0)) > FAILURE_RATE


REFERENCE_CDFS: dict[str, list[Callable[[np.ndarray], np.ndarray]]] = {
    "Normal": [stats.norm(0.1, 1.0).cdf, stats.norm(-0.2, 0.7).cdf],
    "Beta": [stats.beta(2.0, 1.2).cdf, stats.beta(1.5, 2.5).cdf],
    "Mixture": [lambda x: 0.3 * stats.norm(0.0, 1.0).cdf(x) + 0.7 * stats.norm(3.0, 0.5).cdf(x)],
    "AffineTransform": [stats.norm(0.7, 2.0).cdf, stats.norm(-0.4, 1.4).cdf],
    "TransformedDistribution": [
        lambda y: stats.norm(0.1, 1.0).cdf(np.arctanh(y)),
        lambda y: stats.norm(-0.2, 0.7).cdf(np.arctanh(y)),
    ],
}


def _reference_p_values(
    sampler: Distribution, cdfs: list[Callable[[np.ndarray], np.ndarray]]
) -> list[float]:
    samples, _ = _draws_and_density(sampler, sampler, key=0)
    samples = samples.reshape(DRAWS, -1)
    assert samples.shape[1] == len(cdfs)
    return [float(stats.kstest(samples[:, column], cdf).pvalue) for column, cdf in enumerate(cdfs)]


@pytest.mark.parametrize("name", SCALAR_VALUED)
def test_samples_follow_the_reference_distribution(name: str) -> None:
    assert min(_reference_p_values(SCALAR_VALUED[name](), REFERENCE_CDFS[name])) > FAILURE_RATE


def _count_p_value(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Pearson's test of counts against probabilities that sum to one up to float32 rounding.

    ``scipy.stats.chisquare`` requires the expected counts to sum to the observed ones within
    1.5e-8, tighter than a float32 ``log_prob`` gives, so the probabilities are normalised.
    """
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    return float(stats.chisquare(counts, DRAWS * probabilities / probabilities.sum()).pvalue)


def test_bernoulli_counts_follow_the_probabilities() -> None:
    bernoulli = Bernoulli(logits=jnp.array([0.3, -0.5]))
    samples = np.asarray(bernoulli.sample((DRAWS,), rngs=nnx.Rngs(sample=jax.random.key(0))))
    outcomes = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    probabilities = np.exp(np.asarray(bernoulli.log_prob(outcomes), dtype=np.float64))
    for column in range(samples.shape[1]):
        counts = np.bincount(samples[:, column].astype(int), minlength=2)
        assert _count_p_value(counts, probabilities[:, column]) > FAILURE_RATE


@pytest.mark.parametrize("one_hot", [False, True], ids=["Categorical", "OneHotCategorical"])
def test_categorical_counts_follow_the_probabilities(one_hot: bool) -> None:
    logits = jnp.array([0.2, -0.1, 0.4])
    distribution = OneHotCategorical(logits=logits) if one_hot else Categorical(logits=logits)
    samples = np.asarray(distribution.sample((DRAWS,), rngs=nnx.Rngs(sample=jax.random.key(0))))
    classes = samples.argmax(-1) if one_hot else samples
    outcomes = jnp.eye(3) if one_hot else jnp.arange(3)
    probabilities = np.exp(np.asarray(distribution.log_prob(outcomes), dtype=np.float64))
    assert _count_p_value(np.bincount(classes, minlength=3), probabilities) > FAILURE_RATE


def test_a_density_half_again_too_wide_is_rejected() -> None:
    """Control: the density test tells a sampler from a density 1.5 times too wide."""
    assert max(_elementwise_p_values(_normal(), _normal(scale=1.5))) < FAILURE_RATE


def test_a_reference_a_fifth_too_wide_is_rejected() -> None:
    """Control: the Kolmogorov-Smirnov test tells a sampler from a scale 1.2 times off."""
    too_wide = [stats.norm(0.1, 1.2).cdf, stats.norm(-0.2, 0.84).cdf]
    assert max(_reference_p_values(_normal(), too_wide)) < FAILURE_RATE


def test_a_wrong_categorical_probability_is_rejected() -> None:
    """Control: Pearson's test tells class counts from probabilities that are not their own."""
    samples = np.asarray(
        Categorical(logits=jnp.array([0.2, -0.1, 0.4])).sample(
            (DRAWS,), rngs=nnx.Rngs(sample=jax.random.key(0))
        )
    )
    uniform = np.full(3, 1 / 3)
    assert _count_p_value(np.bincount(samples, minlength=3), uniform) < FAILURE_RATE


def test_a_mixture_that_ignores_its_weights_is_rejected() -> None:
    """Control: samples chosen half and half do not fit a density weighted 0.3 to 0.7."""
    assert max(_elementwise_p_values(_mixture(0.5, 0.5), _mixture(0.3, 0.7))) < FAILURE_RATE
