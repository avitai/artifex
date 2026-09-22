"""Chain diagnostics over BlackJAX's estimators."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.sampling import effective_sample_size


def _ar1_chain(rho: float, length: int, *, seed: int = 0, shape: tuple[int, ...] = ()) -> jax.Array:
    """Chains whose integrated autocorrelation time is ``(1 + rho) / (1 - rho)``."""
    noise = jax.random.normal(jax.random.key(seed), (length, *shape))

    def step(previous: jax.Array, innovation: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = rho * previous + innovation
        return current, current

    _, chain = jax.lax.scan(step, jnp.zeros(shape), noise)
    return chain


class TestEffectiveSampleSize:
    """One chain's effective sample size, and its shape."""

    # The estimate's relative standard deviation is about 2.5 to 3.1 / sqrt(ESS)
    # (Madras & Sokal 1988, through Wolff 2004, Comput. Phys. Commun. 156, 143, eq. 42),
    # so three standard deviations is 9 / sqrt(ESS). Measured over 60 replications of eight
    # pooled chains: 0.017 at rho=0 (ESS 32000), 0.037 at 0.5, 0.082 at 0.9, 0.175 at 0.99
    # (ESS 161), against the rule's 0.05, 0.08, 0.22 and 0.71.
    @pytest.mark.parametrize("rho", [0.99, 0.9, 0.5, 0.0])
    def test_it_follows_the_autocorrelation_time(self, rho: float) -> None:
        length, chains = 4000, 8
        analytic = chains * length * (1.0 - rho) / (1.0 + rho)
        three_sigma = 9.0 / math.sqrt(analytic)
        draws = jnp.stack([_ar1_chain(rho, length, seed=seed) for seed in range(chains)])

        ess = effective_sample_size(draws, chain_axis=0, sample_axis=1)

        assert float(ess) == pytest.approx(analytic, rel=three_sigma)

    def test_independent_draws_reach_about_the_chain_length(self) -> None:
        samples = jax.random.normal(jax.random.key(1), (2000,))

        assert float(effective_sample_size(samples)) > 1500

    def test_it_keeps_the_shape_of_one_sample(self) -> None:
        for shape in [(), (1,), (3,), (2, 4)]:
            samples = _ar1_chain(0.5, 256, shape=shape)

            assert effective_sample_size(samples).shape == shape

    def test_several_chains_are_pooled(self) -> None:
        chains = jnp.stack([_ar1_chain(0.5, 1000, seed=seed, shape=(3,)) for seed in range(4)])

        pooled = effective_sample_size(chains, chain_axis=0, sample_axis=1)
        single = effective_sample_size(chains[0])

        assert pooled.shape == (3,)
        assert jnp.all(pooled > single)

    def test_it_traces_under_jit(self) -> None:
        samples = _ar1_chain(0.5, 512, shape=(3,))

        assert jnp.allclose(jax.jit(effective_sample_size)(samples), effective_sample_size(samples))

    def test_an_antithetic_chain_is_worth_more_than_its_draws(self) -> None:
        """Alternating draws carry more than independent ones, up to the estimator's guard."""
        draws = 2000
        alternating = jnp.tile(jnp.asarray([-1.0, 1.0]), draws // 2)

        ess = float(effective_sample_size(alternating))

        assert ess > draws
        assert ess <= draws * math.log10(draws) * (1.0 + 1e-6)

    def test_draws_that_never_move_are_worth_nothing(self) -> None:
        for chain in (jnp.ones((2000,)), jnp.full((2000,), 1e-30)):
            assert float(effective_sample_size(chain)) == 0.0

    def test_a_component_holding_a_nan_is_undefined_and_leaves_its_neighbour_alone(
        self,
    ) -> None:
        contaminated = jnp.ones((2000,)).at[0].set(jnp.nan)
        draws = jnp.stack([contaminated, _ar1_chain(0.5, 2000)], axis=-1)

        ess = effective_sample_size(draws)

        assert bool(jnp.isnan(ess[0]))
        assert float(ess[1]) > 0.0

    def test_a_chain_shorter_than_four_draws_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            effective_sample_size(jnp.zeros((3,)))
