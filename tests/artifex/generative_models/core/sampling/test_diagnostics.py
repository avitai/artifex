"""Chain diagnostics over BlackJAX's estimators."""

from __future__ import annotations

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

    # One chain's estimate is noisy where the chain is worth few draws: over eight seeds at
    # rho=0.99 (analytic 20.1) it spans 16.7 to 38.4. Eight pooled chains hold it to 24.7
    # there, and to within 1% at rho=0.9, which is what these tolerances allow.
    @pytest.mark.parametrize(("rho", "tolerance"), [(0.99, 0.3), (0.9, 0.1), (0.5, 0.1)])
    def test_it_follows_the_autocorrelation_time(self, rho: float, tolerance: float) -> None:
        length, chains = 4000, 8
        analytic = chains * length * (1.0 - rho) / (1.0 + rho)
        draws = jnp.stack([_ar1_chain(rho, length, seed=seed) for seed in range(chains)])

        ess = effective_sample_size(draws, chain_axis=0, sample_axis=1)

        assert float(ess) == pytest.approx(analytic, rel=tolerance)

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

    def test_a_chain_shorter_than_four_draws_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            effective_sample_size(jnp.zeros((3,)))
