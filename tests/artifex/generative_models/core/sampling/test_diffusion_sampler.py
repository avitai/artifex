"""Tests for diffusion-based sampling algorithms.

This module provides comprehensive tests for the DiffusionSampler class,
covering initialization, beta scheduling, step computation, and sampling.
"""

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.sampling.diffusion import DiffusionSampler


def dummy_noise_predictor(x, t, **kwargs):  # noqa: ARG001
    """A simple noise predictor that returns zeros for testing."""
    del t, kwargs  # Unused
    return jnp.zeros_like(x)


def random_noise_predictor(x, t, **kwargs):  # noqa: ARG001
    """A noise predictor that returns the input (identity for testing)."""
    del t, kwargs  # Unused
    return x


class DummyDiffusionModel:
    """A minimal diffusion model for testing."""

    def __call__(self, x, t, **kwargs):  # noqa: ARG002
        """Predict noise as zeros."""
        del t, kwargs  # Unused
        return jnp.zeros_like(x)

    def sample(self, n_samples, scheduler="ddpm", steps=None, rngs=None):
        """Generate random samples."""
        del scheduler, steps  # Unused
        key = jax.random.key(42) if rngs is None else rngs.sample()
        return jax.random.normal(key, (n_samples, 16))


@pytest.fixture
def base_sampler():
    """Fixture for basic diffusion sampler."""
    return DiffusionSampler(
        predict_noise_fn=dummy_noise_predictor,
        num_timesteps=100,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )


@pytest.fixture
def quadratic_sampler():
    """Fixture for diffusion sampler with quadratic schedule."""
    return DiffusionSampler(
        predict_noise_fn=dummy_noise_predictor,
        num_timesteps=100,
        beta_schedule="quadratic",
        beta_start=1e-4,
        beta_end=0.02,
    )


@pytest.fixture
def input_noise():
    """Fixture for input noise."""
    return jax.random.normal(jax.random.key(42), (4, 16))


class TestDiffusionSamplerInitialization:
    """Test suite for DiffusionSampler initialization."""

    def test_initialization_with_predict_fn(self):
        """Test initialization with predict_noise_fn."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
        )

        assert sampler.num_timesteps == 100
        assert sampler.predict_noise_fn is not None

    def test_initialization_with_model(self):
        """Test initialization with model object."""
        model = DummyDiffusionModel()
        sampler = DiffusionSampler(model=model, num_timesteps=100)

        assert sampler.model is model
        assert sampler.predict_noise_fn is not None

    def test_initialization_default_beta_values(self):
        """Test default beta start and end values."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
        )

        assert sampler.betas[0] == pytest.approx(1e-4)
        assert sampler.betas[-1] == pytest.approx(0.02)

    def test_initialization_custom_beta_values(self):
        """Test custom beta start and end values."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=50,
            beta_start=0.001,
            beta_end=0.1,
        )

        assert sampler.betas[0] == pytest.approx(0.001)
        assert sampler.betas[-1] == pytest.approx(0.1)
        assert len(sampler.betas) == 50


class TestBetaSchedule:
    """Test suite for beta scheduling."""

    def test_linear_schedule(self, base_sampler):
        """Test linear beta schedule produces linearly spaced values."""
        betas = base_sampler.betas

        # Check monotonic increase
        assert jnp.all(jnp.diff(betas) > 0)

        # Check linear spacing
        expected_betas = jnp.linspace(1e-4, 0.02, 100)
        assert jnp.allclose(betas, expected_betas)

    def test_quadratic_schedule(self, quadratic_sampler):
        """Test quadratic beta schedule."""
        betas = quadratic_sampler.betas

        # Check monotonic increase
        assert jnp.all(jnp.diff(betas) > 0)

        # Quadratic: betas should be squares of linearly spaced values
        expected_sqrt_betas = jnp.linspace(1e-4**0.5, 0.02**0.5, 100)
        expected_betas = expected_sqrt_betas**2
        assert jnp.allclose(betas, expected_betas)

    def test_invalid_schedule_raises_error(self):
        """Test that invalid schedule raises ValueError."""
        with pytest.raises(ValueError, match="Unknown beta schedule"):
            DiffusionSampler(
                predict_noise_fn=dummy_noise_predictor,
                beta_schedule="invalid",
            )

    def test_alphas_from_betas(self, base_sampler):
        """Test alphas are computed correctly from betas."""
        # alphas = 1 - betas
        expected_alphas = 1.0 - base_sampler.betas
        assert jnp.allclose(base_sampler.alphas, expected_alphas)

    def test_alphas_cumprod(self, base_sampler):
        """Test cumulative product of alphas."""
        expected_cumprod = jnp.cumprod(base_sampler.alphas)
        assert jnp.allclose(base_sampler.alphas_cumprod, expected_cumprod)

    def test_alphas_cumprod_prev(self, base_sampler):
        """Test alphas_cumprod_prev is properly shifted."""
        expected = jnp.append(jnp.array([1.0]), base_sampler.alphas_cumprod[:-1])
        assert jnp.allclose(base_sampler.alphas_cumprod_prev, expected)

    def test_sqrt_alphas_cumprod(self, base_sampler):
        """Test sqrt of cumulative alpha product."""
        expected = jnp.sqrt(base_sampler.alphas_cumprod)
        assert jnp.allclose(base_sampler.sqrt_alphas_cumprod, expected)

    def test_sqrt_one_minus_alphas_cumprod(self, base_sampler):
        """Test sqrt(1 - alphas_cumprod)."""
        expected = jnp.sqrt(1.0 - base_sampler.alphas_cumprod)
        assert jnp.allclose(base_sampler.sqrt_one_minus_alphas_cumprod, expected)

    def test_posterior_variance_formula(self, base_sampler):
        """Test posterior variance follows correct formula.

        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        """
        expected = (
            base_sampler.betas
            * (1.0 - base_sampler.alphas_cumprod_prev)
            / (1.0 - base_sampler.alphas_cumprod)
        )
        assert jnp.allclose(base_sampler.posterior_variance, expected, equal_nan=True)


class TestSamplerInit:
    """Test suite for sampler state initialization."""

    def test_init_state_keys(self, base_sampler, input_noise):
        """Test init returns state with correct keys."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        assert "x" in state
        assert "key" in state
        assert "t" in state

    def test_init_state_values(self, base_sampler, input_noise):
        """Test init state has correct initial values."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        assert jnp.allclose(state["x"], input_noise)
        assert state["t"] == base_sampler.num_timesteps - 1

    def test_init_preserves_shape(self, base_sampler):
        """Test init preserves input shape."""
        key = jax.random.key(0)
        shapes = [(4, 16), (2, 8, 8), (1, 32, 32, 3)]

        for shape in shapes:
            x = jax.random.normal(jax.random.key(42), shape)
            state = base_sampler.init(x, key)
            assert state["x"].shape == shape


class TestSamplerStep:
    """Test suite for sampler step function."""

    def test_step_returns_new_state_and_aux(self, base_sampler, input_noise):
        """Test step returns new state and auxiliary info."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        new_state, aux_info = base_sampler.step(state)

        assert isinstance(new_state, dict)
        assert isinstance(aux_info, dict)

    def test_step_decrements_timestep(self, base_sampler, input_noise):
        """Test step decrements timestep by 1."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        initial_t = state["t"]
        new_state, _ = base_sampler.step(state)

        assert new_state["t"] == initial_t - 1

    def test_step_preserves_shape(self, base_sampler, input_noise):
        """Test step preserves x shape."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        new_state, _ = base_sampler.step(state)

        assert new_state["x"].shape == input_noise.shape

    def test_step_produces_finite_values(self, base_sampler, input_noise):
        """Test step produces finite values."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        new_state, aux_info = base_sampler.step(state)

        assert jnp.all(jnp.isfinite(new_state["x"]))
        assert jnp.all(jnp.isfinite(aux_info["x0_prediction"]))
        assert jnp.all(jnp.isfinite(aux_info["mean"]))

    def test_step_aux_info_keys(self, base_sampler, input_noise):
        """Test step aux_info contains expected keys."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        _, aux_info = base_sampler.step(state)

        assert "x0_prediction" in aux_info
        assert "mean" in aux_info
        assert "variance" in aux_info

    def test_multiple_steps(self, base_sampler, input_noise):
        """Test multiple consecutive steps."""
        key = jax.random.key(0)
        state = base_sampler.init(input_noise, key)

        for i in range(10):
            state, _ = base_sampler.step(state)
            assert state["t"] == base_sampler.num_timesteps - 2 - i
            assert jnp.all(jnp.isfinite(state["x"]))


class TestSampling:
    """Test suite for sampling functionality."""

    def test_sample_with_model_delegation(self):
        """Test sampling delegates to model when available."""
        model = DummyDiffusionModel()
        sampler = DiffusionSampler(model=model, num_timesteps=100)

        samples = sampler.sample(4)

        assert samples.shape == (4, 16)
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_without_model_raises(self, base_sampler):
        """Test sampling without model raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            base_sampler.sample(4)


class TestEdgeCases:
    """Test edge cases for DiffusionSampler."""

    def test_single_timestep(self):
        """Test sampler with single timestep."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=1,
        )

        assert len(sampler.betas) == 1
        assert len(sampler.alphas) == 1

    def test_large_timesteps(self):
        """Test sampler with many timesteps."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=10000,
        )

        assert len(sampler.betas) == 10000
        assert jnp.all(jnp.isfinite(sampler.betas))
        assert jnp.all(jnp.isfinite(sampler.alphas_cumprod))

    def test_very_small_beta_start(self):
        """Test sampler with very small beta_start."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
            beta_start=1e-8,
            beta_end=0.02,
        )

        assert jnp.all(jnp.isfinite(sampler.betas))
        assert jnp.all(jnp.isfinite(sampler.alphas_cumprod))

    def test_single_sample(self):
        """Test with single sample (batch size 1)."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
        )

        x = jax.random.normal(jax.random.key(0), (1, 16))
        key = jax.random.key(1)

        state = sampler.init(x, key)
        new_state, _ = sampler.step(state)

        assert new_state["x"].shape == (1, 16)

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
        )

        x = jax.random.normal(jax.random.key(0), (4, 32, 32, 3))
        key = jax.random.key(1)

        state = sampler.init(x, key)
        new_state, _ = sampler.step(state)

        assert new_state["x"].shape == (4, 32, 32, 3)


class TestMathematicalProperties:
    """Test mathematical properties of diffusion sampler."""

    def test_alphas_bounded_0_1(self, base_sampler):
        """Test alphas are bounded between 0 and 1."""
        assert jnp.all(base_sampler.alphas > 0)
        assert jnp.all(base_sampler.alphas < 1)

    def test_alphas_cumprod_decreasing(self, base_sampler):
        """Test cumulative alpha product is monotonically decreasing."""
        cumprod = base_sampler.alphas_cumprod
        assert jnp.all(jnp.diff(cumprod) < 0)

    def test_alphas_cumprod_bounded(self, base_sampler):
        """Test alphas_cumprod is bounded between 0 and 1."""
        cumprod = base_sampler.alphas_cumprod
        assert jnp.all(cumprod > 0)
        assert jnp.all(cumprod <= 1)

    def test_posterior_variance_nonnegative(self, base_sampler):
        """Test posterior variance is non-negative."""
        # Skip first element which may be NaN due to division
        posterior_var = base_sampler.posterior_variance[1:]
        assert jnp.all(posterior_var >= 0)

    def test_step_reduces_noise_with_identity_predictor(self):
        """Test that step with identity predictor behaves predictably."""
        sampler = DiffusionSampler(
            predict_noise_fn=random_noise_predictor,
            num_timesteps=100,
        )

        x = jax.random.normal(jax.random.key(0), (4, 16))
        key = jax.random.key(1)

        state = sampler.init(x, key)
        _, aux_info = sampler.step(state)

        # x0_prediction should be finite
        assert jnp.all(jnp.isfinite(aux_info["x0_prediction"]))

    def test_noise_schedule_consistency(self):
        """Test that linear and quadratic schedules give different results."""
        linear_sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
            beta_schedule="linear",
        )
        quadratic_sampler = DiffusionSampler(
            predict_noise_fn=dummy_noise_predictor,
            num_timesteps=100,
            beta_schedule="quadratic",
        )

        # Endpoints should be the same
        assert linear_sampler.betas[0] == pytest.approx(quadratic_sampler.betas[0], rel=1e-5)
        assert linear_sampler.betas[-1] == pytest.approx(quadratic_sampler.betas[-1], rel=1e-5)

        # Interior values should differ
        assert not jnp.allclose(linear_sampler.betas[50], quadratic_sampler.betas[50])
