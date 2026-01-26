"""Tests for noise schedule implementations.

Following TDD approach - tests define the expected behavior.
"""

import jax.numpy as jnp
import pytest
from flax import nnx


class TestNoiseScheduleConfig:
    """Tests for NoiseScheduleConfig dataclass."""

    def test_create_linear_schedule_config(self):
        """Test creating a linear noise schedule config."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        config = NoiseScheduleConfig(
            name="linear_schedule",
            schedule_type="linear",
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
        )

        assert config.name == "linear_schedule"
        assert config.schedule_type == "linear"
        assert config.num_timesteps == 1000
        assert config.beta_start == 1e-4
        assert config.beta_end == 2e-2

    def test_create_cosine_schedule_config(self):
        """Test creating a cosine noise schedule config."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        config = NoiseScheduleConfig(
            name="cosine_schedule",
            schedule_type="cosine",
            num_timesteps=1000,
        )

        assert config.schedule_type == "cosine"

    def test_create_quadratic_schedule_config(self):
        """Test creating a quadratic noise schedule config."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        config = NoiseScheduleConfig(
            name="quad_schedule",
            schedule_type="quadratic",
            num_timesteps=500,
        )

        assert config.schedule_type == "quadratic"

    def test_invalid_schedule_type_raises_error(self):
        """Test that invalid schedule_type raises ValueError."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        with pytest.raises(ValueError, match="schedule_type must be one of"):
            NoiseScheduleConfig(
                name="invalid",
                schedule_type="unknown",
            )

    def test_beta_start_must_be_less_than_beta_end(self):
        """Test that beta_start must be less than beta_end."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        with pytest.raises(ValueError, match="beta_start must be less than beta_end"):
            NoiseScheduleConfig(
                name="invalid",
                beta_start=0.02,
                beta_end=0.0001,
            )


class TestNoiseScheduleBase:
    """Tests for the base NoiseSchedule class."""

    @pytest.fixture
    def linear_config(self):
        """Create a linear schedule config for testing."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        return NoiseScheduleConfig(
            name="test_linear",
            schedule_type="linear",
            num_timesteps=100,
            beta_start=1e-4,
            beta_end=2e-2,
        )

    def test_schedule_is_nnx_module(self, linear_config):
        """Test that NoiseSchedule is an nnx.Module."""
        from artifex.generative_models.core.noise_schedule import create_noise_schedule

        schedule = create_noise_schedule(linear_config)
        assert isinstance(schedule, nnx.Module)

    def test_schedule_has_required_attributes(self, linear_config):
        """Test that schedule has all required attributes."""
        from artifex.generative_models.core.noise_schedule import create_noise_schedule

        schedule = create_noise_schedule(linear_config)

        # Required attributes
        assert hasattr(schedule, "num_timesteps")
        assert hasattr(schedule, "betas")
        assert hasattr(schedule, "alphas")
        assert hasattr(schedule, "alphas_cumprod")
        assert hasattr(schedule, "alphas_cumprod_prev")
        assert hasattr(schedule, "sqrt_alphas_cumprod")
        assert hasattr(schedule, "sqrt_one_minus_alphas_cumprod")
        assert hasattr(schedule, "posterior_variance")
        assert hasattr(schedule, "posterior_mean_coef1")
        assert hasattr(schedule, "posterior_mean_coef2")

    def test_schedule_shapes_match_num_timesteps(self, linear_config):
        """Test that all schedule arrays have correct shape."""
        from artifex.generative_models.core.noise_schedule import create_noise_schedule

        schedule = create_noise_schedule(linear_config)
        num_t = linear_config.num_timesteps

        assert schedule.betas.shape == (num_t,)
        assert schedule.alphas.shape == (num_t,)
        assert schedule.alphas_cumprod.shape == (num_t,)
        assert schedule.sqrt_alphas_cumprod.shape == (num_t,)
        assert schedule.sqrt_one_minus_alphas_cumprod.shape == (num_t,)


class TestLinearNoiseSchedule:
    """Tests for LinearNoiseSchedule."""

    @pytest.fixture
    def config(self):
        """Create a linear schedule config."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        return NoiseScheduleConfig(
            name="linear",
            schedule_type="linear",
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
        )

    def test_betas_are_linearly_spaced(self, config):
        """Test that betas follow linear interpolation."""
        from artifex.generative_models.core.noise_schedule import LinearNoiseSchedule

        schedule = LinearNoiseSchedule(config)

        expected_betas = jnp.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        assert jnp.allclose(schedule.betas, expected_betas)

    def test_betas_start_and_end_values(self, config):
        """Test that betas start and end at correct values."""
        from artifex.generative_models.core.noise_schedule import LinearNoiseSchedule

        schedule = LinearNoiseSchedule(config)

        assert jnp.isclose(schedule.betas[0], config.beta_start, rtol=1e-5)
        assert jnp.isclose(schedule.betas[-1], config.beta_end, rtol=1e-5)

    def test_alphas_equal_one_minus_betas(self, config):
        """Test that alphas = 1 - betas."""
        from artifex.generative_models.core.noise_schedule import LinearNoiseSchedule

        schedule = LinearNoiseSchedule(config)

        expected_alphas = 1.0 - schedule.betas
        assert jnp.allclose(schedule.alphas, expected_alphas)


class TestCosineNoiseSchedule:
    """Tests for CosineNoiseSchedule."""

    @pytest.fixture
    def config(self):
        """Create a cosine schedule config."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        return NoiseScheduleConfig(
            name="cosine",
            schedule_type="cosine",
            num_timesteps=100,
        )

    def test_cosine_betas_are_clipped(self, config):
        """Test that cosine betas are clipped to valid range."""
        from artifex.generative_models.core.noise_schedule import CosineNoiseSchedule

        schedule = CosineNoiseSchedule(config)

        # Betas should be in valid range
        assert jnp.all(schedule.betas >= 0.0001)
        assert jnp.all(schedule.betas <= 0.9999)

    def test_cosine_alpha_cumprod_starts_near_one(self, config):
        """Test that cosine alpha_cumprod starts near 1."""
        from artifex.generative_models.core.noise_schedule import CosineNoiseSchedule

        schedule = CosineNoiseSchedule(config)

        # First alpha_cumprod should be close to 1
        assert schedule.alphas_cumprod[0] > 0.99

    def test_cosine_alpha_cumprod_ends_near_zero(self, config):
        """Test that cosine alpha_cumprod ends near 0."""
        from artifex.generative_models.core.noise_schedule import CosineNoiseSchedule

        schedule = CosineNoiseSchedule(config)

        # Last alpha_cumprod should be close to 0
        assert schedule.alphas_cumprod[-1] < 0.1


class TestQuadraticNoiseSchedule:
    """Tests for QuadraticNoiseSchedule."""

    @pytest.fixture
    def config(self):
        """Create a quadratic schedule config."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig

        return NoiseScheduleConfig(
            name="quadratic",
            schedule_type="quadratic",
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
        )

    def test_quadratic_betas_are_squared(self, config):
        """Test that quadratic betas follow squared interpolation."""
        from artifex.generative_models.core.noise_schedule import QuadraticNoiseSchedule

        schedule = QuadraticNoiseSchedule(config)

        expected_betas = (
            jnp.linspace(config.beta_start**0.5, config.beta_end**0.5, config.num_timesteps) ** 2
        )
        assert jnp.allclose(schedule.betas, expected_betas)

    def test_quadratic_differs_from_linear(self, config):
        """Test that quadratic schedule differs from linear."""
        from artifex.generative_models.core.noise_schedule import (
            LinearNoiseSchedule,
            QuadraticNoiseSchedule,
        )

        linear = LinearNoiseSchedule(config)
        quadratic = QuadraticNoiseSchedule(config)

        # They should have different values (except possibly at endpoints)
        assert not jnp.allclose(linear.betas, quadratic.betas)


class TestCreateNoiseSchedule:
    """Tests for create_noise_schedule factory function."""

    def test_creates_linear_schedule(self):
        """Test factory creates LinearNoiseSchedule for 'linear' type."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig
        from artifex.generative_models.core.noise_schedule import (
            create_noise_schedule,
            LinearNoiseSchedule,
        )

        config = NoiseScheduleConfig(name="linear", schedule_type="linear")
        schedule = create_noise_schedule(config)

        assert isinstance(schedule, LinearNoiseSchedule)

    def test_creates_cosine_schedule(self):
        """Test factory creates CosineNoiseSchedule for 'cosine' type."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig
        from artifex.generative_models.core.noise_schedule import (
            CosineNoiseSchedule,
            create_noise_schedule,
        )

        config = NoiseScheduleConfig(name="cosine", schedule_type="cosine")
        schedule = create_noise_schedule(config)

        assert isinstance(schedule, CosineNoiseSchedule)

    def test_creates_quadratic_schedule(self):
        """Test factory creates QuadraticNoiseSchedule for 'quadratic' type."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig
        from artifex.generative_models.core.noise_schedule import (
            create_noise_schedule,
            QuadraticNoiseSchedule,
        )

        config = NoiseScheduleConfig(name="quad", schedule_type="quadratic")
        schedule = create_noise_schedule(config)

        assert isinstance(schedule, QuadraticNoiseSchedule)

    def test_raises_for_unknown_schedule_type(self):
        """Test that factory raises ValueError for unknown schedule type."""
        from artifex.generative_models.core.noise_schedule import create_noise_schedule

        # Create a mock config with invalid schedule_type
        # Note: This tests the factory, not config validation
        class MockConfig:
            name = "mock"
            schedule_type = "invalid_type"
            num_timesteps = 100
            beta_start = 1e-4
            beta_end = 2e-2
            clip_min = 1e-20

        with pytest.raises(ValueError, match="Unknown schedule_type"):
            create_noise_schedule(MockConfig())


class TestNoiseScheduleQSample:
    """Tests for q_sample method (forward diffusion)."""

    @pytest.fixture
    def schedule(self):
        """Create a linear schedule for testing."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig
        from artifex.generative_models.core.noise_schedule import create_noise_schedule

        config = NoiseScheduleConfig(
            name="test",
            schedule_type="linear",
            num_timesteps=100,
        )
        return create_noise_schedule(config)

    def test_q_sample_returns_correct_shape(self, schedule):
        """Test q_sample returns same shape as input."""
        x_start = jnp.ones((4, 32, 32, 3))
        t = jnp.array([10, 20, 30, 40])
        noise = jnp.zeros_like(x_start)

        x_t = schedule.q_sample(x_start, t, noise)

        assert x_t.shape == x_start.shape

    def test_q_sample_at_t0_returns_original(self, schedule):
        """Test that q_sample at t=0 returns input (minimal noise)."""
        x_start = jnp.ones((2, 8, 8, 3))
        t = jnp.array([0, 0])
        noise = jnp.zeros_like(x_start)

        x_t = schedule.q_sample(x_start, t, noise)

        # At t=0, sqrt_alpha_cumprod is close to 1, so x_t ≈ x_start
        assert jnp.allclose(x_t, x_start, atol=1e-2)

    def test_q_sample_formula(self, schedule):
        """Test that q_sample follows the correct formula."""
        x_start = jnp.ones((2, 8, 8, 3))
        t = jnp.array([50, 50])
        noise = jnp.ones_like(x_start) * 0.5

        x_t = schedule.q_sample(x_start, t, noise)

        # x_t = sqrt(alpha_cumprod_t) * x_start + sqrt(1 - alpha_cumprod_t) * noise
        sqrt_alpha = schedule.sqrt_alphas_cumprod[50]
        sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[50]
        expected = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

        assert jnp.allclose(x_t, expected)


class TestNoiseSchedulePosterior:
    """Tests for posterior computation methods."""

    @pytest.fixture
    def schedule(self):
        """Create a linear schedule for testing."""
        from artifex.generative_models.core.configuration import NoiseScheduleConfig
        from artifex.generative_models.core.noise_schedule import create_noise_schedule

        config = NoiseScheduleConfig(
            name="test",
            schedule_type="linear",
            num_timesteps=100,
        )
        return create_noise_schedule(config)

    def test_posterior_mean_coefs_shape(self, schedule):
        """Test posterior mean coefficients have correct shape."""
        assert schedule.posterior_mean_coef1.shape == (100,)
        assert schedule.posterior_mean_coef2.shape == (100,)

    def test_posterior_variance_shape(self, schedule):
        """Test posterior variance has correct shape."""
        assert schedule.posterior_variance.shape == (100,)

    def test_posterior_variance_is_non_negative(self, schedule):
        """Test posterior variance is non-negative."""
        assert jnp.all(schedule.posterior_variance >= 0)
