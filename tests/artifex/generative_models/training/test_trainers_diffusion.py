"""TDD tests for Diffusion-specific trainer.

These tests define the expected behavior for diffusion model training based on
state-of-the-art techniques (2024-2025):
- Prediction types (epsilon, v-prediction, x-prediction)
- Timestep sampling strategies (uniform, logit-normal, mode)
- Loss weighting (uniform, SNR, min-SNR, EDM)
- EMA model updates

References:
    - DDPM: https://arxiv.org/abs/2006.11239
    - v-prediction: https://arxiv.org/abs/2202.00512
    - min-SNR: https://arxiv.org/abs/2303.09556
    - EDM: https://arxiv.org/abs/2206.00364
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleDiffusionModel(nnx.Module):
    """Simple diffusion model for testing trainer functionality."""

    def __init__(self, dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(dim, 32, rngs=rngs)
        self.time_embed = nnx.Linear(1, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        # Simple architecture: concat time embedding with features
        h = nnx.relu(self.fc1(x))
        t_embed = nnx.relu(self.time_embed(t[:, None].astype(jnp.float32)))
        h = h + t_embed
        return self.fc2(h)


class SimpleNoiseSchedule:
    """Simple noise schedule for testing."""

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        # Linear beta schedule
        betas = jnp.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas)

    def add_noise(self, x: jax.Array, noise: jax.Array, t: jax.Array) -> jax.Array:
        """Add noise to data at timestep t."""
        sqrt_alpha = jnp.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha = jnp.sqrt(1.0 - self.alphas_cumprod[t])[:, None]
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise


@pytest.fixture
def diffusion_model() -> SimpleDiffusionModel:
    """Create a simple diffusion model for testing."""
    return SimpleDiffusionModel(rngs=nnx.Rngs(0))


@pytest.fixture
def noise_schedule() -> SimpleNoiseSchedule:
    """Create a simple noise schedule for testing."""
    return SimpleNoiseSchedule(num_timesteps=1000)


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch for testing."""
    return {"image": jax.random.normal(jax.random.key(0), (8, 16))}


@pytest.fixture
def rng_key() -> jax.Array:
    """Create PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# DiffusionTrainingConfig Tests
# =============================================================================


class TestDiffusionTrainingConfig:
    """Tests for DiffusionTrainingConfig dataclass."""

    def test_config_exists(self) -> None:
        """DiffusionTrainingConfig should be importable."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainingConfig,
        )

        assert DiffusionTrainingConfig is not None

    def test_config_default_values(self) -> None:
        """DiffusionTrainingConfig should have SOTA defaults."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        # 2025 SOTA defaults
        assert config.prediction_type == "epsilon"
        assert config.timestep_sampling == "uniform"
        assert config.loss_weighting == "uniform"
        assert config.snr_gamma == 5.0
        assert config.ema_decay == 0.9999
        assert config.ema_update_every == 10

    def test_config_custom_values(self) -> None:
        """DiffusionTrainingConfig should accept custom values."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(
            prediction_type="v_prediction",
            timestep_sampling="logit_normal",
            loss_weighting="min_snr",
            snr_gamma=3.0,
            ema_decay=0.999,
        )
        assert config.prediction_type == "v_prediction"
        assert config.timestep_sampling == "logit_normal"
        assert config.loss_weighting == "min_snr"
        assert config.snr_gamma == 3.0
        assert config.ema_decay == 0.999

    def test_config_all_prediction_types(self) -> None:
        """DiffusionTrainingConfig should support all prediction types."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainingConfig,
        )

        for pred_type in ["epsilon", "v_prediction", "x_start"]:
            config = DiffusionTrainingConfig(prediction_type=pred_type)
            assert config.prediction_type == pred_type

    def test_config_all_timestep_sampling(self) -> None:
        """DiffusionTrainingConfig should support all timestep sampling strategies."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainingConfig,
        )

        for sampling in ["uniform", "logit_normal", "mode"]:
            config = DiffusionTrainingConfig(timestep_sampling=sampling)
            assert config.timestep_sampling == sampling

    def test_config_all_loss_weighting(self) -> None:
        """DiffusionTrainingConfig should support all loss weighting schemes."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainingConfig,
        )

        for weighting in ["uniform", "snr", "min_snr", "edm"]:
            config = DiffusionTrainingConfig(loss_weighting=weighting)
            assert config.loss_weighting == weighting


# =============================================================================
# Timestep Sampling Tests
# =============================================================================


class TestTimestepSampling:
    """Tests for timestep sampling strategies."""

    def test_uniform_sampling_range(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """Uniform sampling should return timesteps in valid range."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(timestep_sampling="uniform")
        trainer = DiffusionTrainer(noise_schedule, config)

        t = trainer.sample_timesteps(100, rng_key)

        assert t.shape == (100,)
        assert jnp.all(t >= 0)
        assert jnp.all(t < noise_schedule.num_timesteps)

    def test_logit_normal_sampling_range(
        self,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """Logit-normal sampling should return timesteps in valid range."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(timestep_sampling="logit_normal")
        trainer = DiffusionTrainer(noise_schedule, config)

        t = trainer.sample_timesteps(100, rng_key)

        assert t.shape == (100,)
        assert jnp.all(t >= 0)
        assert jnp.all(t < noise_schedule.num_timesteps)

    def test_logit_normal_favors_middle_timesteps(
        self,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """Logit-normal should favor middle timesteps over edges."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(timestep_sampling="logit_normal")
        trainer = DiffusionTrainer(noise_schedule, config)

        # Sample many timesteps
        t = trainer.sample_timesteps(10000, rng_key)

        # Count samples in middle vs edges
        mid = noise_schedule.num_timesteps // 2
        mid_range = noise_schedule.num_timesteps // 4
        mid_count = jnp.sum((t > mid - mid_range) & (t < mid + mid_range))
        edge_count = jnp.sum((t < mid_range) | (t > noise_schedule.num_timesteps - mid_range))

        # Middle should have more samples than edges
        assert mid_count > edge_count

    def test_mode_sampling_range(
        self,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """Mode sampling should return timesteps in valid range."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(timestep_sampling="mode")
        trainer = DiffusionTrainer(noise_schedule, config)

        t = trainer.sample_timesteps(100, rng_key)

        assert t.shape == (100,)
        assert jnp.all(t >= 0)
        assert jnp.all(t < noise_schedule.num_timesteps)


# =============================================================================
# Loss Weighting Tests
# =============================================================================


class TestLossWeighting:
    """Tests for loss weighting schemes."""

    def test_uniform_weighting_returns_ones(
        self,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """Uniform weighting should return ones for all timesteps."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(loss_weighting="uniform")
        trainer = DiffusionTrainer(noise_schedule, config)

        t = jnp.array([0, 100, 500, 900, 999])
        weights = trainer.get_loss_weight(t)

        assert jnp.allclose(weights, 1.0)

    def test_snr_weighting_decreases_with_noise(
        self,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """SNR weighting should decrease as noise increases (higher timesteps)."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(loss_weighting="snr")
        trainer = DiffusionTrainer(noise_schedule, config)

        t_low = jnp.array([10])  # Low noise
        t_high = jnp.array([900])  # High noise

        weight_low = trainer.get_loss_weight(t_low)
        weight_high = trainer.get_loss_weight(t_high)

        # SNR is higher at low noise, so weight should be higher
        assert weight_low > weight_high

    def test_min_snr_clamps_high_snr(
        self,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """Min-SNR weighting should clamp weights at high SNR (low noise)."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(loss_weighting="min_snr", snr_gamma=5.0)
        trainer = DiffusionTrainer(noise_schedule, config)

        # At very low noise (high SNR), min-SNR should clamp
        t_low_noise = jnp.array([10])
        weight = trainer.get_loss_weight(t_low_noise)

        # Weight should be <= 1.0 due to clamping
        assert weight[0] <= 1.0

    def test_min_snr_preserves_low_snr(
        self,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """Min-SNR weighting should preserve weights at low SNR (high noise)."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(loss_weighting="min_snr", snr_gamma=5.0)
        trainer = DiffusionTrainer(noise_schedule, config)

        # At high noise (low SNR < gamma), weight should be ~1
        t_high_noise = jnp.array([900])
        weight = trainer.get_loss_weight(t_high_noise)

        # Weight should be close to 1.0
        assert weight[0] > 0.5  # Should be reasonably close to 1


# =============================================================================
# Prediction Target Tests
# =============================================================================


class TestPredictionTargets:
    """Tests for prediction target computation."""

    def test_epsilon_target_is_noise(
        self,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """Epsilon prediction target should be the added noise."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(prediction_type="epsilon")
        trainer = DiffusionTrainer(noise_schedule, config)

        x0 = jnp.ones((2, 16))
        noise = jax.random.normal(rng_key, x0.shape)
        t = jnp.array([500, 500])

        target = trainer.compute_target(x0, noise, t)

        assert jnp.allclose(target, noise)

    def test_v_prediction_target_formula(
        self,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """V-prediction target should be sqrt(alpha)*noise - sqrt(1-alpha)*x0."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(prediction_type="v_prediction")
        trainer = DiffusionTrainer(noise_schedule, config)

        x0 = jnp.ones((2, 16))
        noise = jax.random.normal(rng_key, x0.shape)
        t = jnp.array([500, 500])

        target = trainer.compute_target(x0, noise, t)

        # Compute expected v-prediction target
        alpha = noise_schedule.alphas_cumprod[t][:, None]
        expected = jnp.sqrt(alpha) * noise - jnp.sqrt(1.0 - alpha) * x0

        assert jnp.allclose(target, expected)

    def test_x_start_target_is_original(
        self,
        noise_schedule: SimpleNoiseSchedule,
        rng_key: jax.Array,
    ) -> None:
        """X-start prediction target should be the original clean data."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(prediction_type="x_start")
        trainer = DiffusionTrainer(noise_schedule, config)

        x0 = jnp.ones((2, 16)) * 0.5
        noise = jax.random.normal(rng_key, x0.shape)
        t = jnp.array([500, 500])

        target = trainer.compute_target(x0, noise, t)

        assert jnp.allclose(target, x0)


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestDiffusionLossComputation:
    """Tests for diffusion loss computation."""

    def test_compute_loss_returns_scalar(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """compute_loss should return scalar loss."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)

        loss, _ = trainer.compute_loss(diffusion_model, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_compute_loss_returns_metrics(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """compute_loss should return loss and metrics dict."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)

        _, metrics = trainer.compute_loss(diffusion_model, sample_batch, rng_key)

        assert "loss" in metrics
        assert "loss_unweighted" in metrics


# =============================================================================
# Training Step Tests
# =============================================================================


class TestDiffusionTrainStep:
    """Tests for diffusion training step."""

    def test_train_step_updates_model(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should update model parameters."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        optimizer = nnx.Optimizer(diffusion_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = DiffusionTrainer(noise_schedule, config)

        # Get initial params
        initial_params = nnx.state(diffusion_model, nnx.Param)
        initial_fc1_kernel = initial_params["fc1"]["kernel"].value.copy()

        # Run train step with new signature
        trainer.train_step(diffusion_model, optimizer, sample_batch, rng_key)

        # Get updated params
        updated_params = nnx.state(diffusion_model, nnx.Param)
        updated_fc1_kernel = updated_params["fc1"]["kernel"].value

        # Params should have changed
        assert not jnp.allclose(initial_fc1_kernel, updated_fc1_kernel)

    def test_train_step_returns_metrics(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should return loss and metrics."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        optimizer = nnx.Optimizer(diffusion_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DiffusionTrainer(noise_schedule, config)

        loss, metrics = trainer.train_step(diffusion_model, optimizer, sample_batch, rng_key)

        assert isinstance(loss, jax.Array)
        assert "loss" in metrics


# =============================================================================
# EMA Tests
# =============================================================================


class TestEMAUpdates:
    """Tests for EMA model updates."""

    def test_ema_initialized_on_first_update(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """EMA params should be initialized on first update call."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(ema_update_every=1)
        trainer = DiffusionTrainer(noise_schedule, config)

        assert trainer._ema_params is None

        # EMA now takes model as argument
        trainer.update_ema(diffusion_model)

        assert trainer._ema_params is not None

    def test_ema_respects_update_frequency(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """EMA should only update every N steps."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(ema_update_every=5)
        trainer = DiffusionTrainer(noise_schedule, config)

        # First 4 calls shouldn't initialize EMA
        for _ in range(4):
            trainer.update_ema(diffusion_model)
        assert trainer._ema_params is None

        # 5th call should initialize
        trainer.update_ema(diffusion_model)
        assert trainer._ema_params is not None


# =============================================================================
# DRY Integration Tests
# =============================================================================


class TestDiffusionDRYIntegration:
    """Tests for DRY integration with base Trainer."""

    def test_create_loss_fn_signature(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """DiffusionTrainer should provide loss_fn compatible with base Trainer."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)

        # Should be able to create a loss function for the base Trainer
        loss_fn = trainer.create_loss_fn()

        # Loss function should have correct signature: (model, batch, rng) -> (loss, metrics)
        loss, metrics = loss_fn(diffusion_model, sample_batch, rng_key)

        assert isinstance(loss, jax.Array)
        assert "loss" in metrics


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestDiffusionTrainerExports:
    """Tests for Diffusion trainer exports."""

    def test_exports_from_trainers_init(self) -> None:
        """Diffusion trainer classes should be exported from trainers __init__."""
        from artifex.generative_models.training.trainers import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        assert DiffusionTrainer is not None
        assert DiffusionTrainingConfig is not None
