"""TDD tests for VAE-specific trainer.

These tests define the expected behavior for VAE training based on
state-of-the-art techniques (2024-2025):
- KL annealing schedules (linear, sigmoid, cyclical)
- Free bits constraint for posterior collapse prevention
- Beta-VAE weighting
- Integration with base Trainer via loss functions

References:
    - Cyclical KL Annealing: https://arxiv.org/abs/1903.10145
    - Free Bits: https://arxiv.org/abs/1606.04934
    - Beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl
    - DVAE (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12026048/
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


class SimpleVAE(nnx.Module):
    """Simple VAE for testing trainer functionality."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.encoder = nnx.Linear(16, 8, rngs=rngs)
        self.mean_layer = nnx.Linear(8, 4, rngs=rngs)
        self.logvar_layer = nnx.Linear(8, 4, rngs=rngs)
        self.decoder = nnx.Linear(4, 16, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = nnx.relu(self.encoder(x))
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        # Reparameterization (simplified - no stochasticity for testing)
        z = mean
        recon = self.decoder(z)
        return recon, mean, logvar


@pytest.fixture
def simple_vae() -> SimpleVAE:
    """Create a simple VAE for testing."""
    return SimpleVAE(rngs=nnx.Rngs(0))


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch for testing."""
    return {"data": jax.random.normal(jax.random.key(0), (8, 16))}


# =============================================================================
# VAETrainingConfig Tests
# =============================================================================


class TestVAETrainingConfig:
    """Tests for VAETrainingConfig dataclass."""

    def test_config_exists(self) -> None:
        """VAETrainingConfig should be importable."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainingConfig,
        )

        assert VAETrainingConfig is not None

    def test_config_default_values(self) -> None:
        """VAETrainingConfig should have sensible defaults."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        assert config.kl_annealing == "linear"
        assert config.kl_warmup_steps == 10000
        assert config.beta == 1.0
        assert config.free_bits == 0.0
        assert config.cyclical_period == 10000

    def test_config_custom_values(self) -> None:
        """VAETrainingConfig should accept custom values."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainingConfig,
        )

        config = VAETrainingConfig(
            kl_annealing="cyclical",
            kl_warmup_steps=5000,
            beta=4.0,
            free_bits=0.5,
            cyclical_period=2000,
        )
        assert config.kl_annealing == "cyclical"
        assert config.kl_warmup_steps == 5000
        assert config.beta == 4.0
        assert config.free_bits == 0.5
        assert config.cyclical_period == 2000

    def test_config_all_annealing_types(self) -> None:
        """VAETrainingConfig should support all annealing types."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainingConfig,
        )

        for annealing_type in ["none", "linear", "sigmoid", "cyclical"]:
            config = VAETrainingConfig(kl_annealing=annealing_type)
            assert config.kl_annealing == annealing_type


# =============================================================================
# KL Annealing Tests
# =============================================================================


class TestKLAnnealing:
    """Tests for KL annealing schedules."""

    def test_no_annealing_returns_full_beta(self) -> None:
        """No annealing should return full beta from step 0."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="none", beta=4.0)
        trainer = VAETrainer(config)

        assert trainer.get_kl_weight(0) == 4.0
        assert trainer.get_kl_weight(1000) == 4.0
        assert trainer.get_kl_weight(100000) == 4.0

    def test_linear_annealing_starts_at_zero(self) -> None:
        """Linear annealing should start at zero."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="linear", beta=1.0, kl_warmup_steps=1000)
        trainer = VAETrainer(config)

        assert trainer.get_kl_weight(0) == 0.0

    def test_linear_annealing_reaches_beta(self) -> None:
        """Linear annealing should reach beta at warmup_steps."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="linear", beta=2.0, kl_warmup_steps=1000)
        trainer = VAETrainer(config)

        assert trainer.get_kl_weight(1000) == 2.0
        assert trainer.get_kl_weight(2000) == 2.0  # Should stay at beta

    def test_linear_annealing_midpoint(self) -> None:
        """Linear annealing should be at 50% at halfway point."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="linear", beta=1.0, kl_warmup_steps=1000)
        trainer = VAETrainer(config)

        assert trainer.get_kl_weight(500) == pytest.approx(0.5)

    def test_sigmoid_annealing_s_shaped(self) -> None:
        """Sigmoid annealing should have S-shaped curve."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="sigmoid", beta=1.0, kl_warmup_steps=1000)
        trainer = VAETrainer(config)

        # Start should be near 0
        assert trainer.get_kl_weight(0) < 0.1
        # Middle should be near 0.5
        assert 0.4 < trainer.get_kl_weight(500) < 0.6
        # End should be near 1.0
        assert trainer.get_kl_weight(1000) > 0.9

    def test_cyclical_annealing_resets(self) -> None:
        """Cyclical annealing should reset after each period."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="cyclical", beta=1.0, cyclical_period=100)
        trainer = VAETrainer(config)

        # Start of first cycle
        assert trainer.get_kl_weight(0) == 0.0
        # End of first cycle (reaches beta)
        assert trainer.get_kl_weight(50) == pytest.approx(1.0)
        # Start of second cycle (resets)
        assert trainer.get_kl_weight(100) == 0.0
        # Middle of second cycle
        assert trainer.get_kl_weight(125) == pytest.approx(0.5)


# =============================================================================
# Free Bits Tests
# =============================================================================


class TestFreeBits:
    """Tests for free bits constraint."""

    def test_free_bits_disabled_when_zero(self) -> None:
        """Free bits should not modify KL when set to 0."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(free_bits=0.0)
        trainer = VAETrainer(config)

        kl_per_dim = jnp.array([0.1, 0.2, 0.3])
        result = trainer.apply_free_bits(kl_per_dim)
        assert jnp.allclose(result, kl_per_dim)

    def test_free_bits_clamps_low_values(self) -> None:
        """Free bits should clamp KL values below threshold."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(free_bits=0.5)
        trainer = VAETrainer(config)

        kl_per_dim = jnp.array([0.1, 0.5, 0.9])
        result = trainer.apply_free_bits(kl_per_dim)
        expected = jnp.array([0.5, 0.5, 0.9])
        assert jnp.allclose(result, expected)

    def test_free_bits_prevents_posterior_collapse(self) -> None:
        """Free bits should ensure minimum KL per dimension."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(free_bits=0.25)
        trainer = VAETrainer(config)

        # Simulate posterior collapse (very low KL)
        kl_per_dim = jnp.array([0.001, 0.002, 0.003, 0.004])
        result = trainer.apply_free_bits(kl_per_dim)
        assert jnp.all(result >= 0.25)


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestVAELossComputation:
    """Tests for VAE loss computation."""

    def test_compute_kl_loss_shape(self, simple_vae: SimpleVAE) -> None:
        """compute_kl_loss should return scalar loss and per-sample losses."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)

        mean = jnp.zeros((8, 4))
        logvar = jnp.zeros((8, 4))
        kl_loss, kl_per_sample = trainer.compute_kl_loss(mean, logvar)

        assert kl_loss.shape == ()
        assert kl_per_sample.shape == (8,)

    def test_compute_kl_loss_zero_for_standard_normal(self, simple_vae: SimpleVAE) -> None:
        """KL loss should be zero when posterior matches prior."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(free_bits=0.0)
        trainer = VAETrainer(config)

        # Standard normal: mean=0, logvar=0 (var=1)
        mean = jnp.zeros((8, 4))
        logvar = jnp.zeros((8, 4))
        kl_loss, _ = trainer.compute_kl_loss(mean, logvar)

        assert kl_loss == pytest.approx(0.0, abs=1e-6)

    def test_compute_reconstruction_loss_mse(self, simple_vae: SimpleVAE) -> None:
        """MSE reconstruction loss should compute correctly."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)

        x = jnp.ones((8, 16))
        recon_x = jnp.zeros((8, 16))
        loss = trainer.compute_reconstruction_loss(x, recon_x, "mse")

        # With batch_sum reduction (standard for VAE ELBO):
        # MSE = 1.0 per element, sum over 16 features = 16.0 per sample, mean over batch = 16.0
        assert loss == pytest.approx(16.0)

    def test_compute_loss_returns_metrics(self, simple_vae: SimpleVAE, sample_batch: dict) -> None:
        """compute_loss should return loss and metrics dict."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)

        loss, metrics = trainer.compute_loss(simple_vae, sample_batch, step=100)

        assert "loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics
        assert "kl_weight" in metrics
        assert isinstance(loss, jax.Array)

    def test_loss_includes_kl_weight(self, simple_vae: SimpleVAE, sample_batch: dict) -> None:
        """Loss should apply KL weight based on annealing schedule."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="linear", kl_warmup_steps=100, beta=1.0)
        trainer = VAETrainer(config)

        _, metrics_early = trainer.compute_loss(simple_vae, sample_batch, step=10)
        _, metrics_late = trainer.compute_loss(simple_vae, sample_batch, step=100)

        # Early step should have lower KL weight
        assert metrics_early["kl_weight"] < metrics_late["kl_weight"]


# =============================================================================
# Training Step Tests
# =============================================================================


class TestVAETrainStep:
    """Tests for VAE training step."""

    def test_train_step_updates_model(self, simple_vae: SimpleVAE, sample_batch: dict) -> None:
        """train_step should update model parameters."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(kl_annealing="none", beta=1.0)
        optimizer = nnx.Optimizer(simple_vae, optax.adam(1e-3), wrt=nnx.Param)
        trainer = VAETrainer(config)

        # Get initial params
        initial_params = nnx.state(simple_vae, nnx.Param)
        initial_encoder_kernel = initial_params["encoder"]["kernel"].value.copy()

        # Run train step
        trainer.train_step(simple_vae, optimizer, sample_batch, step=0)

        # Get updated params
        updated_params = nnx.state(simple_vae, nnx.Param)
        updated_encoder_kernel = updated_params["encoder"]["kernel"].value

        # Params should have changed
        assert not jnp.allclose(initial_encoder_kernel, updated_encoder_kernel)

    def test_train_step_returns_metrics(self, simple_vae: SimpleVAE, sample_batch: dict) -> None:
        """train_step should return loss and metrics."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        optimizer = nnx.Optimizer(simple_vae, optax.adam(1e-3), wrt=nnx.Param)
        trainer = VAETrainer(config)

        loss, metrics = trainer.train_step(simple_vae, optimizer, sample_batch, step=0)

        assert isinstance(loss, jax.Array)
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics


# =============================================================================
# DRY Integration Tests - Loss Function for Base Trainer
# =============================================================================


class TestVAELossFunctionIntegration:
    """Tests for using VAE loss with base Trainer (DRY principle)."""

    def test_create_loss_fn_for_base_trainer(self, simple_vae: SimpleVAE) -> None:
        """VAETrainer should provide loss_fn compatible with base Trainer."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)

        # Should be able to create a loss function for the base Trainer
        # Note: create_loss_fn() no longer takes step - step is passed dynamically
        loss_fn = trainer.create_loss_fn()

        # Loss function signature: (model, batch, rng, step) -> (loss, metrics)
        # Step is passed dynamically to support KL annealing inside JIT-compiled loops
        batch = {"data": jax.random.normal(jax.random.key(0), (8, 16))}
        rng = jax.random.key(42)
        step = jnp.array(100)
        loss, metrics = loss_fn(simple_vae, batch, rng, step)

        assert isinstance(loss, jax.Array)
        assert "kl_loss" in metrics


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestVAETrainerExports:
    """Tests for VAE trainer exports."""

    def test_exports_from_trainers_init(self) -> None:
        """VAE trainer classes should be exported from trainers __init__."""
        from artifex.generative_models.training.trainers import (
            VAETrainer,
            VAETrainingConfig,
        )

        assert VAETrainer is not None
        assert VAETrainingConfig is not None
