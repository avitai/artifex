"""TDD tests for Energy-based model trainer.

These tests define the expected behavior for EBM training including:
- Contrastive Divergence (CD) training
- Persistent Contrastive Divergence (PCD) with replay buffer
- Langevin dynamics MCMC sampling
- Score matching training
- Gradient penalty regularization

References:
    - CD: https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
    - Langevin Dynamics: https://arxiv.org/abs/1903.08689
    - IGEBM: https://arxiv.org/abs/1903.08689
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


class SimpleEnergyModel(nnx.Module):
    """Simple energy model for testing trainer functionality.

    Returns a scalar energy for each input sample.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.layer1 = nnx.Linear(16, 32, rngs=rngs)
        self.layer2 = nnx.Linear(32, 16, rngs=rngs)
        self.energy_head = nnx.Linear(16, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass returning energy values."""
        h = nnx.relu(self.layer1(x))
        h = nnx.relu(self.layer2(h))
        energy = self.energy_head(h)
        return energy.squeeze(-1)  # (batch,)


@pytest.fixture
def simple_energy_model() -> SimpleEnergyModel:
    """Create a simple energy model for testing."""
    return SimpleEnergyModel(rngs=nnx.Rngs(0))


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch for testing."""
    return {"data": jax.random.normal(jax.random.key(0), (8, 16))}


@pytest.fixture
def sample_key() -> jax.Array:
    """Create a sample PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# EnergyTrainingConfig Tests
# =============================================================================


class TestEnergyTrainingConfig:
    """Tests for EnergyTrainingConfig dataclass."""

    def test_config_exists(self) -> None:
        """EnergyTrainingConfig should be importable."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainingConfig,
        )

        assert EnergyTrainingConfig is not None

    def test_config_default_values(self) -> None:
        """EnergyTrainingConfig should have sensible defaults."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        assert config.training_method == "cd"
        assert config.mcmc_sampler == "langevin"
        assert config.mcmc_steps == 20
        assert config.step_size == 0.01
        assert config.noise_scale == 0.005
        assert config.gradient_clipping == 1.0
        assert config.replay_buffer_size == 10000
        assert config.replay_buffer_init_prob == 0.95
        assert config.energy_regularization == 0.0
        assert config.gradient_penalty_weight == 0.0

    def test_config_custom_values(self) -> None:
        """EnergyTrainingConfig should accept custom values."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(
            training_method="pcd",
            mcmc_sampler="langevin",
            mcmc_steps=50,
            step_size=0.02,
            noise_scale=0.01,
            gradient_clipping=2.0,
            replay_buffer_size=5000,
            replay_buffer_init_prob=0.9,
            energy_regularization=0.01,
            gradient_penalty_weight=10.0,
        )
        assert config.training_method == "pcd"
        assert config.mcmc_steps == 50
        assert config.step_size == 0.02
        assert config.noise_scale == 0.01
        assert config.gradient_clipping == 2.0
        assert config.replay_buffer_size == 5000
        assert config.energy_regularization == 0.01
        assert config.gradient_penalty_weight == 10.0

    def test_config_all_training_methods(self) -> None:
        """EnergyTrainingConfig should support all training methods."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainingConfig,
        )

        for method in ["cd", "pcd", "score_matching"]:
            config = EnergyTrainingConfig(training_method=method)
            assert config.training_method == method


# =============================================================================
# ReplayBuffer Tests
# =============================================================================


class TestReplayBuffer:
    """Tests for PCD replay buffer."""

    def test_buffer_initialization(self) -> None:
        """ReplayBuffer should initialize empty."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            ReplayBuffer,
        )

        buffer = ReplayBuffer(max_size=1000)
        assert buffer.size == 0

    def test_buffer_add_samples(self) -> None:
        """ReplayBuffer should add samples correctly."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            ReplayBuffer,
        )

        buffer = ReplayBuffer(max_size=1000)
        samples = jnp.ones((10, 16))
        buffer.add(samples)
        assert buffer.size == 10

    def test_buffer_wraps_around(self) -> None:
        """ReplayBuffer should wrap around when full."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            ReplayBuffer,
        )

        buffer = ReplayBuffer(max_size=20)
        for i in range(5):
            samples = jnp.ones((10, 16)) * i
            buffer.add(samples)

        # Buffer should still have max_size items
        assert buffer.size == 20

    def test_buffer_sample(self) -> None:
        """ReplayBuffer should sample correctly."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            ReplayBuffer,
        )

        buffer = ReplayBuffer(max_size=1000)
        samples = jax.random.normal(jax.random.key(0), (100, 16))
        buffer.add(samples)

        key = jax.random.key(42)
        sampled = buffer.sample(8, key)
        assert sampled.shape == (8, 16)

    def test_buffer_sample_empty_raises(self) -> None:
        """ReplayBuffer should raise when sampling from empty buffer."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            ReplayBuffer,
        )

        buffer = ReplayBuffer(max_size=1000)

        with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
            buffer.sample(8, jax.random.key(0))


# =============================================================================
# Langevin Dynamics Tests
# =============================================================================


class TestLangevinDynamics:
    """Tests for Langevin dynamics MCMC sampling."""

    def test_langevin_step_output_shape(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """Langevin step should preserve input shape."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        x = sample_batch["data"]
        x_new = trainer.langevin_step(simple_energy_model, x, sample_key)

        assert x_new.shape == x.shape

    def test_langevin_step_modifies_input(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """Langevin step should modify the input samples."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(step_size=0.1)  # Larger step for visibility
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        x = sample_batch["data"]
        x_new = trainer.langevin_step(simple_energy_model, x, sample_key)

        assert not jnp.allclose(x, x_new)

    def test_mcmc_chain_multiple_steps(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """MCMC chain should run multiple steps."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(mcmc_steps=10)
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        x = sample_batch["data"]
        x_final = trainer.run_mcmc_chain(simple_energy_model, x, sample_key)

        assert x_final.shape == x.shape
        assert not jnp.allclose(x, x_final)


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestEnergyLossComputation:
    """Tests for energy model loss computation."""

    def test_compute_energy_shape(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
    ) -> None:
        """compute_energy should return scalar per sample."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        x = sample_batch["data"]
        energy = trainer.compute_energy(simple_energy_model, x)

        assert energy.shape == (8,)  # batch size

    def test_compute_loss_cd(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """CD loss should compute correctly."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(training_method="cd", mcmc_steps=5)
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        loss, metrics = trainer.compute_loss(simple_energy_model, sample_batch, sample_key)

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()
        assert "loss" in metrics
        assert "energy_data" in metrics
        assert "energy_neg" in metrics
        assert "energy_gap" in metrics

    def test_compute_loss_pcd(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """PCD loss should use replay buffer."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(
            training_method="pcd",
            mcmc_steps=5,
            replay_buffer_size=100,
        )
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        # Run a few steps to populate buffer
        for i in range(5):
            key = jax.random.fold_in(sample_key, i)
            trainer.compute_loss(simple_energy_model, sample_batch, key)

        # Buffer should be populated
        assert trainer._replay_buffer is not None
        assert trainer._replay_buffer.size > 0

    def test_compute_loss_score_matching(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """Score matching loss should compute correctly."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(training_method="score_matching")
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        loss, metrics = trainer.compute_loss(simple_energy_model, sample_batch, sample_key)

        assert isinstance(loss, jax.Array)
        assert "loss" in metrics
        assert "score_norm" in metrics

    def test_energy_regularization(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """Energy regularization should be added to loss."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(
            training_method="cd",
            mcmc_steps=3,
            energy_regularization=0.1,
        )
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        _, metrics = trainer.compute_loss(simple_energy_model, sample_batch, sample_key)

        assert "energy_reg" in metrics
        assert metrics["energy_reg"] > 0

    def test_gradient_penalty(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """Gradient penalty should be added to loss."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(
            training_method="cd",
            mcmc_steps=3,
            gradient_penalty_weight=1.0,
        )
        nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        _, metrics = trainer.compute_loss(simple_energy_model, sample_batch, sample_key)

        assert "gradient_penalty" in metrics


# =============================================================================
# Training Step Tests
# =============================================================================


class TestEnergyTrainStep:
    """Tests for energy model training step."""

    def test_train_step_updates_model(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """train_step should update model parameters."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(training_method="cd", mcmc_steps=3)
        optimizer = nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        # Get initial params
        initial_params = nnx.state(simple_energy_model, nnx.Param)
        initial_kernel = initial_params["layer1"]["kernel"].value.copy()

        # Run train step
        trainer.train_step(simple_energy_model, optimizer, sample_batch, sample_key)

        # Get updated params
        updated_params = nnx.state(simple_energy_model, nnx.Param)
        updated_kernel = updated_params["layer1"]["kernel"].value

        # Params should have changed
        assert not jnp.allclose(initial_kernel, updated_kernel)

    def test_train_step_returns_metrics(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_batch: dict,
        sample_key: jax.Array,
    ) -> None:
        """train_step should return loss and metrics."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(training_method="cd", mcmc_steps=3)
        optimizer = nnx.Optimizer(simple_energy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = EnergyTrainer(config)

        loss, metrics = trainer.train_step(simple_energy_model, optimizer, sample_batch, sample_key)

        assert isinstance(loss, jax.Array)
        assert "energy_data" in metrics
        assert "energy_neg" in metrics


# =============================================================================
# Generation Tests
# =============================================================================


class TestEnergyGeneration:
    """Tests for sample generation from energy models."""

    def test_generate_samples_shape(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_key: jax.Array,
    ) -> None:
        """generate_samples should return correct shape."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        trainer = EnergyTrainer(config)

        samples = trainer.generate_samples(
            simple_energy_model,
            batch_size=4,
            key=sample_key,
            num_steps=10,
            shape=(16,),
        )

        assert samples.shape == (4, 16)

    def test_generate_samples_from_init(
        self,
        simple_energy_model: SimpleEnergyModel,
        sample_key: jax.Array,
    ) -> None:
        """generate_samples should accept initial samples."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        trainer = EnergyTrainer(config)

        x_init = jnp.zeros((4, 16))
        samples = trainer.generate_samples(
            simple_energy_model,
            batch_size=4,
            key=sample_key,
            num_steps=10,
            x_init=x_init,
        )

        assert samples.shape == (4, 16)
        # Samples should have moved from initialization
        assert not jnp.allclose(samples, x_init)


# =============================================================================
# DRY Integration Tests
# =============================================================================


class TestEnergyLossFunctionIntegration:
    """Tests for using Energy loss with base Trainer."""

    def test_create_loss_fn_for_base_trainer(
        self,
        simple_energy_model: SimpleEnergyModel,
    ) -> None:
        """EnergyTrainer should provide loss_fn compatible with base Trainer."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(training_method="cd", mcmc_steps=3)
        trainer = EnergyTrainer(config)

        # Should be able to create a loss function for the base Trainer
        loss_fn = trainer.create_loss_fn()

        # Loss function should have correct signature: (model, batch, rng) -> (loss, metrics)
        batch = {"data": jax.random.normal(jax.random.key(0), (8, 16))}
        rng = jax.random.key(42)
        loss, metrics = loss_fn(simple_energy_model, batch, rng)

        assert isinstance(loss, jax.Array)
        assert "energy_data" in metrics


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestEnergyTrainerExports:
    """Tests for Energy trainer exports."""

    def test_exports_from_trainers_init(self) -> None:
        """Energy trainer classes should be exported from trainers __init__."""
        from artifex.generative_models.training.trainers import (
            EnergyTrainer,
            EnergyTrainingConfig,
            ReplayBuffer,
        )

        assert EnergyTrainer is not None
        assert EnergyTrainingConfig is not None
        assert ReplayBuffer is not None
