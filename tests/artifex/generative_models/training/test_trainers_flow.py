"""TDD tests for Flow-specific trainer.

These tests define the expected behavior for flow matching training based on
state-of-the-art techniques (2024-2025):
- Conditional Flow Matching (CFM)
- Optimal Transport CFM (OT-CFM)
- Rectified Flow
- Time sampling strategies (uniform, logit-normal, u-shaped)

References:
    - Flow Matching: https://arxiv.org/abs/2210.02747
    - OT-CFM: https://arxiv.org/abs/2302.00482
    - Rectified Flow: https://arxiv.org/abs/2209.03003
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


class SimpleFlowModel(nnx.Module):
    """Simple flow model (velocity field) for testing trainer functionality."""

    def __init__(self, dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(dim, 32, rngs=rngs)
        self.time_embed = nnx.Linear(1, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        # Simple architecture: concat time embedding with features
        h = nnx.relu(self.fc1(x))
        t_embed = nnx.relu(self.time_embed(t[:, None] if t.ndim == 1 else t))
        h = h + t_embed
        return self.fc2(h)


@pytest.fixture
def flow_model() -> SimpleFlowModel:
    """Create a simple flow model for testing."""
    return SimpleFlowModel(rngs=nnx.Rngs(0))


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch for testing."""
    return {"image": jax.random.normal(jax.random.key(0), (8, 16))}


@pytest.fixture
def rng_key() -> jax.Array:
    """Create PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# FlowTrainingConfig Tests
# =============================================================================


class TestFlowTrainingConfig:
    """Tests for FlowTrainingConfig dataclass."""

    def test_config_exists(self) -> None:
        """FlowTrainingConfig should be importable."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainingConfig,
        )

        assert FlowTrainingConfig is not None

    def test_config_default_values(self) -> None:
        """FlowTrainingConfig should have sensible defaults."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        assert config.flow_type == "cfm"
        assert config.sigma_min == 0.001
        assert config.time_sampling == "uniform"
        assert config.use_ot is False

    def test_config_custom_values(self) -> None:
        """FlowTrainingConfig should accept custom values."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(
            flow_type="ot_cfm",
            sigma_min=0.01,
            time_sampling="logit_normal",
            use_ot=True,
            ot_regularization=0.05,
        )
        assert config.flow_type == "ot_cfm"
        assert config.sigma_min == 0.01
        assert config.time_sampling == "logit_normal"
        assert config.use_ot is True
        assert config.ot_regularization == 0.05

    def test_config_all_flow_types(self) -> None:
        """FlowTrainingConfig should support all flow types."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainingConfig,
        )

        for flow_type in ["cfm", "ot_cfm", "rectified_flow"]:
            config = FlowTrainingConfig(flow_type=flow_type)
            assert config.flow_type == flow_type

    def test_config_all_time_sampling(self) -> None:
        """FlowTrainingConfig should support all time sampling strategies."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainingConfig,
        )

        for sampling in ["uniform", "logit_normal", "u_shaped"]:
            config = FlowTrainingConfig(time_sampling=sampling)
            assert config.time_sampling == sampling


# =============================================================================
# Time Sampling Tests
# =============================================================================


class TestTimeSampling:
    """Tests for time sampling strategies."""

    def test_uniform_sampling_range(
        self,
        flow_model: SimpleFlowModel,
        rng_key: jax.Array,
    ) -> None:
        """Uniform sampling should return time values in [0, 1]."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(time_sampling="uniform")
        trainer = FlowTrainer(config)

        t = trainer.sample_time(100, rng_key)

        assert t.shape == (100, 1)
        assert jnp.all(t >= 0)
        assert jnp.all(t <= 1)

    def test_logit_normal_sampling_range(
        self,
        flow_model: SimpleFlowModel,
        rng_key: jax.Array,
    ) -> None:
        """Logit-normal sampling should return time values in (0, 1)."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(time_sampling="logit_normal")
        trainer = FlowTrainer(config)

        t = trainer.sample_time(100, rng_key)

        assert t.shape == (100, 1)
        assert jnp.all(t > 0)
        assert jnp.all(t < 1)

    def test_logit_normal_favors_middle(
        self,
        flow_model: SimpleFlowModel,
        rng_key: jax.Array,
    ) -> None:
        """Logit-normal should favor middle time values."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(time_sampling="logit_normal")
        trainer = FlowTrainer(config)

        t = trainer.sample_time(1000, rng_key)

        # Count samples in middle vs edges
        mid_count = jnp.sum((t > 0.3) & (t < 0.7))
        edge_count = jnp.sum((t < 0.2) | (t > 0.8))

        # Middle should have more samples
        assert mid_count > edge_count

    def test_u_shaped_sampling_range(
        self,
        flow_model: SimpleFlowModel,
        rng_key: jax.Array,
    ) -> None:
        """U-shaped sampling should return time values in [0, 1]."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(time_sampling="u_shaped")
        trainer = FlowTrainer(config)

        t = trainer.sample_time(100, rng_key)

        assert t.shape == (100, 1)
        assert jnp.all(t >= 0)
        assert jnp.all(t <= 1)

    def test_u_shaped_favors_endpoints(
        self,
        flow_model: SimpleFlowModel,
        rng_key: jax.Array,
    ) -> None:
        """U-shaped should favor endpoint time values for rectified flows."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(time_sampling="u_shaped")
        trainer = FlowTrainer(config)

        t = trainer.sample_time(1000, rng_key)

        # Count samples at edges vs middle
        edge_count = jnp.sum((t < 0.2) | (t > 0.8))
        mid_count = jnp.sum((t > 0.4) & (t < 0.6))

        # Edges should have more samples than middle
        assert edge_count > mid_count


# =============================================================================
# Conditional Vector Field Tests
# =============================================================================


class TestConditionalVectorField:
    """Tests for conditional vector field computation."""

    def test_linear_interpolation(
        self,
        flow_model: SimpleFlowModel,
    ) -> None:
        """Linear interpolation: x_t = (1-t)x0 + t*x1."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        x0 = jnp.zeros((2, 16))  # Source (noise)
        x1 = jnp.ones((2, 16))  # Target (data)
        t = jnp.array([[0.5], [0.5]])

        x_t, u_t = trainer.compute_conditional_vector_field(x0, x1, t)

        # At t=0.5, x_t should be 0.5 (midpoint)
        assert jnp.allclose(x_t, 0.5 * jnp.ones_like(x_t))

    def test_target_velocity_is_difference(
        self,
        flow_model: SimpleFlowModel,
    ) -> None:
        """Target velocity should be x1 - x0."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        x0 = jnp.zeros((2, 16))
        x1 = jnp.ones((2, 16)) * 2.0
        t = jnp.array([[0.3], [0.7]])

        x_t, u_t = trainer.compute_conditional_vector_field(x0, x1, t)

        # Target velocity should be x1 - x0 = 2.0
        assert jnp.allclose(u_t, 2.0 * jnp.ones_like(u_t))

    def test_interpolation_at_endpoints(
        self,
        flow_model: SimpleFlowModel,
    ) -> None:
        """At t=0, x_t=x0; at t=1, x_t=x1."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        x0 = jnp.zeros((2, 16))
        x1 = jnp.ones((2, 16))

        # At t=0
        x_t_0, _ = trainer.compute_conditional_vector_field(x0, x1, jnp.array([[0.0], [0.0]]))
        assert jnp.allclose(x_t_0, x0)

        # At t=1
        x_t_1, _ = trainer.compute_conditional_vector_field(x0, x1, jnp.array([[1.0], [1.0]]))
        assert jnp.allclose(x_t_1, x1)


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestFlowLossComputation:
    """Tests for flow matching loss computation."""

    def test_compute_loss_returns_scalar(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """compute_loss should return scalar loss."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        loss, metrics = trainer.compute_loss(flow_model, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_compute_loss_returns_metrics(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """compute_loss should return metrics dict."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        loss, metrics = trainer.compute_loss(flow_model, sample_batch, rng_key)

        assert "loss" in metrics

    def test_cfm_loss_is_mse(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """CFM loss should be MSE between predicted and target velocity."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(flow_type="cfm")
        trainer = FlowTrainer(config)

        loss, _ = trainer.compute_loss(flow_model, sample_batch, rng_key)

        # Loss should be non-negative (MSE)
        assert loss >= 0


# =============================================================================
# Training Step Tests
# =============================================================================


class TestFlowTrainStep:
    """Tests for flow training step."""

    def test_train_step_updates_model(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should update model parameters."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        optimizer = nnx.Optimizer(flow_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = FlowTrainer(config)

        # Get initial params
        initial_params = nnx.state(flow_model, nnx.Param)
        initial_fc1_kernel = initial_params["fc1"]["kernel"].value.copy()

        # Run train step
        trainer.train_step(flow_model, optimizer, sample_batch, rng_key)

        # Get updated params
        updated_params = nnx.state(flow_model, nnx.Param)
        updated_fc1_kernel = updated_params["fc1"]["kernel"].value

        # Params should have changed
        assert not jnp.allclose(initial_fc1_kernel, updated_fc1_kernel)

    def test_train_step_returns_metrics(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should return loss and metrics."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        optimizer = nnx.Optimizer(flow_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = FlowTrainer(config)

        loss, metrics = trainer.train_step(flow_model, optimizer, sample_batch, rng_key)

        assert isinstance(loss, jax.Array)
        assert "loss" in metrics


# =============================================================================
# DRY Integration Tests
# =============================================================================


class TestFlowDRYIntegration:
    """Tests for DRY integration with base Trainer."""

    def test_create_loss_fn_signature(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """FlowTrainer should provide loss_fn compatible with base Trainer."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        # Should be able to create a loss function for the base Trainer
        loss_fn = trainer.create_loss_fn()

        # Loss function should have correct signature: (model, batch, rng) -> (loss, metrics)
        loss, metrics = loss_fn(flow_model, sample_batch, rng_key)

        assert isinstance(loss, jax.Array)
        assert "loss" in metrics


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestFlowTrainerExports:
    """Tests for Flow trainer exports."""

    def test_exports_from_trainers_init(self) -> None:
        """Flow trainer classes should be exported from trainers __init__."""
        from artifex.generative_models.training.trainers import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        assert FlowTrainer is not None
        assert FlowTrainingConfig is not None
