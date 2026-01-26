"""TDD tests for gradient accumulation and mixed-precision training.

These tests define the expected behavior for advanced training features:
- GradientAccumulator: Process multiple microbatches before optimizer update
- DynamicLossScaler: Automatic loss scaling for mixed-precision training

References:
    - Gradient accumulation for effective batch sizes
    - Mixed-precision training with dynamic loss scaling
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


class SimpleModel(nnx.Module):
    """Simple model for testing gradient accumulation."""

    def __init__(self, dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc(x)


@pytest.fixture
def model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel(rngs=nnx.Rngs(0))


@pytest.fixture
def optimizer(model: SimpleModel) -> nnx.Optimizer:
    """Create optimizer for testing."""
    return nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create sample batch for testing."""
    return {"data": jax.random.normal(jax.random.key(0), (8, 16))}


@pytest.fixture
def rng_key() -> jax.Array:
    """Create PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# GradientAccumulatorConfig Tests
# =============================================================================


class TestGradientAccumulatorConfig:
    """Tests for GradientAccumulatorConfig dataclass."""

    def test_config_exists(self) -> None:
        """GradientAccumulatorConfig should be importable."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulatorConfig,
        )

        assert GradientAccumulatorConfig is not None

    def test_config_default_values(self) -> None:
        """GradientAccumulatorConfig should have sensible defaults."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig()
        assert config.accumulation_steps == 1
        assert config.normalize_gradients is True

    def test_config_custom_values(self) -> None:
        """GradientAccumulatorConfig should accept custom values."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(
            accumulation_steps=4,
            normalize_gradients=False,
        )
        assert config.accumulation_steps == 4
        assert config.normalize_gradients is False


# =============================================================================
# GradientAccumulator Core Tests
# =============================================================================


class TestGradientAccumulator:
    """Tests for GradientAccumulator class."""

    def test_accumulator_exists(self) -> None:
        """GradientAccumulator should be importable."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
        )

        assert GradientAccumulator is not None

    def test_accumulator_initialization(self) -> None:
        """GradientAccumulator should initialize with config."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4)
        accumulator = GradientAccumulator(config)

        assert accumulator.config.accumulation_steps == 4
        assert accumulator.current_step == 0

    def test_accumulator_reset(self) -> None:
        """GradientAccumulator should reset accumulated gradients."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4)
        accumulator = GradientAccumulator(config)

        # Simulate some accumulation
        accumulator._step_count = 2

        accumulator.reset()

        assert accumulator.current_step == 0
        assert accumulator.accumulated_grads is None

    def test_should_update_after_accumulation_steps(self) -> None:
        """should_update should return True after accumulation_steps."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4)
        accumulator = GradientAccumulator(config)

        # Steps 0, 1, 2 should not update
        for step in range(3):
            assert not accumulator.should_update(step)

        # Step 3 (4th step) should update
        assert accumulator.should_update(3)

        # Step 7 (8th step) should also update
        assert accumulator.should_update(7)

    def test_should_update_every_step_when_accumulation_is_one(self) -> None:
        """should_update should return True every step when accumulation_steps=1."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=1)
        accumulator = GradientAccumulator(config)

        for step in range(10):
            assert accumulator.should_update(step)


# =============================================================================
# Gradient Accumulation Logic Tests
# =============================================================================


class TestGradientAccumulationLogic:
    """Tests for gradient accumulation computation."""

    def test_accumulate_adds_gradients(self, model: SimpleModel) -> None:
        """accumulate should add gradients to accumulated gradients."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4)
        accumulator = GradientAccumulator(config)

        # Create fake gradients as pure JAX arrays (nnx.State leaves are arrays)
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.ones_like(x), state)

        # First accumulation
        accumulator.accumulate(grads)
        assert accumulator.accumulated_grads is not None

        # Second accumulation - should add
        accumulator.accumulate(grads)

        # Check that values are doubled
        for leaf in jax.tree.leaves(accumulator.accumulated_grads):
            assert jnp.allclose(leaf, 2.0)

    def test_get_gradients_with_normalization(self, model: SimpleModel) -> None:
        """get_gradients should normalize by accumulation_steps when enabled."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4, normalize_gradients=True)
        accumulator = GradientAccumulator(config)

        # Create and accumulate gradients (all ones) 4 times
        # Note: jax.tree.map on nnx.State operates on raw arrays, not Param objects
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.ones_like(x), state)

        for _ in range(4):
            accumulator.accumulate(grads)

        # Get normalized gradients
        normalized_grads = accumulator.get_gradients()

        # Should be 1.0 (4.0 / 4)
        for leaf in jax.tree.leaves(normalized_grads):
            assert jnp.allclose(leaf, 1.0)

    def test_get_gradients_without_normalization(self, model: SimpleModel) -> None:
        """get_gradients should not normalize when disabled."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4, normalize_gradients=False)
        accumulator = GradientAccumulator(config)

        # Create and accumulate gradients (all ones) 4 times
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.ones_like(x), state)

        for _ in range(4):
            accumulator.accumulate(grads)

        # Get unnormalized gradients
        unnormalized_grads = accumulator.get_gradients()

        # Should be 4.0 (not normalized)
        for leaf in jax.tree.leaves(unnormalized_grads):
            assert jnp.allclose(leaf, 4.0)

    def test_get_gradients_resets_accumulator(self, model: SimpleModel) -> None:
        """get_gradients should reset the accumulator."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=4)
        accumulator = GradientAccumulator(config)

        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.ones_like(x), state)

        for _ in range(4):
            accumulator.accumulate(grads)

        _ = accumulator.get_gradients()

        # Accumulator should be reset
        assert accumulator.accumulated_grads is None
        assert accumulator.current_step == 0


# =============================================================================
# DynamicLossScalerConfig Tests
# =============================================================================


class TestDynamicLossScalerConfig:
    """Tests for DynamicLossScalerConfig dataclass."""

    def test_config_exists(self) -> None:
        """DynamicLossScalerConfig should be importable."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScalerConfig,
        )

        assert DynamicLossScalerConfig is not None

    def test_config_default_values(self) -> None:
        """DynamicLossScalerConfig should have sensible defaults."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig()
        assert config.initial_scale == 2**15  # 32768
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
        assert config.growth_interval == 2000
        assert config.min_scale == 1.0
        assert config.max_scale == 2**24

    def test_config_custom_values(self) -> None:
        """DynamicLossScalerConfig should accept custom values."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(
            initial_scale=2**10,
            growth_factor=1.5,
            backoff_factor=0.25,
            growth_interval=1000,
        )
        assert config.initial_scale == 2**10
        assert config.growth_factor == 1.5
        assert config.backoff_factor == 0.25
        assert config.growth_interval == 1000


# =============================================================================
# DynamicLossScaler Core Tests
# =============================================================================


class TestDynamicLossScaler:
    """Tests for DynamicLossScaler class."""

    def test_scaler_exists(self) -> None:
        """DynamicLossScaler should be importable."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
        )

        assert DynamicLossScaler is not None

    def test_scaler_initialization(self) -> None:
        """DynamicLossScaler should initialize with config."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(initial_scale=2**10)
        scaler = DynamicLossScaler(config)

        assert scaler.scale == 2**10
        assert scaler.steps_since_growth == 0

    def test_scale_loss(self) -> None:
        """scale_loss should multiply loss by current scale."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(initial_scale=1000.0)
        scaler = DynamicLossScaler(config)

        loss = jnp.array(0.5)
        scaled_loss = scaler.scale_loss(loss)

        assert jnp.allclose(scaled_loss, 500.0)

    def test_unscale_gradients(self, model: SimpleModel) -> None:
        """unscale_gradients should divide gradients by current scale."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(initial_scale=1000.0)
        scaler = DynamicLossScaler(config)

        # Create scaled gradients (all 1000.0)
        # Note: jax.tree.map on nnx.State operates on raw arrays
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 1000.0, state)

        unscaled_grads = scaler.unscale_gradients(grads)

        # Should be 1.0
        for leaf in jax.tree.leaves(unscaled_grads):
            assert jnp.allclose(leaf, 1.0)


# =============================================================================
# Overflow Detection Tests
# =============================================================================


class TestOverflowDetection:
    """Tests for gradient overflow detection."""

    def test_check_overflow_no_overflow(self, model: SimpleModel) -> None:
        """check_overflow should return False for finite gradients."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig()
        scaler = DynamicLossScaler(config)

        # Create finite gradients (jax.tree.map on State operates on raw arrays)
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.ones_like(x), state)

        has_overflow = scaler.check_overflow(grads)

        assert not has_overflow

    def test_check_overflow_with_inf(self, model: SimpleModel) -> None:
        """check_overflow should return True for gradients with inf."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig()
        scaler = DynamicLossScaler(config)

        # Create gradients with inf
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), state)

        has_overflow = scaler.check_overflow(grads)

        assert has_overflow

    def test_check_overflow_with_nan(self, model: SimpleModel) -> None:
        """check_overflow should return True for gradients with nan."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig()
        scaler = DynamicLossScaler(config)

        # Create gradients with nan
        state = nnx.state(model, nnx.Param)
        grads = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), state)

        has_overflow = scaler.check_overflow(grads)

        assert has_overflow


# =============================================================================
# Scale Adjustment Tests
# =============================================================================


class TestScaleAdjustment:
    """Tests for dynamic scale adjustment."""

    def test_update_scale_on_overflow_reduces_scale(self) -> None:
        """update_scale should reduce scale when overflow detected."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(initial_scale=1000.0, backoff_factor=0.5)
        scaler = DynamicLossScaler(config)

        scaler.update_scale(overflow_detected=True)

        assert scaler.scale == 500.0
        assert scaler.steps_since_growth == 0

    def test_update_scale_respects_min_scale(self) -> None:
        """update_scale should not reduce scale below min_scale."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(initial_scale=2.0, backoff_factor=0.5, min_scale=1.0)
        scaler = DynamicLossScaler(config)

        # Multiple overflows should not go below min_scale
        for _ in range(10):
            scaler.update_scale(overflow_detected=True)

        assert scaler.scale >= 1.0

    def test_update_scale_grows_after_interval(self) -> None:
        """update_scale should increase scale after growth_interval successful steps."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(
            initial_scale=1000.0,
            growth_factor=2.0,
            growth_interval=3,
        )
        scaler = DynamicLossScaler(config)

        # 3 successful steps should trigger growth
        for _ in range(3):
            scaler.update_scale(overflow_detected=False)

        assert scaler.scale == 2000.0

    def test_update_scale_respects_max_scale(self) -> None:
        """update_scale should not increase scale above max_scale."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(
            initial_scale=2**23,
            growth_factor=2.0,
            growth_interval=1,
            max_scale=2**24,
        )
        scaler = DynamicLossScaler(config)

        # Multiple growth steps should not exceed max_scale
        for _ in range(10):
            scaler.update_scale(overflow_detected=False)

        assert scaler.scale <= 2**24


# =============================================================================
# Integration with Training Tests
# =============================================================================


class TestIntegration:
    """Tests for integration with training loop."""

    def test_accumulator_with_real_gradients(
        self,
        model: SimpleModel,
        sample_batch: dict,
    ) -> None:
        """GradientAccumulator should work with real computed gradients."""
        from artifex.generative_models.training.gradient_accumulation import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        config = GradientAccumulatorConfig(accumulation_steps=2)
        accumulator = GradientAccumulator(config)

        def loss_fn(m: nnx.Module) -> jax.Array:
            out = m(sample_batch["data"])
            return jnp.mean(out**2)

        # Compute real gradients
        _, grads = nnx.value_and_grad(loss_fn)(model)

        # Accumulate
        accumulator.accumulate(grads)
        assert accumulator.accumulated_grads is not None

    def test_scaler_with_real_loss(self, model: SimpleModel, sample_batch: dict) -> None:
        """DynamicLossScaler should work with real loss computation."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )

        config = DynamicLossScalerConfig(initial_scale=1000.0)
        scaler = DynamicLossScaler(config)

        def loss_fn(m: nnx.Module) -> jax.Array:
            out = m(sample_batch["data"])
            return jnp.mean(out**2)

        # Compute loss
        loss = loss_fn(model)

        # Scale loss
        scaled_loss = scaler.scale_loss(loss)

        assert jnp.isfinite(scaled_loss)
        assert scaled_loss == loss * 1000.0


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestExports:
    """Tests for module exports."""

    def test_exports_from_gradient_accumulation(self) -> None:
        """Classes should be exported from gradient_accumulation module."""
        from artifex.generative_models.training.gradient_accumulation import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        assert GradientAccumulator is not None
        assert GradientAccumulatorConfig is not None
        assert DynamicLossScaler is not None
        assert DynamicLossScalerConfig is not None

    def test_exports_from_training_init(self) -> None:
        """Classes should be exported from training __init__."""
        from artifex.generative_models.training import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
            GradientAccumulator,
            GradientAccumulatorConfig,
        )

        assert GradientAccumulator is not None
        assert GradientAccumulatorConfig is not None
        assert DynamicLossScaler is not None
        assert DynamicLossScalerConfig is not None
