"""Tests for trainer extension integration.

These tests define the expected behavior for extension integration
with the Trainer base class following TDD principles.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    CallbackExtensionConfig,
    ExtensionConfig,
    LossExtensionConfig,
    OptimizerConfig,
    TrainingConfig,
)
from artifex.generative_models.extensions.base import (
    CallbackExtension,
    LossExtension,
    ModelExtension,
)
from artifex.generative_models.training.trainer import Trainer


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class SimpleModel(nnx.Module):
    """Simple model for testing."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.dense = nnx.Linear(in_features=4, out_features=2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)

    def loss_fn(self, batch: dict[str, jax.Array], _rng: jax.Array) -> tuple[jax.Array, dict]:
        """Compute loss for training."""
        x = batch["input"]
        y = batch["target"]
        pred = self(x)
        loss = jnp.mean(jnp.square(pred - y))
        return loss, {"mse": loss}  # Return JAX array, not Python float (JIT compatible)


class TestModelExtension(ModelExtension):
    """Test extension that computes additional loss."""

    def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs, loss_value: float = 0.1):
        super().__init__(config, rngs=rngs)
        # Store value to add to loss
        self.loss_value = loss_value

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model outputs."""
        return {"processed": model_outputs}

    def loss_fn(self, batch: dict[str, Any], model_outputs: Any, **kwargs: Any) -> jax.Array:
        """Compute extension loss."""
        del batch, kwargs  # Unused
        if not self.is_enabled():
            return jnp.array(0.0)
        # Return a simple loss based on model output variance
        return jnp.array(self.loss_value) * jnp.var(model_outputs)


class TestLossExtension(LossExtension):
    """Test loss extension for trainer integration."""

    def __init__(
        self, config: LossExtensionConfig, *, rngs: nnx.Rngs, base_loss_value: float = 0.05
    ):
        super().__init__(config, rngs=rngs)
        self.base_loss_value = base_loss_value

    def compute_loss(
        self, predictions: jax.Array, targets: jax.Array, context: dict
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute loss with metrics."""
        del context  # Unused
        loss = jnp.array(self.base_loss_value) * jnp.mean(jnp.abs(predictions - targets))
        return loss, {"l1_loss": loss}

    def loss_fn(self, batch: dict[str, Any], model_outputs: Any, **kwargs: Any) -> jax.Array:
        """Compute extension loss for trainer integration."""
        del kwargs  # Unused
        if not self.is_enabled():
            return jnp.array(0.0)
        # Compute loss based on model output variance
        if model_outputs is not None:
            return jnp.array(self.base_loss_value) * jnp.var(model_outputs)
        return jnp.array(0.0)


class TestCallbackExtension(CallbackExtension):
    """Test callback extension for lifecycle hooks."""

    def __init__(self, config: CallbackExtensionConfig, *, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.train_begin_called = False
        self.train_end_called = False
        self.batch_begin_count = 0
        self.batch_end_count = 0

    def on_train_begin(self, trainer: Any) -> None:
        del trainer  # Unused
        self.train_begin_called = True

    def on_train_end(self, trainer: Any) -> None:
        del trainer  # Unused
        self.train_end_called = True

    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        del trainer, batch_idx  # Unused
        self.batch_begin_count += 1

    def on_batch_end(self, trainer: Any, batch_idx: int, logs: dict) -> None:
        del trainer, batch_idx, logs  # Unused
        self.batch_end_count += 1


@pytest.fixture
def rngs():
    """Create RNGs for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def simple_model(rngs):
    """Create a simple model for testing."""
    return SimpleModel(rngs=rngs)


@pytest.fixture
def training_config():
    """Create training config for testing."""
    return TrainingConfig(
        name="test-trainer",
        batch_size=4,
        num_epochs=2,
        optimizer=OptimizerConfig(
            name="adam",
            optimizer_type="adam",
            learning_rate=0.001,
        ),
    )


@pytest.fixture
def extension_config():
    """Create extension config for testing."""
    return ExtensionConfig(
        name="test-ext",
        weight=1.0,
        enabled=True,
    )


@pytest.fixture
def loss_extension_config():
    """Create loss extension config for testing."""
    return LossExtensionConfig(
        name="test-loss-ext",
        weight=0.5,
        enabled=True,
        weight_schedule="constant",
        warmup_steps=0,
    )


@pytest.fixture
def callback_extension_config():
    """Create callback extension config for testing."""
    return CallbackExtensionConfig(
        name="test-callback",
        weight=1.0,
        enabled=True,
        frequency=1,
    )


@pytest.fixture
def test_batch():
    """Create test batch for training."""
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    return {
        "input": jax.random.normal(key1, (4, 4)),
        "target": jax.random.normal(key2, (4, 2)),
    }


# =============================================================================
# Trainer Extensions Parameter Tests
# =============================================================================


class TestTrainerExtensionsParameter:
    """Tests for trainer extensions parameter."""

    def test_trainer_accepts_extensions_parameter(
        self, simple_model, training_config, extension_config, rngs
    ):
        """Trainer should accept extensions parameter."""
        extension = TestModelExtension(extension_config, rngs=rngs)
        extensions = {"test_ext": extension}

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions=extensions,
        )

        assert hasattr(trainer, "extensions")
        assert trainer.extensions == extensions

    def test_trainer_extensions_default_none(self, simple_model, training_config):
        """Trainer extensions should default to empty dict when None."""
        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
        )

        assert hasattr(trainer, "extensions")
        assert trainer.extensions == {}

    def test_trainer_accepts_empty_extensions(self, simple_model, training_config):
        """Trainer should accept empty extensions dict."""
        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions={},
        )

        assert trainer.extensions == {}

    def test_trainer_accepts_multiple_extensions(
        self, simple_model, training_config, extension_config, loss_extension_config, rngs
    ):
        """Trainer should accept multiple extensions."""
        model_ext = TestModelExtension(extension_config, rngs=rngs)
        loss_ext = TestLossExtension(loss_extension_config, rngs=rngs)

        extensions = {
            "model_ext": model_ext,
            "loss_ext": loss_ext,
        }

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions=extensions,
        )

        assert len(trainer.extensions) == 2
        assert "model_ext" in trainer.extensions
        assert "loss_ext" in trainer.extensions


# =============================================================================
# Extension Loss Aggregation Tests
# =============================================================================


class TestExtensionLossAggregation:
    """Tests for extension loss aggregation in training."""

    def test_train_step_includes_extension_loss(
        self, simple_model, training_config, extension_config, rngs, test_batch
    ):
        """Train step should include extension losses in total loss."""
        extension = TestModelExtension(extension_config, rngs=rngs)
        extensions = {"test_ext": extension}

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions=extensions,
        )

        metrics = trainer.train_step(test_batch)

        # Metrics should include extension loss
        assert "test_ext_loss" in metrics
        assert metrics["test_ext_loss"] >= 0.0

    def test_train_step_aggregates_multiple_extension_losses(
        self,
        simple_model,
        training_config,
        extension_config,
        loss_extension_config,
        rngs,
        test_batch,
    ):
        """Train step should aggregate multiple extension losses."""
        model_ext = TestModelExtension(extension_config, rngs=rngs)
        loss_ext = TestLossExtension(loss_extension_config, rngs=rngs)

        extensions = {
            "model_ext": model_ext,
            "loss_ext": loss_ext,
        }

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions=extensions,
        )

        metrics = trainer.train_step(test_batch)

        # Both extension losses should be in metrics
        assert "model_ext_loss" in metrics
        assert "loss_ext_loss" in metrics

    def test_total_loss_includes_weighted_extension_losses(
        self, simple_model, training_config, rngs, test_batch
    ):
        """Total loss should include extension losses weighted by config."""
        config_weight_2 = ExtensionConfig(
            name="weighted-ext",
            weight=2.0,
            enabled=True,
        )
        extension = TestModelExtension(config_weight_2, rngs=rngs, loss_value=0.1)

        # Trainer without extensions
        trainer_no_ext = Trainer(
            model=simple_model,
            training_config=training_config,
        )

        # Trainer with extensions
        trainer_with_ext = Trainer(
            model=SimpleModel(rngs=rngs),  # Fresh model
            training_config=training_config,
            extensions={"test_ext": extension},
        )

        # Both trainers should compute different total losses
        # (due to extension contribution)
        metrics_no_ext = trainer_no_ext.train_step(test_batch)
        metrics_with_ext = trainer_with_ext.train_step(test_batch)

        # The trainer with extension should have additional loss components
        assert "test_ext_loss" in metrics_with_ext
        assert "test_ext_loss" not in metrics_no_ext

    def test_disabled_extension_not_included_in_loss(
        self, simple_model, training_config, rngs, test_batch
    ):
        """Disabled extensions should not contribute to loss."""
        disabled_config = ExtensionConfig(
            name="disabled-ext",
            weight=1.0,
            enabled=False,
        )
        extension = TestModelExtension(disabled_config, rngs=rngs, loss_value=0.1)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions={"disabled_ext": extension},
        )

        metrics = trainer.train_step(test_batch)

        # Disabled extension loss should be zero
        assert metrics.get("disabled_ext_loss", 0.0) == 0.0


# =============================================================================
# Extension Gradients Tests
# =============================================================================


class TestExtensionGradients:
    """Tests for gradient flow through extensions."""

    def test_gradients_flow_through_extension_loss(
        self, simple_model, training_config, extension_config, rngs, test_batch
    ):
        """Gradients should flow through extension loss to model."""
        extension = TestModelExtension(extension_config, rngs=rngs)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions={"test_ext": extension},
        )

        # Get initial params
        initial_params = nnx.state(trainer.model, nnx.Param)
        initial_dense_kernel = initial_params["dense"]["kernel"].value.copy()

        # Run train step
        trainer.train_step(test_batch)

        # Get updated params
        updated_params = nnx.state(trainer.model, nnx.Param)
        updated_dense_kernel = updated_params["dense"]["kernel"].value

        # Params should have changed (gradients flowed)
        assert not jnp.allclose(initial_dense_kernel, updated_dense_kernel)


# =============================================================================
# Callback Extension Tests
# =============================================================================


class TestCallbackExtensionIntegration:
    """Tests for callback extension integration."""

    def test_callback_on_batch_end_called(
        self, simple_model, training_config, callback_extension_config, rngs, test_batch
    ):
        """Callback on_batch_end should be called after each train step."""
        callback_ext = TestCallbackExtension(callback_extension_config, rngs=rngs)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions={"callback": callback_ext},
        )

        # Run multiple train steps
        trainer.train_step(test_batch)
        trainer.train_step(test_batch)

        # Callback should have been called twice
        assert callback_ext.batch_end_count == 2


# =============================================================================
# Extension State Serialization Tests
# =============================================================================


class TestExtensionStateSerialization:
    """Tests for extension state in checkpoints."""

    def test_checkpoint_includes_extension_state(
        self, simple_model, training_config, extension_config, rngs, test_batch, tmp_path
    ):
        """Checkpoint should include extension state."""
        extension = TestModelExtension(extension_config, rngs=rngs)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions={"test_ext": extension},
            checkpoint_dir=str(tmp_path),
        )

        # Run a train step
        trainer.train_step(test_batch)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pkl"
        trainer.save_checkpoint(str(checkpoint_path))

        # Load checkpoint and verify extension state is included
        import pickle

        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        assert "extensions_state" in checkpoint


# =============================================================================
# JIT Compatibility Tests
# =============================================================================


class TestTrainerExtensionJITCompatibility:
    """Tests for JIT compatibility with extensions."""

    def test_train_step_jit_compatible_with_extensions(
        self, simple_model, training_config, extension_config, rngs, test_batch
    ):
        """Train step should be JIT compatible with extensions."""
        extension = TestModelExtension(extension_config, rngs=rngs)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            extensions={"test_ext": extension},
        )

        # Should not raise any JIT compilation errors
        metrics = trainer.train_step(test_batch)
        assert "loss" in metrics
        assert not jnp.isnan(metrics["loss"])
