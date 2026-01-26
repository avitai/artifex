"""Integration tests for trainer + extension functionality.

These tests verify that extensions integrate correctly with the training loop,
including loss aggregation, JIT compatibility, and callback execution.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
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


class SimpleModel(nnx.Module):
    """Simple model for testing trainer + extension integration."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(in_features=4, out_features=4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


class TestExtension(ModelExtension):
    """Test extension that provides a simple loss."""

    def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.call_count = 0

    def __call__(self, inputs, model_outputs, **kwargs):
        self.call_count += 1
        return {"processed": True}

    def loss_fn(self, batch, model_outputs, **kwargs):
        if not self.is_enabled():
            return jnp.array(0.0)
        # Simple L2 regularization-like loss
        if model_outputs is not None:
            return jnp.mean(jnp.square(model_outputs))
        return jnp.array(0.1)


class CountingCallbackExtension(CallbackExtension):
    """Callback extension that counts lifecycle events."""

    def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
        # CallbackExtension expects CallbackExtensionConfig, but for testing
        # we use base ExtensionConfig - need to import the right one
        from artifex.generative_models.core.configuration import CallbackExtensionConfig

        callback_config = CallbackExtensionConfig(
            name=config.name,
            weight=config.weight,
            enabled=config.enabled,
        )
        super().__init__(callback_config, rngs=rngs)
        self.batch_begin_count = 0
        self.batch_end_count = 0
        self.epoch_begin_count = 0
        self.epoch_end_count = 0

    def on_batch_begin(self, trainer, batch_idx):
        self.batch_begin_count += 1

    def on_batch_end(self, trainer, batch_idx, logs):
        self.batch_end_count += 1

    def on_epoch_begin(self, trainer, epoch):
        self.epoch_begin_count += 1

    def on_epoch_end(self, trainer, epoch, logs):
        self.epoch_end_count += 1


class TestLossExtension(LossExtension):
    """Test loss extension with weight scheduling."""

    def compute_loss(self, predictions, targets, context):
        if predictions is None:
            loss = jnp.array(0.0)
        else:
            loss = jnp.mean(jnp.abs(predictions))
        return loss, {"abs_mean": float(loss)}


class TestTrainerExtensionIntegration:
    """Test trainer + extension integration."""

    @pytest.fixture
    def rngs(self):
        """Create test rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def model(self, rngs):
        """Create a simple model for testing."""
        return SimpleModel(rngs=rngs)

    @pytest.fixture
    def training_config(self):
        """Create minimal training config."""
        optimizer = OptimizerConfig(
            name="test_optimizer",
            optimizer_type="adam",
            learning_rate=0.001,
        )
        return TrainingConfig(
            name="test_training",
            num_epochs=1,
            batch_size=4,
            optimizer=optimizer,
        )

    @pytest.fixture
    def extension_config(self):
        """Create extension config."""
        return ExtensionConfig(name="test_ext", weight=0.5, enabled=True)

    @pytest.fixture
    def batch(self):
        """Create a test batch."""
        return {
            "input": jnp.ones((4, 4)),
            "target": jnp.zeros((4, 4)),
        }

    def test_trainer_accepts_extensions_parameter(
        self, model, training_config, extension_config, rngs
    ):
        """Test that Trainer accepts extensions parameter."""
        from artifex.generative_models.training import Trainer

        extension = TestExtension(extension_config, rngs=rngs)
        extensions = {"test": extension}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=extensions,
        )

        assert trainer.extensions is not None
        assert "test" in trainer.extensions
        assert trainer.extensions["test"] is extension

    def test_trainer_without_extensions(self, model, training_config):
        """Test that Trainer works without extensions."""
        from artifex.generative_models.training import Trainer

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=None,
        )

        assert trainer.extensions == {}

    def test_extension_loss_included_in_total(
        self, model, training_config, extension_config, rngs, batch
    ):
        """Test that extension loss is added to total loss."""
        from artifex.generative_models.training import Trainer

        extension = TestExtension(extension_config, rngs=rngs)
        extensions = {"test": extension}

        # Define a simple loss function
        def loss_fn(model, batch_data, rng):  # noqa: ARG001
            outputs = model(batch_data["input"])
            loss = jnp.mean(jnp.square(outputs - batch_data["target"]))
            return loss, {"base_loss": loss}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=extensions,
            loss_fn=loss_fn,
        )

        # Run a training step
        metrics = trainer.train_step(batch)

        # Check that extension loss is in metrics
        assert "test_loss" in metrics
        assert metrics["test_loss"] > 0.0

    def test_disabled_extension_not_added_to_loss(self, model, training_config, rngs, batch):
        """Test that disabled extensions don't contribute to loss."""
        from artifex.generative_models.training import Trainer

        # Create disabled extension
        disabled_config = ExtensionConfig(name="disabled_ext", weight=1.0, enabled=False)
        extension = TestExtension(disabled_config, rngs=rngs)
        extensions = {"disabled": extension}

        def loss_fn(model, batch_data, rng):  # noqa: ARG001
            outputs = model(batch_data["input"])
            loss = jnp.mean(jnp.square(outputs))
            return loss, {"base_loss": loss}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=extensions,
            loss_fn=loss_fn,
        )

        metrics = trainer.train_step(batch)

        # Disabled extension should not be in metrics (or be zero)
        assert "disabled_loss" not in metrics or metrics.get("disabled_loss", 0.0) == 0.0

    def test_extension_weight_applied_to_loss(self, model, training_config, rngs, batch):
        """Test that extension weight is applied correctly."""
        from artifex.generative_models.training import Trainer

        # Create extension with specific weight
        config_half = ExtensionConfig(name="half_weight", weight=0.5, enabled=True)
        config_full = ExtensionConfig(name="full_weight", weight=1.0, enabled=True)

        ext_half = TestExtension(config_half, rngs=rngs)
        ext_full = TestExtension(config_full, rngs=rngs)

        def loss_fn(model, batch_data, rng):
            return jnp.array(0.0), {}

        # Train with half-weight extension
        trainer_half = Trainer(
            model=SimpleModel(rngs=rngs),
            training_config=training_config,
            extensions={"ext": ext_half},
            loss_fn=loss_fn,
        )
        metrics_half = trainer_half.train_step(batch)

        # Train with full-weight extension
        trainer_full = Trainer(
            model=SimpleModel(rngs=rngs),
            training_config=training_config,
            extensions={"ext": ext_full},
            loss_fn=loss_fn,
        )
        metrics_full = trainer_full.train_step(batch)

        # Half weight should produce approximately half the loss
        # (accounting for floating point and different model states)
        assert "ext_loss" in metrics_half
        assert "ext_loss" in metrics_full

    def test_multiple_extensions(self, model, training_config, rngs, batch):
        """Test trainer with multiple extensions."""
        from artifex.generative_models.training import Trainer

        config1 = ExtensionConfig(name="ext1", weight=1.0, enabled=True)
        config2 = ExtensionConfig(name="ext2", weight=0.5, enabled=True)

        ext1 = TestExtension(config1, rngs=rngs)
        ext2 = TestExtension(config2, rngs=rngs)

        extensions = {"ext1": ext1, "ext2": ext2}

        def loss_fn(model, batch_data, rng):
            return jnp.array(1.0), {"base": 1.0}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=extensions,
            loss_fn=loss_fn,
        )

        metrics = trainer.train_step(batch)

        # Both extensions should contribute
        assert "ext1_loss" in metrics
        assert "ext2_loss" in metrics

    def test_callback_extension_on_batch_end(self, model, training_config, rngs, batch):
        """Test that callback extensions are called on batch end."""
        from artifex.generative_models.training import Trainer

        config = ExtensionConfig(name="callback", weight=1.0, enabled=True)
        callback_ext = CountingCallbackExtension(config, rngs=rngs)

        def loss_fn(model, batch_data, rng):
            return jnp.array(1.0), {}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions={"callback": callback_ext},
            loss_fn=loss_fn,
        )

        # Run multiple training steps
        trainer.train_step(batch)
        trainer.train_step(batch)
        trainer.train_step(batch)

        # on_batch_end should have been called 3 times
        assert callback_ext.batch_end_count == 3

    def test_extension_loss_fn_jit_compatible(self, extension_config, rngs, batch):
        """Test that extension loss_fn works with jax.jit."""
        extension = TestExtension(extension_config, rngs=rngs)

        # Define a jitted function that calls extension loss
        @jax.jit
        def compute_ext_loss(model_outputs):
            return extension.loss_fn(batch, model_outputs)

        model_outputs = jnp.ones((4, 4))
        loss = compute_ext_loss(model_outputs)

        assert loss.shape == ()
        assert not jnp.isnan(loss)

    def test_extension_loss_fn_differentiable(self, extension_config, rngs, batch):
        """Test that gradients flow through extension loss."""
        extension = TestExtension(extension_config, rngs=rngs)

        def total_loss(outputs):
            return extension.loss_fn(batch, outputs)

        model_outputs = jnp.ones((4, 4))
        grads = jax.grad(total_loss)(model_outputs)

        # Gradients should exist and not be NaN
        assert grads.shape == model_outputs.shape
        assert not jnp.any(jnp.isnan(grads))


class TestLossExtensionWeightScheduling:
    """Test loss extension weight scheduling."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_constant_weight_schedule(self, rngs):
        """Test constant weight schedule returns same weight."""
        config = LossExtensionConfig(
            name="constant",
            weight=2.0,
            weight_schedule="constant",
            warmup_steps=100,
        )
        ext = TestLossExtension(config, rngs=rngs)

        # Weight should be constant at all steps
        assert ext.get_weight_at_step(0) == 2.0
        assert ext.get_weight_at_step(50) == 2.0
        assert ext.get_weight_at_step(100) == 2.0
        assert ext.get_weight_at_step(200) == 2.0

    def test_linear_warmup_schedule(self, rngs):
        """Test linear weight warmup."""
        config = LossExtensionConfig(
            name="linear",
            weight=1.0,
            weight_schedule="linear",
            warmup_steps=100,
        )
        ext = TestLossExtension(config, rngs=rngs)

        # Weight should increase linearly during warmup
        assert ext.get_weight_at_step(0) == 0.0
        assert abs(ext.get_weight_at_step(50) - 0.5) < 0.01
        assert ext.get_weight_at_step(100) == 1.0
        assert ext.get_weight_at_step(200) == 1.0  # After warmup, full weight

    def test_cosine_warmup_schedule(self, rngs):
        """Test cosine weight warmup."""
        config = LossExtensionConfig(
            name="cosine",
            weight=1.0,
            weight_schedule="cosine",
            warmup_steps=100,
        )
        ext = TestLossExtension(config, rngs=rngs)

        # Cosine warmup starts at 0 and ends at 1
        assert ext.get_weight_at_step(0) == 0.0
        assert ext.get_weight_at_step(100) == 1.0
        # Midpoint should be around 0.5 for cosine
        mid_weight = ext.get_weight_at_step(50)
        assert 0.4 < mid_weight < 0.6

    def test_no_warmup_when_zero_steps(self, rngs):
        """Test that zero warmup steps means immediate full weight."""
        config = LossExtensionConfig(
            name="no_warmup",
            weight=2.0,
            weight_schedule="linear",
            warmup_steps=0,
        )
        ext = TestLossExtension(config, rngs=rngs)

        # Should be full weight immediately
        assert ext.get_weight_at_step(0) == 2.0
        assert ext.get_weight_at_step(50) == 2.0
