"""Tests for optimizer factory.

Following TDD principles - these tests define the expected behavior
for the centralized optimizer factory.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from artifex.generative_models.core.configuration import (
    OptimizerConfig,
)


class TestOptimizerFactoryBasic:
    """Test basic optimizer creation via factory."""

    def test_create_adam_optimizer(self):
        """Factory should create Adam optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

        # Verify it works with a simple param tree
        params = {"w": jnp.zeros((4, 4))}
        opt_state = optimizer.init(params)
        grads = {"w": jnp.ones((4, 4))}
        updates, new_state = optimizer.update(grads, opt_state, params)

        # Updates should be computed
        assert "w" in updates
        assert not jnp.allclose(updates["w"], jnp.zeros((4, 4)))

    def test_create_adamw_optimizer(self):
        """Factory should create AdamW optimizer with weight decay."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay=0.01,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_sgd_optimizer(self):
        """Factory should create SGD optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="sgd",
            learning_rate=1e-2,
            momentum=0.9,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_sgd_with_nesterov(self):
        """Factory should create SGD with Nesterov momentum."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="sgd",
            learning_rate=1e-2,
            momentum=0.9,
            nesterov=True,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_rmsprop_optimizer(self):
        """Factory should create RMSProp optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="rmsprop",
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_adagrad_optimizer(self):
        """Factory should create AdaGrad optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adagrad",
            learning_rate=1e-2,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_lamb_optimizer(self):
        """Factory should create LAMB optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="lamb",
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_radam_optimizer(self):
        """Factory should create RAdam optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="radam",
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

    def test_create_nadam_optimizer(self):
        """Factory should create NAdam optimizer."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="nadam",
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None


class TestOptimizerFactoryGradientClipping:
    """Test gradient clipping options."""

    def test_optimizer_with_gradient_clip_norm(self):
        """Factory should apply gradient clipping by norm."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=1e-3,
            gradient_clip_norm=1.0,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None

        # Test that clipping is applied
        params = {"w": jnp.zeros((4, 4))}
        opt_state = optimizer.init(params)

        # Large gradients should be clipped
        large_grads = {"w": jnp.ones((4, 4)) * 100.0}
        updates, _ = optimizer.update(large_grads, opt_state, params)

        # Updates should be bounded
        update_norm = jnp.sqrt(jnp.sum(updates["w"] ** 2))
        # After clipping and Adam scaling, norm should be reasonable
        assert jnp.isfinite(update_norm)

    def test_optimizer_with_gradient_clip_value(self):
        """Factory should apply gradient clipping by value."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=1e-3,
            gradient_clip_value=1.0,
        )

        optimizer = create_optimizer(config)
        assert optimizer is not None


class TestOptimizerFactoryWithSchedule:
    """Test optimizer creation with learning rate schedules."""

    def test_optimizer_with_constant_schedule(self):
        """Factory should work with constant learning rate."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=1e-3,
        )

        # Pass constant as schedule
        optimizer = create_optimizer(config, schedule=1e-3)
        assert optimizer is not None

    def test_optimizer_with_callable_schedule(self):
        """Factory should accept a callable schedule."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=1e-3,  # Base, but schedule overrides
        )

        # Create a custom schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=1000,
        )

        optimizer = create_optimizer(config, schedule=schedule)
        assert optimizer is not None

    def test_optimizer_uses_schedule_over_config_lr(self):
        """When schedule is provided, it should override config learning_rate."""
        from artifex.generative_models.training.optimizers import create_optimizer

        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=1e-3,  # This should be ignored when schedule provided
        )

        # Schedule that produces different learning rates
        schedule = optax.constant_schedule(5e-4)

        optimizer = create_optimizer(config, schedule=schedule)
        assert optimizer is not None


class TestOptimizerFactoryIntegrationWithTrainer:
    """Test that optimizer factory integrates with Trainer correctly."""

    def test_trainer_can_use_factory_created_optimizer(self):
        """Trainer should work with factory-created optimizers."""
        from artifex.generative_models.core.configuration import (
            TrainingConfig,
        )
        from artifex.generative_models.training.optimizers import create_optimizer
        from artifex.generative_models.training.trainer import Trainer

        # Create a simple model
        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.linear = nnx.Linear(4, 4, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(42))

        optimizer_config = OptimizerConfig(
            name="test_optimizer",
            optimizer_type="adam",
            learning_rate=1e-3,
        )

        training_config = TrainingConfig(
            name="test_training",
            optimizer=optimizer_config,
            batch_size=4,
            num_epochs=2,
        )

        # Create optimizer via factory
        optimizer = create_optimizer(optimizer_config)

        def loss_fn(model, batch, rng):
            x = batch["x"]
            y = model(x)
            loss = jnp.mean(y**2)
            return loss, {"loss": loss}

        # Trainer should accept the factory-created optimizer
        trainer = Trainer(
            model=model,
            training_config=training_config,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        # Should be able to train
        batch = {"x": jax.random.normal(jax.random.PRNGKey(0), (4, 4))}
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert jnp.isfinite(metrics["loss"])


class TestOptimizerFactoryExport:
    """Test that optimizer factory is properly exported."""

    def test_create_optimizer_exported_from_optimizers_module(self):
        """create_optimizer should be importable from optimizers module."""
        from artifex.generative_models.training.optimizers import create_optimizer

        assert callable(create_optimizer)

    def test_create_optimizer_exported_from_training_module(self):
        """create_optimizer should be importable from training module."""
        from artifex.generative_models.training import create_optimizer

        assert callable(create_optimizer)
