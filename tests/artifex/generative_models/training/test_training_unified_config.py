"""Test training system with unified configuration."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training.trainer import Trainer


class MockModel(GenerativeModel):
    """Mock model for testing."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__(rngs=rngs)
        self.dense = nnx.Linear(10, 5, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs, **kwargs):
        """Generate samples."""
        key = rngs.sample() if rngs is not None and "sample" in rngs else jax.random.key(0)
        return jax.random.normal(key, (n_samples, 5))

    def loss_fn(self, batch, model_outputs, **kwargs):
        """Compute loss."""
        return jnp.mean((model_outputs - batch["target"]) ** 2)


class TestTrainingUnifiedConfiguration:
    """Test that training system uses unified configuration properly."""

    def test_trainer_requires_training_configuration(self):
        """Test that Trainer requires TrainingConfig, not dict."""
        # Create a simple model
        rngs = nnx.Rngs(42)
        model = MockModel(rngs=rngs)

        # Test with dict config - should raise TypeError
        dict_config = {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "num_epochs": 10,
        }

        with pytest.raises(TypeError, match="training_config must be a TrainingConfig"):
            Trainer(model=model, training_config=dict_config)

    def test_optimizer_configuration_structure(self):
        """Test that optimizer configuration is properly typed."""
        # Create optimizer config
        optimizer_config = OptimizerConfig(
            name="adam_optimizer",
            optimizer_type="adam",
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )

        assert optimizer_config.optimizer_type == "adam"
        assert optimizer_config.learning_rate == 1e-3
        assert optimizer_config.beta1 == 0.9
        assert optimizer_config.beta2 == 0.999
        assert optimizer_config.eps == 1e-8
        assert optimizer_config.weight_decay == 0.0

    def test_scheduler_configuration_structure(self):
        """Test that scheduler configuration is properly typed."""
        # Create scheduler config
        scheduler_config = SchedulerConfig(
            name="cosine_scheduler",
            scheduler_type="cosine",
            warmup_steps=1000,
            min_lr_ratio=0.1,
            cycle_length=None,
        )

        assert scheduler_config.scheduler_type == "cosine"
        assert scheduler_config.warmup_steps == 1000
        assert scheduler_config.min_lr_ratio == 0.1
        assert scheduler_config.cycle_length is None

    def test_training_configuration_with_typed_optimizer(self):
        """Test TrainingConfig with typed optimizer config."""
        # Create optimizer config
        optimizer_config = OptimizerConfig(
            name="adam_optimizer",
            optimizer_type="adam",
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )

        # Create scheduler config
        scheduler_config = SchedulerConfig(
            name="cosine_scheduler",
            scheduler_type="cosine",
            warmup_steps=1000,
            min_lr_ratio=0.1,
            cycle_length=10000,
        )

        # Create training config with typed optimizer and scheduler
        training_config = TrainingConfig(
            name="vae_training",
            batch_size=32,
            num_epochs=100,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            gradient_clip_norm=1.0,
            save_frequency=1000,
        )

        assert training_config.optimizer.learning_rate == 1e-3
        assert training_config.scheduler.warmup_steps == 1000

    def test_trainer_creates_optimizer_from_config(self):
        """Test that Trainer creates optimizer from OptimizerConfiguration."""
        # Create model
        rngs = nnx.Rngs(42)
        model = MockModel(rngs=rngs)

        # Create optimizer config
        optimizer_config = OptimizerConfig(
            name="adam_optimizer",
            optimizer_type="adam",
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )

        # Create training config
        training_config = TrainingConfig(
            name="vae_training",
            batch_size=32,
            num_epochs=100,
            optimizer=optimizer_config,
        )

        # Create trainer
        trainer = Trainer(model=model, training_config=training_config)

        # Verify optimizer was created
        assert trainer.optimizer is not None
        # The optimizer should be an optax optimizer
        assert hasattr(trainer.optimizer, "init")
        assert hasattr(trainer.optimizer, "update")

    def test_backward_compatibility_rejection(self):
        """Test that old dict-based optimizer params are rejected."""
        # Instantiating TrainingConfig without an optimizer should fail validation
        with pytest.raises(Exception):  # Pydantic will raise validation error
            TrainingConfig(
                name="bad_config",
                batch_size=32,
                num_epochs=100,
                # Missing required optimizer field
            )

    def test_scheduler_factory_from_config(self):
        """Test creating LR scheduler from SchedulerConfiguration."""
        from artifex.generative_models.training.schedulers import create_scheduler

        scheduler_config = SchedulerConfig(
            name="cosine_scheduler",
            scheduler_type="cosine",
            warmup_steps=1000,
            min_lr_ratio=0.1,
            cycle_length=10000,
        )

        # Create scheduler
        scheduler = create_scheduler(scheduler_config, base_lr=1e-3)

        # Test scheduler behavior
        assert scheduler(0) < 1e-3  # Warmup phase
        assert scheduler(1000) == 1e-3  # End of warmup
        assert scheduler(10000) < 1e-3  # Decay phase

    def test_exponential_scheduler_config(self):
        """Test exponential scheduler configuration."""
        scheduler_config = SchedulerConfig(
            name="exp_scheduler",
            scheduler_type="exponential",
            decay_rate=0.95,
            decay_steps=1000,
        )

        assert scheduler_config.scheduler_type == "exponential"
        assert scheduler_config.decay_rate == 0.95
        assert scheduler_config.decay_steps == 1000

    def test_linear_scheduler_config(self):
        """Test linear scheduler configuration."""
        scheduler_config = SchedulerConfig(
            name="linear_scheduler",
            scheduler_type="linear",
            warmup_steps=500,
            total_steps=10000,
            min_lr_ratio=0.0,
        )

        assert scheduler_config.scheduler_type == "linear"
        assert scheduler_config.total_steps == 10000
        assert scheduler_config.min_lr_ratio == 0.0

    def test_trainer_rejects_optional_training_config(self):
        """Test that Trainer now requires TrainingConfig."""
        rngs = nnx.Rngs(42)
        model = MockModel(rngs=rngs)

        # Should raise error when training_config is None
        with pytest.raises(TypeError, match="training_config is required"):
            Trainer(model=model, training_config=None)
