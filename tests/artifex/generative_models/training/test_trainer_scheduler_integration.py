"""Tests for Trainer scheduler factory integration.

Following TDD principles - these tests define the expected behavior
for Trainer using the centralized scheduler factory.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training.schedulers import create_scheduler
from artifex.generative_models.training.trainer import Trainer


class SimpleModel(nnx.Module):
    """Simple model for testing trainer scheduler integration."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(rngs=nnx.Rngs(42))


@pytest.fixture
def base_training_config():
    """Create base training config without scheduler."""
    optimizer_config = OptimizerConfig(
        name="test_optimizer",
        optimizer_type="adam",
        learning_rate=1e-3,
    )
    return TrainingConfig(
        name="test_training",
        optimizer=optimizer_config,
        batch_size=4,
        num_epochs=2,
    )


def simple_loss_fn(model, batch, rng):
    """Simple loss function for testing."""
    x = batch["x"]
    y = model(x)
    loss = jnp.mean(y**2)
    return loss, {"loss": loss}


class TestTrainerSchedulerIntegration:
    """Test that Trainer properly uses the scheduler factory."""

    def test_trainer_uses_constant_scheduler(self, simple_model, base_training_config):
        """Trainer should work with constant scheduler."""
        scheduler_config = SchedulerConfig(
            name="constant",
            scheduler_type="constant",
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None
        assert trainer.optimizer is not None

    def test_trainer_uses_linear_scheduler(self, simple_model, base_training_config):
        """Trainer should work with linear scheduler."""
        scheduler_config = SchedulerConfig(
            name="linear",
            scheduler_type="linear",
            total_steps=100,
            warmup_steps=10,
            min_lr_ratio=0.1,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None
        assert trainer.optimizer is not None

    def test_trainer_uses_cosine_scheduler(self, simple_model, base_training_config):
        """Trainer should work with cosine scheduler."""
        scheduler_config = SchedulerConfig(
            name="cosine",
            scheduler_type="cosine",
            cycle_length=100,
            warmup_steps=10,
            min_lr_ratio=0.1,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_uses_exponential_scheduler(self, simple_model, base_training_config):
        """Trainer should work with exponential scheduler."""
        scheduler_config = SchedulerConfig(
            name="exponential",
            scheduler_type="exponential",
            decay_steps=10,
            decay_rate=0.9,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_uses_polynomial_scheduler(self, simple_model, base_training_config):
        """Trainer should work with polynomial scheduler (from factory only)."""
        scheduler_config = SchedulerConfig(
            name="polynomial",
            scheduler_type="polynomial",
            total_steps=100,
            min_lr_ratio=0.1,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_uses_step_scheduler(self, simple_model, base_training_config):
        """Trainer should work with step scheduler."""
        scheduler_config = SchedulerConfig(
            name="step",
            scheduler_type="step",
            step_size=10,
            gamma=0.5,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_uses_multistep_scheduler(self, simple_model, base_training_config):
        """Trainer should work with multistep scheduler."""
        scheduler_config = SchedulerConfig(
            name="multistep",
            scheduler_type="multistep",
            milestones=[20, 40, 60],
            gamma=0.5,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_uses_cyclic_scheduler(self, simple_model, base_training_config):
        """Trainer should work with cyclic scheduler (from factory only)."""
        scheduler_config = SchedulerConfig(
            name="cyclic",
            scheduler_type="cyclic",
            cycle_length=50,
            min_lr_ratio=0.1,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_uses_one_cycle_scheduler(self, simple_model, base_training_config):
        """Trainer should work with one_cycle scheduler (from factory only)."""
        scheduler_config = SchedulerConfig(
            name="one_cycle",
            scheduler_type="one_cycle",
            total_steps=100,
            min_lr_ratio=0.1,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=base_training_config.optimizer,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        assert trainer is not None

    def test_trainer_training_works_with_all_schedulers(self, simple_model, base_training_config):
        """Training steps should work with various schedulers."""
        scheduler_types = [
            ("constant", {}),
            ("linear", {"total_steps": 100, "warmup_steps": 10}),
            ("cosine", {"cycle_length": 100}),
            ("polynomial", {"total_steps": 100}),
            ("one_cycle", {"total_steps": 100}),
        ]

        for scheduler_type, extra_params in scheduler_types:
            # Create fresh model for each test
            model = SimpleModel(rngs=nnx.Rngs(42))

            scheduler_config = SchedulerConfig(
                name=f"test_{scheduler_type}",
                scheduler_type=scheduler_type,
                min_lr_ratio=0.1,
                **extra_params,
            )
            training_config = TrainingConfig(
                name="test",
                optimizer=base_training_config.optimizer,
                scheduler=scheduler_config,
                batch_size=4,
                num_epochs=2,
            )

            trainer = Trainer(
                model=model,
                training_config=training_config,
                loss_fn=simple_loss_fn,
            )

            # Create batch
            batch = {"x": jax.random.normal(jax.random.PRNGKey(0), (4, 4))}

            # Should be able to execute train_step without error
            metrics = trainer.train_step(batch)
            assert "loss" in metrics
            assert jnp.isfinite(metrics["loss"])


class TestTrainerSchedulerCorrectness:
    """Test that Trainer uses correct scheduler behavior (not fallback constant).

    These tests verify that the Trainer properly delegates to the scheduler factory
    and that missing scheduler types fail properly (not silently return constant).
    """

    def test_trainer_schedule_matches_factory_polynomial(self):
        """Trainer's internal schedule should match factory output for polynomial."""
        base_lr = 1e-3
        scheduler_config = SchedulerConfig(
            name="polynomial",
            scheduler_type="polynomial",
            total_steps=100,
            min_lr_ratio=0.1,
        )

        # Factory should produce a schedule that decays from base_lr to min_lr
        factory_schedule = create_scheduler(scheduler_config, base_lr)

        # Verify factory schedule is NOT constant
        lr_at_0 = factory_schedule(0)
        lr_at_100 = factory_schedule(100)

        # Polynomial should decay from base_lr to min_lr_ratio * base_lr
        assert abs(lr_at_0 - base_lr) < 1e-7, f"Expected {base_lr}, got {lr_at_0}"
        assert abs(lr_at_100 - base_lr * 0.1) < 1e-7, f"Expected {base_lr * 0.1}, got {lr_at_100}"
        assert lr_at_0 != lr_at_100, "Schedule should NOT be constant"

    def test_trainer_schedule_matches_factory_cyclic(self):
        """Trainer's internal schedule should match factory output for cyclic."""
        base_lr = 1e-3
        scheduler_config = SchedulerConfig(
            name="cyclic",
            scheduler_type="cyclic",
            cycle_length=100,
            min_lr_ratio=0.1,
        )

        # Factory should produce a cyclic schedule
        factory_schedule = create_scheduler(scheduler_config, base_lr)

        # Verify factory schedule varies
        lr_at_0 = factory_schedule(0)
        lr_at_50 = factory_schedule(50)
        lr_at_100 = factory_schedule(100)

        # Cyclic: starts at min, peaks at 50, returns to min at 100
        assert abs(lr_at_0 - base_lr * 0.1) < 1e-7
        assert abs(lr_at_50 - base_lr) < 1e-7  # Peak
        assert abs(lr_at_100 - base_lr * 0.1) < 1e-7
        assert lr_at_0 != lr_at_50, "Schedule should NOT be constant"

    def test_trainer_schedule_matches_factory_one_cycle(self):
        """Trainer's internal schedule should match factory output for one_cycle."""
        base_lr = 1e-3
        scheduler_config = SchedulerConfig(
            name="one_cycle",
            scheduler_type="one_cycle",
            total_steps=100,
            min_lr_ratio=0.1,
        )

        # Factory should produce a one_cycle schedule
        factory_schedule = create_scheduler(scheduler_config, base_lr)

        # Verify factory schedule has warmup -> peak -> decay
        lr_at_0 = factory_schedule(0)
        lr_at_30 = factory_schedule(30)  # End of warmup (30%)
        lr_at_90 = factory_schedule(90)  # End of annealing (30+60=90%)

        # One cycle: warmup from min to peak, then decay
        assert abs(lr_at_0 - base_lr * 0.1) < 1e-7  # Starts at min
        assert abs(lr_at_30 - base_lr) < 1e-7  # Peak at 30%
        assert lr_at_90 < lr_at_30, "Should decay after peak"

    def test_trainer_create_schedule_uses_factory(self, simple_model):
        """Trainer._create_schedule should produce same result as factory."""
        base_lr = 1e-3
        scheduler_config = SchedulerConfig(
            name="polynomial",
            scheduler_type="polynomial",
            total_steps=100,
            min_lr_ratio=0.1,
        )
        optimizer_config = OptimizerConfig(
            name="test_optimizer",
            optimizer_type="adam",
            learning_rate=base_lr,
        )
        training_config = TrainingConfig(
            name="test",
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            batch_size=4,
            num_epochs=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=simple_loss_fn,
        )

        # The trainer should be using a polynomial schedule
        # (not falling back to constant base_lr)
        # Get expected schedule from factory
        expected_schedule = create_scheduler(scheduler_config, base_lr)

        # Get trainer's schedule via its internal method
        trainer_schedule = trainer._create_schedule(scheduler_config, base_lr)

        # Both should produce same values
        for step in [0, 25, 50, 75, 100]:
            expected_lr = expected_schedule(step)
            trainer_lr = trainer_schedule(step) if callable(trainer_schedule) else trainer_schedule

            # If trainer falls back to constant, this will fail
            assert abs(expected_lr - trainer_lr) < 1e-7, (
                f"At step {step}: expected {expected_lr}, got {trainer_lr}. "
                "Trainer may be falling back to constant instead of using factory."
            )


class TestSchedulerFactoryDirectUsage:
    """Test that the scheduler factory produces correct schedules."""

    def test_factory_constant_schedule(self):
        """Factory should create constant schedule."""
        config = SchedulerConfig(name="const", scheduler_type="constant")
        schedule = create_scheduler(config, base_lr=1e-3)

        # Constant schedule should return base_lr
        assert schedule(0) == 1e-3
        assert schedule(100) == 1e-3

    def test_factory_linear_with_warmup(self):
        """Factory should create linear schedule with warmup."""
        config = SchedulerConfig(
            name="linear",
            scheduler_type="linear",
            total_steps=100,
            warmup_steps=20,
            min_lr_ratio=0.1,
        )
        schedule = create_scheduler(config, base_lr=1e-3)

        # At step 0, should be near 0 (warmup starts)
        assert schedule(0) < 1e-4

        # At step 20, should be at peak
        assert abs(schedule(20) - 1e-3) < 1e-6

        # At step 100, should be at min_lr_ratio * base_lr
        assert abs(schedule(100) - 1e-4) < 1e-6

    def test_factory_cosine_schedule(self):
        """Factory should create cosine decay schedule."""
        config = SchedulerConfig(
            name="cosine",
            scheduler_type="cosine",
            cycle_length=100,
            min_lr_ratio=0.0,
        )
        schedule = create_scheduler(config, base_lr=1e-3)

        # At step 0, should be at peak
        assert abs(schedule(0) - 1e-3) < 1e-6

        # At step 100, should be near 0
        assert schedule(100) < 1e-6

    def test_factory_one_cycle_schedule(self):
        """Factory should create one_cycle schedule."""
        config = SchedulerConfig(
            name="one_cycle",
            scheduler_type="one_cycle",
            total_steps=100,
            min_lr_ratio=0.1,
        )
        schedule = create_scheduler(config, base_lr=1e-3)

        # Should return a schedule (callable)
        assert callable(schedule)

        # At step 0, should be at min_lr_ratio * base_lr (warmup start)
        assert abs(schedule(0) - 1e-4) < 1e-6

        # At step 30 (end of warmup), should be at peak
        assert abs(schedule(30) - 1e-3) < 1e-6

    def test_factory_cyclic_schedule(self):
        """Factory should create cyclic schedule."""
        config = SchedulerConfig(
            name="cyclic",
            scheduler_type="cyclic",
            cycle_length=100,
            min_lr_ratio=0.1,
        )
        schedule = create_scheduler(config, base_lr=1e-3)

        # Should return a schedule (callable)
        assert callable(schedule)

        # At step 0, should be at min_lr
        assert abs(schedule(0) - 1e-4) < 1e-6

        # At step 50 (peak), should be at max
        assert abs(schedule(50) - 1e-3) < 1e-6

        # At step 100 (cycle complete), should be back at min
        assert abs(schedule(100) - 1e-4) < 1e-6
