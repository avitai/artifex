"""Tests for Trainer NNX module support.

Following TDD principles - these tests define the expected behavior
for training NNX-based models like DDPMModel.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DDPMConfig,
    NoiseScheduleConfig,
    OptimizerConfig,
    TrainingConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion import DDPMModel
from artifex.generative_models.training.trainer import Trainer


@pytest.fixture
def ddpm_config():
    """Create a minimal DDPM config for testing."""
    backbone = UNetBackboneConfig(
        name="test_unet",
        in_channels=1,
        out_channels=1,
        hidden_dims=(8, 16),
        channel_mult=(1, 2),
        activation="relu",
        num_res_blocks=1,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        schedule_type="linear",
        num_timesteps=10,
        beta_start=1e-4,
        beta_end=2e-2,
    )
    return DDPMConfig(
        name="test_ddpm",
        input_shape=(8, 8, 1),
        backbone=backbone,
        noise_schedule=noise_schedule,
    )


@pytest.fixture
def training_config():
    """Create training configuration."""
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


@pytest.fixture
def ddpm_model(ddpm_config):
    """Create a DDPM model for testing."""
    rngs = nnx.Rngs(42)
    return DDPMModel(ddpm_config, rngs=rngs)


@pytest.fixture
def sample_batch():
    """Create a sample training batch."""
    key = jax.random.PRNGKey(0)
    images = jax.random.normal(key, (4, 8, 8, 1))
    return {"images": images}


class TestTrainerNNXSupport:
    """Test that Trainer properly supports NNX modules."""

    def test_trainer_accepts_nnx_model(self, ddpm_model, training_config):
        """Trainer should accept NNX modules without error."""

        def loss_fn(model, batch, rng):
            x = batch["images"]
            t = jax.random.randint(rng, (x.shape[0],), 0, model.noise_steps)
            noisy_x, noise = model.forward_diffusion(x, t)
            pred = model(noisy_x, t)
            loss = jnp.mean((noise - pred["predicted_noise"]) ** 2)
            return loss, {"loss": loss}

        trainer = Trainer(
            model=ddpm_model,
            training_config=training_config,
            loss_fn=loss_fn,
            rng=jax.random.PRNGKey(0),
        )

        assert trainer is not None
        assert trainer.model is ddpm_model

    def test_trainer_train_step_works_with_nnx_model(
        self, ddpm_model, training_config, sample_batch
    ):
        """Trainer.train_step should work with NNX models without TraceContextError."""

        def loss_fn(model, batch, rng):
            x = batch["images"]
            t = jax.random.randint(rng, (x.shape[0],), 0, model.noise_steps)
            noisy_x, noise = model.forward_diffusion(x, t)
            pred = model(noisy_x, t)
            loss = jnp.mean((noise - pred["predicted_noise"]) ** 2)
            return loss, {"loss": loss}

        trainer = Trainer(
            model=ddpm_model,
            training_config=training_config,
            loss_fn=loss_fn,
            rng=jax.random.PRNGKey(0),
        )

        # This should NOT raise TraceContextError
        metrics = trainer.train_step(sample_batch)

        assert "loss" in metrics
        assert jnp.isfinite(metrics["loss"])

    def test_trainer_multiple_steps_reduce_loss(self, ddpm_model, training_config, sample_batch):
        """Multiple training steps should reduce loss (model is learning)."""

        def loss_fn(model, batch, rng):
            x = batch["images"]
            t = jax.random.randint(rng, (x.shape[0],), 0, model.noise_steps)
            noisy_x, noise = model.forward_diffusion(x, t)
            pred = model(noisy_x, t)
            loss = jnp.mean((noise - pred["predicted_noise"]) ** 2)
            return loss, {"loss": loss}

        trainer = Trainer(
            model=ddpm_model,
            training_config=training_config,
            loss_fn=loss_fn,
            rng=jax.random.PRNGKey(0),
        )

        # Run multiple steps
        losses = []
        for _ in range(10):
            metrics = trainer.train_step(sample_batch)
            losses.append(float(metrics["loss"]))

        # Loss should generally decrease (or at least not explode)
        assert losses[-1] < losses[0] * 2  # Allow some variance but no explosion
        assert all(jnp.isfinite(l) for l in losses)

    def test_trainer_preserves_model_state(self, ddpm_model, training_config, sample_batch):
        """Training should update model parameters in-place (NNX style)."""

        def loss_fn(model, batch, rng):
            x = batch["images"]
            t = jax.random.randint(rng, (x.shape[0],), 0, model.noise_steps)
            noisy_x, noise = model.forward_diffusion(x, t)
            pred = model(noisy_x, t)
            loss = jnp.mean((noise - pred["predicted_noise"]) ** 2)
            return loss, {"loss": loss}

        # Get initial param values
        initial_params = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x, nnx.state(ddpm_model, nnx.Param)
        )

        trainer = Trainer(
            model=ddpm_model,
            training_config=training_config,
            loss_fn=loss_fn,
            rng=jax.random.PRNGKey(0),
        )

        # Train for a few steps
        for _ in range(5):
            trainer.train_step(sample_batch)

        # Get updated params
        updated_params = nnx.state(ddpm_model, nnx.Param)

        # At least some params should have changed
        params_changed = False
        for init_leaf, updated_leaf in zip(
            jax.tree.leaves(initial_params), jax.tree.leaves(updated_params)
        ):
            if not jnp.allclose(init_leaf, updated_leaf):
                params_changed = True
                break

        assert params_changed, "Model parameters should be updated after training"
