"""TDD tests for JIT-compatible trainer signatures.

These tests define the expected behavior for the refactored trainers with
functional `train_step` signatures that can be wrapped with `nnx.jit`.

The key changes:
1. Trainers no longer store model/optimizer in __init__
2. train_step takes model, optimizer as explicit arguments
3. EMA updates are separate methods taking model as argument

This design allows users to:
- Use train_step directly (non-JIT)
- Wrap train_step with nnx.jit for performance
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleDiffusionModel(nnx.Module):
    """Simple diffusion model for testing trainer functionality."""

    def __init__(self, dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(dim, 32, rngs=rngs)
        self.time_embed = nnx.Linear(1, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        h = nnx.relu(self.fc1(x))
        t_embed = nnx.relu(self.time_embed(t[:, None].astype(jnp.float32)))
        h = h + t_embed
        return self.fc2(h)


class SimpleVAEModel(nnx.Module):
    """Simple VAE model for testing trainer functionality."""

    def __init__(self, dim: int = 16, latent_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.encoder = nnx.Linear(dim, latent_dim * 2, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, dim, rngs=rngs)
        self.latent_dim = latent_dim

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Encode
        h = self.encoder(x)
        mean, logvar = jnp.split(h, 2, axis=-1)
        # Reparameterize
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(jax.random.key(0), mean.shape)
        z = mean + std * eps
        # Decode
        recon = self.decoder(z)
        return recon, mean, logvar


class SimpleFlowModel(nnx.Module):
    """Simple flow model (velocity field) for testing trainer functionality."""

    def __init__(self, dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(dim, 32, rngs=rngs)
        self.time_embed = nnx.Linear(1, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        h = nnx.relu(self.fc1(x))
        t_embed = nnx.relu(self.time_embed(t[:, None].astype(jnp.float32)))
        h = h + t_embed
        return self.fc2(h)


class SimpleEnergyModel(nnx.Module):
    """Simple energy model for testing trainer functionality."""

    def __init__(self, dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(dim, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class SimpleAutoregressiveModel(nnx.Module):
    """Simple autoregressive model for testing trainer functionality."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.embed = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.fc = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        h = self.embed(x)
        return self.fc(h)


class SimpleNoiseSchedule:
    """Simple noise schedule for testing."""

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        betas = jnp.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas)

    def add_noise(self, x: jax.Array, noise: jax.Array, t: jax.Array) -> jax.Array:
        sqrt_alpha = jnp.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha = jnp.sqrt(1.0 - self.alphas_cumprod[t])[:, None]
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise


@pytest.fixture
def diffusion_model() -> SimpleDiffusionModel:
    """Create a simple diffusion model for testing."""
    return SimpleDiffusionModel(rngs=nnx.Rngs(0))


@pytest.fixture
def vae_model() -> SimpleVAEModel:
    """Create a simple VAE model for testing."""
    return SimpleVAEModel(rngs=nnx.Rngs(0))


@pytest.fixture
def flow_model() -> SimpleFlowModel:
    """Create a simple flow model for testing."""
    return SimpleFlowModel(rngs=nnx.Rngs(0))


@pytest.fixture
def energy_model() -> SimpleEnergyModel:
    """Create a simple energy model for testing."""
    return SimpleEnergyModel(rngs=nnx.Rngs(0))


@pytest.fixture
def autoregressive_model() -> SimpleAutoregressiveModel:
    """Create a simple autoregressive model for testing."""
    return SimpleAutoregressiveModel(rngs=nnx.Rngs(0))


@pytest.fixture
def noise_schedule() -> SimpleNoiseSchedule:
    """Create a simple noise schedule for testing."""
    return SimpleNoiseSchedule(num_timesteps=1000)


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch for testing."""
    return {"image": jax.random.normal(jax.random.key(0), (8, 16))}


@pytest.fixture
def token_batch() -> dict[str, jax.Array]:
    """Create a token batch for autoregressive testing."""
    return {"input_ids": jax.random.randint(jax.random.key(0), (4, 32), 0, 100)}


@pytest.fixture
def rng_key() -> jax.Array:
    """Create PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# DiffusionTrainer JIT-Compatible Tests
# =============================================================================


class TestDiffusionTrainerJITCompatible:
    """Tests for JIT-compatible DiffusionTrainer signature."""

    def test_trainer_init_without_model_optimizer(
        self,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """DiffusionTrainer should initialize without model/optimizer."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)

        # Trainer should only store config and noise_schedule
        assert trainer.noise_schedule is noise_schedule
        assert trainer.config is config
        # Should NOT have model or optimizer attributes
        assert not hasattr(trainer, "model") or trainer.model is None
        assert not hasattr(trainer, "optimizer") or trainer.optimizer is None

    def test_train_step_signature(
        self,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """train_step should take model, optimizer, batch, key as arguments."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)

        sig = inspect.signature(trainer.train_step)
        params = list(sig.parameters.keys())

        # Should accept model, optimizer, batch, key
        assert "model" in params
        assert "optimizer" in params
        assert "batch" in params
        assert "key" in params

    def test_train_step_executes(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)
        optimizer = nnx.Optimizer(diffusion_model, optax.adam(1e-3), wrt=nnx.Param)

        loss, metrics = trainer.train_step(diffusion_model, optimizer, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "loss" in metrics

    def test_train_step_updates_model_params(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should update model parameters."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)
        optimizer = nnx.Optimizer(diffusion_model, optax.adam(1e-3), wrt=nnx.Param)

        # Get initial params - nnx.state returns raw arrays in newer Flax versions
        initial_state = nnx.state(diffusion_model, nnx.Param)

        # Handle both Variable objects (older) and raw arrays (newer)
        def extract_value(x):
            if hasattr(x, "value"):
                return jnp.copy(x.value)
            return jnp.copy(x)

        initial_params = jax.tree.map(extract_value, initial_state)

        trainer.train_step(diffusion_model, optimizer, sample_batch, rng_key)

        # Get updated params
        final_state = nnx.state(diffusion_model, nnx.Param)
        final_params = jax.tree.map(extract_value, final_state)

        # At least some params should change
        changed = jax.tree.map(lambda i, f: not jnp.allclose(i, f), initial_params, final_params)
        assert any(jax.tree.leaves(changed))

    def test_train_step_jit_wrappable(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig()
        trainer = DiffusionTrainer(noise_schedule, config)
        optimizer = nnx.Optimizer(diffusion_model, optax.adam(1e-3), wrt=nnx.Param)

        # This should work without error
        jit_step = nnx.jit(trainer.train_step)
        loss, metrics = jit_step(diffusion_model, optimizer, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "loss" in metrics

    def test_update_ema_takes_model_argument(
        self,
        diffusion_model: SimpleDiffusionModel,
        noise_schedule: SimpleNoiseSchedule,
    ) -> None:
        """update_ema should take model as argument."""
        from artifex.generative_models.training.trainers.diffusion_trainer import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        config = DiffusionTrainingConfig(ema_update_every=1)
        trainer = DiffusionTrainer(noise_schedule, config)

        sig = inspect.signature(trainer.update_ema)
        params = list(sig.parameters.keys())

        assert "model" in params

        # Should work when called
        trainer.update_ema(diffusion_model)
        assert trainer._ema_params is not None


# =============================================================================
# VAETrainer JIT-Compatible Tests
# =============================================================================


class TestVAETrainerJITCompatible:
    """Tests for JIT-compatible VAETrainer signature."""

    def test_trainer_init_without_model_optimizer(self) -> None:
        """VAETrainer should initialize without model/optimizer."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)

        assert trainer.config is config
        assert not hasattr(trainer, "model") or trainer.model is None
        assert not hasattr(trainer, "optimizer") or trainer.optimizer is None

    def test_train_step_signature(self) -> None:
        """train_step should take model, optimizer, batch, step as arguments."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)

        sig = inspect.signature(trainer.train_step)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "optimizer" in params
        assert "batch" in params
        assert "step" in params

    def test_train_step_executes(
        self,
        vae_model: SimpleVAEModel,
        sample_batch: dict,
    ) -> None:
        """train_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)
        optimizer = nnx.Optimizer(vae_model, optax.adam(1e-3), wrt=nnx.Param)

        loss, metrics = trainer.train_step(vae_model, optimizer, sample_batch, step=0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "loss" in metrics

    def test_train_step_jit_wrappable(
        self,
        vae_model: SimpleVAEModel,
        sample_batch: dict,
    ) -> None:
        """train_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.vae_trainer import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig()
        trainer = VAETrainer(config)
        optimizer = nnx.Optimizer(vae_model, optax.adam(1e-3), wrt=nnx.Param)

        jit_step = nnx.jit(trainer.train_step)
        loss, metrics = jit_step(vae_model, optimizer, sample_batch, step=0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)


# =============================================================================
# FlowTrainer JIT-Compatible Tests
# =============================================================================


class TestFlowTrainerJITCompatible:
    """Tests for JIT-compatible FlowTrainer signature."""

    def test_trainer_init_without_model_optimizer(self) -> None:
        """FlowTrainer should initialize without model/optimizer."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        assert trainer.config is config
        assert not hasattr(trainer, "model") or trainer.model is None
        assert not hasattr(trainer, "optimizer") or trainer.optimizer is None

    def test_train_step_signature(self) -> None:
        """train_step should take model, optimizer, batch, key as arguments."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)

        sig = inspect.signature(trainer.train_step)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "optimizer" in params
        assert "batch" in params
        assert "key" in params

    def test_train_step_executes(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)
        optimizer = nnx.Optimizer(flow_model, optax.adam(1e-3), wrt=nnx.Param)

        loss, metrics = trainer.train_step(flow_model, optimizer, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "loss" in metrics

    def test_train_step_jit_wrappable(
        self,
        flow_model: SimpleFlowModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.flow_trainer import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig()
        trainer = FlowTrainer(config)
        optimizer = nnx.Optimizer(flow_model, optax.adam(1e-3), wrt=nnx.Param)

        jit_step = nnx.jit(trainer.train_step)
        loss, metrics = jit_step(flow_model, optimizer, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)


# =============================================================================
# EnergyTrainer JIT-Compatible Tests
# =============================================================================


class TestEnergyTrainerJITCompatible:
    """Tests for JIT-compatible EnergyTrainer signature."""

    def test_trainer_init_without_model_optimizer(self) -> None:
        """EnergyTrainer should initialize without model/optimizer."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        trainer = EnergyTrainer(config)

        assert trainer.config is config
        assert not hasattr(trainer, "model") or trainer.model is None
        assert not hasattr(trainer, "optimizer") or trainer.optimizer is None

    def test_train_step_signature(self) -> None:
        """train_step should take model, optimizer, batch, key as arguments."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig()
        trainer = EnergyTrainer(config)

        sig = inspect.signature(trainer.train_step)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "optimizer" in params
        assert "batch" in params
        assert "key" in params

    def test_train_step_executes(
        self,
        energy_model: SimpleEnergyModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        # Use score_matching for simpler test (no MCMC)
        config = EnergyTrainingConfig(training_method="score_matching")
        trainer = EnergyTrainer(config)
        optimizer = nnx.Optimizer(energy_model, optax.adam(1e-3), wrt=nnx.Param)

        loss, metrics = trainer.train_step(energy_model, optimizer, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "loss" in metrics

    def test_train_step_jit_wrappable(
        self,
        energy_model: SimpleEnergyModel,
        sample_batch: dict,
        rng_key: jax.Array,
    ) -> None:
        """train_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.energy_trainer import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(training_method="score_matching")
        trainer = EnergyTrainer(config)
        optimizer = nnx.Optimizer(energy_model, optax.adam(1e-3), wrt=nnx.Param)

        jit_step = nnx.jit(trainer.train_step)
        loss, metrics = jit_step(energy_model, optimizer, sample_batch, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)


# =============================================================================
# AutoregressiveTrainer JIT-Compatible Tests
# =============================================================================


class TestAutoregressiveTrainerJITCompatible:
    """Tests for JIT-compatible AutoregressiveTrainer signature."""

    def test_trainer_init_without_model_optimizer(self) -> None:
        """AutoregressiveTrainer should initialize without model/optimizer."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        assert trainer.config is config
        assert not hasattr(trainer, "model") or trainer.model is None
        assert not hasattr(trainer, "optimizer") or trainer.optimizer is None

    def test_train_step_signature(self) -> None:
        """train_step should take model, optimizer, batch, step, key as arguments."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        sig = inspect.signature(trainer.train_step)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "optimizer" in params
        assert "batch" in params

    def test_train_step_executes(
        self,
        autoregressive_model: SimpleAutoregressiveModel,
        token_batch: dict,
    ) -> None:
        """train_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(use_causal_mask=False)
        trainer = AutoregressiveTrainer(config)
        optimizer = nnx.Optimizer(autoregressive_model, optax.adam(1e-3), wrt=nnx.Param)

        loss, metrics = trainer.train_step(autoregressive_model, optimizer, token_batch, step=0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "loss" in metrics

    def test_train_step_jit_wrappable(
        self,
        autoregressive_model: SimpleAutoregressiveModel,
        token_batch: dict,
    ) -> None:
        """train_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(use_causal_mask=False)
        trainer = AutoregressiveTrainer(config)
        optimizer = nnx.Optimizer(autoregressive_model, optax.adam(1e-3), wrt=nnx.Param)

        jit_step = nnx.jit(trainer.train_step)
        loss, metrics = jit_step(autoregressive_model, optimizer, token_batch, step=0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)


# =============================================================================
# GANTrainer JIT-Compatible Tests
# =============================================================================


class SimpleGenerator(nnx.Module):
    """Simple generator for testing."""

    def __init__(self, latent_dim: int = 8, out_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc = nnx.Linear(latent_dim, out_dim, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        return nnx.tanh(self.fc(z))


class SimpleDiscriminator(nnx.Module):
    """Simple discriminator for testing."""

    def __init__(self, in_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc = nnx.Linear(in_dim, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc(x)


class TestGANTrainerJITCompatible:
    """Tests for JIT-compatible GANTrainer signature."""

    def test_trainer_init_without_models_optimizers(self) -> None:
        """GANTrainer should initialize without models/optimizers."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        assert trainer.config is config
        assert not hasattr(trainer, "generator") or trainer.generator is None
        assert not hasattr(trainer, "discriminator") or trainer.discriminator is None

    def test_discriminator_step_signature(self) -> None:
        """discriminator_step should take models, optimizers, data, key."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        sig = inspect.signature(trainer.discriminator_step)
        params = list(sig.parameters.keys())

        assert "generator" in params
        assert "discriminator" in params
        assert "d_optimizer" in params
        assert "real" in params
        assert "z" in params
        assert "key" in params

    def test_generator_step_signature(self) -> None:
        """generator_step should take models, optimizers, z."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        sig = inspect.signature(trainer.generator_step)
        params = list(sig.parameters.keys())

        assert "generator" in params
        assert "discriminator" in params
        assert "g_optimizer" in params
        assert "z" in params

    def test_discriminator_step_executes(self, rng_key: jax.Array) -> None:
        """discriminator_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        generator = SimpleGenerator(rngs=nnx.Rngs(0))
        discriminator = SimpleDiscriminator(rngs=nnx.Rngs(1))
        d_optimizer = nnx.Optimizer(discriminator, optax.adam(1e-3), wrt=nnx.Param)

        real = jax.random.normal(jax.random.key(0), (8, 16))
        z = jax.random.normal(jax.random.key(1), (8, 8))

        loss, metrics = trainer.discriminator_step(
            generator, discriminator, d_optimizer, real, z, rng_key
        )

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "d_loss" in metrics

    def test_generator_step_executes(self) -> None:
        """generator_step should execute and return loss, metrics."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        generator = SimpleGenerator(rngs=nnx.Rngs(0))
        discriminator = SimpleDiscriminator(rngs=nnx.Rngs(1))
        g_optimizer = nnx.Optimizer(generator, optax.adam(1e-3), wrt=nnx.Param)

        z = jax.random.normal(jax.random.key(1), (8, 8))

        loss, metrics = trainer.generator_step(generator, discriminator, g_optimizer, z)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "g_loss" in metrics

    def test_discriminator_step_jit_wrappable(self, rng_key: jax.Array) -> None:
        """discriminator_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        generator = SimpleGenerator(rngs=nnx.Rngs(0))
        discriminator = SimpleDiscriminator(rngs=nnx.Rngs(1))
        d_optimizer = nnx.Optimizer(discriminator, optax.adam(1e-3), wrt=nnx.Param)

        real = jax.random.normal(jax.random.key(0), (8, 16))
        z = jax.random.normal(jax.random.key(1), (8, 8))

        jit_step = nnx.jit(trainer.discriminator_step)
        loss, metrics = jit_step(generator, discriminator, d_optimizer, real, z, rng_key)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_generator_step_jit_wrappable(self) -> None:
        """generator_step should be wrappable with nnx.jit."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        trainer = GANTrainer(config)

        generator = SimpleGenerator(rngs=nnx.Rngs(0))
        discriminator = SimpleDiscriminator(rngs=nnx.Rngs(1))
        g_optimizer = nnx.Optimizer(generator, optax.adam(1e-3), wrt=nnx.Param)

        z = jax.random.normal(jax.random.key(1), (8, 8))

        jit_step = nnx.jit(trainer.generator_step)
        loss, metrics = jit_step(generator, discriminator, g_optimizer, z)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
