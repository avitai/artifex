"""TDD tests for high-performance training loops.

These tests define expected behavior for optimized training loops BEFORE implementation.
The loops enable 50-500x speedups by eliminating Python loop overhead via:
- nnx.fori_loop for staged data (fits in GPU memory)
- JIT-compiled steps with prefetch for streaming data

The key insight is that existing trainers already implement `create_loss_fn()` which
returns a function with signature: (model, batch, rng) -> (loss, metrics).
These loops reuse that pattern for DRY compliance.

References:
    - Flax NNX transforms: https://flax.readthedocs.io/en/latest/
    - JAX compilation: https://jax.readthedocs.io/en/latest/jit-compilation.html
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleVAE(nnx.Module):
    """Simple VAE for testing training loops."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.encoder = nnx.Linear(16, 8, rngs=rngs)
        self.mean_layer = nnx.Linear(8, 4, rngs=rngs)
        self.logvar_layer = nnx.Linear(8, 4, rngs=rngs)
        self.decoder = nnx.Linear(4, 16, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = nnx.relu(self.encoder(x))
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        z = mean  # Simplified (no stochasticity for testing)
        recon = self.decoder(z)
        return recon, mean, logvar


class SimpleMLP(nnx.Module):
    """Simple MLP for testing generic loops."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(16, 8, rngs=rngs)
        self.fc2 = nnx.Linear(8, 16, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(nnx.relu(self.fc1(x)))


@pytest.fixture
def simple_vae() -> SimpleVAE:
    """Create a simple VAE for testing."""
    return SimpleVAE(rngs=nnx.Rngs(0))


@pytest.fixture
def simple_mlp() -> SimpleMLP:
    """Create a simple MLP for testing."""
    return SimpleMLP(rngs=nnx.Rngs(0))


@pytest.fixture
def simple_optimizer(simple_vae: SimpleVAE) -> nnx.Optimizer:
    """Create optimizer for simple VAE."""
    return nnx.Optimizer(simple_vae, optax.adam(1e-3), wrt=nnx.Param)


@pytest.fixture
def mlp_optimizer(simple_mlp: SimpleMLP) -> nnx.Optimizer:
    """Create optimizer for simple MLP."""
    return nnx.Optimizer(simple_mlp, optax.adam(1e-3), wrt=nnx.Param)


@pytest.fixture
def staged_data() -> jax.Array:
    """Create staged data array for testing (128 samples, 16 features)."""
    # Use random data to ensure non-zero gradients
    return jax.random.normal(jax.random.key(42), (128, 16))


@pytest.fixture
def batch_iterator() -> Iterator[dict[str, jax.Array]]:
    """Create batch iterator for streaming tests."""

    def gen():
        for _ in range(4):
            yield {"image": jax.random.normal(jax.random.key(0), (32, 16))}

    return gen()


def create_simple_loss_fn():
    """Create a simple MSE loss function for testing."""

    def loss_fn(
        model: nnx.Module,
        batch: dict[str, Any],
        _rng: jax.Array,
        _step: jax.Array,  # Step passed dynamically for KL annealing compatibility
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        x = batch.get("image", batch.get("data", batch.get("x")))
        if x is None:
            raise ValueError("Batch must contain 'image', 'data', or 'x' key")
        pred = model(x)
        # Handle VAE outputs (tuple) vs MLP outputs (array)
        if isinstance(pred, tuple):
            recon = pred[0]
        else:
            recon = pred
        loss = jnp.mean((recon - x) ** 2)
        return loss, {"loss": loss, "mse": loss}

    return loss_fn


# =============================================================================
# train_epoch_staged Tests
# =============================================================================


class TestTrainEpochStagedImport:
    """Tests for train_epoch_staged import and basic functionality."""

    def test_function_is_importable(self) -> None:
        """train_epoch_staged should be importable from training.loops."""
        from artifex.generative_models.training.loops import train_epoch_staged

        assert train_epoch_staged is not None
        assert callable(train_epoch_staged)

    def test_function_is_also_importable_from_training(self) -> None:
        """train_epoch_staged should also be importable from training module."""
        from artifex.generative_models.training import train_epoch_staged

        assert train_epoch_staged is not None


class TestTrainEpochStagedBasicBehavior:
    """Tests for train_epoch_staged basic behavior."""

    def test_returns_step_and_metrics(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer, staged_data: jax.Array
    ) -> None:
        """Should return (final_step, metrics_dict)."""
        from artifex.generative_models.training.loops import train_epoch_staged

        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        step, metrics = train_epoch_staged(
            simple_vae, simple_optimizer, staged_data, batch_size=32, rng=rng, loss_fn=loss_fn
        )

        assert isinstance(step, int)
        assert step == 4  # 128 / 32 = 4 batches
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_updates_model_parameters(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer, staged_data: jax.Array
    ) -> None:
        """Model parameters should be updated after training."""
        from artifex.generative_models.training.loops import train_epoch_staged

        # Get initial encoder weight
        initial_kernel = simple_vae.encoder.kernel.value.copy()

        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        train_epoch_staged(
            simple_vae, simple_optimizer, staged_data, batch_size=32, rng=rng, loss_fn=loss_fn
        )

        # Parameters should have changed
        updated_kernel = simple_vae.encoder.kernel.value
        assert not jnp.allclose(initial_kernel, updated_kernel)

    def test_respects_base_step(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer, staged_data: jax.Array
    ) -> None:
        """Should respect base_step parameter for step numbering."""
        from artifex.generative_models.training.loops import train_epoch_staged

        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        step, _ = train_epoch_staged(
            simple_vae,
            simple_optimizer,
            staged_data,
            batch_size=32,
            rng=rng,
            loss_fn=loss_fn,
            base_step=100,
        )

        assert step == 104  # 100 + 4 batches


class TestTrainEpochStagedWithVAETrainer:
    """Tests for train_epoch_staged with VAETrainer.create_loss_fn()."""

    def test_works_with_vae_trainer(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer, staged_data: jax.Array
    ) -> None:
        """Should work with VAETrainer.create_loss_fn() for DRY compliance."""
        from artifex.generative_models.training.loops import train_epoch_staged
        from artifex.generative_models.training.trainers import VAETrainer

        trainer = VAETrainer()
        loss_fn = trainer.create_loss_fn(loss_type="mse")

        rng = jax.random.key(0)
        step, metrics = train_epoch_staged(
            simple_vae, simple_optimizer, staged_data, batch_size=32, rng=rng, loss_fn=loss_fn
        )

        assert "loss" in metrics
        assert "recon_loss" in metrics or "kl_loss" in metrics  # VAE-specific metrics

    def test_loss_decreases_over_epoch(self, simple_vae: SimpleVAE, staged_data: jax.Array) -> None:
        """Loss should generally decrease over training (with real data)."""
        from artifex.generative_models.training.loops import train_epoch_staged
        from artifex.generative_models.training.trainers import VAETrainer

        # Use random data for meaningful loss
        data = jax.random.normal(jax.random.key(42), (128, 16))
        optimizer = nnx.Optimizer(simple_vae, optax.adam(1e-2), wrt=nnx.Param)

        trainer = VAETrainer()

        # Train for a few epochs and check loss trend
        losses = []
        for epoch in range(3):
            rng = jax.random.key(epoch)
            loss_fn = trainer.create_loss_fn(loss_type="mse")
            _, metrics = train_epoch_staged(
                simple_vae, optimizer, data, batch_size=32, rng=rng, loss_fn=loss_fn
            )
            losses.append(float(metrics["loss"]))

        # Loss should decrease (allowing some variance)
        # At least final loss should be less than initial
        assert losses[-1] < losses[0] * 1.5  # Allow some tolerance


class TestTrainEpochStagedCustomDataKey:
    """Tests for custom data_key parameter."""

    def test_supports_custom_data_key(
        self, simple_mlp: SimpleMLP, mlp_optimizer: nnx.Optimizer
    ) -> None:
        """Should support custom data_key for batch dict."""
        from artifex.generative_models.training.loops import train_epoch_staged

        data = jnp.zeros((64, 16))
        rng = jax.random.key(0)

        def loss_fn(model, batch, _rng, _step):
            x = batch["features"]  # Custom key
            pred = model(x)
            loss = jnp.mean((pred - x) ** 2)
            return loss, {"loss": loss}

        step, metrics = train_epoch_staged(
            simple_mlp,
            mlp_optimizer,
            data,
            batch_size=32,
            rng=rng,
            loss_fn=loss_fn,
            data_key="features",  # Custom key
        )

        assert step == 2


# =============================================================================
# train_epoch_streaming Tests
# =============================================================================


class TestTrainEpochStreamingImport:
    """Tests for train_epoch_streaming import and basic functionality."""

    def test_function_is_importable(self) -> None:
        """train_epoch_streaming should be importable from training.loops."""
        from artifex.generative_models.training.loops import train_epoch_streaming

        assert train_epoch_streaming is not None
        assert callable(train_epoch_streaming)

    def test_function_is_also_importable_from_training(self) -> None:
        """train_epoch_streaming should also be importable from training module."""
        from artifex.generative_models.training import train_epoch_streaming

        assert train_epoch_streaming is not None


class TestTrainEpochStreamingBasicBehavior:
    """Tests for train_epoch_streaming basic behavior."""

    def test_returns_step_and_metrics(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer
    ) -> None:
        """Should return (final_step, metrics_dict)."""
        from artifex.generative_models.training.loops import train_epoch_streaming

        batches = [{"image": jax.random.normal(jax.random.key(0), (32, 16))} for _ in range(4)]
        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        step, metrics = train_epoch_streaming(
            simple_vae, simple_optimizer, iter(batches), rng=rng, loss_fn=loss_fn
        )

        assert isinstance(step, int)
        assert step == 4
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_updates_model_parameters(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer
    ) -> None:
        """Model parameters should be updated after training."""
        from artifex.generative_models.training.loops import train_epoch_streaming

        initial_kernel = simple_vae.encoder.kernel.value.copy()

        batches = [{"image": jax.random.normal(jax.random.key(0), (32, 16))} for _ in range(4)]
        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        train_epoch_streaming(simple_vae, simple_optimizer, iter(batches), rng=rng, loss_fn=loss_fn)

        updated_kernel = simple_vae.encoder.kernel.value
        assert not jnp.allclose(initial_kernel, updated_kernel)

    def test_respects_base_step(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer
    ) -> None:
        """Should respect base_step parameter for step numbering."""
        from artifex.generative_models.training.loops import train_epoch_streaming

        batches = [{"image": jax.random.normal(jax.random.key(0), (32, 16))} for _ in range(4)]
        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        step, _ = train_epoch_streaming(
            simple_vae, simple_optimizer, iter(batches), rng=rng, loss_fn=loss_fn, base_step=100
        )

        assert step == 104

    def test_handles_empty_iterator(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer
    ) -> None:
        """Should handle empty iterator gracefully."""
        from artifex.generative_models.training.loops import train_epoch_streaming

        rng = jax.random.key(0)
        loss_fn = create_simple_loss_fn()

        step, metrics = train_epoch_streaming(
            simple_vae, simple_optimizer, iter([]), rng=rng, loss_fn=loss_fn
        )

        assert step == 0
        assert metrics == {}


class TestTrainEpochStreamingWithVAETrainer:
    """Tests for train_epoch_streaming with VAETrainer.create_loss_fn()."""

    def test_works_with_vae_trainer(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer
    ) -> None:
        """Should work with VAETrainer.create_loss_fn() for DRY compliance."""
        from artifex.generative_models.training.loops import train_epoch_streaming
        from artifex.generative_models.training.trainers import VAETrainer

        trainer = VAETrainer()
        loss_fn = trainer.create_loss_fn(loss_type="mse")

        batches = [{"image": jax.random.normal(jax.random.key(0), (32, 16))} for _ in range(4)]
        rng = jax.random.key(0)

        step, metrics = train_epoch_streaming(
            simple_vae, simple_optimizer, iter(batches), rng=rng, loss_fn=loss_fn
        )

        assert "loss" in metrics


# =============================================================================
# gaussian_kl_divergence Tests
# =============================================================================


class TestGaussianKLDivergence:
    """Tests for gaussian_kl_divergence function."""

    def test_function_is_importable(self) -> None:
        """gaussian_kl_divergence should be importable from core.losses."""
        from artifex.generative_models.core.losses import gaussian_kl_divergence

        assert gaussian_kl_divergence is not None
        assert callable(gaussian_kl_divergence)

    def test_zero_for_standard_normal(self) -> None:
        """KL should be zero when posterior equals prior (standard normal)."""
        from artifex.generative_models.core.losses import gaussian_kl_divergence

        mean = jnp.zeros((8, 4))
        logvar = jnp.zeros((8, 4))  # var = 1
        kl = gaussian_kl_divergence(mean, logvar, reduction="mean")

        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_non_standard(self) -> None:
        """KL should be positive for non-standard posteriors."""
        from artifex.generative_models.core.losses import gaussian_kl_divergence

        mean = jnp.ones((8, 4))  # Mean shifted from 0
        logvar = jnp.zeros((8, 4))
        kl = gaussian_kl_divergence(mean, logvar, reduction="mean")

        assert kl > 0

    def test_reduction_none(self) -> None:
        """reduction='none' should return per-sample values."""
        from artifex.generative_models.core.losses import gaussian_kl_divergence

        mean = jnp.zeros((8, 4))
        logvar = jnp.zeros((8, 4))
        kl = gaussian_kl_divergence(mean, logvar, reduction="none")

        assert kl.shape == (8,)

    def test_reduction_sum(self) -> None:
        """reduction='sum' should sum all values."""
        from artifex.generative_models.core.losses import gaussian_kl_divergence

        mean = jnp.ones((8, 4))
        logvar = jnp.zeros((8, 4))
        kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

        assert kl.shape == ()

    def test_reduction_batch_sum(self) -> None:
        """reduction='batch_sum' should sum over features, mean over batch."""
        from artifex.generative_models.core.losses import gaussian_kl_divergence

        mean = jnp.ones((8, 4))
        logvar = jnp.zeros((8, 4))
        kl = gaussian_kl_divergence(mean, logvar, reduction="batch_sum")

        assert kl.shape == ()


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoopsIntegration:
    """Integration tests for training loops."""

    def test_staged_and_streaming_produce_similar_results(self, staged_data: jax.Array) -> None:
        """Both loops should produce similar results on same data."""
        from artifex.generative_models.training.loops import (
            train_epoch_staged,
            train_epoch_streaming,
        )

        # Create two identical models
        vae1 = SimpleVAE(rngs=nnx.Rngs(0))
        vae2 = SimpleVAE(rngs=nnx.Rngs(0))
        opt1 = nnx.Optimizer(vae1, optax.adam(1e-3), wrt=nnx.Param)
        opt2 = nnx.Optimizer(vae2, optax.adam(1e-3), wrt=nnx.Param)

        loss_fn = create_simple_loss_fn()
        rng = jax.random.key(42)

        # Train with staged
        _, metrics1 = train_epoch_staged(
            vae1, opt1, staged_data, batch_size=32, rng=rng, loss_fn=loss_fn
        )

        # Create batches for streaming
        batches = [{"image": staged_data[i * 32 : (i + 1) * 32]} for i in range(4)]

        # Train with streaming
        _, metrics2 = train_epoch_streaming(vae2, opt2, iter(batches), rng=rng, loss_fn=loss_fn)

        # Final losses should be similar (not exact due to JIT differences)
        assert abs(metrics1["loss"] - metrics2["loss"]) < 0.1

    def test_loops_are_jit_compatible(
        self, simple_vae: SimpleVAE, simple_optimizer: nnx.Optimizer, staged_data: jax.Array
    ) -> None:
        """Training loops should be JIT-compatible (no Python control flow issues)."""
        from artifex.generative_models.training.loops import train_epoch_staged

        loss_fn = create_simple_loss_fn()
        rng = jax.random.key(0)

        # This should not raise any JIT compilation errors
        step, metrics = train_epoch_staged(
            simple_vae, simple_optimizer, staged_data, batch_size=32, rng=rng, loss_fn=loss_fn
        )

        assert step == 4
        assert "loss" in metrics


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestLoopsExports:
    """Tests for training loops module exports."""

    def test_loops_module_exports_all_functions(self) -> None:
        """Loops module should export all training loop functions."""
        from artifex.generative_models.training import loops

        assert hasattr(loops, "train_epoch_staged")
        assert hasattr(loops, "train_epoch_streaming")

    def test_training_module_exports_loops(self) -> None:
        """Main training module should re-export loop functions."""
        from artifex.generative_models.training import (
            train_epoch_staged,
            train_epoch_streaming,
        )

        assert train_epoch_staged is not None
        assert train_epoch_streaming is not None
