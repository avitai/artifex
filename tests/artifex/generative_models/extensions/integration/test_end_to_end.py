"""End-to-end integration tests for the extension system.

These tests verify the complete pipeline from configuration to training
with extensions across different model types.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    ExtensionConfig,
    OptimizerConfig,
    TrainingConfig,
    VAEConfig,
)
from artifex.generative_models.extensions.base import ModelExtension
from artifex.generative_models.factory.core import create_model, create_model_with_extensions


class RegularizationExtension(ModelExtension):
    """Extension that adds L2 regularization to any model."""

    def __call__(self, inputs, model_outputs, **kwargs):
        return {"regularized": True}

    def loss_fn(self, batch, model_outputs, **kwargs):
        if not self.is_enabled() or model_outputs is None:
            return jnp.array(0.0)

        # L2 regularization on outputs
        if isinstance(model_outputs, dict):
            if "z_mean" in model_outputs:
                return jnp.mean(jnp.square(model_outputs["z_mean"]))
            elif "reconstruction" in model_outputs:
                return jnp.mean(jnp.square(model_outputs["reconstruction"])) * 0.001
        elif isinstance(model_outputs, jax.Array):
            return jnp.mean(jnp.square(model_outputs)) * 0.001
        return jnp.array(0.0)


class TestEndToEndVAEWithExtensions:
    """End-to-end tests for VAE with extensions."""

    @pytest.fixture
    def rngs(self):
        """Create test rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def vae_config(self):
        """Create VAE configuration."""
        encoder = EncoderConfig(
            name="test_encoder",
            input_shape=(28, 28, 1),
            latent_dim=16,
            hidden_dims=(64, 32),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="test_decoder",
            latent_dim=16,
            output_shape=(28, 28, 1),
            hidden_dims=(32, 64),
            activation="relu",
        )
        return VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
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
    def batch(self):
        """Create test batch."""
        return {
            "input": jnp.ones((4, 28, 28, 1)),
            "target": jnp.ones((4, 28, 28, 1)),
        }

    def test_create_model_returns_model(self, vae_config, rngs):
        """Test that create_model creates a valid model."""
        model = create_model(vae_config, rngs=rngs)

        assert model is not None
        assert isinstance(model, nnx.Module)

    def test_create_model_with_extensions_returns_tuple(self, vae_config, rngs):
        """Test that create_model_with_extensions returns (model, extensions)."""
        ext_configs = {
            "regularization": ExtensionConfig(name="regularization", weight=0.01, enabled=True),
        }

        # Register the extension
        from artifex.generative_models.extensions.registry import get_extensions_registry

        registry = get_extensions_registry()

        # Register if not already registered
        if "regularization" not in registry.list_all_extensions():
            registry.register_extension(
                "regularization",
                RegularizationExtension,
                modalities=["image"],
                capabilities=["regularization"],
            )

        model, extensions = create_model_with_extensions(
            vae_config,
            extensions_config=ext_configs,
            rngs=rngs,
        )

        assert model is not None
        assert isinstance(extensions, dict)
        assert "regularization" in extensions
        assert isinstance(extensions["regularization"], RegularizationExtension)

    def test_vae_with_extension_training_step(self, vae_config, training_config, rngs, batch):
        """Test VAE training step with extension."""
        from artifex.generative_models.training import Trainer

        # Create model
        model = create_model(vae_config, rngs=rngs)

        # Create extension
        ext_config = ExtensionConfig(name="reg", weight=0.01, enabled=True)
        extension = RegularizationExtension(ext_config, rngs=rngs)
        extensions = {"reg": extension}

        # Define VAE loss function
        def vae_loss_fn(model, batch_data, rng):  # noqa: ARG001
            x = batch_data["input"]
            # Flatten input for dense encoder
            x_flat = x.reshape(x.shape[0], -1)
            # VAE forward pass (no rngs needed in __call__)
            outputs = model(x_flat)
            # Simple reconstruction loss
            if isinstance(outputs, dict) and "reconstruction" in outputs:
                recon = outputs["reconstruction"]
                loss = jnp.mean(jnp.square(recon - x_flat))
            else:
                loss = jnp.array(0.0)
            return loss, {"recon_loss": loss}

        # Create trainer with extension
        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=extensions,
            loss_fn=vae_loss_fn,
        )

        # Run training step
        metrics = trainer.train_step(batch)

        # Verify metrics
        assert "loss" in metrics
        assert "reg_loss" in metrics

    def test_full_training_loop_with_extensions(self, vae_config, training_config, rngs):
        """Test full training loop with extensions."""
        from artifex.generative_models.training import Trainer

        # Create model
        model = create_model(vae_config, rngs=rngs)

        # Create extension
        ext_config = ExtensionConfig(name="reg", weight=0.01, enabled=True)
        extension = RegularizationExtension(ext_config, rngs=rngs)
        extensions = {"reg": extension}

        # Simple loss function
        def loss_fn(model, batch_data, rng):  # noqa: ARG001
            x = batch_data["input"].reshape(batch_data["input"].shape[0], -1)
            outputs = model(x)  # VAE doesn't take rngs in __call__
            if isinstance(outputs, dict) and "reconstruction" in outputs:
                loss = jnp.mean(jnp.square(outputs["reconstruction"] - x))
            else:
                loss = jnp.array(1.0)
            return loss, {}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            extensions=extensions,
            loss_fn=loss_fn,
        )

        # Run multiple training steps
        losses = []
        for _ in range(5):
            batch = {
                "input": jax.random.normal(jax.random.key(0), (4, 28, 28, 1)),
            }
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        # Should have trained for 5 steps
        assert trainer.step == 5
        assert len(losses) == 5


class TestExtensionStateManagement:
    """Tests for extension state serialization and restoration."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_extension_state_extraction(self, rngs):
        """Test that extension state can be extracted with nnx.state."""
        config = ExtensionConfig(name="stateful", weight=1.0, enabled=True)
        extension = RegularizationExtension(config, rngs=rngs)

        # Extract state
        state = nnx.state(extension)

        # State should not be None
        assert state is not None

    def test_extension_state_update(self, rngs):
        """Test that extension state can be updated."""
        config = ExtensionConfig(name="stateful", weight=1.0, enabled=True)
        ext1 = RegularizationExtension(config, rngs=rngs)
        ext2 = RegularizationExtension(config, rngs=nnx.Rngs(99))

        # Extract state from ext1
        state1 = nnx.state(ext1)

        # Update ext2 with state from ext1
        nnx.update(ext2, state1)

        # States should now match
        state2 = nnx.state(ext2)

        # Compare state dictionaries
        for key in state1:
            if hasattr(state1[key], "value") and hasattr(state2[key], "value"):
                assert jnp.allclose(state1[key].value, state2[key].value)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_empty_extensions_dict(self, rngs):
        """Test trainer with empty extensions dictionary."""
        from artifex.generative_models.training import Trainer

        # Create simple model
        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs):
                super().__init__()
                self.linear = nnx.Linear(4, 4, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=rngs)
        optimizer = OptimizerConfig(name="test_opt", optimizer_type="adam", learning_rate=0.001)
        config = TrainingConfig(name="test", num_epochs=1, batch_size=4, optimizer=optimizer)

        trainer = Trainer(
            model=model,
            training_config=config,
            extensions={},  # Empty dict
            loss_fn=lambda m, b, r: (jnp.array(1.0), {}),  # noqa: ARG005
        )

        batch = {"input": jnp.ones((4, 4))}
        metrics = trainer.train_step(batch)

        assert "loss" in metrics

    def test_extension_returns_nan_handled_gracefully(self, rngs):
        """Test that NaN from extension doesn't crash training."""

        class NaNExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def loss_fn(self, batch, model_outputs, **kwargs):
                # This could happen in edge cases
                return jnp.array(float("nan"))

        from artifex.generative_models.training import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs):
                super().__init__()
                self.linear = nnx.Linear(4, 4, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=rngs)
        optimizer = OptimizerConfig(name="test_opt", optimizer_type="adam", learning_rate=0.001)
        config = TrainingConfig(name="test", num_epochs=1, batch_size=4, optimizer=optimizer)
        ext_config = ExtensionConfig(name="nan_ext", weight=1.0, enabled=True)
        extension = NaNExtension(ext_config, rngs=rngs)

        trainer = Trainer(
            model=model,
            training_config=config,
            extensions={"nan": extension},
            loss_fn=lambda m, b, r: (jnp.array(1.0), {}),  # noqa: ARG005
        )

        batch = {"input": jnp.ones((4, 4))}

        # Should not raise, but loss will be NaN
        metrics = trainer.train_step(batch)
        assert "nan_loss" in metrics

    def test_extension_with_zero_weight(self, rngs):
        """Test extension with zero weight has no effect on loss."""
        from artifex.generative_models.training import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs):
                super().__init__()
                self.linear = nnx.Linear(4, 4, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        class LargeExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def loss_fn(self, batch, model_outputs, **kwargs):
                return jnp.array(1000.0)  # Large loss

        model = SimpleModel(rngs=rngs)
        optimizer = OptimizerConfig(name="test_opt", optimizer_type="adam", learning_rate=0.001)
        config = TrainingConfig(name="test", num_epochs=1, batch_size=4, optimizer=optimizer)

        # Zero weight should make this not contribute
        ext_config = ExtensionConfig(name="large", weight=0.0, enabled=True)
        extension = LargeExtension(ext_config, rngs=rngs)

        trainer = Trainer(
            model=model,
            training_config=config,
            extensions={"large": extension},
            loss_fn=lambda m, b, r: (jnp.array(1.0), {}),  # noqa: ARG005
        )

        batch = {"input": jnp.ones((4, 4))}
        metrics = trainer.train_step(batch)

        # Extension loss should be 0 (weight * 1000 = 0)
        assert metrics.get("large_loss", 0.0) == 0.0

    def test_vmap_compatible_extension_loss(self, rngs):
        """Test that extension loss_fn works with jax.vmap."""
        config = ExtensionConfig(name="vmap_test", weight=1.0, enabled=True)
        extension = RegularizationExtension(config, rngs=rngs)

        # Create batched function
        def single_loss(output):
            return extension.loss_fn({}, output)

        # This should work with vmap
        outputs = jnp.ones((8, 4))  # 8 samples

        # vmap over first axis
        batched_loss = jax.vmap(single_loss)(outputs)

        assert batched_loss.shape == (8,)
        assert not jnp.any(jnp.isnan(batched_loss))
