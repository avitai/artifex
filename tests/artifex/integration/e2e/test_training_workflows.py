"""End-to-end tests for training workflows."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from tests.utils.test_helpers import should_run_diffusion_tests


@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingWorkflows:
    """End-to-end tests for complete training workflows."""

    def test_vae_complete_workflow(self, e2e_config, sample_dataset, model_save_path, results_path):
        """Test complete VAE training workflow from data to evaluation."""
        try:
            from artifex.generative_models.core.configuration.network_configs import (
                DecoderConfig,
                EncoderConfig,
            )
            from artifex.generative_models.core.configuration.vae_config import VAEConfig
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Initialize model
        rng = jax.random.key(42)
        encoder_config = EncoderConfig(
            name="test_encoder",
            input_shape=e2e_config["image_size"],
            hidden_dims=tuple(e2e_config["hidden_dims"]),
            latent_dim=e2e_config["latent_dim"],
            activation="relu",
        )
        decoder_config = DecoderConfig(
            name="test_decoder",
            latent_dim=e2e_config["latent_dim"],
            hidden_dims=tuple(reversed(e2e_config["hidden_dims"])),
            output_shape=e2e_config["image_size"],
            activation="relu",
        )
        config = VAEConfig(
            name="test_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=1.0,
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))

        # Test forward pass
        batch = sample_dataset["images"][: e2e_config["batch_size"]]

        # Encode
        mu, logvar = model.encode(batch)
        expected_shape = (e2e_config["batch_size"], e2e_config["latent_dim"])
        assert mu.shape == expected_shape
        assert logvar.shape == expected_shape

        # Sample latent
        z = model.reparameterize(mu, logvar)
        assert z.shape == (e2e_config["batch_size"], e2e_config["latent_dim"])

        # Decode
        reconstruction = model.decode(z)
        assert reconstruction.shape == batch.shape

        # Test loss computation
        # VAE uses loss_fn which returns a dictionary with 'loss', 'reconstruction_loss', 'kl_loss'
        loss_dict = model.loss_fn(x=batch)
        assert jnp.isfinite(loss_dict["loss"])
        assert "reconstruction_loss" in loss_dict
        assert "kl_loss" in loss_dict

        # Test sampling
        samples = model.sample(n_samples=e2e_config["num_samples"])
        expected_samples_shape = (e2e_config["num_samples"],) + e2e_config["image_size"]
        assert samples.shape == expected_samples_shape

        # Verify all outputs are finite
        assert jnp.all(jnp.isfinite(reconstruction))
        assert jnp.all(jnp.isfinite(samples))

    def test_diffusion_complete_workflow(
        self, e2e_config, sample_dataset, model_save_path, results_path
    ):
        """Test complete diffusion model workflow."""
        if not should_run_diffusion_tests():
            pytest.skip("Diffusion tests disabled due to GroupNorm issues")

        try:
            from artifex.generative_models.models.diffusion import DDPMModel
        except ImportError:
            pytest.skip("Diffusion model not available")

        # Initialize model
        rng = jax.random.key(42)
        config = {
            "input_dim": e2e_config["image_size"],
            "noise_steps": 20,  # Small for testing
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "hidden_dims": e2e_config["hidden_dims"],
            "time_embedding_dim": 32,
        }

        model = DDPMModel(config, rngs=nnx.Rngs(params=rng))

        # Test forward diffusion process
        batch = sample_dataset["images"][: e2e_config["batch_size"]]
        t = jnp.array([5, 10, 15, 18])  # Different timesteps

        # Add noise
        noise = jax.random.normal(rng, batch.shape)
        noisy_batch = model.q_sample(batch, t, noise)
        assert noisy_batch.shape == batch.shape

        # Test noise prediction
        pred_noise = model(noisy_batch, t)
        assert pred_noise.shape == noise.shape
        assert jnp.all(jnp.isfinite(pred_noise))

        # Test sampling process
        samples = model.sample(n_samples=e2e_config["num_samples"], rngs=nnx.Rngs(params=rng))
        expected_shape = (e2e_config["num_samples"],) + e2e_config["image_size"]
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    def test_gan_complete_workflow(self, e2e_config, sample_dataset, model_save_path, results_path):
        """Test complete GAN training workflow."""
        try:
            from artifex.generative_models.models.gan import GANModel
        except ImportError:
            pytest.skip("GAN model not available")

        # Initialize model
        rng = jax.random.key(42)
        config = {
            "input_dim": e2e_config["image_size"],
            "latent_dim": e2e_config["latent_dim"],
            "generator_hidden_dims": e2e_config["hidden_dims"],
            "discriminator_hidden_dims": e2e_config["hidden_dims"],
        }

        model = GANModel(config, rngs=nnx.Rngs(params=rng))

        # Test generator
        batch_size = e2e_config["batch_size"]
        z = jax.random.normal(rng, (batch_size, e2e_config["latent_dim"]))
        fake_images = model.generator(z)
        assert fake_images.shape == (batch_size,) + e2e_config["image_size"]

        # Test discriminator
        real_batch = sample_dataset["images"][:batch_size]
        real_logits = model.discriminator(real_batch)
        fake_logits = model.discriminator(fake_images)

        assert real_logits.shape == (batch_size, 1)
        assert fake_logits.shape == (batch_size, 1)

        # Test loss computation
        g_loss = model.generator_loss(fake_logits)
        d_loss = model.discriminator_loss(real_logits, fake_logits)

        assert jnp.isfinite(g_loss)
        assert jnp.isfinite(d_loss)

        # Test sampling
        samples = model.sample(n_samples=e2e_config["num_samples"], rngs=nnx.Rngs(params=rng))
        expected_shape = (e2e_config["num_samples"],) + e2e_config["image_size"]
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    def test_flow_complete_workflow(
        self, e2e_config, sample_dataset, model_save_path, results_path
    ):
        """Test complete normalizing flow workflow."""
        try:
            from artifex.generative_models.models.flow import FlowModel
        except ImportError:
            pytest.skip("Flow model not available")

        # Initialize model
        rng = jax.random.key(42)
        config = {
            "input_dim": jnp.prod(jnp.array(e2e_config["image_size"])),
            "num_flows": 4,
            "hidden_dims": e2e_config["hidden_dims"],
        }

        model = FlowModel(config, rngs=nnx.Rngs(params=rng))

        # Flatten images for flow model
        batch = sample_dataset["images"][: e2e_config["batch_size"]]
        flat_batch = batch.reshape(batch.shape[0], -1)

        # Test forward pass (data to noise)
        z, log_det = model.forward(flat_batch)
        assert z.shape == flat_batch.shape
        assert log_det.shape == (e2e_config["batch_size"],)

        # Test inverse pass (noise to data)
        x_reconstructed = model.inverse(z)
        assert x_reconstructed.shape == flat_batch.shape

        # Test log probability computation
        log_prob = model.log_prob(flat_batch)
        assert log_prob.shape == (e2e_config["batch_size"],)
        assert jnp.all(jnp.isfinite(log_prob))

        # Test sampling
        samples = model.sample(n_samples=e2e_config["num_samples"], rngs=nnx.Rngs(params=rng))
        expected_shape = (e2e_config["num_samples"], config["input_dim"])
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    def test_multi_model_comparison_workflow(self, e2e_config, sample_dataset, results_path):
        """Test workflow comparing multiple models on the same data."""
        models_to_test = []

        # Try to load available models
        rng = jax.random.key(42)

        try:
            from artifex.generative_models.core.configuration.network_configs import (
                DecoderConfig,
                EncoderConfig,
            )
            from artifex.generative_models.core.configuration.vae_config import VAEConfig
            from artifex.generative_models.factory import create_model

            encoder_config = EncoderConfig(
                name="test_encoder",
                input_shape=e2e_config["image_size"],
                hidden_dims=tuple(e2e_config["hidden_dims"]),
                latent_dim=e2e_config["latent_dim"],
                activation="relu",
            )
            decoder_config = DecoderConfig(
                name="test_decoder",
                latent_dim=e2e_config["latent_dim"],
                hidden_dims=tuple(reversed(e2e_config["hidden_dims"])),
                output_shape=e2e_config["image_size"],
                activation="relu",
            )
            vae_config = VAEConfig(
                name="test_vae",
                encoder=encoder_config,
                decoder=decoder_config,
                kl_weight=1.0,
            )
            vae_model = create_model(vae_config, rngs=nnx.Rngs(params=rng))
            models_to_test.append(("VAE", vae_model))
        except ImportError:
            pass

        try:
            from artifex.generative_models.models.gan import GANModel

            gan_config = {
                "input_dim": e2e_config["image_size"],
                "latent_dim": e2e_config["latent_dim"],
                "generator_hidden_dims": e2e_config["hidden_dims"],
                "discriminator_hidden_dims": e2e_config["hidden_dims"],
            }
            gan_model = GANModel(gan_config, rngs=nnx.Rngs(params=rng))
            models_to_test.append(("GAN", gan_model))
        except ImportError:
            pass

        if not models_to_test:
            pytest.skip("No models available for comparison")

        # Test each model on the same data
        results = {}

        for model_name, model in models_to_test:
            # Generate samples (VAE and GAN use stored rngs, no need to pass)
            samples = model.sample(n_samples=e2e_config["num_samples"])

            # Store results
            results[model_name] = {
                "samples_shape": samples.shape,
                "samples_finite": jnp.all(jnp.isfinite(samples)),
                "samples_mean": jnp.mean(samples),
                "samples_std": jnp.std(samples),
            }

        # Verify all models produced valid outputs
        for model_name, result in results.items():
            assert result["samples_finite"], f"{model_name} produced non-finite samples"
            expected_shape = (e2e_config["num_samples"],) + e2e_config["image_size"]
            assert result["samples_shape"] == expected_shape, f"{model_name} wrong shape"

        # Results should be different between models (sanity check)
        if len(results) > 1:
            model_names = list(results.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    # Means should be different (with some tolerance)
                    mean_diff = abs(
                        results[model1]["samples_mean"] - results[model2]["samples_mean"]
                    )
                    assert mean_diff > 1e-6, f"{model1} and {model2} too similar"
