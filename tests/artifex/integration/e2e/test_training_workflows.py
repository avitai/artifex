"""End-to-end tests for training workflows."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    CouplingNetworkConfig,
    DCGANConfig,
    DDPMConfig,
    DecoderConfig,
    EncoderConfig,
    NoiseScheduleConfig,
    RealNVPConfig,
    UNetBackboneConfig,
    VAEConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.models.diffusion import DDPMModel
from artifex.generative_models.models.flow import RealNVP
from artifex.generative_models.models.gan import DCGAN


def _create_vae_model(e2e_config: dict[str, object]) -> object:
    rng = jax.random.key(42)
    image_size = e2e_config["image_size"]
    hidden_dims = tuple(e2e_config["hidden_dims"])
    latent_dim = e2e_config["latent_dim"]

    encoder_config = EncoderConfig(
        name="test_encoder",
        input_shape=image_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation="relu",
    )
    decoder_config = DecoderConfig(
        name="test_decoder",
        latent_dim=latent_dim,
        hidden_dims=tuple(reversed(hidden_dims)),
        output_shape=image_size,
        activation="relu",
    )
    config = VAEConfig(
        name="test_vae",
        encoder=encoder_config,
        decoder=decoder_config,
        kl_weight=1.0,
    )
    return create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))


def _create_dcgan_model(e2e_config: dict[str, object]) -> DCGAN:
    image_height, image_width, channels = e2e_config["image_size"]
    latent_dim = e2e_config["latent_dim"]
    hidden_dims = tuple(e2e_config["hidden_dims"])

    generator = ConvGeneratorConfig(
        name="test_generator",
        latent_dim=latent_dim,
        hidden_dims=tuple(reversed(hidden_dims)),
        output_shape=(channels, image_height, image_width),
        activation="relu",
        batch_norm=True,
        kernel_size=(4, 4),
        stride=(2, 2),
        padding="SAME",
    )
    discriminator = ConvDiscriminatorConfig(
        name="test_discriminator",
        hidden_dims=hidden_dims,
        input_shape=(channels, image_height, image_width),
        activation="leaky_relu",
        leaky_relu_slope=0.2,
        batch_norm=False,
        kernel_size=(4, 4),
        stride=(2, 2),
        padding="SAME",
    )
    config = DCGANConfig(
        name="test_dcgan",
        generator=generator,
        discriminator=discriminator,
    )
    return DCGAN(
        config,
        rngs=nnx.Rngs(
            params=jax.random.key(42),
            sample=jax.random.key(43),
            dropout=jax.random.key(44),
        ),
    )


def _create_realnvp_model(e2e_config: dict[str, object]) -> tuple[RealNVP, RealNVPConfig]:
    image_size = e2e_config["image_size"]
    input_dim = int(jnp.prod(jnp.array(image_size)))
    coupling = CouplingNetworkConfig(
        name="test_coupling",
        hidden_dims=(32, 32),
        activation="relu",
        network_type="mlp",
    )
    config = RealNVPConfig(
        name="test_realnvp",
        coupling_network=coupling,
        input_dim=input_dim,
        num_coupling_layers=4,
        mask_type="checkerboard",
    )
    model = RealNVP(config, rngs=nnx.Rngs(params=jax.random.key(52)))
    return model, config


@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingWorkflows:
    """End-to-end tests for complete training workflows."""

    def test_vae_complete_workflow(self, e2e_config, sample_dataset):
        """Test complete VAE training workflow from data to evaluation."""
        model = _create_vae_model(e2e_config)
        batch_size = e2e_config["batch_size"]
        latent_dim = e2e_config["latent_dim"]
        num_samples = e2e_config["num_samples"]
        image_size = e2e_config["image_size"]

        batch = sample_dataset["images"][:batch_size]

        mu, logvar = model.encode(batch)
        expected_shape = (batch_size, latent_dim)
        assert mu.shape == expected_shape
        assert logvar.shape == expected_shape

        z = model.reparameterize(mu, logvar)
        assert z.shape == expected_shape

        reconstruction = model.decode(z)
        assert reconstruction.shape == batch.shape

        model_outputs = model(batch)
        loss_dict = model.loss_fn(batch, model_outputs)
        assert jnp.isfinite(loss_dict["total_loss"])
        assert "reconstruction_loss" in loss_dict
        assert "kl_loss" in loss_dict

        samples = model.sample(n_samples=num_samples)
        assert samples.shape == (num_samples, *image_size)
        assert jnp.all(jnp.isfinite(reconstruction))
        assert jnp.all(jnp.isfinite(samples))

    def test_diffusion_complete_workflow(self, e2e_config, sample_dataset):
        """Test complete diffusion model workflow."""
        rng = jax.random.key(42)
        image_size = e2e_config["image_size"]
        hidden_dims = tuple(e2e_config["hidden_dims"])
        in_channels = image_size[-1]

        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=hidden_dims,
            activation="relu",
            in_channels=in_channels,
            out_channels=in_channels,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=20,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        config = DDPMConfig(
            name="test_ddpm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=image_size,
            loss_type="mse",
            clip_denoised=True,
        )

        model = DDPMModel(
            config,
            rngs=nnx.Rngs(
                params=rng,
                sample=rng,
                noise=rng,
                timestep=rng,
                dropout=rng,
            ),
        )

        batch = sample_dataset["images"][: e2e_config["batch_size"]]
        t = jnp.array([5, 10, 15, 18])
        noise = jax.random.normal(rng, batch.shape)
        noisy_batch = model.q_sample(batch, t, noise)
        assert noisy_batch.shape == batch.shape

        model_output = model(noisy_batch, t)
        pred_noise = model_output["predicted_noise"]
        assert pred_noise.shape == noise.shape
        assert jnp.all(jnp.isfinite(pred_noise))

        samples = model.sample(e2e_config["num_samples"])
        assert samples.shape == (e2e_config["num_samples"], *tuple(image_size))
        assert jnp.all(jnp.isfinite(samples))

    def test_dcgan_complete_workflow(self, e2e_config, sample_dataset):
        """Test complete DCGAN workflow using the live public GAN API."""
        model = _create_dcgan_model(e2e_config)
        batch_size = e2e_config["batch_size"]
        num_samples = e2e_config["num_samples"]
        image_size = e2e_config["image_size"]
        expected_shape = (batch_size, image_size[-1], image_size[0], image_size[1])

        fake_images = model.generate(batch_size, rngs=nnx.Rngs(sample=jax.random.key(60)))
        assert fake_images.shape == expected_shape
        assert jnp.all(jnp.isfinite(fake_images))

        real_batch = sample_dataset["images"][:batch_size]
        real_batch = jnp.transpose(real_batch, (0, 3, 1, 2))
        real_logits = model.discriminator(real_batch)
        fake_logits = model.discriminator(fake_images)

        assert real_logits.shape == (batch_size, 1)
        assert fake_logits.shape == (batch_size, 1)
        assert jnp.all(jnp.isfinite(real_logits))
        assert jnp.all(jnp.isfinite(fake_logits))

        generator_metrics = model.generator_objective(real_batch)
        discriminator_metrics = model.discriminator_objective(real_batch)
        assert jnp.isfinite(generator_metrics["total_loss"])
        assert jnp.isfinite(discriminator_metrics["total_loss"])

        samples = model.generate(num_samples, rngs=nnx.Rngs(sample=jax.random.key(61)))
        assert samples.shape == (num_samples, image_size[-1], image_size[0], image_size[1])
        assert jnp.all(jnp.isfinite(samples))

    def test_realnvp_complete_workflow(self, e2e_config, sample_dataset):
        """Test complete RealNVP workflow using typed configs."""
        model, config = _create_realnvp_model(e2e_config)
        batch_size = e2e_config["batch_size"]
        num_samples = e2e_config["num_samples"]

        batch = sample_dataset["images"][:batch_size]
        flat_batch = batch.reshape(batch.shape[0], -1)

        outputs = model(flat_batch)
        if isinstance(outputs, tuple):
            z, log_det = outputs
        else:
            z = outputs["z"]
            if "logdet" in outputs:
                log_det = outputs["logdet"]
            else:
                log_det = outputs["log_det_jacobian"]

        assert z.shape == flat_batch.shape
        assert log_det.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_det))

        reconstructed = model.inverse(z)
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]
        elif isinstance(reconstructed, dict):
            reconstructed = reconstructed["x"]

        assert reconstructed.shape == flat_batch.shape
        assert jnp.all(jnp.isfinite(reconstructed))

        log_prob = model.log_prob(flat_batch)
        assert log_prob.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(log_prob))

        samples = model.sample(num_samples, rngs=nnx.Rngs(params=jax.random.key(62)))
        assert samples.shape == (num_samples, config.input_dim)
        assert jnp.all(jnp.isfinite(samples))

    def test_multi_model_comparison_workflow(self, e2e_config):
        """Test comparing multiple live model families on the same sample count."""
        vae_model = _create_vae_model(e2e_config)
        dcgan_model = _create_dcgan_model(e2e_config)
        realnvp_model, _ = _create_realnvp_model(e2e_config)

        rng = jax.random.key(42)
        image_size = e2e_config["image_size"]
        backbone = UNetBackboneConfig(
            name="comparison_unet",
            hidden_dims=tuple(e2e_config["hidden_dims"]),
            activation="relu",
            in_channels=image_size[-1],
            out_channels=image_size[-1],
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="comparison_schedule",
            num_timesteps=20,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        diffusion_model = DDPMModel(
            DDPMConfig(
                name="comparison_ddpm",
                backbone=backbone,
                noise_schedule=noise_schedule,
                input_shape=image_size,
                loss_type="mse",
                clip_denoised=True,
            ),
            rngs=nnx.Rngs(
                params=rng,
                sample=rng,
                noise=rng,
                timestep=rng,
                dropout=rng,
            ),
        )

        num_samples = e2e_config["num_samples"]
        results = {
            "VAE": vae_model.sample(n_samples=num_samples),
            "Diffusion": diffusion_model.sample(num_samples),
            "DCGAN": dcgan_model.generate(num_samples, rngs=nnx.Rngs(sample=jax.random.key(70))),
            "RealNVP": realnvp_model.sample(num_samples, rngs=nnx.Rngs(params=jax.random.key(71))),
        }

        for model_name, samples in results.items():
            assert samples.shape[0] == num_samples, f"{model_name} returned the wrong batch size"
            assert jnp.all(jnp.isfinite(samples)), f"{model_name} produced non-finite samples"
            assert jnp.isfinite(jnp.mean(samples)), f"{model_name} mean should be finite"
            assert jnp.isfinite(jnp.std(samples)), f"{model_name} std should be finite"
