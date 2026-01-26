"""Performance tests for model inference and training."""

import time

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


def _create_vae_config(
    name: str,
    input_shape: tuple,
    latent_dim: int,
    hidden_dims: tuple,
    use_batch_norm: bool = True,
):
    """Create a VAEConfig with proper nested encoder/decoder configs.

    Args:
        name: Name of the config
        input_shape: Shape of input data
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        use_batch_norm: Whether to use batch normalization (default True).
            Set to False when comparing batch vs individual processing
            since BatchNorm statistics differ by batch size.
    """
    from artifex.generative_models.core.configuration import (
        DecoderConfig,
        EncoderConfig,
        VAEConfig,
    )

    encoder = EncoderConfig(
        name=f"{name}_encoder",
        input_shape=input_shape,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        use_batch_norm=use_batch_norm,
    )

    decoder = DecoderConfig(
        name=f"{name}_decoder",
        latent_dim=latent_dim,
        output_shape=input_shape,
        hidden_dims=tuple(reversed(hidden_dims)),
        activation="relu",
    )

    return VAEConfig(
        name=name,
        encoder=encoder,
        decoder=decoder,
        encoder_type="dense",
        kl_weight=1.0,
    )


@pytest.mark.performance
@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for model operations."""

    def test_vae_inference_performance(self, benchmark):
        """Benchmark VAE inference latency."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        batch_size = 8
        test_batch = jnp.ones((batch_size, 32, 32, 3))

        # Warm up
        _ = model.encode(test_batch)

        # Benchmark encoding
        def encode_batch():
            return model.encode(test_batch)

        result = benchmark(encode_batch)

        # Verify result shape
        mu, logvar = result
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)

    def test_vae_sampling_performance(self, benchmark):
        """Benchmark VAE sampling performance."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        n_samples = 8

        # Warm up
        _ = model.sample(n_samples=2)

        # Benchmark sampling
        def sample_batch():
            return model.sample(n_samples=n_samples)

        result = benchmark(sample_batch)

        # Verify result shape
        assert result.shape == (n_samples, 32, 32, 3)

    def test_diffusion_inference_performance(self, benchmark):
        """Benchmark diffusion model inference performance."""
        try:
            from tests.utils.test_helpers import should_run_diffusion_tests

            from artifex.generative_models.core.configuration import (
                DDPMConfig,
                NoiseScheduleConfig,
                UNetBackboneConfig,
            )
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("Diffusion model not available")

        if not should_run_diffusion_tests():
            pytest.skip("Diffusion tests disabled")

        # Setup
        rng = jax.random.key(42)

        # Create proper nested configs for DDPM
        backbone = UNetBackboneConfig(
            name="test_backbone",
            hidden_dims=(16, 32),
            activation="gelu",
            in_channels=3,
            out_channels=3,
            time_embedding_dim=16,
            num_res_blocks=1,
            attention_resolutions=(),  # No attention for speed
            channel_mult=(1, 2),
        )

        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            schedule_type="linear",
            num_timesteps=10,  # Fewer steps for testing
            beta_start=1e-4,
            beta_end=0.02,
        )

        config = DDPMConfig(
            name="test_ddpm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(16, 16, 3),  # Smaller for performance testing
            loss_type="mse",
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        batch_size = 4
        test_batch = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([2, 4, 6, 8])

        # Warm up
        _ = model(test_batch, t)

        # Benchmark noise prediction
        def predict_noise():
            return model(test_batch, t)

        result = benchmark(predict_noise)

        # Verify result shape
        assert result.shape == test_batch.shape

    def test_gan_generator_performance(self, benchmark):
        """Benchmark GAN generator performance."""
        try:
            from artifex.generative_models.core.configuration import (
                ConvDiscriminatorConfig,
                ConvGeneratorConfig,
                DCGANConfig,
            )
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("GAN model not available")

        # Setup
        rng = jax.random.key(42)

        # Create proper nested configs for DCGAN
        generator = ConvGeneratorConfig(
            name="test_generator",
            latent_dim=16,
            output_shape=(3, 32, 32),  # channels first for GAN
            hidden_dims=(64, 32),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="test_discriminator",
            input_shape=(3, 32, 32),  # channels first for GAN
            hidden_dims=(32, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = DCGANConfig(
            name="test_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, dropout=rng, sample=rng))
        batch_size = 8
        z = jax.random.normal(rng, (batch_size, 16))

        # Warm up
        _ = model.generator(z)

        # Benchmark generation
        def generate_batch():
            return model.generator(z)

        result = benchmark(generate_batch)

        # Verify result shape - GAN models typically return (batch, channels, height, width)
        assert result.shape == (batch_size, 3, 32, 32)

    def test_flow_forward_performance(self, benchmark):
        """Benchmark normalizing flow forward pass performance."""
        try:
            from artifex.generative_models.core.configuration import (
                CouplingNetworkConfig,
                RealNVPConfig,
            )
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("Flow model not available")

        # Setup
        rng = jax.random.key(42)
        input_dim = 32 * 32 * 3

        # Create proper nested config for RealNVP
        coupling_network = CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 128),
            activation="relu",
            network_type="mlp",
            scale_activation="tanh",
        )

        config = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=input_dim,
            num_coupling_layers=4,
            mask_type="checkerboard",
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        batch_size = 8
        test_batch = jax.random.normal(rng, (batch_size, input_dim))

        # Warm up
        _ = model.forward(test_batch)

        # Benchmark forward pass
        def forward_pass():
            return model.forward(test_batch)

        z, log_det = benchmark(forward_pass)

        # Verify result shapes
        assert z.shape == test_batch.shape
        assert log_det.shape == (batch_size,)


@pytest.mark.performance
class TestMemoryUsage:
    """Tests for memory usage and efficiency."""

    def test_vae_memory_footprint(self):
        """Test VAE model memory usage doesn't exceed thresholds."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(64, 64, 3),  # Larger model for memory testing
            latent_dim=32,
            hidden_dims=(64, 128, 256),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))

        # Test with different batch sizes
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            test_batch = jnp.ones((batch_size, 64, 64, 3))

            # Should not raise memory errors
            mu, logvar = model.encode(test_batch)
            z = model.reparameterize(mu, logvar)
            reconstruction = model.decode(z)

            # Verify shapes
            assert mu.shape == (batch_size, 32)
            assert reconstruction.shape == test_batch.shape

    @pytest.mark.parametrize(
        "batch_size,min_speedup",
        [
            (4, 2.0),  # Small batch: expect at least 2x speedup
            (8, 4.0),  # Medium batch: expect at least 4x speedup
            (16, 8.0),  # Larger batch: expect at least 8x speedup
            (32, 16.0),  # Large batch: expect at least 16x speedup
        ],
    )
    def test_batch_processing_efficiency(self, batch_size: int, min_speedup: float):
        """Test that batch processing is more efficient than individual items.

        Batch processing should provide significant speedup over processing
        items individually. The speedup scales roughly linearly with batch size
        as the overhead of kernel launches is amortized.
        """
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Use float32 precision for numerical equivalence testing
        # (bfloat16 default causes slight differences between batch/individual)
        with jax.default_matmul_precision("float32"):
            self._run_batch_efficiency_test(batch_size, min_speedup, create_model)

    def _run_batch_efficiency_test(self, batch_size: int, min_speedup: float, create_model):
        """Implementation of batch efficiency test."""
        # Setup
        rng = jax.random.key(42)
        # Disable batch norm to ensure numerical equivalence between batch
        # and individual processing (BatchNorm statistics differ by batch size)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
            use_batch_norm=False,
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        # Put model in eval mode for deterministic inference
        model.eval()

        # Create test data
        test_batch = jnp.ones((batch_size, 32, 32, 3))
        individual_items = [test_batch[i : i + 1] for i in range(batch_size)]

        # Warmup runs to trigger JIT compilation for both batch sizes
        _ = model.encode(test_batch)
        _ = model.encode(individual_items[0])

        # Time batch processing (multiple runs for stability)
        n_runs = 5
        start_time = time.time()
        for _ in range(n_runs):
            batch_result = model.encode(test_batch)
        batch_time = (time.time() - start_time) / n_runs

        # Time individual processing
        start_time = time.time()
        for _ in range(n_runs):
            individual_results = []
            for item in individual_items:
                result = model.encode(item)
                individual_results.append(result)
        individual_time = (time.time() - start_time) / n_runs

        # Batch processing should be significantly more efficient than individual processing
        efficiency_ratio = individual_time / batch_time
        assert efficiency_ratio > min_speedup, (
            f"Batch processing not efficient enough for batch_size={batch_size}: "
            f"{efficiency_ratio:.2f}x (expected >{min_speedup}x)"
        )

        # Results should be exactly equivalent (with strict floating point tolerance)
        batch_mu, batch_logvar = batch_result
        for i, (ind_mu, ind_logvar) in enumerate(individual_results):
            assert jnp.allclose(batch_mu[i : i + 1], ind_mu, atol=1e-6)
            assert jnp.allclose(batch_logvar[i : i + 1], ind_logvar, atol=1e-6)


@pytest.mark.performance
@pytest.mark.slow
class TestRegressionBenchmarks:
    """Regression tests to ensure performance doesn't degrade."""

    def test_vae_training_step_performance(self):
        """Test VAE training step performance baseline."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        batch_size = 8
        test_batch = jnp.ones((batch_size, 32, 32, 3))

        # Warm up
        for _ in range(3):
            _ = model.loss_fn(x=test_batch, rng=jax.random.key(42))

        # Time training step
        start_time = time.time()
        loss_dict = model.loss_fn(x=test_batch, rng=jax.random.key(42))
        step_time = time.time() - start_time

        # Performance baseline: should complete within reasonable time
        max_step_time = 1.0  # 1 second max for small model
        assert step_time < max_step_time, f"Training step too slow: {step_time:.3f}s"

        # Verify outputs
        assert jnp.isfinite(loss_dict["loss"])
        assert "reconstruction_loss" in loss_dict
        assert "kl_loss" in loss_dict

    def test_model_compilation_time(self):
        """Test that model compilation doesn't take too long."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Time model creation and first forward pass
        start_time = time.time()

        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng))
        test_batch = jnp.ones((4, 32, 32, 3))

        # First forward pass (includes compilation)
        _ = model.encode(test_batch)

        compilation_time = time.time() - start_time

        # Should compile within reasonable time
        max_compilation_time = 10.0  # 10 seconds max
        assert compilation_time < max_compilation_time, (
            f"Model compilation too slow: {compilation_time:.3f}s"
        )
