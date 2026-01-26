"""Integration tests for energy-based models.

These tests verify that all components work together correctly in realistic scenarios,
including training workflows, generation pipelines, and cross-component interactions.

Updated to use dataclass-based configs (EBMConfig, DeepEBMConfig) following Principle #4.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import (
    create_cifar_ebm,
    create_mnist_ebm,
    create_simple_ebm,
    DeepEBM,
    EBM,
    langevin_dynamics,
    persistent_contrastive_divergence,
    SampleBuffer,
)
from tests.artifex.generative_models.models.energy.conftest import jax_required


# =============================================================================
# Helper fixtures for dataclass configs
# =============================================================================


@pytest.fixture
def mcmc_config():
    """Create MCMCConfig for testing."""
    return MCMCConfig(
        name="test_mcmc",
        n_steps=60,
        step_size=0.01,
        noise_scale=0.005,
    )


@pytest.fixture
def sample_buffer_config():
    """Create SampleBufferConfig for testing."""
    return SampleBufferConfig(
        name="test_buffer",
        capacity=64,
        reinit_prob=0.05,
    )


class TestEnergyModelWorkflows:
    """Test complete energy model workflows from initialization to generation."""

    @jax_required
    def test_mlp_ebm_training_workflow(self, energy_rngs, mcmc_config, sample_buffer_config):
        """Test complete MLP EBM training workflow."""
        # Create synthetic dataset
        key = jax.random.key(42)
        dataset_size = 32
        input_dim = 16

        # Generate synthetic mixture data in input_dim dimensions
        n_modes = 4
        key, subkey = jax.random.split(key)
        mode_centers = 2.0 * jax.random.normal(subkey, (n_modes, input_dim))

        all_data = []
        for i in range(dataset_size):
            mode_idx = i % n_modes
            center = mode_centers[mode_idx]
            key, subkey = jax.random.split(key)
            point = center + 0.3 * jax.random.normal(subkey, (input_dim,))
            all_data.append(point)

        dataset = jnp.stack(all_data)

        # Create EBM with dataclass configuration
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(32, 16),
            activation="gelu",
            network_type="mlp",
            use_bias=True,
        )

        config = EBMConfig(
            name="test_ebm",
            input_dim=input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Training loop simulation
        batch_size = 8
        n_training_steps = 5

        for step in range(n_training_steps):
            # Create mini-batch
            batch_start = (step * batch_size) % dataset_size
            batch_end = min(batch_start + batch_size, dataset_size)
            batch_data = dataset[batch_start:batch_end]

            batch = {
                "data": batch_data,
                "batch_size": batch_data.shape[0],
            }

            # Training step (uses internal rngs)
            loss_dict = ebm.train_step(batch)

            # Verify training step produces valid results
            assert jnp.isfinite(loss_dict["loss"])
            assert loss_dict["real_energy_mean"] != loss_dict["fake_energy_mean"]

        # Generate samples after training (uses internal rngs)
        generated_samples = ebm.generate(
            n_samples=16,
            shape=(input_dim,),
            n_steps=20,
        )

        # Verify generated samples
        assert generated_samples.shape == (16, input_dim)
        assert jnp.all(jnp.isfinite(generated_samples))

        # Test that buffer has been populated
        assert len(ebm.sample_buffer.buffer) > 0

    @jax_required
    def test_cnn_ebm_image_workflow(self, energy_rngs, mcmc_config, sample_buffer_config):
        """Test complete CNN EBM workflow with image data."""
        # Create synthetic image dataset
        key = jax.random.key(123)
        batch_size = 8
        height, width, channels = 16, 16, 1

        # Generate synthetic images with simple patterns
        images = []
        for i in range(batch_size):
            key, subkey = jax.random.split(key)
            # Create checkerboard patterns with noise
            image = jnp.zeros((height, width, channels))
            for h in range(height):
                for w in range(width):
                    if (h + w) % 4 == i % 4:
                        image = image.at[h, w, 0].set(1.0)

            # Add noise
            noise = 0.1 * jax.random.normal(subkey, (height, width, channels))
            image = image + noise
            images.append(jnp.clip(image, -1, 1))

        dataset = jnp.stack(images)

        # Create CNN EBM with dataclass configuration (using DeepEBM for CNN)
        energy_config = EnergyNetworkConfig(
            name="test_cnn_energy",
            hidden_dims=(8, 16),
            activation="gelu",
            network_type="cnn",
            use_bias=True,
        )

        config = DeepEBMConfig(
            name="test_cnn_ebm",
            input_shape=(height, width, channels),
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Training workflow
        n_steps = 3
        for step in range(n_steps):
            batch = {
                "data": dataset,
                "batch_size": batch_size,
            }

            loss_dict = ebm.train_step(batch)

            assert jnp.isfinite(loss_dict["loss"])

        # Generate new images (uses internal rngs)
        generated_images = ebm.generate(
            n_samples=4,
            shape=(height, width, channels),
            n_steps=15,
        )

        # Verify generated images
        assert generated_images.shape == (4, height, width, channels)
        assert jnp.all(jnp.isfinite(generated_images))

    @jax_required
    def test_deep_ebm_advanced_workflow(self, energy_rngs, mcmc_config):
        """Test advanced Deep EBM workflow with multiple features."""
        # Create Deep EBM for complex images with dataclass configuration
        energy_config = EnergyNetworkConfig(
            name="test_deep_energy",
            hidden_dims=(16, 32, 64),
            activation="gelu",
            network_type="cnn",
            use_residual=True,
            use_spectral_norm=True,
        )

        buffer_config = SampleBufferConfig(
            name="test_buffer",
            capacity=128,
            reinit_prob=0.05,
        )

        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=buffer_config,
        )
        deep_ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Create synthetic RGB data
        key = jax.random.key(456)
        batch_size = 4
        height, width, channels = 16, 16, 3

        # Generate colorful synthetic data
        images = []
        for i in range(batch_size):
            key, subkey = jax.random.split(key)
            # Create colored patches
            image = jnp.zeros((height, width, channels))
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        if (h + w + c) % 3 == i % 3:
                            image = image.at[h, w, c].set(1.0)

            # Add noise
            noise = 0.05 * jax.random.normal(subkey, (height, width, channels))
            image = image + noise
            images.append(jnp.clip(image, -1, 1))

        dataset = jnp.stack(images)

        # Training with multiple steps
        n_steps = 4
        loss_history = []

        for step in range(n_steps):
            batch = {
                "data": dataset,
                "batch_size": batch_size,
            }

            loss_dict = deep_ebm.train_step(batch)

            loss_history.append(loss_dict["loss"])
            assert jnp.isfinite(loss_dict["loss"])

        # Generate samples (uses internal rngs)
        generated = deep_ebm.generate(
            n_samples=2,
            shape=(height, width, channels),
            n_steps=10,
        )

        assert generated.shape == (2, height, width, channels)
        assert jnp.all(jnp.isfinite(generated))

    @jax_required
    def test_factory_function_workflows(self, energy_rngs):
        """Test workflows using factory functions."""
        # Test MNIST EBM workflow
        mnist_ebm = create_mnist_ebm(rngs=energy_rngs)

        # Create MNIST-like data
        key = jax.random.key(789)
        mnist_data = jax.random.normal(key, (4, 28, 28, 1))
        mnist_data = jnp.clip(mnist_data, -1, 1)

        batch = {"data": mnist_data, "batch_size": 4}
        loss_dict = mnist_ebm.train_step(batch)
        assert jnp.isfinite(loss_dict["loss"])

        # Test CIFAR EBM workflow
        cifar_ebm = create_cifar_ebm(rngs=energy_rngs)

        # Create CIFAR-like data
        cifar_data = jax.random.normal(key, (4, 32, 32, 3))
        cifar_data = jnp.clip(cifar_data, -1, 1)

        batch = {"data": cifar_data, "batch_size": 4}
        loss_dict = cifar_ebm.train_step(batch)
        assert jnp.isfinite(loss_dict["loss"])

        # Test simple EBM workflow
        simple_ebm = create_simple_ebm(input_dim=10, rngs=energy_rngs)

        # Create vector data
        vector_data = jax.random.normal(key, (4, 10))
        batch = {"data": vector_data, "batch_size": 4}
        loss_dict = simple_ebm.train_step(batch)
        assert jnp.isfinite(loss_dict["loss"])


class TestCrossComponentIntegration:
    """Test integration between different energy model components."""

    @jax_required
    def test_mcmc_ebm_integration(self, energy_rngs, mcmc_config, sample_buffer_config):
        """Test integration between MCMC utilities and EBM models."""
        # Create EBM with dataclass configuration
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_mcmc_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Create test data
        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (8, 10))

        # Set eval mode to disable dropout during MCMC (prevents TraceContextError)
        ebm.energy_fn.eval()

        # Test direct MCMC with EBM energy function
        final_samples = langevin_dynamics(
            energy_fn=ebm.energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.01,
            rng_key=key,
        )

        assert final_samples.shape == initial_samples.shape
        assert jnp.all(jnp.isfinite(final_samples))

        # Test persistent contrastive divergence integration
        sample_buffer = SampleBuffer(capacity=32, reinit_prob=0.1)
        sample_buffer.update_buffer(initial_samples)

        _, pcd_samples = persistent_contrastive_divergence(
            energy_fn=ebm.energy_fn,
            real_samples=initial_samples,
            sample_buffer=sample_buffer,
            rng_key=key,
            n_mcmc_steps=10,
        )

        assert pcd_samples.shape == initial_samples.shape
        assert jnp.all(jnp.isfinite(pcd_samples))

    @jax_required
    def test_buffer_ebm_integration(self, energy_rngs, mcmc_config):
        """Test integration between sample buffer and EBM training."""
        # Create EBM with dataclass configuration
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        buffer_config = SampleBufferConfig(
            name="test_buffer",
            capacity=64,
            reinit_prob=0.05,
        )

        config = EBMConfig(
            name="test_buffer_ebm",
            input_dim=12,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Simulate multiple training steps with buffer evolution
        key = jax.random.key(42)

        for step in range(5):
            # Generate batch data
            key, subkey = jax.random.split(key)
            batch_data = jax.random.normal(subkey, (8, 12))
            batch = {"data": batch_data, "batch_size": 8}

            # Training step (uses internal rngs)
            loss_dict = ebm.train_step(batch)

            assert jnp.isfinite(loss_dict["loss"])

            # Check buffer growth
            assert len(ebm.sample_buffer.buffer) >= 1

        # Test sampling from populated buffer (uses internal rngs)
        buffer_samples = ebm.sample_from_buffer(4)

        assert buffer_samples.shape == (4, 12)
        assert jnp.all(jnp.isfinite(buffer_samples))


class TestErrorHandlingIntegration:
    """Test error handling and edge cases in integrated workflows."""

    @jax_required
    def test_mismatched_dimensions_error_handling(
        self, energy_rngs, mcmc_config, sample_buffer_config
    ):
        """Test error handling for mismatched dimensions."""
        # Create EBM for specific input dimension with dataclass configuration
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_mismatch_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Try to use data with wrong dimension
        key = jax.random.key(42)
        wrong_dim_data = jax.random.normal(key, (4, 5))  # Should be (4, 10)

        # This should raise an error or produce invalid results
        with pytest.raises((ValueError, TypeError, AttributeError)):
            ebm.energy_fn(wrong_dim_data)

    @jax_required
    def test_empty_buffer_error_handling(self, energy_rngs, mcmc_config):
        """Test error handling for empty sample buffer."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        buffer_config = SampleBufferConfig(
            name="test_buffer",
            capacity=32,
            reinit_prob=0.05,
        )

        config = EBMConfig(
            name="test_empty_buffer_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Try to sample from empty buffer (uses internal rngs)
        with pytest.raises(RuntimeError, match="Sample buffer is empty"):
            ebm.sample_from_buffer(4)

    @jax_required
    def test_invalid_mcmc_parameters(self, energy_rngs, mcmc_config, sample_buffer_config):
        """Test handling of invalid MCMC parameters."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_invalid_mcmc_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        key = jax.random.key(42)
        test_data = jax.random.normal(key, (4, 10))

        # Set eval mode to disable dropout during MCMC (prevents TraceContextError)
        ebm.energy_fn.eval()

        # Test with extreme parameters that might cause numerical issues
        try:
            samples = langevin_dynamics(
                energy_fn=ebm.energy_fn,
                initial_samples=test_data,
                n_steps=1000,  # Very many steps
                step_size=1.0,  # Large step size
                noise_scale=10.0,  # Large noise
                rng_key=key,
            )
            # If it doesn't crash, samples should still be finite
            assert jnp.all(jnp.isfinite(samples))
        except (ValueError, FloatingPointError):
            # It's okay if extreme parameters cause errors
            pass


class TestPerformanceIntegration:
    """Test performance characteristics in integrated scenarios."""

    @jax_required
    def test_batch_size_scaling(self, energy_rngs, mcmc_config, sample_buffer_config):
        """Test that models handle different batch sizes correctly."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_batch_scaling_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        key = jax.random.key(42)

        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            test_data = jax.random.normal(key, (batch_size, 10))

            # Forward pass
            outputs = ebm.energy_outputs(test_data)
            assert outputs["energy"].shape == (batch_size,)

            # Training step
            batch = {"data": test_data, "batch_size": batch_size}
            loss_dict = ebm.train_step(batch)
            assert jnp.isfinite(loss_dict["loss"])

    @jax_required
    def test_memory_efficiency_large_buffers(self, energy_rngs, mcmc_config):
        """Test memory efficiency with large sample buffers."""
        # Create EBM with large buffer using dataclass configuration
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(8, 4),
            activation="gelu",
            network_type="mlp",
        )

        buffer_config = SampleBufferConfig(
            name="test_buffer",
            capacity=128,
            reinit_prob=0.05,
        )

        config = EBMConfig(
            name="test_memory_ebm",
            input_dim=4,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        key = jax.random.key(42)

        # Simulate many training steps to fill buffer
        for _ in range(10):
            key, subkey = jax.random.split(key)
            batch_data = jax.random.normal(subkey, (8, 4))
            batch = {"data": batch_data, "batch_size": 8}

            loss_dict = ebm.train_step(batch)
            assert jnp.isfinite(loss_dict["loss"])

        # Verify buffer doesn't exceed capacity
        assert len(ebm.sample_buffer.buffer) <= ebm.sample_buffer.capacity

        # Test that sampling still works efficiently (uses internal rngs)
        samples = ebm.sample_from_buffer(16)
        assert samples.shape == (16, 4)


class TestReproducibilityIntegration:
    """Test reproducibility across integrated workflows."""

    @jax_required
    def test_deterministic_training_with_fixed_seeds(self, mcmc_config, sample_buffer_config):
        """Test that training produces consistent results with fixed seeds.

        Note: Due to independent sample buffers evolving differently, we test
        for reasonable consistency rather than exact determinism.
        """
        # Create two identical EBMs with fresh RNG objects (same seed)
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_deterministic_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )

        ebm1 = EBM(config=config, rngs=nnx.Rngs(12345))
        ebm2 = EBM(config=config, rngs=nnx.Rngs(12345))

        # Create identical test data
        key = jax.random.key(42)
        test_data = jax.random.normal(key, (8, 10))
        batch = {"data": test_data, "batch_size": 8}

        # Train both models
        loss_dict1 = ebm1.train_step(batch)
        loss_dict2 = ebm2.train_step(batch)

        # Results should be finite and reasonable
        assert jnp.isfinite(loss_dict1["loss"])
        assert jnp.isfinite(loss_dict2["loss"])
        assert jnp.isfinite(loss_dict1["real_energy_mean"])
        assert jnp.isfinite(loss_dict2["real_energy_mean"])
        assert jnp.isfinite(loss_dict1["fake_energy_mean"])
        assert jnp.isfinite(loss_dict2["fake_energy_mean"])

        # Real energy means should be similar
        assert jnp.allclose(
            loss_dict1["real_energy_mean"], loss_dict2["real_energy_mean"], atol=1e-5
        )

    @jax_required
    def test_generation_reproducibility(self, mcmc_config, sample_buffer_config):
        """Test that generation is reproducible with fixed seeds."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_generation_reproducibility_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )

        # Create two EBMs with same seed for reproducibility
        ebm1 = EBM(config=config, rngs=nnx.Rngs(2000))
        ebm2 = EBM(config=config, rngs=nnx.Rngs(2000))

        # Generate samples from each (uses internal rngs)
        samples1 = ebm1.generate(
            n_samples=4,
            shape=(10,),
            n_steps=5,
        )

        samples2 = ebm2.generate(
            n_samples=4,
            shape=(10,),
            n_steps=5,
        )

        # Samples should be identical (same init seed = same RNG sequence)
        assert jnp.allclose(samples1, samples2, atol=1e-6)

    @jax_required
    def test_mcmc_reproducibility(self, energy_rngs, mcmc_config, sample_buffer_config):
        """Test MCMC reproducibility in integrated workflow."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_mcmc_reproducibility_ebm",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Set eval mode to disable dropout during MCMC
        ebm.energy_fn.eval()

        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (4, 10))

        # Run MCMC twice with same seed
        mcmc_key1 = jax.random.key(3000)
        mcmc_key2 = jax.random.key(3000)

        final_samples1 = langevin_dynamics(
            energy_fn=ebm.energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.01,
            rng_key=mcmc_key1,
        )

        final_samples2 = langevin_dynamics(
            energy_fn=ebm.energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.01,
            rng_key=mcmc_key2,
        )

        # Results should be identical
        assert jnp.allclose(final_samples1, final_samples2, atol=1e-6)
