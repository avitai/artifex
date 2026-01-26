"""Unit tests for Energy-Based Model implementations.

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
from artifex.generative_models.models.energy.ebm import (
    create_cifar_ebm,
    create_mnist_ebm,
    create_simple_ebm,
    DeepCNNEnergyFunction,
    DeepEBM,
    EBM,
    EnergyBlock,
)
from artifex.generative_models.models.energy.mcmc import SampleBuffer
from tests.artifex.generative_models.models.energy.conftest import jax_required


# =============================================================================
# Helper fixtures for dataclass configs
# =============================================================================


@pytest.fixture
def mlp_energy_network_config():
    """Create EnergyNetworkConfig for MLP."""
    return EnergyNetworkConfig(
        name="test_mlp_energy",
        hidden_dims=(32, 16),
        activation="gelu",
        network_type="mlp",
        use_bias=True,
    )


@pytest.fixture
def cnn_energy_network_config():
    """Create EnergyNetworkConfig for CNN."""
    return EnergyNetworkConfig(
        name="test_cnn_energy",
        hidden_dims=(16, 32),
        activation="silu",
        network_type="cnn",
        use_bias=True,
    )


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
        capacity=8192,
        reinit_prob=0.05,
    )


class TestEBM:
    """Test the main EBM implementation."""

    @jax_required
    def test_ebm_initialization_mlp(
        self, energy_rngs, mlp_energy_network_config, mcmc_config, sample_buffer_config
    ):
        """Test EBM initialization with MLP energy function."""
        config = EBMConfig(
            name="test_ebm_mlp",
            input_dim=10,
            energy_network=mlp_energy_network_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
            alpha=0.01,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Check basic attributes
        assert ebm.sample_buffer is not None
        assert isinstance(ebm.sample_buffer, SampleBuffer)
        assert ebm.mcmc_steps == 60
        assert ebm.mcmc_step_size == 0.01
        assert ebm.alpha == 0.01

    @jax_required
    def test_ebm_initialization_cnn(
        self, energy_rngs, cnn_energy_network_config, mcmc_config, sample_buffer_config
    ):
        """Test EBM initialization with CNN energy function."""
        # For CNN, we use DeepEBM with input_shape
        config = DeepEBMConfig(
            name="test_ebm_cnn",
            input_shape=(32, 32, 3),
            energy_network=cnn_energy_network_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Check that CNN energy function was created
        assert ebm.energy_fn is not None

    @jax_required
    def test_ebm_train_step(
        self,
        energy_rngs: nnx.Rngs,
        energy_training_batch: dict[str, jax.Array],
        mlp_energy_network_config,
        mcmc_config,
        sample_buffer_config,
    ):
        """Test EBM training step."""
        input_dim = energy_training_batch["data"].shape[-1]

        # Update energy network config for this input_dim
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_ebm_mlp",
            input_dim=input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Perform training step (uses internal rngs)
        loss_dict = ebm.train_step(energy_training_batch)

        # Check loss components
        assert isinstance(loss_dict, dict)
        expected_keys = [
            "loss",
            "contrastive_divergence",
            "regularization",
            "real_energy_mean",
            "fake_energy_mean",
        ]
        for key in expected_keys:
            assert key in loss_dict
            assert jnp.isfinite(loss_dict[key])

    @jax_required
    def test_ebm_sample_from_buffer(
        self,
        energy_rngs: nnx.Rngs,
        energy_test_mlp_data: jax.Array,
        mcmc_config,
        sample_buffer_config,
    ):
        """Test sampling from EBM sample buffer."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_ebm_mlp",
            input_dim=input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Add some samples to the buffer first
        ebm.sample_buffer.update_buffer(energy_test_mlp_data)

        # Sample from buffer (uses internal rngs)
        n_samples = 4
        samples = ebm.sample_from_buffer(n_samples)

        # Check output
        assert samples.shape == (n_samples, input_dim)
        assert jnp.all(jnp.isfinite(samples))

    @jax_required
    def test_ebm_sample_from_empty_buffer(
        self, energy_rngs: nnx.Rngs, energy_test_input_dim: int, mcmc_config, sample_buffer_config
    ):
        """Test sampling from empty buffer raises error."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_ebm_mlp",
            input_dim=energy_test_input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Try to sample from empty buffer (uses internal rngs)
        with pytest.raises(RuntimeError, match="Sample buffer is empty"):
            ebm.sample_from_buffer(4)

    @jax_required
    def test_ebm_get_config(self, energy_rngs: nnx.Rngs, mcmc_config):
        """Test EBM configuration retrieval."""
        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(32, 16),
            activation="gelu",
            network_type="mlp",
        )

        buffer_config = SampleBufferConfig(
            name="test_buffer",
            capacity=128,
            reinit_prob=0.1,
        )

        config = EBMConfig(
            name="test_ebm_mlp",
            input_dim=10,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        retrieved_config = ebm.get_config()

        # Check configuration components
        assert isinstance(retrieved_config, dict)
        assert retrieved_config["sample_buffer_capacity"] == 128
        assert retrieved_config["sample_buffer_reinit_prob"] == 0.1
        assert retrieved_config["mcmc_steps"] == 60
        assert retrieved_config["mcmc_step_size"] == 0.01
        assert retrieved_config["mcmc_noise_scale"] == 0.005
        assert retrieved_config["alpha"] == 0.01

    @jax_required
    def test_ebm_different_activations(
        self,
        energy_rngs: nnx.Rngs,
        energy_test_mlp_data: jax.Array,
        mcmc_config,
        sample_buffer_config,
    ):
        """Test EBM with different activation functions."""
        input_dim = energy_test_mlp_data.shape[-1]
        activations = ["gelu", "silu", "relu"]

        for activation_name in activations:
            energy_config = EnergyNetworkConfig(
                name=f"test_energy_{activation_name}",
                hidden_dims=(16, 8),
                activation=activation_name,
                network_type="mlp",
            )

            config = EBMConfig(
                name=f"test_ebm_{activation_name}",
                input_dim=input_dim,
                energy_network=energy_config,
                mcmc=mcmc_config,
                sample_buffer=sample_buffer_config,
            )
            ebm = EBM(config=config, rngs=energy_rngs)

            # Test forward pass
            energies = ebm.energy(energy_test_mlp_data)
            assert jnp.all(jnp.isfinite(energies))
            assert energies.shape == (energy_test_mlp_data.shape[0],)

    @jax_required
    def test_ebm_dropout_configuration(
        self,
        energy_rngs: nnx.Rngs,
        energy_test_mlp_data: jax.Array,
        mcmc_config,
        sample_buffer_config,
    ):
        """Test EBM with dropout configuration."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_config = EnergyNetworkConfig(
            name="test_energy_dropout",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
            dropout_rate=0.5,
        )

        config = EBMConfig(
            name="test_ebm_dropout",
            input_dim=input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Test forward pass
        energies = ebm.energy(energy_test_mlp_data)
        assert jnp.all(jnp.isfinite(energies))


class TestDeepEBM:
    """Test the Deep EBM implementation."""

    @jax_required
    def test_deep_ebm_initialization(
        self, energy_rngs: nnx.Rngs, cnn_energy_network_config, mcmc_config, sample_buffer_config
    ):
        """Test Deep EBM initialization."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=cnn_energy_network_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        deep_ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Check that it's a specialized EBM
        assert isinstance(deep_ebm, EBM)
        assert deep_ebm.energy_fn is not None

    @jax_required
    def test_deep_ebm_forward_pass(self, energy_rngs: nnx.Rngs, mcmc_config, sample_buffer_config):
        """Test Deep EBM forward pass."""
        energy_config = EnergyNetworkConfig(
            name="test_cnn_energy",
            hidden_dims=(8, 16),
            activation="silu",
            network_type="cnn",
        )

        config = DeepEBMConfig(
            name="test_deep_ebm_forward",
            input_shape=(16, 16, 1),
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        deep_ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Create test image data
        key = jax.random.key(42)
        test_images = jax.random.normal(key, (4, 16, 16, 1))

        # Forward pass
        energies = deep_ebm.energy(test_images)

        # Check output
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    @jax_required
    def test_deep_ebm_configuration_options(
        self, energy_rngs: nnx.Rngs, mcmc_config, sample_buffer_config
    ):
        """Test Deep EBM with various configuration options."""
        energy_config = EnergyNetworkConfig(
            name="test_cnn_energy_options",
            hidden_dims=(16, 32),
            activation="silu",
            network_type="cnn",
            use_residual=True,
            use_spectral_norm=True,
        )

        config = DeepEBMConfig(
            name="test_deep_ebm_options",
            input_shape=(8, 8, 3),
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        deep_ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Test with different input
        key = jax.random.key(42)
        test_images = jax.random.normal(key, (2, 8, 8, 3))

        energies = deep_ebm.energy(test_images)
        assert jnp.all(jnp.isfinite(energies))


class TestDeepCNNEnergyFunction:
    """Test the Deep CNN Energy Function implementation."""

    @jax_required
    def test_deep_cnn_energy_function_initialization(self, energy_rngs: nnx.Rngs):
        """Test Deep CNN energy function initialization."""
        deep_cnn = DeepCNNEnergyFunction(
            hidden_dims=[16, 32, 64],
            input_channels=3,
            rngs=energy_rngs,
        )

        # Check basic attributes
        assert deep_cnn.input_channels == 3
        assert deep_cnn.hidden_dims == [16, 32, 64]
        assert deep_cnn.use_residual is True
        assert deep_cnn.use_spectral_norm is True

    @jax_required
    def test_deep_cnn_energy_function_forward_pass(self, energy_rngs: nnx.Rngs):
        """Test Deep CNN energy function forward pass."""
        deep_cnn = DeepCNNEnergyFunction(
            hidden_dims=[8, 16],
            input_channels=1,
            rngs=energy_rngs,
        )

        # Create test data
        key = jax.random.key(42)
        test_images = jax.random.normal(key, (4, 16, 16, 1))

        # Forward pass
        energies = deep_cnn(test_images)

        # Check output
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    @jax_required
    def test_deep_cnn_energy_function_options(self, energy_rngs: nnx.Rngs):
        """Test Deep CNN energy function with various options."""
        # Test without residual connections
        deep_cnn_no_res = DeepCNNEnergyFunction(
            hidden_dims=[8, 16],
            input_channels=1,
            use_residual=False,
            use_spectral_norm=False,
            activation=nnx.gelu,
            rngs=energy_rngs,
        )

        # Create test data
        key = jax.random.key(42)
        test_images = jax.random.normal(key, (2, 8, 8, 1))

        energies = deep_cnn_no_res(test_images)
        assert jnp.all(jnp.isfinite(energies))


class TestEnergyBlock:
    """Test the Energy Block implementation."""

    @jax_required
    def test_energy_block_initialization(self, energy_rngs: nnx.Rngs):
        """Test Energy Block initialization."""
        block = EnergyBlock(
            in_channels=16,
            out_channels=32,
            rngs=energy_rngs,
        )

        # Check basic attributes
        assert block.in_channels == 16
        assert block.out_channels == 32
        assert block.kernel_size == 3
        assert block.stride == 1
        assert block.use_residual is False

    @jax_required
    def test_energy_block_forward_pass(self, energy_rngs: nnx.Rngs):
        """Test Energy Block forward pass."""
        block = EnergyBlock(
            in_channels=8,
            out_channels=16,
            rngs=energy_rngs,
        )

        # Create test data
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (4, 16, 16, 8))

        # Forward pass
        output = block(test_input)

        # Check output shape
        expected_shape = (4, 16, 16, 16)  # Same H, W, different C
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    @jax_required
    def test_energy_block_with_residual(self, energy_rngs: nnx.Rngs):
        """Test Energy Block with residual connections."""
        # For residual connections, in_channels == out_channels
        block = EnergyBlock(
            in_channels=16,
            out_channels=16,
            use_residual=True,
            rngs=energy_rngs,
        )

        # Create test data
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (4, 8, 8, 16))

        # Forward pass
        output = block(test_input)

        # Check output shape (should be same as input)
        assert output.shape == test_input.shape
        assert jnp.all(jnp.isfinite(output))

    @jax_required
    def test_energy_block_stride(self, energy_rngs: nnx.Rngs):
        """Test Energy Block with different strides."""
        block = EnergyBlock(
            in_channels=8,
            out_channels=16,
            stride=2,
            rngs=energy_rngs,
        )

        # Create test data
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (4, 16, 16, 8))

        # Forward pass
        output = block(test_input)

        # Check output shape (should be downsampled by stride)
        expected_shape = (4, 8, 8, 16)  # H, W halved due to stride=2
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    @jax_required
    def test_energy_block_different_activations(self, energy_rngs: nnx.Rngs):
        """Test Energy Block with different activations."""
        activations = [nnx.silu, nnx.gelu, nnx.relu]

        # Create test data
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (2, 8, 8, 4))

        for activation in activations:
            block = EnergyBlock(
                in_channels=4,
                out_channels=8,
                activation=activation,
                rngs=energy_rngs,
            )

            output = block(test_input)
            assert jnp.all(jnp.isfinite(output))
            assert output.shape == (2, 8, 8, 8)


class TestFactoryFunctions:
    """Test the factory functions for creating EBMs."""

    @jax_required
    def test_create_mnist_ebm(self, energy_rngs: nnx.Rngs):
        """Test MNIST EBM factory function."""
        ebm = create_mnist_ebm(rngs=energy_rngs)

        # Check that it's an EBM instance
        assert isinstance(ebm, EBM)

        # Test with MNIST-like data (28x28x1)
        key = jax.random.key(42)
        mnist_data = jax.random.normal(key, (4, 28, 28, 1))

        energies = ebm.energy(mnist_data)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    @jax_required
    def test_create_cifar_ebm(self, energy_rngs: nnx.Rngs):
        """Test CIFAR EBM factory function."""
        ebm = create_cifar_ebm(rngs=energy_rngs)

        # Check that it's a DeepEBM instance
        assert isinstance(ebm, DeepEBM)

        # Test with CIFAR-like data (32x32x3)
        key = jax.random.key(42)
        cifar_data = jax.random.normal(key, (4, 32, 32, 3))

        energies = ebm.energy(cifar_data)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    @jax_required
    def test_create_simple_ebm(self, energy_rngs: nnx.Rngs):
        """Test simple EBM factory function."""
        input_dim = 20
        ebm = create_simple_ebm(input_dim=input_dim, rngs=energy_rngs)

        # Check that it's an EBM instance
        assert isinstance(ebm, EBM)

        # Test with vector data
        key = jax.random.key(42)
        vector_data = jax.random.normal(key, (4, input_dim))

        energies = ebm.energy(vector_data)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    @jax_required
    def test_factory_functions_with_kwargs(self, energy_rngs: nnx.Rngs):
        """Test factory functions with additional kwargs."""
        # Test MNIST with custom parameters
        ebm_mnist = create_mnist_ebm(
            rngs=energy_rngs,
            sample_buffer_capacity=256,
        )
        assert isinstance(ebm_mnist, EBM)

        # Test CIFAR with custom parameters
        ebm_cifar = create_cifar_ebm(
            rngs=energy_rngs,
            sample_buffer_capacity=512,
        )
        assert isinstance(ebm_cifar, DeepEBM)

        # Test simple with custom parameters
        ebm_simple = create_simple_ebm(
            input_dim=10,
            rngs=energy_rngs,
            sample_buffer_capacity=128,
        )
        assert isinstance(ebm_simple, EBM)


class TestEBMIntegration:
    """Integration tests for EBM components."""

    @jax_required
    def test_ebm_complete_workflow(
        self, energy_rngs: nnx.Rngs, energy_training_batch: dict[str, jax.Array], mcmc_config
    ):
        """Test complete EBM workflow including training and sampling."""
        input_dim = energy_training_batch["data"].shape[-1]

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
            name="test_ebm_workflow",
            input_dim=input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Training step (uses internal rngs)
        loss_dict = ebm.train_step(energy_training_batch)
        assert jnp.isfinite(loss_dict["loss"])

        # Add samples to buffer (simulate training)
        ebm.sample_buffer.update_buffer(energy_training_batch["data"])

        # Sample from buffer (uses internal rngs)
        samples = ebm.sample_from_buffer(2)
        assert samples.shape == (2, input_dim)

        # Generate new samples (uses internal rngs)
        generated = ebm.generate(
            n_samples=2,
            shape=(input_dim,),
            n_steps=5,
        )
        assert generated.shape == (2, input_dim)

    @jax_required
    def test_deep_ebm_complete_workflow(
        self, energy_rngs: nnx.Rngs, mcmc_config, sample_buffer_config
    ):
        """Test complete Deep EBM workflow."""
        energy_config = EnergyNetworkConfig(
            name="test_cnn_energy",
            hidden_dims=(8, 16),
            activation="silu",
            network_type="cnn",
        )

        config = DeepEBMConfig(
            name="test_deep_ebm_workflow",
            input_shape=(16, 16, 1),
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        deep_ebm = DeepEBM(config=config, rngs=energy_rngs)

        # Create image batch
        key = jax.random.key(42)
        image_batch = {
            "data": jax.random.normal(key, (4, 16, 16, 1)),
            "batch_size": 4,
        }

        # Training step (uses internal rngs)
        loss_dict = deep_ebm.train_step(image_batch)
        assert jnp.isfinite(loss_dict["loss"])

        # Generate samples (uses internal rngs)
        generated = deep_ebm.generate(
            n_samples=2,
            shape=(16, 16, 1),
            n_steps=5,
        )
        assert generated.shape == (2, 16, 16, 1)

    @jax_required
    def test_ebm_energy_consistency(
        self,
        energy_rngs: nnx.Rngs,
        energy_test_mlp_data: jax.Array,
        mcmc_config,
        sample_buffer_config,
    ):
        """Test energy computation consistency across different methods."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_config = EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(16, 8),
            activation="gelu",
            network_type="mlp",
        )

        config = EBMConfig(
            name="test_ebm_mlp",
            input_dim=input_dim,
            energy_network=energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )
        ebm = EBM(config=config, rngs=energy_rngs)

        # Set eval mode for deterministic results
        # This ensures consistent behavior when comparing energy computations
        ebm.energy_fn.eval()

        # Compute energies using different methods
        energies_direct = ebm.energy(energy_test_mlp_data)
        outputs = ebm(energy_test_mlp_data)
        energies_from_outputs = outputs["energy"]

        # Check consistency
        assert jnp.allclose(energies_direct, energies_from_outputs, atol=1e-6)

        # Check score relationship
        scores = outputs["score"]
        assert scores.shape == energy_test_mlp_data.shape

        # Check log prob relationship
        log_probs = outputs["unnormalized_log_prob"]
        assert jnp.allclose(log_probs, -energies_direct, atol=1e-6)
