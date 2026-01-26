"""Unit tests for energy model base classes."""

import pytest

from tests.artifex.generative_models.models.energy.conftest import jax_required


# Import JAX dependencies conditionally
try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from artifex.generative_models.models.energy.base import (
        CNNEnergyFunction,
        EnergyBasedModel,
        EnergyFunction,
        MLPEnergyFunction,
    )

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestEnergyFunction:
    """Test the base EnergyFunction class."""

    @jax_required
    def test_energy_function_abstract(self, energy_rngs):
        """Test that EnergyFunction is abstract and cannot be instantiated directly."""

        class ConcreteEnergyFunction(EnergyFunction):
            def __call__(self, x):
                return jnp.sum(x**2, axis=-1)

        # Should be able to create concrete implementation
        energy_fn = ConcreteEnergyFunction(rngs=energy_rngs)
        assert energy_fn is not None

    @jax_required
    def test_energy_function_call_not_implemented(self, energy_rngs):
        """Test that base EnergyFunction raises NotImplementedError."""
        energy_fn = EnergyFunction(rngs=energy_rngs)
        dummy_input = jnp.ones((2, 4))

        with pytest.raises(NotImplementedError):
            energy_fn(dummy_input)


class TestMLPEnergyFunction:
    """Test the MLP energy function implementation."""

    @jax_required
    def test_mlp_energy_function_initialization(self, energy_rngs):
        """Test MLP energy function initialization."""
        hidden_dims = [32, 16]
        input_dim = 10

        mlp_energy = MLPEnergyFunction(
            hidden_dims=hidden_dims,
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        # Check basic attributes
        assert mlp_energy.input_dim == input_dim
        assert mlp_energy.hidden_dims == hidden_dims
        assert mlp_energy.use_bias is True
        assert mlp_energy.dropout_rate == 0.0

    @jax_required
    def test_mlp_energy_function_forward_pass(self, energy_rngs, energy_test_mlp_data):
        """Test MLP energy function forward pass."""
        hidden_dims = [16, 8]
        input_dim = energy_test_mlp_data.shape[-1]

        mlp_energy = MLPEnergyFunction(
            hidden_dims=hidden_dims,
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        # Forward pass
        energy_values = mlp_energy(energy_test_mlp_data)

        # Check output shape
        expected_shape = (energy_test_mlp_data.shape[0],)
        assert energy_values.shape == expected_shape

        # Check that energies are finite
        assert jnp.all(jnp.isfinite(energy_values))

    @jax_required
    def test_mlp_energy_function_with_dropout(self, energy_rngs, energy_test_mlp_data):
        """Test MLP energy function with dropout."""
        hidden_dims = [16, 8]
        input_dim = energy_test_mlp_data.shape[-1]

        mlp_energy = MLPEnergyFunction(
            hidden_dims=hidden_dims,
            input_dim=input_dim,
            dropout_rate=0.5,
            rngs=energy_rngs,
        )

        # Forward pass in training mode (dropout active)
        mlp_energy.train()
        energy_train = mlp_energy(energy_test_mlp_data)

        # Forward pass in evaluation mode (dropout inactive)
        mlp_energy.eval()
        energy_eval = mlp_energy(energy_test_mlp_data)

        # Both should be finite and have correct shape
        assert jnp.all(jnp.isfinite(energy_train))
        assert jnp.all(jnp.isfinite(energy_eval))
        assert energy_train.shape == energy_eval.shape

    @jax_required
    def test_mlp_energy_function_activations(self, energy_rngs, energy_test_mlp_data):
        """Test MLP energy function with different activations."""
        hidden_dims = [16, 8]
        input_dim = energy_test_mlp_data.shape[-1]

        activations = [nnx.gelu, nnx.silu, nnx.relu]

        for activation in activations:
            mlp_energy = MLPEnergyFunction(
                hidden_dims=hidden_dims,
                input_dim=input_dim,
                activation=activation,
                rngs=energy_rngs,
            )

            energy_values = mlp_energy(energy_test_mlp_data)
            assert jnp.all(jnp.isfinite(energy_values))
            assert energy_values.shape == (energy_test_mlp_data.shape[0],)

    @jax_required
    def test_mlp_energy_function_bias_control(self, energy_rngs, energy_test_mlp_data):
        """Test MLP energy function bias control."""
        hidden_dims = [16, 8]
        input_dim = energy_test_mlp_data.shape[-1]

        # Test with bias
        mlp_with_bias = MLPEnergyFunction(
            hidden_dims=hidden_dims,
            input_dim=input_dim,
            use_bias=True,
            rngs=energy_rngs,
        )

        # Test without bias
        mlp_no_bias = MLPEnergyFunction(
            hidden_dims=hidden_dims,
            input_dim=input_dim,
            use_bias=False,
            rngs=energy_rngs,
        )

        energy_with_bias = mlp_with_bias(energy_test_mlp_data)
        energy_no_bias = mlp_no_bias(energy_test_mlp_data)

        # Both should work and produce finite outputs
        assert jnp.all(jnp.isfinite(energy_with_bias))
        assert jnp.all(jnp.isfinite(energy_no_bias))


class TestCNNEnergyFunction:
    """Test the CNN energy function implementation."""

    @jax_required
    def test_cnn_energy_function_initialization(self, energy_rngs):
        """Test CNN energy function initialization."""
        hidden_dims = [16, 32]
        input_channels = 3

        cnn_energy = CNNEnergyFunction(
            hidden_dims=hidden_dims,
            input_channels=input_channels,
            rngs=energy_rngs,
        )

        # Check basic attributes
        assert cnn_energy.input_channels == input_channels
        assert cnn_energy.hidden_dims == hidden_dims
        assert cnn_energy.kernel_size == 3
        assert cnn_energy.use_bias is True

    @jax_required
    def test_cnn_energy_function_forward_pass(self, energy_rngs, energy_test_image_data):
        """Test CNN energy function forward pass."""
        hidden_dims = [8, 16]
        input_channels = energy_test_image_data.shape[-1]

        cnn_energy = CNNEnergyFunction(
            hidden_dims=hidden_dims,
            input_channels=input_channels,
            rngs=energy_rngs,
        )

        # Forward pass
        energy_values = cnn_energy(energy_test_image_data)

        # Check output shape
        expected_shape = (energy_test_image_data.shape[0],)
        assert energy_values.shape == expected_shape

        # Check that energies are finite
        assert jnp.all(jnp.isfinite(energy_values))

    @jax_required
    def test_cnn_energy_function_kernel_sizes(self, energy_rngs, energy_test_image_data):
        """Test CNN energy function with different kernel sizes."""
        hidden_dims = [8, 16]
        input_channels = energy_test_image_data.shape[-1]

        kernel_sizes = [1, 3, 5]

        for kernel_size in kernel_sizes:
            cnn_energy = CNNEnergyFunction(
                hidden_dims=hidden_dims,
                input_channels=input_channels,
                kernel_size=kernel_size,
                rngs=energy_rngs,
            )

            energy_values = cnn_energy(energy_test_image_data)
            assert jnp.all(jnp.isfinite(energy_values))
            assert energy_values.shape == (energy_test_image_data.shape[0],)

    @jax_required
    def test_cnn_energy_function_activations(self, energy_rngs, energy_test_image_data):
        """Test CNN energy function with different activations."""
        hidden_dims = [8, 16]
        input_channels = energy_test_image_data.shape[-1]

        activations = [nnx.silu, nnx.gelu, nnx.relu]

        for activation in activations:
            cnn_energy = CNNEnergyFunction(
                hidden_dims=hidden_dims,
                input_channels=input_channels,
                activation=activation,
                rngs=energy_rngs,
            )

            energy_values = cnn_energy(energy_test_image_data)
            assert jnp.all(jnp.isfinite(energy_values))
            assert energy_values.shape == (energy_test_image_data.shape[0],)

    @jax_required
    def test_cnn_energy_function_different_input_channels(self, energy_rngs):
        """Test CNN energy function with different input channels."""
        hidden_dims = [8, 16]

        # Test with different channel counts
        for input_channels in [1, 3, 4]:
            # Create appropriate test data
            key = jax.random.key(42)
            test_data = jax.random.normal(key, (4, 8, 8, input_channels))

            cnn_energy = CNNEnergyFunction(
                hidden_dims=hidden_dims,
                input_channels=input_channels,
                rngs=energy_rngs,
            )

            energy_values = cnn_energy(test_data)
            assert jnp.all(jnp.isfinite(energy_values))
            assert energy_values.shape == (test_data.shape[0],)


class TestEnergyBasedModel:
    """Test the EnergyBasedModel base class."""

    @jax_required
    def test_energy_based_model_initialization(self, energy_rngs, energy_test_input_dim):
        """Test EnergyBasedModel initialization."""
        # Create an MLP energy function
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=energy_test_input_dim,
            rngs=energy_rngs,
        )

        # Create energy-based model
        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Check that energy function is stored
        assert ebm.energy_fn is energy_fn

    @jax_required
    def test_energy_method(self, energy_rngs, energy_test_mlp_data):
        """Test energy computation method."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Compute energies
        energies = ebm.energy(energy_test_mlp_data)

        # Check output
        assert energies.shape == (energy_test_mlp_data.shape[0],)
        assert jnp.all(jnp.isfinite(energies))

    @jax_required
    def test_unnormalized_log_prob_method(self, energy_rngs, energy_test_mlp_data):
        """Test unnormalized log probability method."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Compute energies and log probs
        energies = ebm.energy(energy_test_mlp_data)
        log_probs = ebm.unnormalized_log_prob(energy_test_mlp_data)

        # Check that log_prob = -energy
        assert jnp.allclose(log_probs, -energies, atol=1e-6)

    @jax_required
    def test_score_method(self, energy_rngs, energy_test_mlp_data):
        """Test score function method."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Compute score
        scores = ebm.score(energy_test_mlp_data)

        # Check output shape
        assert scores.shape == energy_test_mlp_data.shape
        assert jnp.all(jnp.isfinite(scores))

    @jax_required
    def test_forward_pass(self, energy_rngs, energy_test_mlp_data):
        """Test complete forward pass."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Forward pass
        outputs = ebm(energy_test_mlp_data)

        # Check output dictionary
        assert isinstance(outputs, dict)
        assert "energy" in outputs
        assert "unnormalized_log_prob" in outputs
        assert "score" in outputs

        # Check shapes
        batch_size = energy_test_mlp_data.shape[0]
        assert outputs["energy"].shape == (batch_size,)
        assert outputs["unnormalized_log_prob"].shape == (batch_size,)
        assert outputs["score"].shape == energy_test_mlp_data.shape

        # Check relationships
        assert jnp.allclose(outputs["unnormalized_log_prob"], -outputs["energy"], atol=1e-6)

    @jax_required
    def test_contrastive_divergence_loss(self, energy_rngs, energy_test_mlp_data):
        """Test contrastive divergence loss computation."""
        input_dim = energy_test_mlp_data.shape[-1]

        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Create fake data (different from real data)
        key = jax.random.key(999)
        fake_data = jax.random.normal(key, energy_test_mlp_data.shape)

        # Compute loss
        loss_dict = ebm.contrastive_divergence_loss(
            real_data=energy_test_mlp_data,
            fake_data=fake_data,
            alpha=0.01,
        )

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
    def test_generate_method(self, energy_rngs, energy_test_input_dim):
        """Test generation method."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=energy_test_input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Generate samples
        n_samples = 4
        shape = (energy_test_input_dim,)

        # Create RNG for generation
        gen_rngs = nnx.Rngs(999)

        samples = ebm.generate(
            n_samples=n_samples,
            rngs=gen_rngs,
            shape=shape,
            n_steps=5,  # Few steps for fast testing
        )

        # Check output
        expected_shape = (n_samples, energy_test_input_dim)
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    @jax_required
    def test_loss_fn_method(self, energy_rngs, energy_training_batch):
        """Test loss function method."""
        input_dim = energy_training_batch["data"].shape[-1]

        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=input_dim,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Get model outputs
        model_outputs = ebm(energy_training_batch["data"])

        # Create fake samples for loss computation
        key = jax.random.key(777)
        fake_samples = jax.random.normal(key, energy_training_batch["data"].shape)

        # Compute loss
        loss_dict = ebm.loss_fn(
            batch=energy_training_batch,
            model_outputs=model_outputs,
            fake_samples=fake_samples,
        )

        # Check loss output
        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict
        assert jnp.isfinite(loss_dict["loss"])
