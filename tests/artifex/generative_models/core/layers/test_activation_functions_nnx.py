"""Test for replacing JAX activation functions with NNX ones.

This test ensures we use NNX activation functions instead of JAX ones
in NNX modules.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class TestActivationFunctionCompliance:
    """Test that activation functions use NNX instead of JAX."""

    def test_nnx_activation_functions_available(self):
        """Test that NNX provides activation function modules."""
        # Check that NNX has common activation functions
        assert hasattr(nnx, "relu")
        assert hasattr(nnx, "gelu")
        assert hasattr(nnx, "sigmoid")
        assert hasattr(nnx, "tanh")
        assert hasattr(nnx, "softmax")
        assert hasattr(nnx, "leaky_relu")
        assert hasattr(nnx, "elu")
        assert hasattr(nnx, "silu")

    def test_jax_vs_nnx_activation_difference(self):
        """Test the difference between JAX and NNX activations."""
        # JAX activations are functions
        assert callable(jax.nn.relu)

        # NNX activations are also functions (per critical guidelines)
        assert callable(nnx.relu)
        assert not isinstance(nnx.relu, type)

    def test_using_nnx_activation_in_module(self):
        """Test proper usage of NNX activation in a module."""

        class ProperNNXModule(nnx.Module):
            """Module using NNX activation functions properly."""

            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.dense1 = nnx.Linear(10, 20, rngs=rngs)
                self.dense2 = nnx.Linear(20, 10, rngs=rngs)

            def __call__(self, x):
                x = self.dense1(x)
                x = nnx.relu(x)  # NNX activation function
                x = self.dense2(x)
                x = nnx.gelu(x)  # Another NNX activation
                return x

        # Test the module
        rngs = nnx.Rngs(42)
        module = ProperNNXModule(rngs=rngs)
        x = jnp.ones((4, 10))
        output = module(x)
        assert output.shape == (4, 10)

    def test_improper_jax_activation_usage(self):
        """Test what NOT to do - using JAX activations in NNX modules."""

        class ImproperModule(nnx.Module):
            """Module incorrectly using JAX activation functions."""

            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.dense1 = nnx.Linear(10, 20, rngs=rngs)
                # This is WRONG - storing JAX function directly
                self.activation = jax.nn.relu
                self.dense2 = nnx.Linear(20, 10, rngs=rngs)

            def __call__(self, x):
                x = self.dense1(x)
                x = self.activation(x)  # Works but not proper NNX pattern
                x = self.dense2(x)
                return x

        # This will work but is not the correct pattern
        rngs = nnx.Rngs(42)
        module = ImproperModule(rngs=rngs)
        x = jnp.ones((4, 10))
        output = module(x)
        assert output.shape == (4, 10)

        # Check that activation is not an NNX module
        assert not isinstance(module.activation, nnx.Module)

    def test_sequential_with_nnx_activations(self):
        """Test using NNX activations in Sequential."""

        # Proper way with NNX - Sequential accepts functions directly
        rngs = nnx.Rngs(42)
        model = nnx.Sequential(
            nnx.Linear(10, 20, rngs=rngs),
            nnx.relu,  # NNX activation function
            nnx.Linear(20, 30, rngs=rngs),
            nnx.gelu,  # NNX activation function
            nnx.Linear(30, 10, rngs=rngs),
        )

        x = jnp.ones((4, 10))
        output = model(x)
        assert output.shape == (4, 10)

        # Check that layers are accessible
        assert len(model.layers) == 5
        assert isinstance(model.layers[0], nnx.Linear)
        assert callable(model.layers[1])  # relu function
        assert isinstance(model.layers[2], nnx.Linear)
        assert callable(model.layers[3])  # gelu function
        assert isinstance(model.layers[4], nnx.Linear)

    def test_list_based_module_with_activations(self):
        """Test pattern often seen in geometric models."""

        class ListBasedModule(nnx.Module):
            """Module that builds layers in a list."""

            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                # Use nnx.List for Flax NNX 0.12.0+ compatibility
                layers = []
                # Build layers (only Linear layers, activations applied in __call__)
                for i in range(3):
                    layers.append(nnx.Linear(10, 10, rngs=rngs))
                self.layers = nnx.List(layers)

            def __call__(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    # Apply activation after all but last layer
                    if i < len(self.layers) - 1:
                        x = nnx.relu(x)  # Apply activation function
                return x

        rngs = nnx.Rngs(42)
        module = ListBasedModule(rngs=rngs)
        x = jnp.ones((4, 10))
        output = module(x)
        assert output.shape == (4, 10)

        # Verify we only have Linear layers (no activation modules)
        assert len(module.layers) == 3
        for layer in module.layers:
            assert isinstance(layer, nnx.Linear)


class TestActivationFunctionPatterns:
    """Test common patterns for activation function usage."""

    def test_parametric_activations(self):
        """Test parametric activations like LeakyReLU."""
        # NNX parametric activations are functions with parameters
        x = jnp.array([-1.0, 0.0, 1.0])
        output = nnx.leaky_relu(x, negative_slope=0.2)

        # Check negative values are scaled
        assert output[0] == -0.2  # -1.0 * 0.2
        assert output[1] == 0.0
        assert output[2] == 1.0

    def test_activation_in_residual_block(self):
        """Test activation usage in residual connections."""

        class ResidualBlock(nnx.Module):
            """Residual block with NNX activations."""

            def __init__(self, dim: int, *, rngs: nnx.Rngs):
                super().__init__()
                self.conv1 = nnx.Conv(dim, dim, kernel_size=3, rngs=rngs)
                self.conv2 = nnx.Conv(dim, dim, kernel_size=3, rngs=rngs)

            def __call__(self, x):
                residual = x
                x = self.conv1(x)
                x = nnx.relu(x)  # Apply activation function
                x = self.conv2(x)
                x = x + residual  # Residual connection
                x = nnx.relu(x)  # Apply activation function
                return x

        rngs = nnx.Rngs(42)
        block = ResidualBlock(64, rngs=rngs)

        # Test with dummy input
        x = jnp.ones((1, 32, 32, 64))
        output = block(x)
        assert output.shape == x.shape
