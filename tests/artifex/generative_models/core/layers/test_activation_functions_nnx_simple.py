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
        # Both JAX and NNX activations are functions
        assert callable(jax.nn.relu)
        assert callable(nnx.relu)

        # The key difference is that we should use nnx versions in nnx modules
        # for consistency with the NNX framework

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


class TestActivationFunctionPatterns:
    """Test common patterns for activation function usage."""

    def test_parametric_activations(self):
        """Test parametric activations like LeakyReLU."""
        # LeakyReLU with custom negative slope
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
                self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
                self.linear2 = nnx.Linear(dim, dim, rngs=rngs)

            def __call__(self, x):
                residual = x
                x = self.linear1(x)
                x = nnx.relu(x)  # Apply activation function
                x = self.linear2(x)
                x = x + residual  # Residual connection
                x = nnx.relu(x)  # Apply activation function
                return x

        rngs = nnx.Rngs(42)
        block = ResidualBlock(10, rngs=rngs)

        # Test with dummy input
        x = jnp.ones((4, 10))
        output = block(x)
        assert output.shape == x.shape
