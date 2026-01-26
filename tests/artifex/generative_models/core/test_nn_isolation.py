"""Minimal isolation tests to debug segmentation faults."""

import jax
import jax.numpy as jnp
import pytest


def test_jax_basic():
    """Test basic JAX functionality without NNX."""
    # Simple JAX operation
    x = jnp.ones((2, 10))
    y = jnp.ones((10, 5))

    # Simple matrix multiplication
    result = jnp.matmul(x, y)

    # Check output shape
    assert result.shape == (2, 5)


def test_jax_random():
    """Test JAX random number generation."""
    # Create a random key
    key = jax.random.key(0)

    # Generate random numbers
    random_nums = jax.random.normal(key, (3, 3))

    # Check shape and finite values
    assert random_nums.shape == (3, 3)
    assert jnp.all(jnp.isfinite(random_nums))


# Try with NNX
try:
    from flax import nnx

    def test_nnx_basic():
        """Test basic NNX functionality with minimal usage."""
        # Create parameters directly instead of using a module
        params = nnx.Param(jnp.ones((10, 5)))

        # Use the parameters in a simple JAX operation
        x = jnp.ones((2, 10))
        result = jnp.matmul(x, params.value)

        # Check output shape
        assert result.shape == (2, 5)

    def test_nnx_module_creation():
        """Test creating an NNX module without calling it."""
        # Create a simple Linear layer without calling it
        linear = nnx.Linear(in_features=10, out_features=5, rngs=nnx.Rngs(0))

        # Check module has parameters
        assert hasattr(linear, "kernel")
        assert hasattr(linear, "bias")
        assert linear.kernel.value.shape == (10, 5)

    def test_nnx_linear_call():
        """Test calling an NNX Linear layer."""
        # Create random number generators
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

        try:
            # Create a simple Linear layer
            linear = nnx.Linear(in_features=10, out_features=5, rngs=rngs)

            # Create input data
            x = jnp.ones((2, 10))

            # Call the layer
            result = linear(x)

            # Check output shape
            assert result.shape == (2, 5)
        except Exception as e:
            # Catch any exception to avoid segmentation fault
            pytest.fail(f"Exception in NNX Linear call: {str(e)}")

    # Test with GenerativeModule
    try:
        from artifex.generative_models.core.base import GenerativeModule

        class SimpleGenerativeModule(GenerativeModule):
            """Simple GenerativeModule for testing."""

            def __init__(
                self,
                *,
                rngs: nnx.Rngs,
            ):
                try:
                    super().__init__(rngs=rngs)
                    self.linear = nnx.Linear(in_features=10, out_features=5, rngs=rngs)
                except Exception as e:
                    pytest.fail(f"Exception in init: {str(e)}")

            def __call__(self, x, *, rngs=None):
                activation = self._get_default_activation()
                return activation(self.linear(x))

        def test_generative_module_init():
            """Test initialization of GenerativeModule."""
            rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

            try:
                # Create the module
                module = SimpleGenerativeModule(rngs=rngs)

                # Check attributes
                assert hasattr(module, "linear")

            except Exception as e:
                pytest.fail(f"Exception in GenerativeModule init: {str(e)}")

        def test_generative_module_call():
            """Test calling GenerativeModule."""
            rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

            try:
                # Create the module
                module = SimpleGenerativeModule(rngs=rngs)

                # Create input data
                x = jnp.ones((2, 10))

                # Call the module with verbose error handling
                try:
                    # Get the activation function
                    activation = module._get_default_activation()
                    print(f"Activation function: {activation}")

                    # Call the linear layer
                    linear_output = module.linear(x)
                    print(f"Linear output shape: {linear_output.shape}")

                    # Apply activation
                    result = activation(linear_output)
                    print(f"Final output shape: {result.shape}")

                    # Check output shape
                    assert result.shape == (2, 5)
                except Exception as e:
                    pytest.fail(f"Exception in GenerativeModule call: {str(e)}")
            except Exception as e:
                pytest.fail(f"Exception in test setup: {str(e)}")
    except ImportError:
        print("GenerativeModule not available")
except ImportError:
    print("NNX not available")
