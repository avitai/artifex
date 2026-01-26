"""Tests for model adapters for benchmarks."""

from typing import Protocol, runtime_checkable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.model_adapters import (
    adapt_model,
    BenchmarkModelAdapter,
    NNXModelAdapter,
    register_adapter,
)


class MockNNXModel(nnx.Module):
    """Mock NNX model for testing."""

    def __init__(self, name="mock_model", *, rngs=None):
        """Initialize the model.

        Args:
            name: Model name
            rngs: RNG dict for NNX compatibility
        """
        self.model_name = nnx.Variable(name)
        self.weights = nnx.Param(jnp.ones((10, 10)))
        self.called = nnx.Variable(False)

    def __call__(self, x, *, rngs):
        """Mock call method.

        Args:
            x: Input data
            rngs: Required RNG dict for NNX compatibility

        Returns:
            Model output
        """
        self.called.value = True
        return jnp.ones((x.shape[0], 5))


# Create concrete adapter without using a constructor that interferes with pytest
class CustomTestAdapter(BenchmarkModelAdapter):
    """Concrete adapter for testing abstract methods."""

    def __init__(self, model):
        """Initialize the adapter with a model."""
        super().__init__(model)

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Mock predict method."""
        return jnp.ones((x.shape[0], 5))

    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Mock sample method."""
        return jnp.ones((batch_size, 10))


class TestBenchmarkModelAdapter:
    """Tests for the BenchmarkModelAdapter base class."""

    def setup_method(self):
        """Set up the test environment."""
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        # Test that the abstract class can't be instantiated directly
        with pytest.raises(TypeError):
            BenchmarkModelAdapter(MockNNXModel(rngs=self.rngs))

        # Test that a concrete implementation works
        adapter = CustomTestAdapter(MockNNXModel(rngs=self.rngs))
        assert isinstance(adapter, BenchmarkModelAdapter)

    def test_model_name(self):
        """Test that the model name is set correctly."""
        # Test with a model that has a model_name attribute
        model = MockNNXModel(name="test_model", rngs=self.rngs)
        adapter = CustomTestAdapter(model)
        assert adapter.model_name == model.model_name.value

        # Test with a model that doesn't have a model_name attribute
        class SimpleNNXModel(nnx.Module):
            def __call__(self, x, *, rngs):
                return x

        adapter = CustomTestAdapter(SimpleNNXModel())
        assert adapter.model_name == "unknown"


class TestNNXModelAdapter:
    """Tests for the NNXModelAdapter."""

    def setup_method(self):
        """Set up test environment."""
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)
        self.test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))

    def test_can_adapt(self):
        """Test that can_adapt method correctly identifies NNX models."""
        # NNX model
        nnx_model = MockNNXModel(rngs=self.rngs)
        assert NNXModelAdapter.can_adapt(nnx_model)

        # Non-NNX model
        class NonNNXModel:
            pass

        non_nnx = NonNNXModel()
        assert not NNXModelAdapter.can_adapt(non_nnx)

    def test_predict(self):
        """Test that predict method calls the model's __call__ method."""
        model = MockNNXModel(rngs=self.rngs)
        adapter = NNXModelAdapter(model)

        x = jnp.ones((5, 10))
        result = adapter.predict(x, rngs=self.test_rngs)

        assert model.called.value
        assert result.shape == (5, 5)

    def test_sample(self):
        """Test that sample method handles rngs correctly."""

        # Create a model with sample method
        class SamplingModel(nnx.Module):
            def __init__(self, *, rngs=None):
                self.sample_called = nnx.Variable(False)

            def sample(self, batch_size=1, *, rngs):
                self.sample_called.value = True
                return jnp.ones((batch_size, 10))

        model = SamplingModel(rngs=self.rngs)
        adapter = NNXModelAdapter(model)

        result = adapter.sample(batch_size=3, rngs=self.test_rngs)

        assert model.sample_called.value
        assert result.shape == (3, 10)


@runtime_checkable
class RuntimeModelProtocol(Protocol):
    """Runtime checkable protocol for model tests."""

    def predict(self, x, *, rngs): ...
    def sample(self, rng_key, batch_size=1, *, rngs): ...


class TestAdapterRegistry:
    """Tests for the adapter registry functions."""

    def setup_method(self):
        """Set up the test environment."""
        # Create RNGs for model initialization
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

        # Patch the model protocol to be runtime checkable
        import artifex.benchmarks

        self.original_protocol = artifex.benchmarks.ModelProtocol
        artifex.benchmarks.ModelProtocol = RuntimeModelProtocol

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original protocol
        import artifex.benchmarks

        artifex.benchmarks.ModelProtocol = self.original_protocol

    def test_register_adapter(self):
        """Test registering an adapter."""

        # Create a custom adapter for NNX models
        class CustomNNXAdapter(BenchmarkModelAdapter):
            @classmethod
            def can_adapt(cls, model):
                return isinstance(model, nnx.Module) and hasattr(model, "custom_attr")

            def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
                return jnp.ones((x.shape[0], 1))

            def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
                return jnp.ones((batch_size, 1))

        # Register the adapter
        register_adapter(CustomNNXAdapter)

        # Create a model that can be adapted by this adapter
        class CustomNNXModel(nnx.Module):
            def __init__(self, *, rngs=None):
                self.custom_attr = True

            def __call__(self, x, *, rngs):
                return x

        # Test the adapter
        model = CustomNNXModel(rngs=self.rngs)
        adapter = adapt_model(model)
        assert isinstance(adapter, CustomNNXAdapter)

    def test_adapt_model(self):
        """Test adapting NNX models."""
        # Test adapting an NNX model
        nnx_adapter = adapt_model(MockNNXModel(rngs=self.rngs))
        assert isinstance(nnx_adapter, NNXModelAdapter)

        # Test error for non-NNX model
        class NonNNXModel:
            pass

        with pytest.raises(ValueError, match="Only Flax NNX models are supported"):
            adapt_model(NonNNXModel())
