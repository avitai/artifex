"""Tests for model adapters for benchmarks."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.model_adapters import (
    adapt_model,
    NNXGenerativeModelAdapter,
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
        self.called.set_value(True)
        return jnp.ones((x.shape[0], 5))


class TestNNXGenerativeModelAdapter:
    """Tests for the NNXGenerativeModelAdapter."""

    def setup_method(self):
        """Set up the test environment."""
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

    def test_model_name_from_variable(self):
        """Model name resolves from nnx.Variable."""
        model = MockNNXModel(name="test_model", rngs=self.rngs)
        adapter = NNXGenerativeModelAdapter(model)
        assert adapter.model_name == "test_model"

    def test_model_name_fallback(self):
        """Model name falls back to 'unknown'."""

        class SimpleNNXModel(nnx.Module):
            def __call__(self, x, *, rngs):
                return x

        adapter = NNXGenerativeModelAdapter(SimpleNNXModel())
        assert adapter.model_name == "unknown"

    def test_can_adapt(self):
        """can_adapt correctly identifies NNX models."""
        nnx_model = MockNNXModel(rngs=self.rngs)
        assert NNXGenerativeModelAdapter.can_adapt(nnx_model)

        class NonNNXModel:
            pass

        assert not NNXGenerativeModelAdapter.can_adapt(NonNNXModel())

    def test_predict(self):
        """predict delegates to model.__call__."""
        model = MockNNXModel(rngs=self.rngs)
        adapter = NNXGenerativeModelAdapter(model)
        test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))

        x = jnp.ones((5, 10))
        result = adapter.predict(x, rngs=test_rngs)

        assert model.called.get_value()
        assert result.shape == (5, 5)

    def test_sample(self):
        """sample delegates to model.sample."""

        class SamplingModel(nnx.Module):
            def __init__(self, *, rngs=None):
                self.sample_called = nnx.Variable(False)

            def sample(self, batch_size=1, *, rngs):
                self.sample_called.set_value(True)
                return jnp.ones((batch_size, 10))

        model = SamplingModel(rngs=self.rngs)
        adapter = NNXGenerativeModelAdapter(model)
        test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))

        result = adapter.sample(batch_size=3, rngs=test_rngs)

        assert model.sample_called.get_value()
        assert result.shape == (3, 10)


class TestAdapterRegistry:
    """Tests for the adapter registry functions."""

    def setup_method(self):
        """Set up the test environment."""
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

    def test_register_adapter(self):
        """Registering a custom adapter gives it higher priority."""

        class CustomNNXAdapter(NNXGenerativeModelAdapter):
            @classmethod
            def can_adapt(cls, model):
                return isinstance(model, nnx.Module) and hasattr(model, "custom_attr")

        register_adapter(CustomNNXAdapter)

        class CustomNNXModel(nnx.Module):
            def __init__(self, *, rngs=None):
                self.custom_attr = True

            def __call__(self, x, *, rngs):
                return x

        model = CustomNNXModel(rngs=self.rngs)
        adapter = adapt_model(model)
        assert isinstance(adapter, CustomNNXAdapter)

    def test_adapt_model(self):
        """adapt_model returns NNXGenerativeModelAdapter for NNX models."""
        nnx_adapter = adapt_model(MockNNXModel(rngs=self.rngs))
        assert isinstance(nnx_adapter, NNXGenerativeModelAdapter)

    def test_adapt_non_nnx_raises(self):
        """adapt_model raises ValueError for non-NNX models."""

        class NonNNXModel:
            pass

        with pytest.raises(ValueError):
            adapt_model(NonNNXModel())
