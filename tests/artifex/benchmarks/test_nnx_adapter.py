"""Tests for NNXModelAdapter in benchmarks."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.model_adapters import (
    adapt_model,
    NNXModelAdapter,
)
from tests.utils.test_models import SimpleNNXModel


# SimpleNNXModel is now imported from tests.utils.test_models


class NNXModelWithPredict(nnx.Module):
    """NNX model with a predict method."""

    def __init__(self, features, *, rngs=None):
        """Initialize the model.

        Args:
            features: Number of output features.
            rngs: Optional RNG dict for initialization.
        """
        self.dense = nnx.Linear(in_features=10, out_features=features, rngs=rngs)

    def predict(self, x, *, rngs=None):
        """Make prediction.

        Args:
            x: Input data.
            rngs: Optional RNG dict.

        Returns:
            The model prediction.
        """
        return self.dense(x)

    def generate(self, batch_size=1, *, rngs=None):
        """Generate samples.

        Args:
            batch_size: Number of samples to generate.
            rngs: Optional RNG dict.

        Returns:
            Generated samples.
        """
        # Proper RNG handling following guidelines
        sample_rng = None
        if rngs is not None and hasattr(rngs, "sample"):
            sample_rng = rngs.sample.key.value
        else:
            sample_rng = jax.random.key(0)

        x = jax.random.normal(sample_rng, (batch_size, 10))
        return self.predict(x, rngs=rngs)


class TestNNXModelAdapter:
    """Tests for the NNXModelAdapter."""

    def setup_method(self):
        """Set up for each test."""
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

    def test_can_adapt(self):
        """Test the can_adapt method."""
        # Should be able to adapt an nnx.Module
        model = SimpleNNXModel(features=5, rngs=self.rngs)
        assert NNXModelAdapter.can_adapt(model)

        # Should not be able to adapt a non-nnx model
        class NonNNXModel:
            pass

        non_nnx_model = NonNNXModel()
        assert not NNXModelAdapter.can_adapt(non_nnx_model)

    def test_adapt_model(self):
        """Test adapting an NNX model."""
        model = SimpleNNXModel(features=5, rngs=self.rngs)
        adapter = adapt_model(model)

        assert isinstance(adapter, NNXModelAdapter)

    def test_predict(self):
        """Test the predict method."""
        model = SimpleNNXModel(features=5, rngs=self.rngs)
        adapter = NNXModelAdapter(model)

        # Test without rngs
        x = jnp.ones((3, 10))
        test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))
        output = adapter.predict(x, rngs=test_rngs)
        assert output.shape == (3, 5)

    def test_predict_with_predict_method(self):
        """Test the predict method with a model that has a predict method."""
        model = NNXModelWithPredict(features=5, rngs=self.rngs)
        adapter = NNXModelAdapter(model)

        x = jnp.ones((3, 10))
        test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))
        output = adapter.predict(x, rngs=test_rngs)
        assert output.shape == (3, 5)

    def test_sample(self):
        """Test the sample method."""
        model = SimpleNNXModel(features=5, rngs=self.rngs)
        adapter = NNXModelAdapter(model)

        # Test with rngs
        test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))
        samples = adapter.sample(batch_size=2, rngs=test_rngs)
        assert samples.shape == (2, 5)

        # Test with sample key in rngs
        sample_key = jax.random.PRNGKey(2)
        test_rngs = nnx.Rngs(sample=sample_key)
        samples = adapter.sample(batch_size=4, rngs=test_rngs)
        assert samples.shape == (4, 5)

    def test_sample_with_generate_method(self):
        """Test the sample method with a model that has a generate method."""
        model = NNXModelWithPredict(features=5, rngs=self.rngs)
        adapter = NNXModelAdapter(model)

        test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))
        samples = adapter.sample(batch_size=3, rngs=test_rngs)
        assert samples.shape == (3, 5)

    def test_no_predict_or_call(self):
        """Test error when model has no predict or call method."""

        class EmptyModel(nnx.Module):
            pass

        model = EmptyModel()
        adapter = NNXModelAdapter(model)

        with pytest.raises(ValueError):
            x = jnp.ones((3, 10))
            test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))
            adapter.predict(x, rngs=test_rngs)

    def test_no_sample_or_generate(self):
        """Test error when model has no sample or generate method."""

        class ModelWithoutSample(nnx.Module):
            def __call__(self, x, *, rngs=None):
                return x

        model = ModelWithoutSample()
        adapter = NNXModelAdapter(model)

        with pytest.raises(ValueError):
            test_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(1))
            adapter.sample(batch_size=3, rngs=test_rngs)
