"""Tests for calibrax-backed model adapters."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from calibrax.core import NNXBenchmarkAdapter

from artifex.benchmarks.model_adapters import (
    adapt_model,
    NNXGenerativeModelAdapter,
    register_adapter,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class _CallableModel(nnx.Module):
    """NNX model with __call__."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(in_features=10, out_features=5, rngs=rngs)

    def __call__(self, x, *, rngs=None):
        return self.dense(x)

    def sample(self, batch_size=1, *, rngs=None):
        key = rngs.sample() if rngs is not None else jax.random.key(0)
        x = jax.random.normal(key, (batch_size, 10))
        return self.dense(x)


class _PredictModel(nnx.Module):
    """NNX model with predict method (no __call__)."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(in_features=10, out_features=5, rngs=rngs)

    def predict(self, x, *, rngs=None):
        return self.dense(x)

    def generate(self, batch_size=1, *, rngs=None):
        key = rngs.sample() if rngs is not None else jax.random.key(0)
        x = jax.random.normal(key, (batch_size, 10))
        return self.dense(x)


class _ModelWithName(nnx.Module):
    """NNX model with model_name as nnx.Variable."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.model_name = nnx.Variable("my_model")

    def __call__(self, x, *, rngs=None):
        return x

    def sample(self, batch_size=1, *, rngs=None):
        return jnp.ones((batch_size, 10))


@pytest.fixture
def rngs():
    return nnx.Rngs(42)


# ---------------------------------------------------------------------------
# NNXGenerativeModelAdapter extends calibrax NNXBenchmarkAdapter
# ---------------------------------------------------------------------------


class TestNNXGenerativeModelAdapter:
    """Verify NNXGenerativeModelAdapter extends calibrax NNXBenchmarkAdapter."""

    def test_extends_calibrax_adapter(self) -> None:
        """NNXGenerativeModelAdapter is a subclass of NNXBenchmarkAdapter."""
        assert issubclass(NNXGenerativeModelAdapter, NNXBenchmarkAdapter)

    def test_is_nnx_module(self) -> None:
        """NNXGenerativeModelAdapter is an nnx.Module."""
        assert issubclass(NNXGenerativeModelAdapter, nnx.Module)

    def test_can_adapt_nnx_model(self, rngs) -> None:
        """can_adapt returns True for nnx.Module instances."""
        model = _CallableModel(rngs=rngs)
        assert NNXGenerativeModelAdapter.can_adapt(model)

    def test_cannot_adapt_non_nnx(self) -> None:
        """can_adapt returns False for non-nnx objects."""

        class Plain:
            pass

        assert not NNXGenerativeModelAdapter.can_adapt(Plain())

    def test_predict_via_call(self, rngs) -> None:
        """predict delegates to model.__call__."""
        model = _CallableModel(rngs=rngs)
        adapter = NNXGenerativeModelAdapter(model)
        x = jnp.ones((3, 10))
        result = adapter.predict(x, rngs=rngs)
        assert result.shape == (3, 5)

    def test_predict_via_predict_method(self, rngs) -> None:
        """predict delegates to model.predict when no __call__."""
        model = _PredictModel(rngs=rngs)
        adapter = NNXGenerativeModelAdapter(model)
        x = jnp.ones((3, 10))
        result = adapter.predict(x, rngs=rngs)
        assert result.shape == (3, 5)

    def test_sample_via_sample_method(self, rngs) -> None:
        """sample delegates to model.sample."""
        model = _CallableModel(rngs=rngs)
        adapter = NNXGenerativeModelAdapter(model)
        result = adapter.sample(batch_size=4, rngs=rngs)
        assert result.shape == (4, 5)

    def test_sample_via_generate_method(self, rngs) -> None:
        """sample delegates to model.generate."""
        model = _PredictModel(rngs=rngs)
        adapter = NNXGenerativeModelAdapter(model)
        result = adapter.sample(batch_size=2, rngs=rngs)
        assert result.shape == (2, 5)

    def test_predict_no_method_raises(self, rngs) -> None:
        """predict raises ValueError when model has no predict or __call__."""

        class EmptyModel(nnx.Module):
            pass

        adapter = NNXGenerativeModelAdapter(EmptyModel())
        with pytest.raises(ValueError, match="no predict or __call__"):
            adapter.predict(jnp.ones((1, 10)), rngs=rngs)

    def test_sample_no_method_raises(self, rngs) -> None:
        """sample raises ValueError when model has no sample or generate."""

        class NoSampleModel(nnx.Module):
            def __call__(self, x, *, rngs=None):
                return x

        adapter = NNXGenerativeModelAdapter(NoSampleModel())
        with pytest.raises(ValueError, match="no sample or generate"):
            adapter.sample(batch_size=1, rngs=rngs)

    def test_model_name_from_variable(self, rngs) -> None:
        """Name resolves through nnx.Variable unwrapping."""
        model = _ModelWithName(rngs=rngs)
        adapter = NNXGenerativeModelAdapter(model)
        assert adapter.model_name == "my_model"

    def test_model_name_fallback(self, rngs) -> None:
        """Name falls back to 'unknown' when model has no name."""
        model = _CallableModel(rngs=rngs)
        adapter = NNXGenerativeModelAdapter(model)
        assert adapter.model_name == "unknown"


# ---------------------------------------------------------------------------
# adapt_model / register_adapter backed by calibrax AdapterRegistry
# ---------------------------------------------------------------------------


class TestAdaptModelCalibrax:
    """Verify adapt_model and register_adapter use calibrax AdapterRegistry."""

    def test_adapt_nnx_model(self, rngs) -> None:
        """adapt_model returns NNXGenerativeModelAdapter for nnx.Module."""
        model = _CallableModel(rngs=rngs)
        adapter = adapt_model(model)
        assert isinstance(adapter, NNXGenerativeModelAdapter)

    def test_adapt_non_nnx_raises(self) -> None:
        """adapt_model raises ValueError for non-NNX models."""

        class Plain:
            pass

        with pytest.raises(ValueError):
            adapt_model(Plain())

    def test_register_custom_adapter(self, rngs) -> None:
        """register_adapter adds a custom adapter with higher priority."""

        class SpecialModel(nnx.Module):
            def __init__(self) -> None:
                self.special = True

            def __call__(self, x, *, rngs=None):
                return x

        class SpecialAdapter(NNXGenerativeModelAdapter):
            @classmethod
            def can_adapt(cls, target) -> bool:
                return isinstance(target, SpecialModel)

        register_adapter(SpecialAdapter)
        adapter = adapt_model(SpecialModel())
        assert isinstance(adapter, SpecialAdapter)
