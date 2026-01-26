"""Tests for base extension classes.

This module contains tests for the base extension classes that provide
extensibility to models.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import (
    ConstraintExtension,
    ModelExtension,
)


# Create mock implementations without using the class definition directly
@pytest.fixture
def mock_model_extension(mock_rngs, config):
    """Create a mock model extension for testing."""

    class _TestModelExtension(ModelExtension):
        """Test implementation of ModelExtension."""

        def __call__(self, inputs, model_outputs, **kwargs):
            """Process model inputs/outputs for testing."""
            self.called = True
            return {"test_output": jnp.array(1.0)}

        def loss_fn(self, batch, model_outputs, **kwargs):
            """Calculate test loss."""
            self.loss_called = True
            if not self.enabled:
                return super().loss_fn(batch, model_outputs, **kwargs)
            return jnp.array(self.weight)

    extension = _TestModelExtension(config, rngs=mock_rngs)
    extension.called = False
    extension.loss_called = False
    return extension


@pytest.fixture
def mock_constraint_extension(mock_rngs, config):
    """Create a mock constraint extension for testing."""

    class _TestConstraintExtension(ConstraintExtension):
        """Test implementation of ConstraintExtension."""

        def __call__(self, inputs, model_outputs, **kwargs):
            """Process model inputs/outputs for testing."""
            return {"test_constraint": jnp.array(1.0)}

        def validate(self, outputs):
            """Validate outputs against constraints for testing."""
            self.validate_called = True
            return {"validity": jnp.array(1.0 if self.enabled else 0.0)}

        def project(self, outputs):
            """Project outputs to satisfy constraints for testing."""
            self.project_called = True
            if not self.enabled:
                return super().project(outputs)
            return {"projected": outputs["original"] * self.weight}

    extension = _TestConstraintExtension(config, rngs=mock_rngs)
    extension.validate_called = False
    extension.project_called = False
    return extension


@pytest.fixture
def mock_rngs():
    """Create mock random number generator keys."""
    return nnx.Rngs(0)


@pytest.fixture
def config():
    """Create a basic configuration for testing."""
    return ExtensionConfig(name="test_extension", weight=2.0, enabled=True)


def test_model_extension_init(mock_model_extension, config):
    """Test initializing a ModelExtension."""
    extension = mock_model_extension

    assert extension.weight == 2.0
    assert extension.enabled is True
    assert extension.called is False
    assert extension.loss_called is False


def test_model_extension_call(mock_model_extension):
    """Test calling a ModelExtension."""
    extension = mock_model_extension

    result = extension({"input": 1}, {"output": 2})

    assert extension.called is True
    assert isinstance(result, dict)
    assert "test_output" in result
    assert jnp.allclose(result["test_output"], jnp.array(1.0))


def test_model_extension_loss(mock_model_extension):
    """Test calculating loss with a ModelExtension."""
    extension = mock_model_extension

    loss = extension.loss_fn({"input": 1}, {"output": 2})

    assert extension.loss_called is True
    assert jnp.allclose(loss, jnp.array(2.0))  # Weight is 2.0


def test_model_extension_disabled(mock_rngs):
    """Test disabled ModelExtension."""
    config = ExtensionConfig(name="test_extension", weight=2.0, enabled=False)

    class _TestModelExtension(ModelExtension):
        def __call__(self, inputs, model_outputs, **kwargs):
            self.called = True
            return {"test_output": jnp.array(1.0)}

        def loss_fn(self, batch, model_outputs, **kwargs):
            self.loss_called = True
            if not self.enabled:
                return super().loss_fn(batch, model_outputs, **kwargs)
            return jnp.array(self.weight)

    extension = _TestModelExtension(config, rngs=mock_rngs)
    extension.called = False
    extension.loss_called = False

    loss = extension.loss_fn({"input": 1}, {"output": 2})

    assert extension.loss_called is True
    assert jnp.allclose(loss, jnp.array(0.0))  # Should return 0 when disabled


def test_model_extension_is_enabled(mock_model_extension, mock_rngs):
    """Test is_enabled method of ModelExtension."""
    extension = mock_model_extension
    assert extension.is_enabled() is True

    config = ExtensionConfig(name="test_extension", weight=2.0, enabled=False)

    class _TestModelExtension(ModelExtension):
        pass

    disabled_extension = _TestModelExtension(config, rngs=mock_rngs)
    assert disabled_extension.is_enabled() is False


def test_constraint_extension_init(mock_constraint_extension, config):
    """Test initializing a ConstraintExtension."""
    extension = mock_constraint_extension

    assert extension.weight == 2.0
    assert extension.enabled is True
    assert extension.validate_called is False
    assert extension.project_called is False


def test_constraint_extension_validate(mock_constraint_extension):
    """Test validating with a ConstraintExtension."""
    extension = mock_constraint_extension

    result = extension.validate({"test": 1})

    assert extension.validate_called is True
    assert isinstance(result, dict)
    assert "validity" in result
    assert jnp.allclose(result["validity"], jnp.array(1.0))


def test_constraint_extension_disabled_validate(mock_rngs):
    """Test validating with a disabled ConstraintExtension."""
    config = ExtensionConfig(name="test_constraint", weight=2.0, enabled=False)

    class _TestConstraintExtension(ConstraintExtension):
        def __call__(self, inputs, model_outputs, **kwargs):
            return {"test_constraint": jnp.array(1.0)}

        def validate(self, outputs):
            self.validate_called = True
            return {"validity": jnp.array(1.0 if self.enabled else 0.0)}

        def project(self, outputs):
            self.project_called = True
            if not self.enabled:
                return super().project(outputs)
            return {"projected": outputs["original"] * self.weight}

    extension = _TestConstraintExtension(config, rngs=mock_rngs)
    extension.validate_called = False
    extension.project_called = False

    result = extension.validate({"test": 1})

    assert extension.validate_called is True
    assert jnp.allclose(result["validity"], jnp.array(0.0))


def test_constraint_extension_project(mock_constraint_extension):
    """Test projecting with a ConstraintExtension."""
    extension = mock_constraint_extension

    original = jnp.array(1.0)
    result = extension.project({"original": original})

    assert extension.project_called is True
    assert isinstance(result, dict)
    assert "projected" in result
    assert jnp.allclose(result["projected"], original * 2.0)  # Weight is 2.0


def test_constraint_extension_disabled_project(mock_rngs):
    """Test projecting with a disabled ConstraintExtension."""
    config = ExtensionConfig(name="test_constraint", weight=2.0, enabled=False)

    class _TestConstraintExtension(ConstraintExtension):
        def __call__(self, inputs, model_outputs, **kwargs):
            return {"test_constraint": jnp.array(1.0)}

        def validate(self, outputs):
            self.validate_called = True
            return {"validity": jnp.array(1.0 if self.enabled else 0.0)}

        def project(self, outputs):
            self.project_called = True
            if not self.enabled:
                return super().project(outputs)
            return {"projected": outputs["original"] * self.weight}

    extension = _TestConstraintExtension(config, rngs=mock_rngs)
    extension.validate_called = False
    extension.project_called = False

    original = {"original": jnp.array(1.0)}
    result = extension.project(original)

    assert extension.project_called is True
    assert result is original  # Should return original without modification
