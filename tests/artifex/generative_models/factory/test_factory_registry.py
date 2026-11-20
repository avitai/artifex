"""Tests for model type registry."""

from typing import Any

import pytest
from calibrax.core import DuplicateRegistrationError, Registry
from flax import nnx

from artifex.generative_models.factory.registry import (
    BuilderNotFoundError,
    ModelBuilder,
    ModelTypeRegistry,
)


class MockBuilder(ModelBuilder):
    """Mock builder for testing."""

    def build(self, config: Any, *, rngs: nnx.Rngs, **kwargs) -> Any:
        """Build a mock model."""
        return {"type": "mock", "config": config, "kwargs": kwargs}


class TestModelTypeRegistry:
    """Test model type registry functionality."""

    def test_register_builder(self):
        """Test registering a model builder."""
        registry = ModelTypeRegistry()
        builder = MockBuilder()

        registry.register("vae", builder)

        assert registry.get_builder("vae") == builder
        assert "vae" in registry.list_names()

    def test_registry_is_the_calibrax_registry(self):
        """The duplicate refusal is calibrax's, written once for the stack."""
        assert issubclass(ModelTypeRegistry, Registry)

    def test_duplicate_registration_is_refused_and_keeps_the_first_builder(self):
        """A taken model type is refused with calibrax's type; the first builder stays."""
        registry = ModelTypeRegistry()
        first = MockBuilder()
        second = MockBuilder()

        registry.register("vae", first)

        with pytest.raises(DuplicateRegistrationError) as exc_info:
            registry.register("vae", second)

        assert str(exc_info.value) == (
            "'vae' is already registered in ModelTypeRegistry; remove it first to replace it."
        )
        assert registry.get_builder("vae") is first

    def test_builder_not_found_error(self):
        """Test error when builder not found."""
        registry = ModelTypeRegistry()

        with pytest.raises(BuilderNotFoundError) as exc_info:
            registry.get_builder("nonexistent")

        assert "No builder registered for type 'nonexistent'" in str(exc_info.value)

    def test_list_builders(self):
        """Test listing all registered builders."""
        registry = ModelTypeRegistry()

        registry.register("vae", MockBuilder())
        registry.register("gan", MockBuilder())
        registry.register("diffusion", MockBuilder())

        builders = registry.list_names()
        assert set(builders) == {"vae", "gan", "diffusion"}

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ModelTypeRegistry()

        registry.register("vae", MockBuilder())
        registry.clear()

        assert len(registry.list_names()) == 0

    def test_registry_has_no_discovery_api(self):
        """The registry should stay explicit rather than auto-discovering builders."""
        registry = ModelTypeRegistry()
        assert not hasattr(registry, "discover_builders")
