"""Tests for model type registry."""

from typing import Any

import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.factory.registry import (
    BuilderNotFoundError,
    DuplicateBuilderError,
    ModelBuilder,
    ModelTypeRegistry,
)


class MockBuilder(ModelBuilder):
    """Mock builder for testing."""

    def build(self, config: ModelConfig, *, rngs: nnx.Rngs, **kwargs) -> Any:
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
        assert "vae" in registry.list_builders()

    def test_duplicate_builder_error(self):
        """Test error on duplicate registration."""
        registry = ModelTypeRegistry()
        builder = MockBuilder()

        registry.register("vae", builder)

        with pytest.raises(DuplicateBuilderError) as exc_info:
            registry.register("vae", builder)

        assert "Builder for 'vae' already registered" in str(exc_info.value)

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

        builders = registry.list_builders()
        assert set(builders) == {"vae", "gan", "diffusion"}

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ModelTypeRegistry()

        registry.register("vae", MockBuilder())
        registry.clear()

        assert len(registry.list_builders()) == 0

    @pytest.mark.skip(reason="Builders not yet implemented")
    def test_builder_discovery(self):
        """Test automatic builder discovery from module."""
        registry = ModelTypeRegistry()
        registry.discover_builders("artifex.generative_models.factory.builders")

        # Should find all builders in the builders module
        builders = registry.list_builders()
        assert len(builders) > 0
