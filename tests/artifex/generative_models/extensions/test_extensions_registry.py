"""Tests for the extensions registry system."""

import pytest
from flax import nnx

from artifex.generative_models.extensions.base import ModelExtension
from artifex.generative_models.extensions.registry import (
    ExtensionsRegistry,
    ExtensionType,
    get_extensions_registry,
)


class TestExtensionsRegistry:
    """Test the extensions registry functionality."""

    @pytest.fixture
    def rngs(self):
        """Create test rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ExtensionsRegistry()

    def test_registry_initialization(self, registry):
        """Test that registry initializes correctly."""
        # Should have core extensions registered
        extensions = registry.list_all_extensions()
        assert isinstance(extensions, dict)

    def test_get_extensions_for_modality(self, registry):
        """Test getting extensions by modality."""
        # Test molecular modality
        molecular_extensions = registry.get_extensions_for_modality("molecular")
        assert isinstance(molecular_extensions, list)

        # If chemical extensions are available, should include them
        if molecular_extensions:
            assert any("chemical" in ext or "molecular" in ext for ext in molecular_extensions)

    def test_get_extensions_by_capability(self, registry):
        """Test getting extensions by capability."""
        # Test augmentation capability
        augmentation_extensions = registry.get_extensions_by_capability("augmentation")
        assert isinstance(augmentation_extensions, list)

    def test_create_extension(self, registry, rngs):
        """Test creating extension instances."""
        extensions = registry.list_all_extensions()

        if extensions:
            # Try to create the first available extension
            ext_name = list(extensions.keys())[0]
            extension = registry.create_extension(ext_name, rngs=rngs)
            assert isinstance(extension, nnx.Module)

    def test_extension_info(self, registry):
        """Test getting extension information."""
        extensions = registry.list_all_extensions()

        if extensions:
            ext_name = list(extensions.keys())[0]
            info = registry.get_extension_info(ext_name)

            assert "modalities" in info
            assert "capabilities" in info
            assert "description" in info
            assert "registered_at" in info

    def test_validate_extension_compatibility(self, registry):
        """Test extension compatibility validation."""
        extensions = registry.list_all_extensions()

        if extensions:
            ext_names = list(extensions.keys())[:2]  # Test first 2 extensions
            compatibility = registry.validate_extension_compatibility(ext_names, "molecular")

            assert isinstance(compatibility, dict)
            assert len(compatibility) == len(ext_names)

    def test_search_extensions(self, registry):
        """Test extension search functionality."""
        # Search by modality
        results = registry.search_extensions(modality="molecular")
        assert isinstance(results, list)

        # Search by capability
        results = registry.search_extensions(capability="validation")
        assert isinstance(results, list)

    def test_get_available_modalities(self, registry):
        """Test getting available modalities."""
        modalities = registry.get_available_modalities()
        assert isinstance(modalities, list)
        assert all(isinstance(mod, str) for mod in modalities)

    def test_get_available_capabilities(self, registry):
        """Test getting available capabilities."""
        capabilities = registry.get_available_capabilities()
        assert isinstance(capabilities, list)
        assert all(isinstance(cap, str) for cap in capabilities)

    def test_global_registry(self):
        """Test the global registry instance."""
        registry1 = get_extensions_registry()
        registry2 = get_extensions_registry()

        # Should return the same instance
        assert registry1 is registry2
        assert isinstance(registry1, ExtensionsRegistry)

    def test_create_extension_pipeline(self, registry, rngs):
        """Test creating an extension pipeline."""
        extensions = registry.list_all_extensions()

        if extensions:
            # Create a simple pipeline config as list of tuples
            ext_name = list(extensions.keys())[0]
            pipeline_config = [(ext_name, None)]  # None for default config

            pipeline = registry.create_extension_pipeline(pipeline_config, rngs=rngs)

            assert isinstance(pipeline, list)
            assert len(pipeline) == 1
            assert isinstance(pipeline[0], nnx.Module)

    def test_invalid_extension_creation(self, registry, rngs):
        """Test error handling for invalid extension creation."""
        with pytest.raises(ValueError):
            registry.create_extension("nonexistent_extension", rngs=rngs)

    def test_invalid_extension_info(self, registry):
        """Test error handling for invalid extension info request."""
        with pytest.raises(ValueError):
            registry.get_extension_info("nonexistent_extension")


class TestExtensionType:
    """Test ExtensionType enum and type-based filtering."""

    @pytest.fixture
    def rngs(self):
        """Create test rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ExtensionsRegistry()

    def test_extension_type_enum_values(self):
        """Test that ExtensionType enum has expected values."""
        assert hasattr(ExtensionType, "MODEL")
        assert hasattr(ExtensionType, "CONSTRAINT")
        assert hasattr(ExtensionType, "AUGMENTATION")
        assert hasattr(ExtensionType, "SAMPLING")
        assert hasattr(ExtensionType, "LOSS")
        assert hasattr(ExtensionType, "EVALUATION")
        assert hasattr(ExtensionType, "CALLBACK")
        assert hasattr(ExtensionType, "MODALITY")

    def test_extension_type_enum_string_values(self):
        """Test ExtensionType enum string representations."""
        assert ExtensionType.MODEL.value == "model"
        assert ExtensionType.CONSTRAINT.value == "constraint"
        assert ExtensionType.AUGMENTATION.value == "augmentation"
        assert ExtensionType.SAMPLING.value == "sampling"
        assert ExtensionType.LOSS.value == "loss"
        assert ExtensionType.EVALUATION.value == "evaluation"
        assert ExtensionType.CALLBACK.value == "callback"
        assert ExtensionType.MODALITY.value == "modality"

    def test_register_extension_with_type(self, registry, rngs):
        """Test registering an extension with explicit type."""

        class TestModelExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        registry.register_extension(
            name="test_model_ext",
            extension_class=TestModelExtension,
            extension_type=ExtensionType.MODEL,
            modalities=["test"],
            capabilities=["test_capability"],
            description="Test model extension",
        )

        info = registry.get_extension_info("test_model_ext")
        assert info["extension_type"] == "model"

    def test_get_extensions_by_type(self, registry, rngs):
        """Test filtering extensions by type."""

        class TestModelExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        class TestConstraintExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        registry.register_extension(
            name="test_model",
            extension_class=TestModelExtension,
            extension_type=ExtensionType.MODEL,
            modalities=["test"],
            capabilities=["test"],
        )

        registry.register_extension(
            name="test_constraint",
            extension_class=TestConstraintExtension,
            extension_type=ExtensionType.CONSTRAINT,
            modalities=["test"],
            capabilities=["test"],
        )

        model_extensions = registry.get_extensions_by_type(ExtensionType.MODEL)
        constraint_extensions = registry.get_extensions_by_type(ExtensionType.CONSTRAINT)

        assert "test_model" in model_extensions
        assert "test_constraint" not in model_extensions
        assert "test_constraint" in constraint_extensions
        assert "test_model" not in constraint_extensions

    def test_search_extensions_with_type(self, registry, rngs):
        """Test searching extensions with type filter."""

        class TestAugExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        registry.register_extension(
            name="test_aug",
            extension_class=TestAugExtension,
            extension_type=ExtensionType.AUGMENTATION,
            modalities=["image"],
            capabilities=["augmentation"],
        )

        results = registry.search_extensions(
            modality="image",
            extension_type=ExtensionType.AUGMENTATION,
        )

        assert "test_aug" in results

    def test_get_available_extension_types(self, registry, rngs):
        """Test getting all available extension types."""

        class TestExt(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        registry.register_extension(
            name="test_ext1",
            extension_class=TestExt,
            extension_type=ExtensionType.MODEL,
            modalities=["test"],
            capabilities=["test"],
        )

        registry.register_extension(
            name="test_ext2",
            extension_class=TestExt,
            extension_type=ExtensionType.SAMPLING,
            modalities=["test"],
            capabilities=["test"],
        )

        types = registry.get_available_extension_types()

        assert isinstance(types, list)
        assert ExtensionType.MODEL in types
        assert ExtensionType.SAMPLING in types

    def test_extension_info_includes_type(self, registry, rngs):
        """Test that extension info includes type field."""

        class TestExt(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        registry.register_extension(
            name="typed_ext",
            extension_class=TestExt,
            extension_type=ExtensionType.LOSS,
            modalities=["test"],
            capabilities=["test"],
        )

        info = registry.get_extension_info("typed_ext")
        assert "extension_type" in info
        assert info["extension_type"] == "loss"

    def test_list_extensions_includes_type(self, registry, rngs):
        """Test that list_all_extensions includes type in metadata."""

        class TestExt(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        registry.register_extension(
            name="list_test_ext",
            extension_class=TestExt,
            extension_type=ExtensionType.EVALUATION,
            modalities=["test"],
            capabilities=["test"],
        )

        all_exts = registry.list_all_extensions()
        assert "list_test_ext" in all_exts
        assert "extension_type" in all_exts["list_test_ext"]
        assert all_exts["list_test_ext"]["extension_type"] == "evaluation"
