"""Tests for modality adapters with unified configuration system.

Following TDD principles - these tests are written FIRST before implementation.
Now updated to use specific dataclass configs instead of ModelConfig.
"""

from typing import Any

import pytest
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration import (
    BaseConfig,
    DecoderConfig,
    EncoderConfig,
    ModalityConfig,
    VAEConfig,
)
from artifex.generative_models.core.protocols.configuration import BaseModalityConfig
from artifex.generative_models.modalities.base import (
    BaseModalityImplementation,
    ModelAdapter,
)


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(params=0, dropout=1, sample=2)


@pytest.fixture
def valid_vae_config():
    """Valid VAEConfig for testing."""
    encoder = EncoderConfig(
        name="encoder",
        input_shape=(28, 28, 1),
        latent_dim=32,
        hidden_dims=(256, 128),
        activation="gelu",
    )
    decoder = DecoderConfig(
        name="decoder",
        latent_dim=32,
        output_shape=(28, 28, 1),
        hidden_dims=(128, 256),
        activation="gelu",
    )
    return VAEConfig(
        name="test_model",
        encoder=encoder,
        decoder=decoder,
        encoder_type="dense",
    )


@pytest.fixture
def valid_modality_config():
    """Valid ModalityConfig for testing."""
    return ModalityConfig(
        name="test_modality",
        modality_name="test",
        supported_models=["vae", "gan"],
        default_metrics=["accuracy", "fid"],
        preprocessing_steps=[{"type": "normalize", "mean": 0.5, "std": 0.5}],
    )


@pytest.fixture
def base_modality_config():
    """BaseModalityConfig for current implementation."""
    from dataclasses import dataclass

    @dataclass
    class TestModalityConfig(BaseModalityConfig):
        name: str = "test"

    return TestModalityConfig()


class MockModelAdapter:
    """Mock adapter for testing that implements ModelAdapter protocol."""

    def create(self, config: VAEConfig, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        """Create a model with typed config."""
        if not isinstance(config, VAEConfig):
            raise TypeError(f"config must be a VAEConfig, got {type(config).__name__}")
        from artifex.generative_models.factory import create_model

        return create_model(config=config, rngs=rngs)


class MockModality(BaseModalityImplementation):
    """Mock modality for testing."""

    def __init__(self, config, *, rngs: nnx.Rngs):
        """Initialize with either BaseModalityConfig or ModalityConfig."""
        if isinstance(config, ModalityConfig):
            from dataclasses import dataclass

            @dataclass
            class TempConfig(BaseModalityConfig):
                name: str = config.name

            super().__init__(TempConfig(), rngs=rngs)
        else:
            super().__init__(config, rngs=rngs)

    def get_adapter(self, model_cls: type[GenerativeModel]) -> ModelAdapter:
        """Get adapter for model class."""
        return MockModelAdapter()

    def get_extensions(self, config: BaseConfig) -> dict[str, Any]:
        """Get modality-specific extensions with typed config."""
        if not isinstance(config, BaseConfig):
            raise TypeError(f"config must be a BaseConfig, got {type(config).__name__}")
        return {}


class TestModalityAdaptersUnifiedConfig:
    """Test modality adapters with unified configuration requirements."""

    def test_model_adapter_requires_dataclass_config(self, rngs):
        """Test that ModelAdapter.create raises TypeError for non-dataclass configs."""
        adapter = MockModelAdapter()

        # Dict config should raise TypeError
        with pytest.raises(TypeError, match="config must be a VAEConfig"):
            adapter.create(config={"latent_dim": 32}, rngs=rngs)

        # Any other type should raise TypeError
        with pytest.raises(TypeError, match="config must be a VAEConfig"):
            adapter.create(config="invalid", rngs=rngs)

        # None should raise TypeError
        with pytest.raises(TypeError, match="config must be a VAEConfig"):
            adapter.create(config=None, rngs=rngs)

    def test_model_adapter_accepts_vae_config(self, valid_vae_config, rngs):
        """Test that ModelAdapter works correctly with VAEConfig."""
        adapter = MockModelAdapter()
        model = adapter.create(config=valid_vae_config, rngs=rngs)

        # Model should be created successfully
        assert model is not None
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")

    def test_modality_get_extensions_requires_typed_config(self, valid_modality_config, rngs):
        """Test that Modality.get_extensions should require typed configuration."""
        modality = MockModality(config=valid_modality_config, rngs=rngs)

        # Test with valid typed config
        extensions = modality.get_extensions(config=valid_modality_config)
        assert isinstance(extensions, dict)

    def test_model_adapter_implementation_enforcement(self, valid_vae_config, rngs):
        """Test that ModelAdapter implementations should enforce dataclass configs."""

        class StrictAdapter:
            def create(self, config: VAEConfig, *, rngs: nnx.Rngs, **kwargs) -> GenerativeModel:
                if not isinstance(config, VAEConfig):
                    raise TypeError(f"config must be a VAEConfig, got {type(config).__name__}")
                from artifex.generative_models.factory import create_model

                return create_model(config=config, rngs=rngs)

        adapter = StrictAdapter()

        # Should reject dict
        with pytest.raises(TypeError, match="config must be a VAEConfig"):
            adapter.create(config={"some": "dict"}, rngs=rngs)

        # Should accept VAEConfig
        model = adapter.create(config=valid_vae_config, rngs=rngs)
        assert model is not None

    def test_legacy_modality_config_rejected(self, rngs):
        """Test that legacy modality config classes are rejected."""

        class LegacyModalityConfig:
            def __init__(self):
                self.name = "legacy"
                self.modality_name = "legacy_modality"

        legacy_config = LegacyModalityConfig()

        try:
            MockModality(config=legacy_config, rngs=rngs)
            pytest.fail("BaseModalityImplementation should reject non-typed configs")
        except (TypeError, AttributeError):
            pass

    def test_modality_create_model_method_with_typed_config(
        self, valid_vae_config, valid_modality_config, rngs
    ):
        """Test that modality.create_model method uses typed configuration."""
        modality = MockModality(config=valid_modality_config, rngs=rngs)

        if hasattr(modality, "create_model"):
            with pytest.raises(TypeError, match="config must be a VAEConfig"):
                modality.create_model(model_type="vae", config={"some": "dict"}, rngs=rngs)

            model = modality.create_model(model_type="vae", config=valid_vae_config, rngs=rngs)
            assert model is not None

    def test_all_modality_methods_enforce_typed_configs(
        self, valid_vae_config, valid_modality_config, rngs
    ):
        """Test that all modality methods enforce typed configurations."""
        modality = MockModality(config=valid_modality_config, rngs=rngs)
        adapter = modality.get_adapter(GenerativeModel)

        # Test adapter.create rejects dict
        with pytest.raises(TypeError, match="config must be a VAEConfig"):
            adapter.create(config={"some": "dict"}, rngs=rngs)

        # Test adapter.create accepts VAEConfig
        model = adapter.create(config=valid_vae_config, rngs=rngs)
        assert model is not None

    def test_modality_implementation_requires_modality_configuration(self, rngs):
        """Test that BaseModalityImplementation requires ModalityConfig."""
        with pytest.raises(TypeError, match="config must be"):
            BaseModalityImplementation(config={"name": "test"}, rngs=rngs)
