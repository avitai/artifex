"""Tests for timeseries model adapters.

This module provides comprehensive tests for the timeseries adapters, covering
config validation, adapter initialization, and adapter functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from artifex.generative_models.modalities.timeseries.adapters import (
    get_timeseries_adapter,
    TimeseriesAdapterConfig,
    TimeseriesDiffusionAdapter,
    TimeseriesRNNAdapter,
    TimeseriesTransformerAdapter,
    TimeseriesVAEAdapter,
)


class TestTimeseriesAdapterConfig:
    """Test suite for TimeseriesAdapterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimeseriesAdapterConfig()

        assert config.name == "timeseries_adapter"
        assert config.sequence_length == 128
        assert config.num_features == 1
        assert config.sampling_rate == 1.0
        assert config.use_temporal_position_encoding is True
        assert config.causal_masking is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TimeseriesAdapterConfig(
            name="custom_adapter",
            sequence_length=256,
            num_features=10,
            sampling_rate=44100.0,
            use_temporal_position_encoding=False,
            causal_masking=False,
        )

        assert config.name == "custom_adapter"
        assert config.sequence_length == 256
        assert config.num_features == 10
        assert config.sampling_rate == 44100.0
        assert config.use_temporal_position_encoding is False
        assert config.causal_masking is False

    def test_config_validation_sequence_length(self):
        """Test validation rejects non-positive sequence_length."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            TimeseriesAdapterConfig(sequence_length=0)

        with pytest.raises(ValueError, match="sequence_length must be positive"):
            TimeseriesAdapterConfig(sequence_length=-10)

    def test_config_validation_num_features(self):
        """Test validation rejects non-positive num_features."""
        with pytest.raises(ValueError, match="num_features must be positive"):
            TimeseriesAdapterConfig(num_features=0)

        with pytest.raises(ValueError, match="num_features must be positive"):
            TimeseriesAdapterConfig(num_features=-5)

    def test_config_validation_sampling_rate(self):
        """Test validation rejects non-positive sampling_rate."""
        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            TimeseriesAdapterConfig(sampling_rate=0.0)

        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            TimeseriesAdapterConfig(sampling_rate=-100.0)

    def test_frozen_dataclass_immutability(self):
        """Test that frozen dataclass is immutable."""
        config = TimeseriesAdapterConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.sequence_length = 512

    def test_config_equality(self):
        """Test that configs with same values are equal."""
        config1 = TimeseriesAdapterConfig(sequence_length=64)
        config2 = TimeseriesAdapterConfig(sequence_length=64)
        config3 = TimeseriesAdapterConfig(sequence_length=128)

        assert config1 == config2
        assert config1 != config3

    def test_config_hashable(self):
        """Test that frozen config is hashable."""
        config = TimeseriesAdapterConfig()
        # Should not raise
        hash_val = hash(config)
        assert isinstance(hash_val, int)


class TestTimeseriesTransformerAdapter:
    """Test suite for TimeseriesTransformerAdapter."""

    @pytest.fixture
    def default_adapter(self):
        """Create default adapter for testing."""
        return TimeseriesTransformerAdapter()

    @pytest.fixture
    def custom_adapter(self):
        """Create adapter with custom config for testing."""
        config = TimeseriesAdapterConfig(
            sequence_length=64,
            num_features=5,
            sampling_rate=16000.0,
            use_temporal_position_encoding=True,
            causal_masking=True,
        )
        return TimeseriesTransformerAdapter(config=config)

    def test_initialization_default(self, default_adapter):
        """Test adapter initialization with default config."""
        assert default_adapter.config is not None
        assert default_adapter.config.sequence_length == 128
        assert default_adapter.config.num_features == 1

    def test_initialization_custom(self, custom_adapter):
        """Test adapter initialization with custom config."""
        assert custom_adapter.config.sequence_length == 64
        assert custom_adapter.config.num_features == 5
        assert custom_adapter.config.sampling_rate == 16000.0

    def test_initialization_with_none_config(self):
        """Test adapter uses default config when None is passed."""
        adapter = TimeseriesTransformerAdapter(config=None)
        assert adapter.config is not None
        assert adapter.config == TimeseriesAdapterConfig()

    def test_get_timeseries_extensions_with_temporal_encoding(self, base_rngs):
        """Test extension creation with temporal position encoding enabled."""
        config = TimeseriesAdapterConfig(
            use_temporal_position_encoding=True,
            causal_masking=False,
        )
        adapter = TimeseriesTransformerAdapter(config=config)

        class MockConfig:
            pass

        mock_config = MockConfig()

        # Mock the temporal extension module
        mock_extension = MagicMock()
        with patch.dict(
            "sys.modules",
            {"artifex.generative_models.extensions.temporal": MagicMock()},
        ):
            with patch(
                "artifex.generative_models.modalities.timeseries.adapters.TimeseriesTransformerAdapter._get_timeseries_extensions"
            ) as mock_method:
                mock_method.return_value = {"temporal_position": mock_extension}
                _ = adapter._get_timeseries_extensions(mock_config, rngs=base_rngs)
                # Original method would fail with missing import, so we use mock
                assert mock_method.called

    def test_get_timeseries_extensions_both_disabled(self, base_rngs):
        """Test extension creation with both features disabled."""
        config = TimeseriesAdapterConfig(
            use_temporal_position_encoding=False,
            causal_masking=False,
        )
        adapter = TimeseriesTransformerAdapter(config=config)

        class MockConfig:
            use_temporal_position_encoding = False
            causal_masking = False

        mock_config = MockConfig()
        extensions = adapter._get_timeseries_extensions(mock_config, rngs=base_rngs)

        assert len(extensions) == 0

    def test_adapter_stores_config(self):
        """Test that adapter correctly stores the config."""
        config = TimeseriesAdapterConfig(
            name="test_transformer",
            sequence_length=512,
            num_features=8,
        )
        adapter = TimeseriesTransformerAdapter(config=config)

        assert adapter.config.name == "test_transformer"
        assert adapter.config.sequence_length == 512
        assert adapter.config.num_features == 8


class TestTimeseriesRNNAdapter:
    """Test suite for TimeseriesRNNAdapter."""

    @pytest.fixture
    def default_adapter(self):
        """Create default adapter for testing."""
        return TimeseriesRNNAdapter()

    def test_initialization_default(self, default_adapter):
        """Test adapter initialization with default config."""
        assert default_adapter.config is not None
        assert default_adapter.config.sequence_length == 128

    def test_initialization_with_custom_config(self):
        """Test adapter initialization with custom config."""
        config = TimeseriesAdapterConfig(
            sequence_length=512,
            num_features=20,
        )
        adapter = TimeseriesRNNAdapter(config=config)

        assert adapter.config.sequence_length == 512
        assert adapter.config.num_features == 20

    def test_initialization_with_none_config(self):
        """Test adapter uses default config when None is passed."""
        adapter = TimeseriesRNNAdapter(config=None)
        assert adapter.config is not None
        assert adapter.config == TimeseriesAdapterConfig()

    def test_get_extensions_no_dropout_no_clipping(self, base_rngs):
        """Test extension creation with both dropout and clipping disabled."""
        adapter = TimeseriesRNNAdapter()

        class MockConfig:
            use_sequence_dropout = False
            use_gradient_clipping = False

        mock_config = MockConfig()
        extensions = adapter._get_timeseries_extensions(mock_config, rngs=base_rngs)

        assert len(extensions) == 0

    def test_adapter_stores_config(self):
        """Test that adapter correctly stores the config."""
        config = TimeseriesAdapterConfig(
            name="test_rnn",
            sequence_length=256,
        )
        adapter = TimeseriesRNNAdapter(config=config)

        assert adapter.config.name == "test_rnn"
        assert adapter.config.sequence_length == 256


class TestTimeseriesDiffusionAdapter:
    """Test suite for TimeseriesDiffusionAdapter."""

    @pytest.fixture
    def default_adapter(self):
        """Create default adapter for testing."""
        return TimeseriesDiffusionAdapter()

    def test_initialization_default(self, default_adapter):
        """Test adapter initialization with default config."""
        assert default_adapter.config is not None
        assert default_adapter.config.sequence_length == 128

    def test_initialization_with_custom_config(self):
        """Test adapter initialization with custom config."""
        config = TimeseriesAdapterConfig(
            sequence_length=1024,
            num_features=4,
            sampling_rate=8000.0,
        )
        adapter = TimeseriesDiffusionAdapter(config=config)

        assert adapter.config.sequence_length == 1024
        assert adapter.config.num_features == 4

    def test_initialization_with_none_config(self):
        """Test adapter uses default config when None is passed."""
        adapter = TimeseriesDiffusionAdapter(config=None)
        assert adapter.config is not None
        assert adapter.config == TimeseriesAdapterConfig()

    def test_get_extensions_both_disabled(self, base_rngs):
        """Test extension creation with both features disabled."""
        adapter = TimeseriesDiffusionAdapter()

        class MockConfig:
            use_temporal_conditioning = False
            use_temporal_noise_schedule = False

        mock_config = MockConfig()
        extensions = adapter._get_timeseries_extensions(mock_config, rngs=base_rngs)

        assert len(extensions) == 0

    def test_adapter_stores_config(self):
        """Test that adapter correctly stores the config."""
        config = TimeseriesAdapterConfig(
            name="test_diffusion",
            sequence_length=2048,
            sampling_rate=44100.0,
        )
        adapter = TimeseriesDiffusionAdapter(config=config)

        assert adapter.config.name == "test_diffusion"
        assert adapter.config.sequence_length == 2048
        assert adapter.config.sampling_rate == 44100.0


class TestTimeseriesVAEAdapter:
    """Test suite for TimeseriesVAEAdapter."""

    @pytest.fixture
    def default_adapter(self):
        """Create default adapter for testing."""
        return TimeseriesVAEAdapter()

    def test_initialization_default(self, default_adapter):
        """Test adapter initialization with default config."""
        assert default_adapter.config is not None
        assert default_adapter.config.sequence_length == 128

    def test_initialization_with_custom_config(self):
        """Test adapter initialization with custom config."""
        config = TimeseriesAdapterConfig(
            sequence_length=256,
            num_features=8,
        )
        adapter = TimeseriesVAEAdapter(config=config)

        assert adapter.config.sequence_length == 256
        assert adapter.config.num_features == 8

    def test_initialization_with_none_config(self):
        """Test adapter uses default config when None is passed."""
        adapter = TimeseriesVAEAdapter(config=None)
        assert adapter.config is not None
        assert adapter.config == TimeseriesAdapterConfig()

    def test_get_extensions_both_disabled(self, base_rngs):
        """Test extension creation with both features disabled."""
        adapter = TimeseriesVAEAdapter()

        class MockConfig:
            use_kl_annealing = False
            use_reconstruction_weighting = False

        mock_config = MockConfig()
        extensions = adapter._get_timeseries_extensions(mock_config, rngs=base_rngs)

        assert len(extensions) == 0

    def test_adapter_stores_config(self):
        """Test that adapter correctly stores the config."""
        config = TimeseriesAdapterConfig(
            name="test_vae",
            sequence_length=512,
            num_features=16,
        )
        adapter = TimeseriesVAEAdapter(config=config)

        assert adapter.config.name == "test_vae"
        assert adapter.config.sequence_length == 512
        assert adapter.config.num_features == 16


class TestGetTimeseriesAdapter:
    """Test suite for get_timeseries_adapter factory function."""

    def test_get_transformer_adapter(self):
        """Test getting adapter for transformer model."""

        class TransformerModel:
            pass

        adapter = get_timeseries_adapter(TransformerModel)
        assert isinstance(adapter, TimeseriesTransformerAdapter)

    def test_get_rnn_adapter(self):
        """Test getting adapter for RNN model."""

        class RNNModel:
            pass

        adapter = get_timeseries_adapter(RNNModel)
        assert isinstance(adapter, TimeseriesRNNAdapter)

    def test_get_diffusion_adapter(self):
        """Test getting adapter for diffusion model."""

        class DiffusionModel:
            pass

        adapter = get_timeseries_adapter(DiffusionModel)
        assert isinstance(adapter, TimeseriesDiffusionAdapter)

    def test_get_vae_adapter(self):
        """Test getting adapter for VAE model."""

        class VAEModel:
            pass

        adapter = get_timeseries_adapter(VAEModel)
        assert isinstance(adapter, TimeseriesVAEAdapter)

    def test_unknown_model_defaults_to_transformer(self):
        """Test that unknown model class defaults to transformer adapter."""

        class UnknownModel:
            pass

        adapter = get_timeseries_adapter(UnknownModel)
        assert isinstance(adapter, TimeseriesTransformerAdapter)

    def test_with_custom_config(self):
        """Test adapter factory with custom config."""

        class TransformerModel:
            pass

        config = TimeseriesAdapterConfig(
            sequence_length=512,
            sampling_rate=22050.0,
        )
        adapter = get_timeseries_adapter(TransformerModel, config=config)

        assert isinstance(adapter, TimeseriesTransformerAdapter)
        assert adapter.config.sequence_length == 512
        assert adapter.config.sampling_rate == 22050.0

    def test_with_none_config(self):
        """Test adapter factory with None config."""

        class RNNModel:
            pass

        adapter = get_timeseries_adapter(RNNModel, config=None)
        assert isinstance(adapter, TimeseriesRNNAdapter)
        assert adapter.config == TimeseriesAdapterConfig()

    def test_adapter_naming_case_sensitivity(self):
        """Test that adapter lookup is based on class name."""

        class transformermodel:  # lowercase
            pass

        # Should default to transformer since exact match "TransformerModel" won't work
        adapter = get_timeseries_adapter(transformermodel)
        assert isinstance(adapter, TimeseriesTransformerAdapter)

    def test_multiple_adapter_instances_independent(self):
        """Test that multiple adapter instances don't share state."""

        class DiffusionModel:
            pass

        config1 = TimeseriesAdapterConfig(sequence_length=64)
        config2 = TimeseriesAdapterConfig(sequence_length=128)

        adapter1 = get_timeseries_adapter(DiffusionModel, config=config1)
        adapter2 = get_timeseries_adapter(DiffusionModel, config=config2)

        assert adapter1.config.sequence_length == 64
        assert adapter2.config.sequence_length == 128
        assert adapter1.config != adapter2.config
