"""Tests for UNet1D backbone configuration."""

import pytest

from artifex.generative_models.core.configuration.backbone_config import (
    UNet1DBackboneConfig,
)


def create_unet1d_config(**kwargs) -> UNet1DBackboneConfig:
    """Create a UNet1D config with required fields."""
    defaults = {
        "name": "test_unet1d",
        "hidden_dims": (32, 64, 128, 256),
        "activation": "gelu",
    }
    defaults.update(kwargs)
    return UNet1DBackboneConfig(**defaults)


class TestUNet1DBackboneConfig:
    """Tests for UNet1D backbone configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = create_unet1d_config()

        assert config.backbone_type == "unet_1d"
        assert config.in_channels == 1
        assert config.time_embedding_dim == 128
        assert config.hidden_dims == (32, 64, 128, 256)

    def test_custom_hidden_dims(self):
        """Test custom hidden dimensions."""
        config = create_unet1d_config(hidden_dims=(16, 32, 64))

        assert config.hidden_dims == (16, 32, 64)

    def test_custom_in_channels(self):
        """Test custom input channels (e.g., stereo audio)."""
        config = create_unet1d_config(in_channels=2)  # Stereo

        assert config.in_channels == 2

    def test_custom_time_embedding_dim(self):
        """Test custom time embedding dimension."""
        config = create_unet1d_config(time_embedding_dim=256)

        assert config.time_embedding_dim == 256

    def test_backbone_type_is_unet_1d(self):
        """Test that backbone_type is always 'unet_1d'."""
        config = create_unet1d_config()
        assert config.backbone_type == "unet_1d"

    def test_invalid_in_channels(self):
        """Test that non-positive in_channels raises error."""
        with pytest.raises(ValueError, match="in_channels"):
            create_unet1d_config(in_channels=0)

        with pytest.raises(ValueError, match="in_channels"):
            create_unet1d_config(in_channels=-1)

    def test_invalid_time_embedding_dim(self):
        """Test that non-positive time_embedding_dim raises error."""
        with pytest.raises(ValueError, match="time_embedding_dim"):
            create_unet1d_config(time_embedding_dim=0)

        with pytest.raises(ValueError, match="time_embedding_dim"):
            create_unet1d_config(time_embedding_dim=-1)

    def test_frozen_dataclass(self):
        """Test that config is immutable."""
        config = create_unet1d_config()

        with pytest.raises(AttributeError):
            config.in_channels = 2  # type: ignore
