"""Tests for backbone configuration dataclasses.

This module tests the backbone configuration system for diffusion models:
- UNetBackboneConfig: Standard convolutional U-Net
- DiTBackboneConfig: Diffusion Transformer
- UViTBackboneConfig: ViT with U-Net-like skip connections
- UNet2DConditionBackboneConfig: Text-conditioned UNet
- BackboneConfig: Union type with backbone_type discriminator
- create_backbone: Factory function for creating backbones

Following TDD principles - these tests define the expected behavior.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.backbone_config import (
    BackboneConfig,
    create_backbone,
    DiTBackboneConfig,
    get_backbone_config_type,
    UNet2DConditionBackboneConfig,
    UNetBackboneConfig,
    UViTBackboneConfig,
)
from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig


# =============================================================================
# UNetBackboneConfig Tests
# =============================================================================


class TestUNetBackboneConfigBasics:
    """Test basic functionality of UNetBackboneConfig."""

    def test_create_minimal(self):
        """Test creating UNetBackboneConfig with minimal required fields."""
        config = UNetBackboneConfig(
            name="unet_backbone",
            hidden_dims=(64, 128, 256),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        assert config.name == "unet_backbone"
        assert config.backbone_type == "unet"
        assert config.hidden_dims == (64, 128, 256)
        assert config.activation == "gelu"
        assert config.in_channels == 3
        assert config.out_channels == 3

    def test_backbone_type_discriminator(self):
        """Test that backbone_type is always 'unet'."""
        config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        assert config.backbone_type == "unet"

    def test_default_values(self):
        """Test default values for optional fields."""
        config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        assert config.time_embedding_dim == 128
        assert config.num_res_blocks == 2
        assert config.attention_resolutions == (16, 8)
        assert config.channel_mult == (1, 2, 4)
        assert config.use_attention is True
        assert config.resamp_with_conv is True

    def test_frozen(self):
        """Test that UNetBackboneConfig is frozen."""
        config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.in_channels = 1  # type: ignore

    def test_inherits_from_base_network_config(self):
        """Test inheritance from BaseNetworkConfig."""
        config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        assert isinstance(config, BaseNetworkConfig)


class TestUNetBackboneConfigValidation:
    """Test validation of UNetBackboneConfig."""

    def test_invalid_in_channels_zero(self):
        """Test that zero in_channels raises ValueError."""
        with pytest.raises(ValueError, match="in_channels must be positive"):
            UNetBackboneConfig(
                name="unet",
                hidden_dims=(64, 128),
                activation="gelu",
                in_channels=0,
                out_channels=3,
            )

    def test_invalid_out_channels_zero(self):
        """Test that zero out_channels raises ValueError."""
        with pytest.raises(ValueError, match="out_channels must be positive"):
            UNetBackboneConfig(
                name="unet",
                hidden_dims=(64, 128),
                activation="gelu",
                in_channels=3,
                out_channels=0,
            )

    def test_invalid_time_embedding_dim_zero(self):
        """Test that zero time_embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="time_embedding_dim must be positive"):
            UNetBackboneConfig(
                name="unet",
                hidden_dims=(64, 128),
                activation="gelu",
                in_channels=3,
                out_channels=3,
                time_embedding_dim=0,
            )

    def test_empty_channel_mult_invalid(self):
        """Test that empty channel_mult raises ValueError."""
        with pytest.raises(ValueError, match="channel_mult cannot be empty"):
            UNetBackboneConfig(
                name="unet",
                hidden_dims=(64, 128),
                activation="gelu",
                in_channels=3,
                out_channels=3,
                channel_mult=(),
            )


class TestUNetBackboneConfigSerialization:
    """Test serialization of UNetBackboneConfig."""

    def test_to_dict_includes_backbone_type(self):
        """Test that to_dict includes backbone_type."""
        config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        d = config.to_dict()
        assert d["backbone_type"] == "unet"

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128, 256),
            activation="silu",
            in_channels=1,
            out_channels=1,
            time_embedding_dim=256,
        )
        d = original.to_dict()
        restored = UNetBackboneConfig.from_dict(d)
        assert restored == original


# =============================================================================
# DiTBackboneConfig Tests
# =============================================================================


class TestDiTBackboneConfigBasics:
    """Test basic functionality of DiTBackboneConfig."""

    def test_create_minimal(self):
        """Test creating DiTBackboneConfig with minimal required fields."""
        config = DiTBackboneConfig(
            name="dit_backbone",
            hidden_dims=(512,),  # Required by BaseNetworkConfig
            activation="gelu",
        )
        assert config.name == "dit_backbone"
        assert config.backbone_type == "dit"
        assert config.hidden_size == 512
        assert config.depth == 12
        assert config.num_heads == 8

    def test_backbone_type_discriminator(self):
        """Test that backbone_type is always 'dit'."""
        config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(768,),
            activation="gelu",
        )
        assert config.backbone_type == "dit"

    def test_default_values(self):
        """Test default values for optional fields."""
        config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.img_size == 32
        assert config.patch_size == 2
        assert config.hidden_size == 512
        assert config.depth == 12
        assert config.num_heads == 8
        assert config.mlp_ratio == 4.0
        assert config.num_classes is None
        assert config.learn_sigma is False
        assert config.cfg_scale == 1.0

    def test_create_with_custom_values(self):
        """Test creating DiTBackboneConfig with custom values."""
        config = DiTBackboneConfig(
            name="dit_xl",
            hidden_dims=(1152,),
            activation="gelu",
            img_size=256,
            patch_size=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=1000,
            learn_sigma=True,
            cfg_scale=4.0,
        )
        assert config.img_size == 256
        assert config.patch_size == 4
        assert config.hidden_size == 1152
        assert config.depth == 28
        assert config.num_heads == 16
        assert config.num_classes == 1000
        assert config.learn_sigma is True
        assert config.cfg_scale == 4.0

    def test_frozen(self):
        """Test that DiTBackboneConfig is frozen."""
        config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(512,),
            activation="gelu",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.depth = 24  # type: ignore


class TestDiTBackboneConfigValidation:
    """Test validation of DiTBackboneConfig."""

    def test_invalid_img_size_zero(self):
        """Test that zero img_size raises ValueError."""
        with pytest.raises(ValueError, match="img_size must be positive"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                img_size=0,
            )

    def test_invalid_patch_size_zero(self):
        """Test that zero patch_size raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                patch_size=0,
            )

    def test_invalid_hidden_size_zero(self):
        """Test that zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                hidden_size=0,
            )

    def test_invalid_depth_zero(self):
        """Test that zero depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be positive"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                depth=0,
            )

    def test_invalid_num_heads_zero(self):
        """Test that zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                num_heads=0,
            )

    def test_hidden_size_divisible_by_num_heads(self):
        """Test that hidden_size must be divisible by num_heads."""
        with pytest.raises(ValueError, match="hidden_size must be divisible by num_heads"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                hidden_size=512,
                num_heads=7,  # 512 not divisible by 7
            )

    def test_img_size_divisible_by_patch_size(self):
        """Test that img_size must be divisible by patch_size."""
        with pytest.raises(ValueError, match="img_size must be divisible by patch_size"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                img_size=32,
                patch_size=5,  # 32 not divisible by 5
            )

    def test_invalid_num_classes_zero(self):
        """Test that zero num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
                num_classes=0,
            )

    def test_num_classes_none_allowed(self):
        """Test that None num_classes is allowed (unconditional)."""
        config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(512,),
            activation="gelu",
            num_classes=None,
        )
        assert config.num_classes is None


class TestDiTBackboneConfigSerialization:
    """Test serialization of DiTBackboneConfig."""

    def test_to_dict_includes_backbone_type(self):
        """Test that to_dict includes backbone_type."""
        config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(512,),
            activation="gelu",
        )
        d = config.to_dict()
        assert d["backbone_type"] == "dit"

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = DiTBackboneConfig(
            name="dit",
            hidden_dims=(768,),
            activation="gelu",
            img_size=64,
            patch_size=4,
            hidden_size=768,
            depth=16,
            num_heads=12,
            num_classes=1000,
        )
        d = original.to_dict()
        restored = DiTBackboneConfig.from_dict(d)
        assert restored == original


# =============================================================================
# UViTBackboneConfig Tests
# =============================================================================


class TestUViTBackboneConfigBasics:
    """Test basic functionality of UViTBackboneConfig."""

    def test_create_minimal(self):
        """Test creating UViTBackboneConfig with minimal required fields."""
        config = UViTBackboneConfig(
            name="uvit_backbone",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.name == "uvit_backbone"
        assert config.backbone_type == "uvit"

    def test_backbone_type_discriminator(self):
        """Test that backbone_type is always 'uvit'."""
        config = UViTBackboneConfig(
            name="uvit",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.backbone_type == "uvit"

    def test_default_values(self):
        """Test default values for optional fields."""
        config = UViTBackboneConfig(
            name="uvit",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.img_size == 32
        assert config.patch_size == 2
        assert config.embed_dim == 512
        assert config.depth == 12
        assert config.num_heads == 8
        assert config.in_channels == 3
        assert config.out_channels == 3
        assert config.mlp_ratio == 4.0
        assert config.use_skip_connection is True


class TestUViTBackboneConfigValidation:
    """Test validation of UViTBackboneConfig."""

    def test_embed_dim_divisible_by_num_heads(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads"):
            UViTBackboneConfig(
                name="uvit",
                hidden_dims=(512,),
                activation="gelu",
                embed_dim=512,
                num_heads=7,  # 512 not divisible by 7
            )


# =============================================================================
# UNet2DConditionBackboneConfig Tests
# =============================================================================


class TestUNet2DConditionBackboneConfigBasics:
    """Test basic functionality of UNet2DConditionBackboneConfig."""

    def test_create_minimal(self):
        """Test creating UNet2DConditionBackboneConfig with minimal fields."""
        config = UNet2DConditionBackboneConfig(
            name="unet2d_cond",
            hidden_dims=(320, 640, 1280),
            activation="gelu",
        )
        assert config.name == "unet2d_cond"
        assert config.backbone_type == "unet2d_condition"

    def test_backbone_type_discriminator(self):
        """Test that backbone_type is always 'unet2d_condition'."""
        config = UNet2DConditionBackboneConfig(
            name="unet2d",
            hidden_dims=(320, 640),
            activation="gelu",
        )
        assert config.backbone_type == "unet2d_condition"

    def test_default_values(self):
        """Test default values for optional fields."""
        config = UNet2DConditionBackboneConfig(
            name="unet2d",
            hidden_dims=(320, 640, 1280),
            activation="gelu",
        )
        assert config.in_channels == 4
        assert config.out_channels == 4
        assert config.cross_attention_dim == 768
        assert config.num_heads == 8
        assert config.num_res_blocks == 2
        assert config.attention_levels == (0, 1, 2)
        assert config.time_embedding_dim == 128


# =============================================================================
# BackboneConfig Union Type Tests
# =============================================================================


class TestBackboneConfigUnion:
    """Test the BackboneConfig union type."""

    def test_unet_is_backbone_config(self):
        """Test that UNetBackboneConfig is a BackboneConfig."""
        config: BackboneConfig = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        assert config.backbone_type == "unet"

    def test_dit_is_backbone_config(self):
        """Test that DiTBackboneConfig is a BackboneConfig."""
        config: BackboneConfig = DiTBackboneConfig(
            name="dit",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.backbone_type == "dit"

    def test_uvit_is_backbone_config(self):
        """Test that UViTBackboneConfig is a BackboneConfig."""
        config: BackboneConfig = UViTBackboneConfig(
            name="uvit",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.backbone_type == "uvit"

    def test_unet2d_condition_is_backbone_config(self):
        """Test that UNet2DConditionBackboneConfig is a BackboneConfig."""
        config: BackboneConfig = UNet2DConditionBackboneConfig(
            name="unet2d",
            hidden_dims=(320, 640),
            activation="gelu",
        )
        assert config.backbone_type == "unet2d_condition"

    def test_backbone_type_discriminates(self):
        """Test that backbone_type can be used to discriminate union types."""
        configs: list[BackboneConfig] = [
            UNetBackboneConfig(
                name="unet",
                hidden_dims=(64,),
                activation="gelu",
                in_channels=3,
                out_channels=3,
            ),
            DiTBackboneConfig(
                name="dit",
                hidden_dims=(512,),
                activation="gelu",
            ),
        ]

        backbone_types = [c.backbone_type for c in configs]
        assert backbone_types == ["unet", "dit"]


# =============================================================================
# get_backbone_config_type Tests
# =============================================================================


class TestGetBackboneConfigType:
    """Test the get_backbone_config_type function."""

    def test_get_unet_config_type(self):
        """Test getting UNetBackboneConfig class."""
        config_class = get_backbone_config_type("unet")
        assert config_class is UNetBackboneConfig

    def test_get_dit_config_type(self):
        """Test getting DiTBackboneConfig class."""
        config_class = get_backbone_config_type("dit")
        assert config_class is DiTBackboneConfig

    def test_get_uvit_config_type(self):
        """Test getting UViTBackboneConfig class."""
        config_class = get_backbone_config_type("uvit")
        assert config_class is UViTBackboneConfig

    def test_get_unet2d_condition_config_type(self):
        """Test getting UNet2DConditionBackboneConfig class."""
        config_class = get_backbone_config_type("unet2d_condition")
        assert config_class is UNet2DConditionBackboneConfig

    def test_invalid_backbone_type(self):
        """Test that invalid backbone_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone_type"):
            get_backbone_config_type("invalid")  # type: ignore


# =============================================================================
# create_backbone Factory Tests
# =============================================================================


class TestCreateBackboneFactory:
    """Test the create_backbone factory function."""

    @pytest.fixture
    def rngs(self):
        """Create rngs for testing."""
        from flax import nnx

        return nnx.Rngs(0)

    def test_create_unet_backbone(self, rngs):
        """Test creating UNet backbone from config."""
        config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(32, 64),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        backbone = create_backbone(config, rngs=rngs)

        # Check it's a UNet
        from artifex.generative_models.models.backbones.unet import UNet

        assert isinstance(backbone, UNet)

    def test_create_dit_backbone(self, rngs):
        """Test creating DiT backbone from config."""
        config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(256,),
            activation="gelu",
            img_size=16,
            patch_size=2,
            hidden_size=256,
            depth=4,
            num_heads=4,
        )
        backbone = create_backbone(config, rngs=rngs)

        # Check it's a DiffusionTransformer
        from artifex.generative_models.models.backbones.dit import DiffusionTransformer

        assert isinstance(backbone, DiffusionTransformer)

    def test_create_uvit_backbone_not_implemented(self, rngs):
        """Test that U-ViT backbone raises NotImplementedError."""
        config = UViTBackboneConfig(
            name="uvit",
            hidden_dims=(256,),
            activation="gelu",
        )
        with pytest.raises(NotImplementedError, match="U-ViT backbone is not yet implemented"):
            create_backbone(config, rngs=rngs)

    def test_create_unet2d_condition_backbone(self, rngs):
        """Test creating UNet2DCondition backbone from config."""
        config = UNet2DConditionBackboneConfig(
            name="unet2d",
            hidden_dims=(32, 64),
            activation="gelu",
            in_channels=3,
            out_channels=3,
            cross_attention_dim=64,
            num_heads=2,
        )
        backbone = create_backbone(config, rngs=rngs)

        # Check it's a UNet2DCondition
        from artifex.generative_models.models.backbones.unet_cross_attention import (
            UNet2DCondition,
        )

        assert isinstance(backbone, UNet2DCondition)

    def test_backbone_type_dispatch(self, rngs):
        """Test that factory dispatches based on backbone_type."""
        unet_config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(32,),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        dit_config = DiTBackboneConfig(
            name="dit",
            hidden_dims=(128,),
            activation="gelu",
            img_size=8,
            patch_size=2,
            hidden_size=128,
            depth=2,
            num_heads=4,
        )

        unet = create_backbone(unet_config, rngs=rngs)
        dit = create_backbone(dit_config, rngs=rngs)

        # Different types
        assert type(unet).__name__ == "UNet"
        assert type(dit).__name__ == "DiffusionTransformer"
