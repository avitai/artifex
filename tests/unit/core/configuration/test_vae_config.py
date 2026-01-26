"""Tests for VAE configuration classes using frozen dataclasses.

This module tests VAEConfig, BetaVAEConfig, ConditionalVAEConfig, and VQVAEConfig,
following the established test patterns from GAN config tests.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration import (
    BaseConfig,
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    ConditionalVAEConfig,
    VAEConfig,
    VQVAEConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def encoder_config() -> EncoderConfig:
    """Create a valid encoder config for testing."""
    return EncoderConfig(
        name="test_encoder",
        input_shape=(28, 28, 1),
        latent_dim=32,
        hidden_dims=(256, 128),
        activation="relu",
        batch_norm=True,
        dropout_rate=0.1,
    )


@pytest.fixture
def decoder_config() -> DecoderConfig:
    """Create a valid decoder config for testing."""
    return DecoderConfig(
        name="test_decoder",
        latent_dim=32,
        output_shape=(28, 28, 1),
        hidden_dims=(128, 256),
        activation="relu",
        batch_norm=True,
        dropout_rate=0.1,
        output_activation="sigmoid",
    )


@pytest.fixture
def vae_config(encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> VAEConfig:
    """Create a valid VAEConfig for testing."""
    return VAEConfig(
        name="test_vae",
        encoder=encoder_config,
        decoder=decoder_config,
        kl_weight=1.0,
    )


# =============================================================================
# VAEConfig Tests
# =============================================================================


class TestVAEConfigBasics:
    """Basic tests for VAEConfig instantiation and properties."""

    def test_create_minimal_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating VAEConfig with minimal required parameters."""
        config = VAEConfig(
            name="minimal_vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.name == "minimal_vae"
        assert config.encoder == encoder_config
        assert config.decoder == decoder_config

    def test_create_full_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating VAEConfig with all parameters."""
        config = VAEConfig(
            name="full_vae",
            description="A test VAE",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=0.5,
            tags=("test", "vae"),
            metadata={"version": "1.0"},
        )
        assert config.name == "full_vae"
        assert config.description == "A test VAE"
        assert config.kl_weight == 0.5
        assert config.tags == ("test", "vae")
        assert config.metadata == {"version": "1.0"}

    def test_config_is_frozen(self, vae_config: VAEConfig) -> None:
        """Test that VAEConfig is immutable (frozen dataclass)."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            vae_config.name = "new_name"  # type: ignore

    def test_config_equality(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that equal configs compare equal."""
        config1 = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        config2 = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config1 == config2

    def test_config_inequality(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that different configs compare unequal."""
        config1 = VAEConfig(
            name="vae1",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        config2 = VAEConfig(
            name="vae2",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config1 != config2


class TestVAEConfigInheritance:
    """Test VAEConfig inheritance from BaseConfig."""

    def test_inherits_from_base_config(self) -> None:
        """Test that VAEConfig inherits from BaseConfig."""
        assert issubclass(VAEConfig, BaseConfig)

    def test_has_base_config_fields(self, vae_config: VAEConfig) -> None:
        """Test that VAEConfig has all BaseConfig fields."""
        assert hasattr(vae_config, "name")
        assert hasattr(vae_config, "description")
        assert hasattr(vae_config, "tags")
        assert hasattr(vae_config, "metadata")


class TestVAEConfigValidation:
    """Test VAEConfig validation rules."""

    def test_encoder_required(self, decoder_config: DecoderConfig) -> None:
        """Test that encoder config is required."""
        with pytest.raises(ValueError, match="encoder.*required"):
            VAEConfig(
                name="vae",
                encoder=None,  # type: ignore
                decoder=decoder_config,
            )

    def test_decoder_required(self, encoder_config: EncoderConfig) -> None:
        """Test that decoder config is required."""
        with pytest.raises(ValueError, match="decoder.*required"):
            VAEConfig(
                name="vae",
                encoder=encoder_config,
                decoder=None,  # type: ignore
            )

    def test_encoder_wrong_type(self, decoder_config: DecoderConfig) -> None:
        """Test that encoder must be EncoderConfig."""
        with pytest.raises(TypeError, match="EncoderConfig"):
            VAEConfig(
                name="vae",
                encoder="not_a_config",  # type: ignore
                decoder=decoder_config,
            )

    def test_decoder_wrong_type(self, encoder_config: EncoderConfig) -> None:
        """Test that decoder must be DecoderConfig."""
        with pytest.raises(TypeError, match="DecoderConfig"):
            VAEConfig(
                name="vae",
                encoder=encoder_config,
                decoder="not_a_config",  # type: ignore
            )

    def test_kl_weight_positive(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that kl_weight must be non-negative."""
        with pytest.raises(ValueError, match="kl_weight"):
            VAEConfig(
                name="vae",
                encoder=encoder_config,
                decoder=decoder_config,
                kl_weight=-0.5,
            )

    def test_kl_weight_zero_allowed(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that kl_weight can be zero."""
        config = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=0.0,
        )
        assert config.kl_weight == 0.0

    def test_latent_dim_consistency_warning(self) -> None:
        """Test that mismatched latent_dim between encoder/decoder is detected."""
        encoder = EncoderConfig(
            name="enc",
            input_shape=(28, 28, 1),
            latent_dim=32,
            hidden_dims=(128,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="dec",
            latent_dim=64,  # Mismatch!
            output_shape=(28, 28, 1),
            hidden_dims=(128,),
            activation="relu",
        )
        with pytest.raises(ValueError, match="latent_dim"):
            VAEConfig(
                name="vae",
                encoder=encoder,
                decoder=decoder,
            )


class TestVAEConfigDefaults:
    """Test VAEConfig default values."""

    def test_default_kl_weight(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default kl_weight value."""
        config = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.kl_weight == 1.0

    def test_default_description(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default description value."""
        config = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.description == ""

    def test_default_tags(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default tags value."""
        config = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.tags == ()


class TestVAEConfigSerialization:
    """Test VAEConfig serialization (to_dict/from_dict)."""

    def test_to_dict(self, vae_config: VAEConfig) -> None:
        """Test converting VAEConfig to dictionary."""
        config_dict = vae_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test_vae"
        assert "encoder" in config_dict
        assert "decoder" in config_dict

    def test_to_dict_preserves_nested_configs(self, vae_config: VAEConfig) -> None:
        """Test that to_dict properly serializes nested configs."""
        config_dict = vae_config.to_dict()
        assert isinstance(config_dict["encoder"], dict)
        assert isinstance(config_dict["decoder"], dict)
        assert config_dict["encoder"]["name"] == "test_encoder"
        assert config_dict["decoder"]["name"] == "test_decoder"

    def test_from_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test creating VAEConfig from dictionary."""
        config_dict = {
            "name": "from_dict_vae",
            "encoder": encoder_config.to_dict(),
            "decoder": decoder_config.to_dict(),
            "kl_weight": 0.75,
        }
        config = VAEConfig.from_dict(config_dict)
        assert config.name == "from_dict_vae"
        assert config.kl_weight == 0.75
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.decoder, DecoderConfig)

    def test_roundtrip_serialization(self, vae_config: VAEConfig) -> None:
        """Test that to_dict -> from_dict preserves all values."""
        config_dict = vae_config.to_dict()
        restored_config = VAEConfig.from_dict(config_dict)
        assert restored_config.name == vae_config.name
        assert restored_config.kl_weight == vae_config.kl_weight
        assert restored_config.encoder.name == vae_config.encoder.name
        assert restored_config.decoder.name == vae_config.decoder.name


class TestVAEConfigEdgeCases:
    """Test VAEConfig edge cases."""

    def test_very_large_kl_weight(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test with very large kl_weight."""
        config = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=1000.0,
        )
        assert config.kl_weight == 1000.0

    def test_very_small_kl_weight(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test with very small kl_weight."""
        config = VAEConfig(
            name="vae",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=1e-8,
        )
        assert config.kl_weight == 1e-8

    def test_latent_dim_property(self, vae_config: VAEConfig) -> None:
        """Test that latent_dim is accessible from encoder."""
        assert vae_config.encoder.latent_dim == 32


# =============================================================================
# BetaVAEConfig Tests
# =============================================================================


class TestBetaVAEConfigBasics:
    """Basic tests for BetaVAEConfig."""

    def test_create_minimal_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating BetaVAEConfig with minimal required parameters."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.name == "beta_vae"
        assert isinstance(config, VAEConfig)

    def test_create_full_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating BetaVAEConfig with all parameters."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            beta_warmup_steps=1000,
            reconstruction_loss_type="bce",
        )
        assert config.beta_default == 4.0
        assert config.beta_warmup_steps == 1000
        assert config.reconstruction_loss_type == "bce"

    def test_inherits_from_vae_config(self) -> None:
        """Test that BetaVAEConfig inherits from VAEConfig."""
        assert issubclass(BetaVAEConfig, VAEConfig)


class TestBetaVAEConfigValidation:
    """Test BetaVAEConfig validation rules."""

    def test_beta_default_positive(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that beta_default must be positive."""
        with pytest.raises(ValueError, match="beta_default"):
            BetaVAEConfig(
                name="beta_vae",
                encoder=encoder_config,
                decoder=decoder_config,
                beta_default=-1.0,
            )

    def test_beta_warmup_steps_non_negative(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that beta_warmup_steps must be non-negative."""
        with pytest.raises(ValueError, match="beta_warmup_steps"):
            BetaVAEConfig(
                name="beta_vae",
                encoder=encoder_config,
                decoder=decoder_config,
                beta_warmup_steps=-100,
            )

    def test_reconstruction_loss_type_valid(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that reconstruction_loss_type must be valid."""
        with pytest.raises(ValueError, match="reconstruction_loss_type"):
            BetaVAEConfig(
                name="beta_vae",
                encoder=encoder_config,
                decoder=decoder_config,
                reconstruction_loss_type="invalid",
            )

    def test_mse_loss_type_valid(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that 'mse' is a valid loss type."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            reconstruction_loss_type="mse",
        )
        assert config.reconstruction_loss_type == "mse"

    def test_bce_loss_type_valid(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that 'bce' is a valid loss type."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            reconstruction_loss_type="bce",
        )
        assert config.reconstruction_loss_type == "bce"


class TestBetaVAEConfigDefaults:
    """Test BetaVAEConfig default values."""

    def test_default_beta_default(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default beta_default value."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.beta_default == 1.0

    def test_default_beta_warmup_steps(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default beta_warmup_steps value."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.beta_warmup_steps == 0

    def test_default_reconstruction_loss_type(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default reconstruction_loss_type value."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.reconstruction_loss_type == "mse"


class TestBetaVAEConfigSerialization:
    """Test BetaVAEConfig serialization."""

    def test_to_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test converting BetaVAEConfig to dictionary."""
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            beta_warmup_steps=1000,
        )
        config_dict = config.to_dict()
        assert config_dict["beta_default"] == 4.0
        assert config_dict["beta_warmup_steps"] == 1000

    def test_from_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test creating BetaVAEConfig from dictionary."""
        config_dict = {
            "name": "beta_vae",
            "encoder": encoder_config.to_dict(),
            "decoder": decoder_config.to_dict(),
            "beta_default": 4.0,
            "beta_warmup_steps": 1000,
            "reconstruction_loss_type": "bce",
        }
        config = BetaVAEConfig.from_dict(config_dict)
        assert config.beta_default == 4.0
        assert config.beta_warmup_steps == 1000
        assert config.reconstruction_loss_type == "bce"

    def test_roundtrip_serialization(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test roundtrip serialization for BetaVAEConfig."""
        original = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            beta_warmup_steps=1000,
        )
        config_dict = original.to_dict()
        restored = BetaVAEConfig.from_dict(config_dict)
        assert restored.beta_default == original.beta_default
        assert restored.beta_warmup_steps == original.beta_warmup_steps


# =============================================================================
# ConditionalVAEConfig Tests
# =============================================================================


class TestConditionalVAEConfigBasics:
    """Basic tests for ConditionalVAEConfig."""

    def test_create_minimal_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating ConditionalVAEConfig with minimal required parameters."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
        )
        assert config.name == "cvae"
        assert config.num_classes == 10

    def test_create_full_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating ConditionalVAEConfig with all parameters."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=100,
            condition_dim=64,
            condition_type="add",
        )
        assert config.num_classes == 100
        assert config.condition_dim == 64
        assert config.condition_type == "add"

    def test_inherits_from_vae_config(self) -> None:
        """Test that ConditionalVAEConfig inherits from VAEConfig."""
        assert issubclass(ConditionalVAEConfig, VAEConfig)


class TestConditionalVAEConfigValidation:
    """Test ConditionalVAEConfig validation rules."""

    def test_num_classes_required(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that num_classes is required and must be positive."""
        with pytest.raises(ValueError, match="num_classes"):
            ConditionalVAEConfig(
                name="cvae",
                encoder=encoder_config,
                decoder=decoder_config,
                num_classes=0,
            )

    def test_num_classes_positive(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that num_classes must be positive."""
        with pytest.raises(ValueError, match="num_classes"):
            ConditionalVAEConfig(
                name="cvae",
                encoder=encoder_config,
                decoder=decoder_config,
                num_classes=-5,
            )

    def test_condition_dim_positive(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that condition_dim must be positive."""
        with pytest.raises(ValueError, match="condition_dim"):
            ConditionalVAEConfig(
                name="cvae",
                encoder=encoder_config,
                decoder=decoder_config,
                num_classes=10,
                condition_dim=-10,
            )

    def test_condition_type_valid(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that condition_type must be valid."""
        with pytest.raises(ValueError, match="condition_type"):
            ConditionalVAEConfig(
                name="cvae",
                encoder=encoder_config,
                decoder=decoder_config,
                num_classes=10,
                condition_type="invalid",
            )

    def test_concat_condition_type_valid(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that 'concat' is a valid condition type."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
            condition_type="concat",
        )
        assert config.condition_type == "concat"

    def test_add_condition_type_valid(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that 'add' is a valid condition type."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
            condition_type="add",
        )
        assert config.condition_type == "add"


class TestConditionalVAEConfigDefaults:
    """Test ConditionalVAEConfig default values."""

    def test_default_condition_dim(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default condition_dim value."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
        )
        assert config.condition_dim == 10  # Default to num_classes

    def test_default_condition_type(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default condition_type value."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
        )
        assert config.condition_type == "concat"


class TestConditionalVAEConfigSerialization:
    """Test ConditionalVAEConfig serialization."""

    def test_to_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test converting ConditionalVAEConfig to dictionary."""
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
            condition_dim=50,
        )
        config_dict = config.to_dict()
        assert config_dict["num_classes"] == 10
        assert config_dict["condition_dim"] == 50

    def test_from_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test creating ConditionalVAEConfig from dictionary."""
        config_dict = {
            "name": "cvae",
            "encoder": encoder_config.to_dict(),
            "decoder": decoder_config.to_dict(),
            "num_classes": 10,
            "condition_dim": 50,
            "condition_type": "add",
        }
        config = ConditionalVAEConfig.from_dict(config_dict)
        assert config.num_classes == 10
        assert config.condition_dim == 50
        assert config.condition_type == "add"

    def test_roundtrip_serialization(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test roundtrip serialization for ConditionalVAEConfig."""
        original = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
            condition_dim=50,
        )
        config_dict = original.to_dict()
        restored = ConditionalVAEConfig.from_dict(config_dict)
        assert restored.num_classes == original.num_classes
        assert restored.condition_dim == original.condition_dim


# =============================================================================
# VQVAEConfig Tests
# =============================================================================


class TestVQVAEConfigBasics:
    """Basic tests for VQVAEConfig."""

    def test_create_minimal_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating VQVAEConfig with minimal required parameters."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.name == "vqvae"
        assert isinstance(config, VAEConfig)

    def test_create_full_config(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test creating VQVAEConfig with all parameters."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_embeddings=1024,
            embedding_dim=128,
            commitment_cost=0.5,
        )
        assert config.num_embeddings == 1024
        assert config.embedding_dim == 128
        assert config.commitment_cost == 0.5

    def test_inherits_from_vae_config(self) -> None:
        """Test that VQVAEConfig inherits from VAEConfig."""
        assert issubclass(VQVAEConfig, VAEConfig)


class TestVQVAEConfigValidation:
    """Test VQVAEConfig validation rules."""

    def test_num_embeddings_positive(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that num_embeddings must be positive."""
        with pytest.raises(ValueError, match="num_embeddings"):
            VQVAEConfig(
                name="vqvae",
                encoder=encoder_config,
                decoder=decoder_config,
                num_embeddings=0,
            )

    def test_embedding_dim_positive(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that embedding_dim must be positive."""
        with pytest.raises(ValueError, match="embedding_dim"):
            VQVAEConfig(
                name="vqvae",
                encoder=encoder_config,
                decoder=decoder_config,
                embedding_dim=0,
            )

    def test_commitment_cost_non_negative(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that commitment_cost must be non-negative."""
        with pytest.raises(ValueError, match="commitment_cost"):
            VQVAEConfig(
                name="vqvae",
                encoder=encoder_config,
                decoder=decoder_config,
                commitment_cost=-0.1,
            )

    def test_commitment_cost_zero_allowed(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test that commitment_cost can be zero."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            commitment_cost=0.0,
        )
        assert config.commitment_cost == 0.0


class TestVQVAEConfigDefaults:
    """Test VQVAEConfig default values."""

    def test_default_num_embeddings(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default num_embeddings value."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.num_embeddings == 512

    def test_default_embedding_dim(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default embedding_dim value."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.embedding_dim == 64

    def test_default_commitment_cost(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test default commitment_cost value."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.commitment_cost == 0.25


class TestVQVAEConfigSerialization:
    """Test VQVAEConfig serialization."""

    def test_to_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test converting VQVAEConfig to dictionary."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_embeddings=1024,
            embedding_dim=128,
        )
        config_dict = config.to_dict()
        assert config_dict["num_embeddings"] == 1024
        assert config_dict["embedding_dim"] == 128

    def test_from_dict(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig) -> None:
        """Test creating VQVAEConfig from dictionary."""
        config_dict = {
            "name": "vqvae",
            "encoder": encoder_config.to_dict(),
            "decoder": decoder_config.to_dict(),
            "num_embeddings": 1024,
            "embedding_dim": 128,
            "commitment_cost": 0.5,
        }
        config = VQVAEConfig.from_dict(config_dict)
        assert config.num_embeddings == 1024
        assert config.embedding_dim == 128
        assert config.commitment_cost == 0.5

    def test_roundtrip_serialization(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test roundtrip serialization for VQVAEConfig."""
        original = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_embeddings=1024,
            embedding_dim=128,
        )
        config_dict = original.to_dict()
        restored = VQVAEConfig.from_dict(config_dict)
        assert restored.num_embeddings == original.num_embeddings
        assert restored.embedding_dim == original.embedding_dim


class TestVQVAEConfigEdgeCases:
    """Test VQVAEConfig edge cases."""

    def test_large_codebook(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test with large codebook size."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_embeddings=8192,
            embedding_dim=256,
        )
        assert config.num_embeddings == 8192
        assert config.embedding_dim == 256

    def test_small_codebook(
        self, encoder_config: EncoderConfig, decoder_config: DecoderConfig
    ) -> None:
        """Test with small codebook size."""
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_embeddings=16,
            embedding_dim=8,
        )
        assert config.num_embeddings == 16
        assert config.embedding_dim == 8
