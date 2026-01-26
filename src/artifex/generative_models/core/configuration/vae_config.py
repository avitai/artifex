"""VAE configuration using frozen dataclasses.

This module provides VAE configuration classes using frozen dataclasses with
nested network configurations for true plug-and-play architecture.
"""

import dataclasses

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.validation import (
    validate_non_negative_float,
    validate_non_negative_int,
    validate_positive_int,
)


@dataclasses.dataclass(frozen=True)
class VAEConfig(BaseConfig):
    """Base configuration for VAE models with nested network configs.

    This is a frozen dataclass that provides immutable configuration for VAEs
    with true plug-and-play architecture using nested EncoderConfig and
    DecoderConfig objects.

    Attributes:
        name: Name of the configuration
        description: Optional description
        encoder: EncoderConfig for the encoder network
        decoder: DecoderConfig for the decoder network
        encoder_type: Type of encoder/decoder architecture (dense, cnn, resnet)
        kl_weight: Weight for KL divergence term (default: 1.0)
        tags: Tuple of string tags
        metadata: Dictionary of arbitrary metadata

    Example:
        encoder = EncoderConfig(
            name="vae_encoder",
            input_shape=(28, 28, 1),
            latent_dim=32,
            hidden_dims=(256, 128),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="vae_decoder",
            latent_dim=32,
            output_shape=(28, 28, 1),
            hidden_dims=(128, 256),
            activation="relu",
        )
        config = VAEConfig(
            name="vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
            kl_weight=1.0,
        )
    """

    # Required nested network configurations
    encoder: EncoderConfig = dataclasses.field(default=None)  # type: ignore
    decoder: DecoderConfig = dataclasses.field(default=None)  # type: ignore

    # VAE-specific parameters
    encoder_type: str = "dense"
    kl_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If encoder or decoder have wrong type
        """
        # Call parent validation first
        super().__post_init__()

        # Validate required nested configs
        if self.encoder is None:
            raise ValueError("encoder config is required")
        if self.decoder is None:
            raise ValueError("decoder config is required")

        # Validate types
        if not isinstance(self.encoder, EncoderConfig):
            raise TypeError(f"encoder must be EncoderConfig, got {type(self.encoder).__name__}")
        if not isinstance(self.decoder, DecoderConfig):
            raise TypeError(f"decoder must be DecoderConfig, got {type(self.decoder).__name__}")

        # Validate latent_dim consistency between encoder and decoder
        if self.encoder.latent_dim != self.decoder.latent_dim:
            raise ValueError(
                f"encoder latent_dim ({self.encoder.latent_dim}) must match "
                f"decoder latent_dim ({self.decoder.latent_dim})"
            )

        # Validate encoder_type
        valid_encoder_types = {"dense", "cnn", "resnet"}
        if self.encoder_type not in valid_encoder_types:
            raise ValueError(
                f"encoder_type must be one of {valid_encoder_types}, got '{self.encoder_type}'"
            )

        # Validate kl_weight
        validate_non_negative_float(self.kl_weight, "kl_weight")

    @classmethod
    def from_dict(cls, data: dict) -> "VAEConfig":
        """Create VAEConfig from dictionary with proper nested config handling.

        Args:
            data: Dictionary representation of the config

        Returns:
            VAEConfig instance

        Raises:
            ValueError: If data is invalid
        """
        # Make a copy to avoid modifying the input
        data = dict(data)

        # Convert encoder field if it's a dict
        if "encoder" in data and isinstance(data["encoder"], dict):
            data["encoder"] = EncoderConfig.from_dict(data["encoder"])

        # Convert decoder field if it's a dict
        if "decoder" in data and isinstance(data["decoder"], dict):
            data["decoder"] = DecoderConfig.from_dict(data["decoder"])

        # Use parent from_dict for remaining fields
        return super(VAEConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class BetaVAEConfig(VAEConfig):
    """Configuration for Beta-VAE.

    Extends base VAEConfig with beta-specific training parameters for
    learning disentangled representations.

    Attributes:
        beta_default: Default weight for KL divergence term (default: 1.0)
        beta_warmup_steps: Steps for beta annealing, 0 for no annealing (default: 0)
        reconstruction_loss_type: Loss type 'mse' or 'bce' (default: "mse")

    Example:
        config = BetaVAEConfig(
            name="beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            beta_warmup_steps=1000,
            reconstruction_loss_type="mse",
        )
    """

    # BetaVAE-specific parameters
    beta_default: float = 1.0
    beta_warmup_steps: int = 0
    reconstruction_loss_type: str = "mse"

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate BetaVAE-specific parameters
        if self.beta_default <= 0:
            raise ValueError(f"beta_default must be positive, got {self.beta_default}")

        validate_non_negative_int(self.beta_warmup_steps, "beta_warmup_steps")

        # Validate reconstruction_loss_type
        valid_loss_types = {"mse", "bce"}
        if self.reconstruction_loss_type not in valid_loss_types:
            raise ValueError(
                f"reconstruction_loss_type must be one of {valid_loss_types}, "
                f"got '{self.reconstruction_loss_type}'"
            )


@dataclasses.dataclass(frozen=True)
class BetaVAEWithCapacityConfig(BetaVAEConfig):
    """Configuration for Beta-VAE with Burgess et al. capacity control.

    Extends BetaVAEConfig with capacity control parameters for progressive
    KL divergence capacity annealing during training.

    Attributes:
        use_capacity_control: Whether to use capacity control (default: False)
        capacity_max: Maximum capacity in nats (default: 25.0)
        capacity_num_iter: Steps to reach max capacity (default: 25000)
        gamma: Weight for capacity loss term (default: 1000.0)

    Example:
        config = BetaVAEWithCapacityConfig(
            name="beta_vae_capacity",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            use_capacity_control=True,
            capacity_max=25.0,
            capacity_num_iter=25000,
            gamma=1000.0,
        )
    """

    # Capacity control parameters
    use_capacity_control: bool = False
    capacity_max: float = 25.0
    capacity_num_iter: int = 25000
    gamma: float = 1000.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate capacity control parameters
        validate_non_negative_float(self.capacity_max, "capacity_max")
        validate_non_negative_int(self.capacity_num_iter, "capacity_num_iter")
        validate_non_negative_float(self.gamma, "gamma")


@dataclasses.dataclass(frozen=True)
class ConditionalVAEConfig(VAEConfig):
    """Configuration for Conditional VAE.

    Extends base VAEConfig with conditioning parameters for class-conditional generation.

    Attributes:
        num_classes: Number of classes for conditioning (required)
        condition_dim: Dimension of condition embedding (default: num_classes)
        condition_type: Type of conditioning - 'concat' or 'add' (default: "concat")

    Example:
        config = ConditionalVAEConfig(
            name="cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=10,
            condition_dim=50,
            condition_type="concat",
        )
    """

    # ConditionalVAE-specific parameters
    num_classes: int = 0  # Required, validated in __post_init__
    condition_dim: int = 0  # Default to num_classes if 0
    condition_type: str = "concat"

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate num_classes (required)
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        # Handle condition_dim default (defaults to num_classes)
        if self.condition_dim == 0:
            # Use object.__setattr__ to bypass frozen
            object.__setattr__(self, "condition_dim", self.num_classes)
        elif self.condition_dim < 0:
            raise ValueError(f"condition_dim must be positive, got {self.condition_dim}")

        # Validate condition_type
        valid_condition_types = {"concat", "add"}
        if self.condition_type not in valid_condition_types:
            raise ValueError(
                f"condition_type must be one of {valid_condition_types}, "
                f"got '{self.condition_type}'"
            )


@dataclasses.dataclass(frozen=True)
class VQVAEConfig(VAEConfig):
    """Configuration for Vector Quantized VAE (VQ-VAE).

    Extends base VAEConfig with vector quantization parameters.

    Attributes:
        num_embeddings: Size of the codebook (number of embeddings) (default: 512)
        embedding_dim: Dimension of each embedding vector (default: 64)
        commitment_cost: Weight for the commitment loss term (default: 0.25)

    Example:
        config = VQVAEConfig(
            name="vqvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_embeddings=512,
            embedding_dim=64,
            commitment_cost=0.25,
        )
    """

    # VQ-VAE-specific parameters
    num_embeddings: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate VQ-VAE-specific parameters
        validate_positive_int(self.num_embeddings, "num_embeddings")
        validate_positive_int(self.embedding_dim, "embedding_dim")
        validate_non_negative_float(self.commitment_cost, "commitment_cost")
