"""Diffusion model configuration using frozen dataclasses.

This module provides diffusion configuration classes using frozen dataclasses with
nested network configurations for the nested config architecture.

Configuration Hierarchy:
- NoiseScheduleConfig: Noise schedule parameters
- DiffusionConfig: Base diffusion config with nested backbone and schedule
  - Uses polymorphic BackboneConfig (UNetBackboneConfig, DiTBackboneConfig, etc.)
  - DDPMConfig: Denoising Diffusion Probabilistic Model (extends DiffusionConfig)
    - DDIMConfig: Denoising Diffusion Implicit Model (extends DDPMConfig)
  - ScoreDiffusionConfig: Score-based diffusion (extends DiffusionConfig)
  - LatentDiffusionConfig: Latent space diffusion with encoder/decoder (extends DiffusionConfig)
- DiTConfig: Diffusion Transformer (uses DiTBackboneConfig, extends BaseConfig)

Backbone System (Principle #4 compliant):
- BackboneConfig union type allows polymorphic backbone selection via config
- Each backbone type has its own config class with backbone_type discriminator
- create_backbone factory creates the appropriate backbone from config
"""

from __future__ import annotations

import dataclasses

from artifex.generative_models.core.configuration.backbone_config import (
    BackboneConfig,
    DiTBackboneConfig,
    UNet1DBackboneConfig,
    UNet2DConditionBackboneConfig,
    UNetBackboneConfig,
    UViTBackboneConfig,
)
from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.validation import (
    validate_non_negative_float,
    validate_positive_float,
    validate_positive_int,
    validate_positive_tuple,
)


@dataclasses.dataclass(frozen=True)
class NoiseScheduleConfig(BaseConfig):
    """Configuration for noise schedule in diffusion models.

    Defines the noise schedule parameters that control how noise is
    added and removed during the diffusion process.

    Attributes:
        name: Name of the configuration
        schedule_type: Type of schedule - 'linear', 'cosine', or 'quadratic'
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting value for noise schedule
        beta_end: Ending value for noise schedule
        clip_min: Minimum clipping value for numerical stability

    Example:
        config = NoiseScheduleConfig(
            name="cosine_schedule",
            schedule_type="cosine",
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
        )
    """

    # Schedule parameters with defaults
    schedule_type: str = "linear"
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    clip_min: float = 1e-20

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate schedule_type
        valid_schedule_types = {"linear", "cosine", "quadratic", "sqrt"}
        if self.schedule_type not in valid_schedule_types:
            raise ValueError(
                f"schedule_type must be one of {valid_schedule_types}, got '{self.schedule_type}'"
            )

        # Validate num_timesteps
        validate_positive_int(self.num_timesteps, "num_timesteps")

        # Validate beta values
        validate_positive_float(self.beta_start, "beta_start")
        validate_positive_float(self.beta_end, "beta_end")

        if self.beta_start >= self.beta_end:
            raise ValueError(
                f"beta_start must be less than beta_end, "
                f"got beta_start={self.beta_start}, beta_end={self.beta_end}"
            )

        # Validate clip_min
        validate_non_negative_float(self.clip_min, "clip_min")


@dataclasses.dataclass(frozen=True)
class DiffusionConfig(BaseConfig):
    """Base configuration for diffusion models with nested configs.

    This is a frozen dataclass that provides immutable configuration for
    diffusion models with nested backbone and NoiseScheduleConfig objects.

    Follows Principle #4: Methods take configs, NOT individual parameters.
    The backbone field accepts any BackboneConfig type (polymorphic):
    - UNetBackboneConfig
    - DiTBackboneConfig
    - UViTBackboneConfig
    - UNet2DConditionBackboneConfig

    Attributes:
        name: Name of the configuration
        backbone: Backbone config (polymorphic - any BackboneConfig type)
        noise_schedule: NoiseScheduleConfig for the noise schedule
        input_shape: Shape of input data (H, W, C) - JAX convention

    Example:
        from artifex.generative_models.core.configuration import (
            UNetBackboneConfig,
            NoiseScheduleConfig,
            DiffusionConfig,
        )

        backbone = UNetBackboneConfig(
            name="unet",
            hidden_dims=(64, 128, 256),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        schedule = NoiseScheduleConfig(name="schedule")
        config = DiffusionConfig(
            name="diffusion",
            backbone=backbone,
            noise_schedule=schedule,
            input_shape=(32, 32, 3),
        )
    """

    # Required nested configurations
    # backbone must be a BackboneConfig type with backbone_type discriminator
    backbone: BackboneConfig = dataclasses.field(default=None)  # type: ignore
    noise_schedule: NoiseScheduleConfig = dataclasses.field(default=None)  # type: ignore

    # Diffusion-specific parameters
    input_shape: tuple[int, ...] = (32, 32, 3)  # (H, W, C) - JAX convention

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If nested configs have wrong type
        """
        super().__post_init__()

        # Validate required nested configs
        if self.backbone is None:
            raise ValueError("backbone config is required")
        if self.noise_schedule is None:
            raise ValueError("noise_schedule config is required")

        # Validate backbone type - must be a BackboneConfig with backbone_type
        valid_backbone_types = (
            UNetBackboneConfig,
            DiTBackboneConfig,
            UViTBackboneConfig,
            UNet2DConditionBackboneConfig,
            UNet1DBackboneConfig,
        )
        if not isinstance(self.backbone, valid_backbone_types):
            raise TypeError(
                f"backbone must be a BackboneConfig type "
                f"(UNetBackboneConfig, DiTBackboneConfig, UViTBackboneConfig, "
                f"UNet2DConditionBackboneConfig, UNet1DBackboneConfig), "
                f"got {type(self.backbone).__name__}"
            )

        if not isinstance(self.noise_schedule, NoiseScheduleConfig):
            raise TypeError(
                f"noise_schedule must be NoiseScheduleConfig, "
                f"got {type(self.noise_schedule).__name__}"
            )

        # Validate input_shape
        if len(self.input_shape) == 0:
            raise ValueError("input_shape cannot be empty")
        validate_positive_tuple(self.input_shape, "input_shape")

    @classmethod
    def from_dict(cls, data: dict) -> "DiffusionConfig":
        """Create DiffusionConfig from dictionary with proper nested config handling.

        Uses backbone_type discriminator to determine which backbone config to create.

        Args:
            data: Dictionary representation of the config

        Returns:
            DiffusionConfig instance

        Raises:
            ValueError: If backbone_type is missing or invalid
        """
        data = dict(data)

        # Convert backbone field if it's a dict
        if "backbone" in data and isinstance(data["backbone"], dict):
            backbone_data = data["backbone"]
            backbone_type = backbone_data.get("backbone_type")

            if backbone_type is None:
                raise ValueError(
                    "backbone dict must have 'backbone_type' field "
                    "(one of: 'unet', 'dit', 'uvit', 'unet2d_condition', 'unet_1d')"
                )

            # Dispatch based on backbone_type discriminator
            if backbone_type == "unet":
                data["backbone"] = UNetBackboneConfig.from_dict(backbone_data)
            elif backbone_type == "dit":
                data["backbone"] = DiTBackboneConfig.from_dict(backbone_data)
            elif backbone_type == "uvit":
                data["backbone"] = UViTBackboneConfig.from_dict(backbone_data)
            elif backbone_type == "unet2d_condition":
                data["backbone"] = UNet2DConditionBackboneConfig.from_dict(backbone_data)
            elif backbone_type == "unet_1d":
                data["backbone"] = UNet1DBackboneConfig.from_dict(backbone_data)
            else:
                raise ValueError(
                    f"Unknown backbone_type: '{backbone_type}'. "
                    f"Must be one of: 'unet', 'dit', 'uvit', 'unet2d_condition', 'unet_1d'"
                )

        # Convert noise_schedule field if it's a dict
        if "noise_schedule" in data and isinstance(data["noise_schedule"], dict):
            data["noise_schedule"] = NoiseScheduleConfig.from_dict(data["noise_schedule"])

        return super(DiffusionConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class ConditionalDiffusionConfig(DiffusionConfig):
    """Configuration for conditional diffusion models.

    Extends base DiffusionConfig with conditioning-specific parameters.
    Used with ConditionalDiffusionMixin for models that support conditioning.

    Follows Principle #4: Methods take configs, NOT individual parameters.

    Attributes:
        conditioning_dim: Dimension of conditioning information (e.g., class embeddings)

    Example:
        config = ConditionalDiffusionConfig(
            name="conditional_diffusion",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            conditioning_dim=128,
        )
    """

    # Conditioning-specific parameters
    conditioning_dim: int = 0  # Dummy default for field ordering

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        super().__post_init__()

        # Validate conditioning_dim is positive
        validate_positive_int(self.conditioning_dim, "conditioning_dim")


@dataclasses.dataclass(frozen=True)
class DDPMConfig(DiffusionConfig):
    """Configuration for Denoising Diffusion Probabilistic Model (DDPM).

    Extends base DiffusionConfig with DDPM-specific parameters.

    Attributes:
        loss_type: Loss function type - 'mse', 'l1', or 'huber'
        clip_denoised: Whether to clip denoised samples to [-1, 1]

    Example:
        config = DDPMConfig(
            name="ddpm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            loss_type="mse",
            clip_denoised=True,
        )
    """

    # DDPM-specific parameters
    loss_type: str = "mse"
    clip_denoised: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate loss_type
        valid_loss_types = {"mse", "l1", "huber"}
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'")


@dataclasses.dataclass(frozen=True)
class DDIMConfig(DDPMConfig):
    """Configuration for Denoising Diffusion Implicit Model (DDIM).

    Extends DDPMConfig with DDIM-specific sampling parameters.
    DDIMModel inherits from DDPMModel, so DDIMConfig inherits from DDPMConfig.

    Inherits from DDPMConfig:
        loss_type: Loss function type - 'mse', 'l1', or 'huber'
        clip_denoised: Whether to clip denoised samples to [-1, 1]

    Attributes:
        eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        num_inference_steps: Number of sampling steps
        skip_type: Timestep skip strategy - 'uniform' or 'quadratic'

    Example:
        config = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            loss_type="mse",
            clip_denoised=True,
            eta=0.0,  # Deterministic
            num_inference_steps=50,
        )
    """

    # DDIM-specific parameters
    eta: float = 0.0
    num_inference_steps: int = 50
    skip_type: str = "uniform"

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate eta (must be in [0, 1])
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"eta must be in range [0, 1], got {self.eta}")

        # Validate num_inference_steps
        validate_positive_int(self.num_inference_steps, "num_inference_steps")

        # Validate skip_type
        valid_skip_types = {"uniform", "quadratic"}
        if self.skip_type not in valid_skip_types:
            raise ValueError(f"skip_type must be one of {valid_skip_types}, got '{self.skip_type}'")


@dataclasses.dataclass(frozen=True)
class ScoreDiffusionConfig(DiffusionConfig):
    """Configuration for Score-based Diffusion Model.

    Extends base DiffusionConfig with score-matching specific parameters.

    Attributes:
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        score_scaling: Scaling factor for score function

    Example:
        config = ScoreDiffusionConfig(
            name="score",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            sigma_min=0.01,
            sigma_max=50.0,
        )
    """

    # Score-based diffusion parameters
    sigma_min: float = 0.01
    sigma_max: float = 50.0
    score_scaling: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate sigma values
        validate_positive_float(self.sigma_min, "sigma_min")
        validate_positive_float(self.sigma_max, "sigma_max")

        if self.sigma_min >= self.sigma_max:
            raise ValueError(
                f"sigma_min must be less than sigma_max, "
                f"got sigma_min={self.sigma_min}, sigma_max={self.sigma_max}"
            )

        # Validate score_scaling
        validate_positive_float(self.score_scaling, "score_scaling")


@dataclasses.dataclass(frozen=True)
class LatentDiffusionConfig(DDPMConfig):
    """Configuration for Latent Diffusion Model.

    Extends DDPMConfig with encoder/decoder for latent space diffusion.
    LDMModel inherits from DDPMModel, so LatentDiffusionConfig inherits from DDPMConfig.

    Attributes:
        encoder: EncoderConfig for compressing data to latent space
        decoder: DecoderConfig for decompressing from latent space
        latent_scale_factor: Scaling factor for latent codes

    Example:
        config = LatentDiffusionConfig(
            name="ldm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            encoder=encoder_config,
            decoder=decoder_config,
            latent_scale_factor=0.18215,
        )
    """

    # Latent diffusion specific nested configs
    encoder: EncoderConfig = dataclasses.field(default=None)  # type: ignore
    decoder: DecoderConfig = dataclasses.field(default=None)  # type: ignore

    # Latent diffusion parameters
    latent_scale_factor: float = 0.18215

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If encoder or decoder have wrong type
        """
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

        # Validate latent_dim consistency
        if self.encoder.latent_dim != self.decoder.latent_dim:
            raise ValueError(
                f"encoder latent_dim ({self.encoder.latent_dim}) must match "
                f"decoder latent_dim ({self.decoder.latent_dim})"
            )

        # Validate latent_scale_factor
        validate_positive_float(self.latent_scale_factor, "latent_scale_factor")

    @classmethod
    def from_dict(cls, data: dict) -> "LatentDiffusionConfig":
        """Create LatentDiffusionConfig from dictionary.

        Args:
            data: Dictionary representation of the config

        Returns:
            LatentDiffusionConfig instance
        """
        data = dict(data)

        # Convert encoder field if it's a dict
        if "encoder" in data and isinstance(data["encoder"], dict):
            data["encoder"] = EncoderConfig.from_dict(data["encoder"])

        # Convert decoder field if it's a dict
        if "decoder" in data and isinstance(data["decoder"], dict):
            data["decoder"] = DecoderConfig.from_dict(data["decoder"])

        # Let parent handle backbone and noise_schedule
        return super(LatentDiffusionConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class DiTConfig(BaseConfig):
    """Configuration for Diffusion Transformer (DiT).

    DiT uses a transformer architecture instead of UNet, so it has
    different configuration parameters. It still uses the noise schedule.

    Attributes:
        name: Name of the configuration
        noise_schedule: NoiseScheduleConfig for the noise schedule
        patch_size: Size of image patches for patchification
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = hidden_size * mlp_ratio
        learn_sigma: Whether to learn variance in addition to noise
        num_classes: Number of classes for conditional generation (None for unconditional)
        cfg_scale: Classifier-free guidance scale
        input_shape: Shape of input images

    Example:
        config = DiTConfig(
            name="dit_b",
            noise_schedule=schedule_config,
            patch_size=2,
            hidden_size=768,
            depth=12,
            num_heads=12,
            num_classes=1000,
        )
    """

    # Required nested config
    noise_schedule: NoiseScheduleConfig = dataclasses.field(default=None)  # type: ignore

    # DiT architecture parameters
    patch_size: int = 2
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    learn_sigma: bool = False

    # Conditional generation parameters
    num_classes: int | None = None
    cfg_scale: float = 1.0

    # Input shape
    input_shape: tuple[int, ...] = (3, 32, 32)

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If noise_schedule has wrong type
        """
        super().__post_init__()

        # Validate noise_schedule
        if self.noise_schedule is None:
            raise ValueError("noise_schedule config is required")
        if not isinstance(self.noise_schedule, NoiseScheduleConfig):
            raise TypeError(
                f"noise_schedule must be NoiseScheduleConfig, "
                f"got {type(self.noise_schedule).__name__}"
            )

        # Validate DiT architecture parameters
        validate_positive_int(self.patch_size, "patch_size")
        validate_positive_int(self.hidden_size, "hidden_size")
        validate_positive_int(self.depth, "depth")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_float(self.mlp_ratio, "mlp_ratio")

        # Validate num_classes (must be positive if specified)
        if self.num_classes is not None:
            validate_positive_int(self.num_classes, "num_classes")

        # Validate cfg_scale
        validate_non_negative_float(self.cfg_scale, "cfg_scale")

        # Validate hidden_size divisibility by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, "
                f"got hidden_size={self.hidden_size}, num_heads={self.num_heads}"
            )

        # Validate input_shape
        if len(self.input_shape) == 0:
            raise ValueError("input_shape cannot be empty")
        validate_positive_tuple(self.input_shape, "input_shape")

    @classmethod
    def from_dict(cls, data: dict) -> "DiTConfig":
        """Create DiTConfig from dictionary.

        Args:
            data: Dictionary representation of the config

        Returns:
            DiTConfig instance
        """
        data = dict(data)

        # Convert noise_schedule field if it's a dict
        if "noise_schedule" in data and isinstance(data["noise_schedule"], dict):
            data["noise_schedule"] = NoiseScheduleConfig.from_dict(data["noise_schedule"])

        return super(DiTConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class StableDiffusionConfig(LatentDiffusionConfig):
    """Configuration for Stable Diffusion Model.

    Extends LatentDiffusionConfig with text conditioning parameters.
    Stable Diffusion operates in latent space with text-based guidance.

    Attributes:
        text_embedding_dim: Dimension of text embeddings
        text_max_length: Maximum text sequence length
        vocab_size: Size of text vocabulary
        guidance_scale: Classifier-free guidance scale
        use_guidance: Whether to use classifier-free guidance

    Example:
        config = StableDiffusionConfig(
            name="stable_diffusion",
            backbone=unet_config,
            noise_schedule=schedule_config,
            encoder=encoder_config,
            decoder=decoder_config,
            input_shape=(32, 32, 3),
            text_embedding_dim=128,
            vocab_size=1000,
        )
    """

    # Text conditioning parameters
    text_embedding_dim: int = 512
    text_max_length: int = 77
    vocab_size: int = 10000

    # Guidance parameters
    guidance_scale: float = 7.5
    use_guidance: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate text parameters
        validate_positive_int(self.text_embedding_dim, "text_embedding_dim")
        validate_positive_int(self.text_max_length, "text_max_length")
        validate_positive_int(self.vocab_size, "vocab_size")

        # Validate guidance scale
        validate_non_negative_float(self.guidance_scale, "guidance_scale")
