"""GAN configuration using frozen dataclasses.

This module provides GAN configuration classes using frozen dataclasses with
nested network configurations for true plug-and-play architecture.
"""

import dataclasses

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    DiscriminatorConfig,
    GeneratorConfig,
)
from artifex.generative_models.core.configuration.validation import (
    validate_non_negative_float,
    validate_positive_float,
    validate_positive_int,
)


@dataclasses.dataclass(frozen=True)
class GANConfig(BaseConfig):
    """Base configuration for GAN models with nested network configs.

    This is a frozen dataclass that provides immutable configuration for GANs
    with true plug-and-play architecture using nested GeneratorConfig and
    DiscriminatorConfig objects.

    Supports both single-generator/single-discriminator GANs (vanilla, WGAN, LSGAN, etc.)
    and multi-network GANs (CycleGAN with 2 generators + 2 discriminators).

    Attributes:
        name: Name of the configuration
        description: Optional description
        generator: Single GeneratorConfig OR dict of {name: GeneratorConfig}
            for multi-generator GANs
        discriminator: Single DiscriminatorConfig OR
            dict of {name: DiscriminatorConfig} for multi-discriminator GANs
        generator_lr: Generator learning rate (must be positive)
        discriminator_lr: Discriminator learning rate (must be positive)
        beta1: Adam beta1 parameter (must be in [0, 1))
        beta2: Adam beta2 parameter (must be in [0, 1))
        loss_type: Type of GAN loss (vanilla, wasserstein, least_squares, hinge)
        gradient_penalty_weight: Weight for gradient penalty (must be non-negative)
        tags: Tuple of string tags
        metadata: Dictionary of arbitrary metadata
        rngs_seeds: Dictionary mapping RNG stream names to seed values

    Example (Single GAN):
        config = GANConfig(
            name="vanilla_gan",
            generator=GeneratorConfig(...),
            discriminator=DiscriminatorConfig(...),
        )

    Example (Multi-network GAN like CycleGAN):
        config = CycleGANConfig(
            name="cyclegan",
            generator={
                "a_to_b": GeneratorConfig(...),
                "b_to_a": GeneratorConfig(...),
            },
            discriminator={
                "disc_a": DiscriminatorConfig(...),
                "disc_b": DiscriminatorConfig(...),
            },
        )
    """

    # Required nested network configurations - can be single or dict
    generator: GeneratorConfig | dict[str, GeneratorConfig] = dataclasses.field(default=None)  # type: ignore
    discriminator: DiscriminatorConfig | dict[str, DiscriminatorConfig] = dataclasses.field(
        default=None
    )  # type: ignore

    # Orchestration parameters with defaults
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    loss_type: str = "vanilla"
    gradient_penalty_weight: float = 0.0
    rngs_seeds: dict[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If generator or discriminator are not provided or have wrong type
        """
        # Call parent validation first
        super().__post_init__()

        # Validate required nested configs
        if self.generator is None:
            raise ValueError("generator config is required")
        if self.discriminator is None:
            raise ValueError("discriminator config is required")

        # Validate types - can be single config or dict of configs
        if isinstance(self.generator, dict):
            if not self.generator:
                raise ValueError("generator dict cannot be empty")
            for name, gen in self.generator.items():
                if not isinstance(gen, GeneratorConfig):
                    raise TypeError(
                        f"generator['{name}'] must be GeneratorConfig, got {type(gen).__name__}"
                    )
        elif not isinstance(self.generator, GeneratorConfig):
            raise TypeError(
                f"generator must be GeneratorConfig or dict[str, GeneratorConfig], "
                f"got {type(self.generator).__name__}"
            )

        if isinstance(self.discriminator, dict):
            if not self.discriminator:
                raise ValueError("discriminator dict cannot be empty")
            for name, disc in self.discriminator.items():
                if not isinstance(disc, DiscriminatorConfig):
                    raise TypeError(
                        f"discriminator['{name}'] must be DiscriminatorConfig, "
                        f"got {type(disc).__name__}"
                    )
        elif not isinstance(self.discriminator, DiscriminatorConfig):
            raise TypeError(
                f"discriminator must be DiscriminatorConfig or dict[str, DiscriminatorConfig], "
                f"got {type(self.discriminator).__name__}"
            )

        # Validate learning rates
        validate_positive_float(self.generator_lr, "generator_lr")
        validate_positive_float(self.discriminator_lr, "discriminator_lr")

        # Validate beta parameters (must be in [0, 1))
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not (0.0 <= self.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")

        # Validate gradient penalty weight
        validate_non_negative_float(self.gradient_penalty_weight, "gradient_penalty_weight")

        # Validate loss type
        valid_loss_types = {"vanilla", "wasserstein", "least_squares", "hinge"}
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'")

    @classmethod
    def from_dict(cls, data: dict) -> "GANConfig":
        """Create GANConfig from dictionary with proper nested config handling.

        This method properly handles the Union[GeneratorConfig, dict[str, GeneratorConfig]]
        type by converting nested dicts to GeneratorConfig/DiscriminatorConfig objects.

        Args:
            data: Dictionary representation of the config

        Returns:
            GANConfig instance

        Raises:
            ValueError: If data is invalid
        """
        # Make a copy to avoid modifying the input
        data = dict(data)

        # Convert generator field
        if "generator" in data and isinstance(data["generator"], dict):
            gen_data = data["generator"]
            # Check if it's a dict of generators (multi-network) or a single generator dict
            if gen_data and all(isinstance(v, dict) for v in gen_data.values()):
                # Could be either:
                # 1. Single generator: {"name": "gen", "latent_dim": 100, ...}
                # 2. Multi-generator: {"gen_a": {...}, "gen_b": {...}}
                # Check if it looks like a single config (has "name", "latent_dim", etc.)
                if "latent_dim" in gen_data or "output_shape" in gen_data:
                    # Single generator config as dict
                    data["generator"] = GeneratorConfig.from_dict(gen_data)
                else:
                    # Multi-generator: dict of {name: config_dict}
                    data["generator"] = {
                        name: GeneratorConfig.from_dict(cfg) for name, cfg in gen_data.items()
                    }
            else:
                # It's a simple dict representing a single generator
                data["generator"] = GeneratorConfig.from_dict(gen_data)

        # Convert discriminator field
        if "discriminator" in data and isinstance(data["discriminator"], dict):
            disc_data = data["discriminator"]
            # Check if it's a dict of discriminators (multi-network) or a single discriminator dict
            if disc_data and all(isinstance(v, dict) for v in disc_data.values()):
                # Could be either:
                # 1. Single discriminator: {"name": "disc", "input_shape": [64, 64, 3], ...}
                # 2. Multi-discriminator: {"disc_a": {...}, "disc_b": {...}}
                # Check if it looks like a single config (has "name", "input_shape", etc.)
                if "input_shape" in disc_data or "hidden_dims" in disc_data:
                    # Single discriminator config as dict
                    data["discriminator"] = DiscriminatorConfig.from_dict(disc_data)
                else:
                    # Multi-discriminator: dict of {name: config_dict}
                    data["discriminator"] = {
                        name: DiscriminatorConfig.from_dict(cfg) for name, cfg in disc_data.items()
                    }
            else:
                # It's a simple dict representing a single discriminator
                data["discriminator"] = DiscriminatorConfig.from_dict(disc_data)

        # Use parent from_dict for remaining fields
        return super(GANConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class WGANConfig(GANConfig):
    """Configuration for Wasserstein GAN (WGAN).

    Extends base GANConfig with WGAN-specific training parameters.
    Requires ConvGeneratorConfig and ConvDiscriminatorConfig since WGAN
    uses convolutional architecture.

    Attributes:
        critic_iterations: Number of critic updates per generator update (default: 5)
        use_gradient_penalty: Whether to use gradient penalty for WGAN-GP (default: True)

    Example:
        generator = ConvGeneratorConfig(
            name="wgan_generator",
            latent_dim=100,
            hidden_dims=(512, 256, 128, 64),
            output_shape=(3, 64, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="wgan_critic",
            hidden_dims=(64, 128, 256, 512),
            input_shape=(3, 64, 64),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )

        config = WGANConfig(
            name="wgan",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=5,
            use_gradient_penalty=True,
        )
    """

    # WGAN-specific training parameters
    critic_iterations: int = 5
    use_gradient_penalty: bool = True

    # Override parent defaults for WGAN-specific values
    loss_type: str = "wasserstein"
    gradient_penalty_weight: float = 10.0
    generator_lr: float = 0.0001
    discriminator_lr: float = 0.0001
    beta1: float = 0.0
    beta2: float = 0.9

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If generator or discriminator have wrong type
        """
        # Call parent validation first
        super().__post_init__()

        # WGAN requires ConvGeneratorConfig and ConvDiscriminatorConfig
        if not isinstance(self.generator, ConvGeneratorConfig):
            raise TypeError(
                f"WGANConfig requires ConvGeneratorConfig, got {type(self.generator).__name__}"
            )
        if not isinstance(self.discriminator, ConvDiscriminatorConfig):
            raise TypeError(
                f"WGANConfig requires ConvDiscriminatorConfig, "
                f"got {type(self.discriminator).__name__}"
            )

        # Validate WGAN-specific parameters
        validate_positive_int(self.critic_iterations, "critic_iterations")

    @classmethod
    def from_dict(cls, data: dict) -> "WGANConfig":
        """Create WGANConfig from dictionary.

        Converts generator/discriminator dicts to ConvGeneratorConfig/ConvDiscriminatorConfig.

        Args:
            data: Dictionary representation of the config

        Returns:
            WGANConfig instance
        """
        data = dict(data)

        # Convert generator to ConvGeneratorConfig
        if "generator" in data and isinstance(data["generator"], dict):
            data["generator"] = ConvGeneratorConfig.from_dict(data["generator"])

        # Convert discriminator to ConvDiscriminatorConfig
        if "discriminator" in data and isinstance(data["discriminator"], dict):
            data["discriminator"] = ConvDiscriminatorConfig.from_dict(data["discriminator"])

        return super(WGANConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class LSGANConfig(GANConfig):
    """Configuration for Least Squares GAN (LSGAN).

    Extends base GANConfig with LSGAN-specific loss parameters.
    Requires ConvGeneratorConfig and ConvDiscriminatorConfig since LSGAN
    uses convolutional architecture.

    Attributes:
        a: Target value for fake samples in discriminator loss (default: 0.0)
        b: Target value for real samples in discriminator loss (default: 1.0)
        c: Target value for fake samples in generator loss (default: 1.0)

    Note:
        Common LSGAN configurations:
        - Standard: a=0, b=1, c=1
        - Alternative: a=-1, b=1, c=0

    Example:
        generator = ConvGeneratorConfig(
            name="lsgan_generator",
            latent_dim=100,
            hidden_dims=(512, 256, 128, 64),
            output_shape=(3, 64, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="lsgan_discriminator",
            hidden_dims=(64, 128, 256, 512),
            input_shape=(3, 64, 64),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = LSGANConfig(
            name="lsgan",
            generator=generator,
            discriminator=discriminator,
            a=0.0,
            b=1.0,
            c=1.0,
        )
    """

    # LSGAN-specific loss parameters
    a: float = 0.0  # Target for fake samples in discriminator
    b: float = 1.0  # Target for real samples in discriminator
    c: float = 1.0  # Target for fake samples in generator

    # Override parent loss_type for LSGAN
    loss_type: str = "least_squares"

    # Note: a, b, c values are not validated as they can be any real numbers
    # according to the LSGAN paper

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            TypeError: If generator or discriminator have wrong type
        """
        # Call parent validation first
        super().__post_init__()

        # LSGAN requires ConvGeneratorConfig and ConvDiscriminatorConfig
        if not isinstance(self.generator, ConvGeneratorConfig):
            raise TypeError(
                f"LSGANConfig requires ConvGeneratorConfig, got {type(self.generator).__name__}"
            )
        if not isinstance(self.discriminator, ConvDiscriminatorConfig):
            raise TypeError(
                f"LSGANConfig requires ConvDiscriminatorConfig, "
                f"got {type(self.discriminator).__name__}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "LSGANConfig":
        """Create LSGANConfig from dictionary.

        Converts generator/discriminator dicts to ConvGeneratorConfig/ConvDiscriminatorConfig.

        Args:
            data: Dictionary representation of the config

        Returns:
            LSGANConfig instance
        """
        data = dict(data)

        # Convert generator to ConvGeneratorConfig
        if "generator" in data and isinstance(data["generator"], dict):
            data["generator"] = ConvGeneratorConfig.from_dict(data["generator"])

        # Convert discriminator to ConvDiscriminatorConfig
        if "discriminator" in data and isinstance(data["discriminator"], dict):
            data["discriminator"] = ConvDiscriminatorConfig.from_dict(data["discriminator"])

        return super(LSGANConfig, cls).from_dict(data)


@dataclasses.dataclass(frozen=True)
class ConditionalGANConfig(GANConfig):
    """Configuration for Conditional GAN.

    Extends base GANConfig for class-conditional generation. Uses composition pattern
    where conditional parameters (num_classes, embedding_dim) are embedded in the
    nested ConditionalGeneratorConfig and ConditionalDiscriminatorConfig via
    ConditionalParams.

    The generator and discriminator configs must be ConditionalGeneratorConfig and
    ConditionalDiscriminatorConfig respectively, which contain the conditional
    parameters via composition (config.conditional.num_classes, etc.).

    Example:
        from artifex.generative_models.core.configuration.network_configs import (
            ConditionalParams,
            ConditionalGeneratorConfig,
            ConditionalDiscriminatorConfig,
        )

        # Create conditional params (reusable component)
        cond_params = ConditionalParams(num_classes=10, embedding_dim=50)

        generator = ConditionalGeneratorConfig(
            name="cgan_generator",
            latent_dim=100,
            hidden_dims=(512, 256, 128, 64),
            output_shape=(1, 28, 28),
            activation="relu",
            batch_norm=True,
            conditional=cond_params,  # Composition!
        )

        discriminator = ConditionalDiscriminatorConfig(
            name="cgan_discriminator",
            hidden_dims=(64, 128, 256, 512),
            input_shape=(1, 28, 28),
            activation="leaky_relu",
            conditional=cond_params,  # Same params for consistency
        )

        config = ConditionalGANConfig(
            name="conditional_gan",
            generator=generator,
            discriminator=discriminator,
        )
    """

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
            TypeError: If generator or discriminator have wrong type
        """
        # Call parent validation first
        super().__post_init__()

        # Conditional GAN requires single generator and discriminator (not dict)
        if isinstance(self.generator, dict):
            raise TypeError(
                "ConditionalGANConfig requires a single ConditionalGeneratorConfig, "
                "not dict of generators. Use GANConfig base class for multi-network GANs."
            )
        if isinstance(self.discriminator, dict):
            raise TypeError(
                "ConditionalGANConfig requires a single ConditionalDiscriminatorConfig, "
                "not dict of discriminators. Use GANConfig base class for multi-network GANs."
            )

        # Validate that configs are the conditional types
        from artifex.generative_models.core.configuration.network_configs import (
            ConditionalDiscriminatorConfig,
            ConditionalGeneratorConfig,
        )

        if not isinstance(self.generator, ConditionalGeneratorConfig):
            raise TypeError(
                f"ConditionalGANConfig requires ConditionalGeneratorConfig, "
                f"got {type(self.generator).__name__}"
            )
        if not isinstance(self.discriminator, ConditionalDiscriminatorConfig):
            raise TypeError(
                f"ConditionalGANConfig requires ConditionalDiscriminatorConfig, "
                f"got {type(self.discriminator).__name__}"
            )


@dataclasses.dataclass(frozen=True)
class CycleGANConfig(GANConfig):
    """Configuration for CycleGAN (Cycle-Consistent Adversarial Networks).

    Extends base GANConfig for unpaired image-to-image translation with cycle consistency.

    CycleGAN requires exactly TWO generators and TWO discriminators:
    - Generator A→B: Translates from domain A to domain B
    - Generator B→A: Translates from domain B to domain A
    - Discriminator A: Distinguishes real/fake domain A images
    - Discriminator B: Distinguishes real/fake domain B images

    All architectural parameters (n_residual_blocks, activation, etc.) should be
    specified in the nested GeneratorConfig and DiscriminatorConfig objects.

    Attributes:
        generator: Dict with exactly 2 generators
            {"a_to_b": GeneratorConfig, "b_to_a": GeneratorConfig}
        discriminator: Dict with exactly 2 discriminators
            {"disc_a": DiscriminatorConfig, "disc_b": DiscriminatorConfig}
        input_shape_a: Shape of domain A images (height, width, channels)
        input_shape_b: Shape of domain B images (height, width, channels)
        lambda_cycle: Weight for cycle consistency loss (default: 10.0)
        lambda_identity: Weight for identity loss (default: 0.5)

    Example:
        config = CycleGANConfig(
            name="cyclegan_horse2zebra",
            generator={
                "a_to_b": CycleGANGeneratorConfig(
                    name="horse_to_zebra",
                    latent_dim=0,  # Not used for image-to-image
                    hidden_dims=(64, 128, 256),
                    output_shape=(256, 256, 3),
                    input_shape=(256, 256, 3),
                    n_residual_blocks=6,
                    activation="relu",
                ),
                "b_to_a": CycleGANGeneratorConfig(
                    name="zebra_to_horse",
                    latent_dim=0,  # Not used for image-to-image
                    hidden_dims=(64, 128, 256),
                    output_shape=(256, 256, 3),
                    input_shape=(256, 256, 3),
                    n_residual_blocks=6,
                    activation="relu",
                ),
            },
            discriminator={
                "disc_a": PatchGANDiscriminatorConfig(
                    name="horse_discriminator",
                    hidden_dims=(64, 128, 256, 512),
                    input_shape=(256, 256, 3),
                    activation="leaky_relu",
                ),
                "disc_b": PatchGANDiscriminatorConfig(
                    name="zebra_discriminator",
                    hidden_dims=(64, 128, 256, 512),
                    input_shape=(256, 256, 3),
                    activation="leaky_relu",
                ),
            },
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
            lambda_cycle=10.0,
            lambda_identity=0.5,
        )
    """

    # Domain shapes (dummy defaults, validated in __post_init__)
    input_shape_a: tuple[int, int, int] = (0, 0, 0)
    input_shape_b: tuple[int, int, int] = (0, 0, 0)

    # CycleGAN-specific loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        # Call parent validation
        super().__post_init__()

        # CycleGAN MUST have dict of generators and discriminators
        if not isinstance(self.generator, dict):
            raise TypeError(
                "CycleGAN requires generator to be a dict with 2 generators "
                f"(a_to_b, b_to_a), got {type(self.generator).__name__}"
            )
        if not isinstance(self.discriminator, dict):
            raise TypeError(
                "CycleGAN requires discriminator to be a dict with 2 "
                f"discriminators (disc_a, disc_b), got {type(self.discriminator).__name__}"
            )

        # Validate exactly 2 generators
        if len(self.generator) != 2:
            raise ValueError(
                f"CycleGAN requires exactly 2 generators (a_to_b, b_to_a), "
                f"got {len(self.generator)}"
            )

        # Validate exactly 2 discriminators
        if len(self.discriminator) != 2:
            raise ValueError(
                f"CycleGAN requires exactly 2 discriminators (disc_a, disc_b), "
                f"got {len(self.discriminator)}"
            )

        # Validate input shapes
        if len(self.input_shape_a) != 3:
            raise ValueError(
                "input_shape_a must be a tuple of 3 positive integers (height, width, channels)"
            )
        if not all(dim > 0 for dim in self.input_shape_a):
            raise ValueError(
                "input_shape_a must be a tuple of 3 positive integers (height, width, channels)"
            )

        if len(self.input_shape_b) != 3:
            raise ValueError(
                "input_shape_b must be a tuple of 3 positive integers (height, width, channels)"
            )
        if not all(dim > 0 for dim in self.input_shape_b):
            raise ValueError(
                "input_shape_b must be a tuple of 3 positive integers (height, width, channels)"
            )

        # Validate loss weights
        validate_non_negative_float(self.lambda_cycle, "lambda_cycle")
        validate_non_negative_float(self.lambda_identity, "lambda_identity")


@dataclasses.dataclass(frozen=True)
class DCGANConfig(GANConfig):
    """Configuration for Deep Convolutional GAN (DCGAN).

    Extends base GANConfig with DCGAN-specific defaults following the original paper.
    DCGAN uses convolutional layers for both generator (transposed conv) and
    discriminator (strided conv).

    Uses ConvGeneratorConfig and ConvDiscriminatorConfig for network configurations
    which include convolutional parameters (kernel_size, stride, padding).

    Default hyperparameters from the DCGAN paper:
    - Adam with beta1=0.5, beta2=0.999
    - Learning rate = 0.0002
    - Vanilla (BCE) loss

    Example:
        generator = ConvGeneratorConfig(
            name="dcgan_generator",
            latent_dim=100,
            hidden_dims=(512, 256, 128, 64),
            output_shape=(3, 64, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="dcgan_discriminator",
            hidden_dims=(64, 128, 256, 512),
            input_shape=(3, 64, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = DCGANConfig(
            name="dcgan",
            generator=generator,
            discriminator=discriminator,
        )
    """

    # DCGAN uses vanilla (BCE) loss by default (from the paper)
    loss_type: str = "vanilla"

    # DCGAN default hyperparameters (from the paper)
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    beta1: float = 0.5  # Lower beta1 for stability with batch norm
    beta2: float = 0.999

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            TypeError: If generator or discriminator have wrong type
        """
        # Call parent validation first
        super().__post_init__()

        # DCGAN requires single generator and discriminator (not dict)
        if isinstance(self.generator, dict):
            raise TypeError(
                "DCGANConfig requires a single GeneratorConfig, not dict of generators. "
                "Use GANConfig base class for multi-network GANs."
            )
        if isinstance(self.discriminator, dict):
            raise TypeError(
                "DCGANConfig requires a single DiscriminatorConfig, not dict of discriminators. "
                "Use GANConfig base class for multi-network GANs."
            )

    @classmethod
    def from_dict(cls, data: dict) -> "DCGANConfig":
        """Create DCGANConfig from dictionary.

        Converts generator/discriminator dicts to ConvGeneratorConfig/ConvDiscriminatorConfig.

        Args:
            data: Dictionary representation of the config

        Returns:
            DCGANConfig instance
        """
        data = dict(data)

        # Convert generator to ConvGeneratorConfig
        if "generator" in data and isinstance(data["generator"], dict):
            data["generator"] = ConvGeneratorConfig.from_dict(data["generator"])

        # Convert discriminator to ConvDiscriminatorConfig
        if "discriminator" in data and isinstance(data["discriminator"], dict):
            data["discriminator"] = ConvDiscriminatorConfig.from_dict(data["discriminator"])

        return super(DCGANConfig, cls).from_dict(data)
