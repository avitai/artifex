"""Network configuration classes for generative model components.

This module provides configuration classes for network components used in generative models.
These follow Principle #4: Methods should accept config objects, not individual parameters.

Hierarchy:
- GeneratorConfig: Base generator configuration
  - ConvGeneratorConfig: Convolutional generator (DCGAN, WGAN, LSGAN, etc.)
- DiscriminatorConfig: Base discriminator configuration
  - ConvDiscriminatorConfig: Convolutional discriminator (DCGAN, WGAN, LSGAN, etc.)
- EncoderConfig: VAE encoder configuration
- DecoderConfig: VAE decoder configuration
"""

import dataclasses

from .base_network import BaseNetworkConfig
from .validation import (
    validate_activation,
    validate_positive_float,
    validate_positive_int,
    validate_positive_tuple,
)


# Valid padding modes for convolutional layers
VALID_PADDING_MODES = {"SAME", "VALID"}


@dataclasses.dataclass(frozen=True)
class GeneratorConfig(BaseNetworkConfig):
    """Configuration for generator networks.

    Inherits common fields from BaseNetworkConfig:
    - hidden_dims: Hidden layer dimensions
    - activation: Activation function name
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate (0.0 to 1.0)

    Generator-specific fields:
    - latent_dim: Dimension of latent/noise vector input
    - output_shape: Shape of generated output (e.g., (28, 28, 1) for MNIST)
    - use_bias: Whether to use bias in layers
    - output_activation: Activation for final output layer
    """

    # Generator-specific required fields (dummy defaults for field ordering)
    latent_dim: int = 0
    output_shape: tuple[int, ...] = ()

    # Generator-specific optional fields
    use_bias: bool = True
    output_activation: str | None = "tanh"

    def __post_init__(self) -> None:
        """Validate generator-specific fields after initialization."""
        # Call parent validation first
        super().__post_init__()

        # Validate latent_dim
        validate_positive_int(self.latent_dim, "latent_dim")

        # Validate output_shape
        if not self.output_shape:
            raise ValueError("output_shape cannot be empty")
        validate_positive_tuple(self.output_shape, "output_shape")

        # Validate output_activation (if provided)
        if self.output_activation is not None:
            validate_activation(self.output_activation)


@dataclasses.dataclass(frozen=True)
class DiscriminatorConfig(BaseNetworkConfig):
    """Configuration for GAN Discriminator network.

    Inherits common fields from BaseNetworkConfig and adds discriminator-specific fields.

    Attributes:
        input_shape: Shape of input data (e.g., (1, 28, 28) for images)
        leaky_relu_slope: Negative slope for leaky ReLU activation
        use_spectral_norm: Whether to use spectral normalization
        hidden_dims: Hidden layer dimensions (from BaseNetworkConfig)
        activation: Activation function name (from BaseNetworkConfig)
        batch_norm: Whether to use batch normalization (from BaseNetworkConfig)
        dropout_rate: Dropout rate (from BaseNetworkConfig)
    """

    # Discriminator-specific fields
    input_shape: tuple[int, ...] = ()
    leaky_relu_slope: float = 0.2
    use_spectral_norm: bool = False

    def __post_init__(self) -> None:
        """Validate discriminator configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().__post_init__()

        # Validate discriminator-specific fields
        if len(self.input_shape) == 0:
            raise ValueError("input_shape is required and cannot be empty")
        validate_positive_tuple(self.input_shape, "input_shape")

        # Validate leaky_relu_slope (must be positive for leaky ReLU to work)
        validate_positive_float(self.leaky_relu_slope, "leaky_relu_slope")


@dataclasses.dataclass(frozen=True)
class EncoderConfig(BaseNetworkConfig):
    """Configuration for VAE Encoder network.

    Inherits common fields from BaseNetworkConfig and adds encoder-specific fields.

    Attributes:
        input_shape: Shape of input data (e.g., (1, 28, 28) for images)
        latent_dim: Dimension of latent space output
        use_batch_norm: Whether to use batch normalization in encoder
        use_layer_norm: Whether to use layer normalization in encoder
        hidden_dims: Hidden layer dimensions (from BaseNetworkConfig)
        activation: Activation function name (from BaseNetworkConfig)
        batch_norm: Whether to use batch normalization (from BaseNetworkConfig)
        dropout_rate: Dropout rate (from BaseNetworkConfig)
    """

    # Encoder-specific required fields (dummy defaults for field ordering)
    input_shape: tuple[int, ...] = ()
    latent_dim: int = 0

    # Encoder-specific optional fields
    use_batch_norm: bool = True
    use_layer_norm: bool = False

    def __post_init__(self) -> None:
        """Validate encoder configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().__post_init__()

        # Validate encoder-specific fields
        if len(self.input_shape) == 0:
            raise ValueError("input_shape is required and cannot be empty")
        validate_positive_tuple(self.input_shape, "input_shape")

        # Validate latent_dim
        validate_positive_int(self.latent_dim, "latent_dim")


@dataclasses.dataclass(frozen=True)
class DecoderConfig(BaseNetworkConfig):
    """Configuration for VAE Decoder network.

    Inherits common fields from BaseNetworkConfig and adds decoder-specific fields.

    Attributes:
        latent_dim: Dimension of latent space input
        output_shape: Shape of generated output (e.g., (28, 28, 1) for images)
        output_activation: Activation function for output layer (default: sigmoid)
        hidden_dims: Hidden layer dimensions (from BaseNetworkConfig)
        activation: Activation function name (from BaseNetworkConfig)
        batch_norm: Whether to use batch normalization (from BaseNetworkConfig)
        dropout_rate: Dropout rate (from BaseNetworkConfig)
    """

    # Decoder-specific required fields (dummy defaults for field ordering)
    latent_dim: int = 0
    output_shape: tuple[int, ...] = ()

    # Decoder-specific optional fields
    output_activation: str | None = "sigmoid"

    def __post_init__(self) -> None:
        """Validate decoder configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().__post_init__()

        # Validate decoder-specific fields
        validate_positive_int(self.latent_dim, "latent_dim")

        if len(self.output_shape) == 0:
            raise ValueError("output_shape is required and cannot be empty")
        validate_positive_tuple(self.output_shape, "output_shape")

        # Validate output_activation (if provided)
        if self.output_activation is not None:
            validate_activation(self.output_activation)


@dataclasses.dataclass(frozen=True)
class ConvGeneratorConfig(GeneratorConfig):
    """Configuration for convolutional generator networks (DCGAN, WGAN, LSGAN, etc.).

    Extends GeneratorConfig with convolutional layer parameters for DCGAN-style
    architectures using transposed convolutions for upsampling.

    Inherits from GeneratorConfig:
    - latent_dim: Dimension of latent/noise vector input
    - output_shape: Shape of generated output
    - hidden_dims: Hidden layer dimensions (number of filters per layer)
    - activation: Activation function name
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate

    Conv-specific fields:
    - kernel_size: Kernel size for conv layers (height, width)
    - stride: Stride for conv layers (height, width)
    - padding: Padding mode ("SAME" or "VALID")
    - batch_norm_momentum: Momentum for batch normalization
    - batch_norm_use_running_avg: Whether to use running average in batch norm
    """

    # Convolutional layer parameters (required)
    kernel_size: tuple[int, int] = (4, 4)
    stride: tuple[int, int] = (2, 2)
    padding: str = "SAME"

    # BatchNorm parameters
    batch_norm_momentum: float = 0.9
    batch_norm_use_running_avg: bool = False

    def __post_init__(self) -> None:
        """Validate convolutional generator configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate kernel_size
        if not isinstance(self.kernel_size, tuple) or len(self.kernel_size) != 2:
            raise TypeError("kernel_size must be a tuple of 2 integers")
        if not all(k > 0 for k in self.kernel_size):
            raise ValueError("kernel_size must have positive dimensions")

        # Validate stride
        if not isinstance(self.stride, tuple) or len(self.stride) != 2:
            raise TypeError("stride must be a tuple of 2 integers")
        if not all(s > 0 for s in self.stride):
            raise ValueError("stride must have positive dimensions")

        # Validate padding
        if self.padding not in VALID_PADDING_MODES:
            raise ValueError(f"padding must be one of {VALID_PADDING_MODES}, got '{self.padding}'")

        # Validate batch_norm_momentum (must be in (0, 1))
        if not (0.0 < self.batch_norm_momentum < 1.0):
            raise ValueError(
                f"batch_norm_momentum must be in (0, 1), got {self.batch_norm_momentum}"
            )


@dataclasses.dataclass(frozen=True)
class ConvDiscriminatorConfig(DiscriminatorConfig):
    """Configuration for convolutional discriminator networks (DCGAN, WGAN, LSGAN, etc.).

    Extends DiscriminatorConfig with convolutional layer parameters for DCGAN-style
    architectures using strided convolutions for downsampling.

    Inherits from DiscriminatorConfig:
    - input_shape: Shape of input data
    - hidden_dims: Hidden layer dimensions (number of filters per layer)
    - activation: Activation function name
    - leaky_relu_slope: Negative slope for leaky ReLU
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate
    - use_spectral_norm: Whether to use spectral normalization

    Conv-specific fields:
    - kernel_size: Kernel size for conv layers (height, width)
    - stride: Stride for conv layers (height, width)
    - padding: Padding mode ("SAME" or "VALID")
    - batch_norm_momentum: Momentum for batch normalization
    - batch_norm_use_running_avg: Whether to use running average in batch norm
    - output_dim: Output dimension (default: 1 for binary real/fake)
    """

    # Convolutional layer parameters (required)
    kernel_size: tuple[int, int] = (4, 4)
    stride: tuple[int, int] = (2, 2)
    padding: str = "SAME"

    # BatchNorm parameters
    batch_norm_momentum: float = 0.9
    batch_norm_use_running_avg: bool = False

    # Instance normalization (used in WGAN-GP instead of BatchNorm)
    use_instance_norm: bool = False

    # Output dimension (typically 1 for real/fake classification)
    output_dim: int = 1

    def __post_init__(self) -> None:
        """Validate convolutional discriminator configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate kernel_size
        if not isinstance(self.kernel_size, tuple) or len(self.kernel_size) != 2:
            raise TypeError("kernel_size must be a tuple of 2 integers")
        if not all(k > 0 for k in self.kernel_size):
            raise ValueError("kernel_size must have positive dimensions")

        # Validate stride
        if not isinstance(self.stride, tuple) or len(self.stride) != 2:
            raise TypeError("stride must be a tuple of 2 integers")
        if not all(s > 0 for s in self.stride):
            raise ValueError("stride must have positive dimensions")

        # Validate padding
        if self.padding not in VALID_PADDING_MODES:
            raise ValueError(f"padding must be one of {VALID_PADDING_MODES}, got '{self.padding}'")

        # Validate batch_norm_momentum (must be in (0, 1))
        if not (0.0 < self.batch_norm_momentum < 1.0):
            raise ValueError(
                f"batch_norm_momentum must be in (0, 1), got {self.batch_norm_momentum}"
            )

        # Validate output_dim
        validate_positive_int(self.output_dim, "output_dim")


@dataclasses.dataclass(frozen=True)
class ConditionalParams:
    """Reusable component for conditional generation parameters.

    This is a composition component (HAS-A relationship) that can be embedded
    in any generator or discriminator config to add conditional capabilities.
    Following the "composition over inheritance" principle.

    Attributes:
        num_classes: Number of classes for conditional generation
        embedding_dim: Dimension of class label embeddings
    """

    num_classes: int = 10
    embedding_dim: int = 100

    def __post_init__(self) -> None:
        """Validate conditional parameters."""
        validate_positive_int(self.num_classes, "num_classes")
        validate_positive_int(self.embedding_dim, "embedding_dim")


@dataclasses.dataclass(frozen=True)
class ConditionalGeneratorConfig(GeneratorConfig):
    """Configuration for Conditional GAN generator networks.

    Extends GeneratorConfig with conditional generation capabilities via composition.
    The generator is conditioned on class labels by concatenating label embeddings
    with the noise vector.

    Inherits from GeneratorConfig:
    - latent_dim: Dimension of latent/noise vector input
    - output_shape: Shape of generated output
    - hidden_dims: Hidden layer dimensions
    - activation: Activation function name
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate

    Conditional-specific (via composition):
    - conditional: ConditionalParams with num_classes and embedding_dim

    Conv layer parameters (for convolutional conditional generators):
    - kernel_size: Kernel size for conv layers
    - stride: Stride for conv layers
    - padding: Padding mode
    - batch_norm_momentum: Momentum for batch normalization
    - batch_norm_use_running_avg: Whether to use running average in batch norm
    """

    # Composition: conditional parameters as nested config
    conditional: ConditionalParams = dataclasses.field(default_factory=lambda: ConditionalParams())

    # Conv layer parameters (optional, for conv-based conditional generators)
    kernel_size: tuple[int, int] = (4, 4)
    stride: tuple[int, int] = (2, 2)
    padding: str = "SAME"
    batch_norm_momentum: float = 0.9
    batch_norm_use_running_avg: bool = False

    def __post_init__(self) -> None:
        """Validate conditional generator configuration."""
        super().__post_init__()

        # Validate kernel_size
        if not isinstance(self.kernel_size, tuple) or len(self.kernel_size) != 2:
            raise TypeError("kernel_size must be a tuple of 2 integers")
        if not all(k > 0 for k in self.kernel_size):
            raise ValueError("kernel_size must have positive dimensions")

        # Validate stride
        if not isinstance(self.stride, tuple) or len(self.stride) != 2:
            raise TypeError("stride must be a tuple of 2 integers")
        if not all(s > 0 for s in self.stride):
            raise ValueError("stride must have positive dimensions")

        # Validate padding
        if self.padding not in VALID_PADDING_MODES:
            raise ValueError(f"padding must be one of {VALID_PADDING_MODES}, got '{self.padding}'")

        # Validate batch_norm_momentum
        if not (0.0 < self.batch_norm_momentum < 1.0):
            raise ValueError(
                f"batch_norm_momentum must be in (0, 1), got {self.batch_norm_momentum}"
            )


@dataclasses.dataclass(frozen=True)
class ConditionalDiscriminatorConfig(DiscriminatorConfig):
    """Configuration for Conditional GAN discriminator networks.

    Extends DiscriminatorConfig with conditional discrimination capabilities via composition.
    The discriminator receives both images and class labels.

    Inherits from DiscriminatorConfig:
    - input_shape: Shape of input images
    - hidden_dims: Hidden layer dimensions
    - activation: Activation function name
    - leaky_relu_slope: Negative slope for leaky ReLU
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate
    - use_spectral_norm: Whether to use spectral normalization

    Conditional-specific (via composition):
    - conditional: ConditionalParams with num_classes and embedding_dim

    Conv layer parameters (for convolutional conditional discriminators):
    - kernel_size: Kernel size for conv layers
    - stride: Stride for conv layers
    - stride_first: Stride for the first convolutional layer
    - padding: Padding mode
    - batch_norm_momentum: Momentum for batch normalization
    - batch_norm_use_running_avg: Whether to use running average in batch norm
    """

    # Composition: conditional parameters as nested config
    conditional: ConditionalParams = dataclasses.field(default_factory=lambda: ConditionalParams())

    # Conv layer parameters (for conv-based conditional discriminators)
    kernel_size: tuple[int, int] = (4, 4)
    stride: tuple[int, int] = (2, 2)
    stride_first: tuple[int, int] = (2, 2)
    padding: str = "SAME"
    batch_norm_momentum: float = 0.9
    batch_norm_use_running_avg: bool = False

    def __post_init__(self) -> None:
        """Validate conditional discriminator configuration."""
        super().__post_init__()

        # Validate kernel_size
        if not isinstance(self.kernel_size, tuple) or len(self.kernel_size) != 2:
            raise TypeError("kernel_size must be a tuple of 2 integers")
        if not all(k > 0 for k in self.kernel_size):
            raise ValueError("kernel_size must have positive dimensions")

        # Validate stride
        if not isinstance(self.stride, tuple) or len(self.stride) != 2:
            raise TypeError("stride must be a tuple of 2 integers")
        if not all(s > 0 for s in self.stride):
            raise ValueError("stride must have positive dimensions")

        # Validate stride_first
        if not isinstance(self.stride_first, tuple) or len(self.stride_first) != 2:
            raise TypeError("stride_first must be a tuple of 2 integers")
        if not all(s > 0 for s in self.stride_first):
            raise ValueError("stride_first must have positive dimensions")

        # Validate padding
        if self.padding not in VALID_PADDING_MODES:
            raise ValueError(f"padding must be one of {VALID_PADDING_MODES}, got '{self.padding}'")

        # Validate batch_norm_momentum
        if not (0.0 < self.batch_norm_momentum < 1.0):
            raise ValueError(
                f"batch_norm_momentum must be in (0, 1), got {self.batch_norm_momentum}"
            )


@dataclasses.dataclass(frozen=True)
class CycleGANGeneratorConfig(GeneratorConfig):
    """Configuration for CycleGAN generator networks.

    Extends GeneratorConfig for image-to-image translation with ResNet-style
    architecture. CycleGAN generators take an image as input (not a latent vector).

    Inherits from GeneratorConfig:
    - latent_dim: Not used (set to 0 for image-to-image)
    - output_shape: Shape of generated output image (H, W, C)
    - hidden_dims: Number of filters in encoder/decoder layers
    - activation: Activation function name
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate

    CycleGAN-specific fields:
    - input_shape: Shape of input images (H, W, C)
    - n_residual_blocks: Number of residual blocks in the middle
    - use_skip_connections: Whether to use U-Net style skip connections
    """

    # CycleGAN-specific required fields
    input_shape: tuple[int, ...] = ()

    # CycleGAN-specific optional fields with defaults
    n_residual_blocks: int = 6
    use_skip_connections: bool = True

    def __post_init__(self) -> None:
        """Validate CycleGAN generator configuration.

        Overrides parent validation to allow latent_dim=0 for image-to-image translation.

        Raises:
            ValueError: If validation fails
        """
        # Skip GeneratorConfig's latent_dim validation by calling BaseNetworkConfig directly
        # CycleGAN is image-to-image so latent_dim should be 0
        BaseNetworkConfig.__post_init__(self)

        # Validate output_shape (from GeneratorConfig)
        if not self.output_shape:
            raise ValueError("output_shape cannot be empty")
        validate_positive_tuple(self.output_shape, "output_shape")

        # Validate output_activation (if provided) - from GeneratorConfig
        if self.output_activation is not None:
            validate_activation(self.output_activation)

        # CycleGAN allows latent_dim=0 since it's image-to-image (no latent space)
        if self.latent_dim < 0:
            raise ValueError("latent_dim must be non-negative")

        # Validate input_shape
        if len(self.input_shape) == 0:
            raise ValueError("input_shape is required and cannot be empty")
        validate_positive_tuple(self.input_shape, "input_shape")

        # Validate n_residual_blocks
        validate_positive_int(self.n_residual_blocks, "n_residual_blocks")


@dataclasses.dataclass(frozen=True)
class PatchGANDiscriminatorConfig(DiscriminatorConfig):
    """Configuration for PatchGAN discriminator networks (used by CycleGAN, Pix2Pix).

    Extends DiscriminatorConfig for patch-based discrimination. PatchGAN classifies
    patches of the input as real or fake rather than the entire image.

    Inherits from DiscriminatorConfig:
    - input_shape: Shape of input images (C, H, W) - channels first
    - hidden_dims: Computed from num_filters and num_layers
    - activation: Activation function name
    - leaky_relu_slope: Negative slope for leaky ReLU
    - batch_norm: Whether to use batch normalization
    - dropout_rate: Dropout rate
    - use_spectral_norm: Whether to use spectral normalization

    PatchGAN-specific fields:
    - num_filters: Base number of filters (doubled each layer)
    - num_layers: Number of convolutional layers
    - kernel_size: Kernel size for conv layers
    - stride: Stride for conv layers
    - padding: Padding mode
    - use_bias: Whether to use bias in conv layers (typically False with BatchNorm)
    - last_kernel_size: Kernel size for the final output layer
    """

    # PatchGAN-specific architecture parameters
    num_filters: int = 64
    num_layers: int = 3
    use_bias: bool = False
    last_kernel_size: tuple[int, int] = (1, 1)

    # PatchGAN convolutional layer parameters
    kernel_size: tuple[int, int] = (4, 4)
    stride: tuple[int, int] = (2, 2)
    padding: str = "SAME"

    def __post_init__(self) -> None:
        """Validate PatchGAN discriminator configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate num_filters
        validate_positive_int(self.num_filters, "num_filters")

        # Validate num_layers
        validate_positive_int(self.num_layers, "num_layers")

        # Validate kernel_size
        if not isinstance(self.kernel_size, tuple) or len(self.kernel_size) != 2:
            raise TypeError("kernel_size must be a tuple of 2 integers")
        if not all(k > 0 for k in self.kernel_size):
            raise ValueError("kernel_size must have positive dimensions")

        # Validate last_kernel_size
        if not isinstance(self.last_kernel_size, tuple) or len(self.last_kernel_size) != 2:
            raise TypeError("last_kernel_size must be a tuple of 2 integers")
        if not all(k > 0 for k in self.last_kernel_size):
            raise ValueError("last_kernel_size must have positive dimensions")

        # Validate stride
        if not isinstance(self.stride, tuple) or len(self.stride) != 2:
            raise TypeError("stride must be a tuple of 2 integers")
        if not all(s > 0 for s in self.stride):
            raise ValueError("stride must have positive dimensions")

        # Validate padding
        if self.padding not in VALID_PADDING_MODES:
            raise ValueError(f"padding must be one of {VALID_PADDING_MODES}, got '{self.padding}'")


@dataclasses.dataclass(frozen=True)
class MultiScalePatchGANConfig:
    """Configuration for Multi-scale PatchGAN discriminator.

    Contains a nested PatchGANDiscriminatorConfig for base discriminator settings
    plus multi-scale specific parameters.

    Attributes:
        discriminator: Base PatchGANDiscriminatorConfig for each discriminator
        num_discriminators: Number of discriminators at different scales
        num_layers_per_disc: Tuple of layer counts per discriminator (or None to use base)
        pooling_method: Pooling method for downsampling ('avg' or 'max')
        minimum_size: Minimum spatial size to avoid over-downsampling
    """

    discriminator: PatchGANDiscriminatorConfig
    num_discriminators: int = 3
    num_layers_per_disc: tuple[int, ...] | None = None
    pooling_method: str = "avg"
    minimum_size: int = 16

    def __post_init__(self) -> None:
        """Validate multi-scale PatchGAN configuration.

        Raises:
            ValueError: If validation fails
            TypeError: If discriminator is not PatchGANDiscriminatorConfig
        """
        # Validate discriminator config type
        if not isinstance(self.discriminator, PatchGANDiscriminatorConfig):
            raise TypeError(
                f"discriminator must be PatchGANDiscriminatorConfig, "
                f"got {type(self.discriminator).__name__}"
            )

        # Validate num_discriminators
        validate_positive_int(self.num_discriminators, "num_discriminators")

        # Validate num_layers_per_disc if provided
        if self.num_layers_per_disc is not None:
            if len(self.num_layers_per_disc) != self.num_discriminators:
                raise ValueError(
                    f"num_layers_per_disc length ({len(self.num_layers_per_disc)}) must match "
                    f"num_discriminators ({self.num_discriminators})"
                )
            for i, layers in enumerate(self.num_layers_per_disc):
                if layers < 1:
                    raise ValueError(f"num_layers_per_disc[{i}] must be positive, got {layers}")

        # Validate pooling_method
        valid_pooling = {"avg", "max"}
        if self.pooling_method not in valid_pooling:
            raise ValueError(
                f"pooling_method must be one of {valid_pooling}, got '{self.pooling_method}'"
            )

        # Validate minimum_size
        validate_positive_int(self.minimum_size, "minimum_size")


@dataclasses.dataclass(frozen=True)
class StyleGAN3GeneratorConfig(GeneratorConfig):
    """Configuration for StyleGAN3 generator networks.

    StyleGAN3 generators use a mapping network to transform latent codes to
    style vectors, and a synthesis network to generate images progressively.

    Inherits from GeneratorConfig:
    - latent_dim: Dimension of latent/noise vector input
    - output_shape: Shape of generated output image (H, W, C)
    - hidden_dims: Not used for StyleGAN3 (architecture is resolution-based)
    - activation: Activation function name
    - batch_norm: Not typically used in StyleGAN3
    - dropout_rate: Dropout rate

    StyleGAN3-specific fields:
    - style_dim: Dimension of style vectors
    - mapping_layers: Number of layers in mapping network
    - img_resolution: Output image resolution (must be power of 2)
    - img_channels: Number of output image channels (e.g., 3 for RGB)
    """

    # StyleGAN3-specific required fields
    style_dim: int = 512
    mapping_layers: int = 8
    img_resolution: int = 256
    img_channels: int = 3

    def __post_init__(self) -> None:
        """Validate StyleGAN3 generator configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate style_dim
        validate_positive_int(self.style_dim, "style_dim")

        # Validate mapping_layers
        validate_positive_int(self.mapping_layers, "mapping_layers")

        # Validate img_resolution (must be power of 2 and >= 4)
        validate_positive_int(self.img_resolution, "img_resolution")
        if self.img_resolution < 4:
            raise ValueError("img_resolution must be at least 4")
        if (self.img_resolution & (self.img_resolution - 1)) != 0:
            raise ValueError("img_resolution must be a power of 2")

        # Validate img_channels
        validate_positive_int(self.img_channels, "img_channels")


@dataclasses.dataclass(frozen=True)
class StyleGAN3DiscriminatorConfig(DiscriminatorConfig):
    """Configuration for StyleGAN3 discriminator networks.

    StyleGAN3 discriminators use progressive downsampling to classify images.

    Inherits from DiscriminatorConfig:
    - input_shape: Shape of input images (H, W, C)
    - hidden_dims: Not used for StyleGAN3 (architecture is resolution-based)
    - activation: Activation function name
    - leaky_relu_slope: Negative slope for leaky ReLU
    - batch_norm: Not typically used in StyleGAN3
    - dropout_rate: Dropout rate
    - use_spectral_norm: Whether to use spectral normalization

    StyleGAN3-specific fields:
    - img_resolution: Input image resolution (must be power of 2)
    - img_channels: Number of input image channels
    - base_channels: Base number of channels (doubles with downsampling)
    - max_channels: Maximum number of channels
    """

    # StyleGAN3-specific required fields
    img_resolution: int = 256
    img_channels: int = 3
    base_channels: int = 64
    max_channels: int = 512

    def __post_init__(self) -> None:
        """Validate StyleGAN3 discriminator configuration.

        Raises:
            ValueError: If validation fails
        """
        # Call parent validation first
        super().__post_init__()

        # Validate img_resolution (must be power of 2 and >= 4)
        validate_positive_int(self.img_resolution, "img_resolution")
        if self.img_resolution < 4:
            raise ValueError("img_resolution must be at least 4")
        if (self.img_resolution & (self.img_resolution - 1)) != 0:
            raise ValueError("img_resolution must be a power of 2")

        # Validate img_channels
        validate_positive_int(self.img_channels, "img_channels")

        # Validate base_channels
        validate_positive_int(self.base_channels, "base_channels")

        # Validate max_channels
        validate_positive_int(self.max_channels, "max_channels")
