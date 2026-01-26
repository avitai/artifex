"""Flow model configuration classes.

This module provides dataclass-based configuration for normalizing flow models,
following the established pattern from GAN, VAE, and Diffusion configs.

Flow models have swappable coupling network architectures (MLP, ResNet, Attention, CNN)
which are configured through the CouplingNetworkConfig class.
"""

import dataclasses
from typing import Any

from .base_dataclass import BaseConfig
from .base_network import BaseNetworkConfig


# Valid options for validation
VALID_NETWORK_TYPES = ("mlp", "resnet", "attention", "cnn")
VALID_SCALE_ACTIVATIONS = ("tanh", "sigmoid", "exp", "softplus")
VALID_BASE_DISTRIBUTIONS = ("normal", "uniform")
VALID_MASK_TYPES = ("checkerboard", "channel-wise")


@dataclasses.dataclass(frozen=True)
class CouplingNetworkConfig(BaseNetworkConfig):
    """Configuration for coupling layer networks in normalizing flows.

    The coupling network is the neural network that computes the scale and shift
    parameters in coupling layers. Different network architectures can be used:
    - MLP: Simple feedforward network for tabular data
    - ResNet: Residual networks for better gradient flow (images)
    - Attention: Self-attention for long-range dependencies
    - CNN: Convolutional networks for image data

    This extends BaseNetworkConfig to inherit hidden_dims, activation, etc.

    Attributes:
        network_type: Type of network architecture ("mlp", "resnet", "attention", "cnn")
        scale_activation: Activation for scale output ("tanh", "sigmoid", "exp", "softplus")
        num_residual_blocks: Number of residual blocks (for resnet type)
        num_attention_heads: Number of attention heads (for attention type)
    """

    # Coupling-specific fields
    network_type: str = "mlp"
    scale_activation: str = "tanh"
    num_residual_blocks: int = 0
    num_attention_heads: int = 4

    def __post_init__(self) -> None:
        """Validate coupling network configuration."""
        super().__post_init__()

        # Validate network_type
        if self.network_type not in VALID_NETWORK_TYPES:
            raise ValueError(
                f"network_type must be one of {VALID_NETWORK_TYPES}, got '{self.network_type}'"
            )

        # Validate scale_activation
        if self.scale_activation not in VALID_SCALE_ACTIVATIONS:
            raise ValueError(
                f"scale_activation must be one of {VALID_SCALE_ACTIVATIONS}, "
                f"got '{self.scale_activation}'"
            )

        # Validate num_residual_blocks
        if self.num_residual_blocks < 0:
            raise ValueError(
                f"num_residual_blocks must be non-negative, got {self.num_residual_blocks}"
            )

        # Validate num_attention_heads
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CouplingNetworkConfig":
        """Create config from dictionary.

        Handles conversion of lists to tuples for hidden_dims.
        """
        data = data.copy()

        # Convert lists to tuples for immutability
        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class FlowConfig(BaseConfig):
    """Base configuration for normalizing flow models.

    This is the base class for all flow model configurations.
    It contains a nested CouplingNetworkConfig for the coupling layer networks.

    Attributes:
        coupling_network: Configuration for the coupling layer networks
        input_dim: Dimension of input data
        latent_dim: Dimension of latent space (defaults to input_dim)
        base_distribution: Type of base distribution ("normal", "uniform")
        base_distribution_params: Parameters for the base distribution
    """

    # Required field - coupling network configuration
    coupling_network: CouplingNetworkConfig | None = None

    # Flow-specific fields
    input_dim: int = 0
    latent_dim: int = 0  # Defaults to input_dim in __post_init__
    base_distribution: str = "normal"
    base_distribution_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"loc": 0.0, "scale": 1.0}
    )

    def __post_init__(self) -> None:
        """Validate flow configuration."""
        super().__post_init__()

        # Validate coupling_network is provided
        if self.coupling_network is None:
            raise ValueError("coupling_network is required and cannot be None")

        # Validate coupling_network type
        if not isinstance(self.coupling_network, CouplingNetworkConfig):
            raise TypeError(
                f"coupling_network must be CouplingNetworkConfig, "
                f"got {type(self.coupling_network).__name__}"
            )

        # Validate input_dim
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")

        # Default latent_dim to input_dim if not set
        if self.latent_dim == 0:
            object.__setattr__(self, "latent_dim", self.input_dim)
        elif self.latent_dim < 0:
            raise ValueError(f"latent_dim must be non-negative, got {self.latent_dim}")

        # Validate base_distribution
        if self.base_distribution not in VALID_BASE_DISTRIBUTIONS:
            raise ValueError(
                f"base_distribution must be one of {VALID_BASE_DISTRIBUTIONS}, "
                f"got '{self.base_distribution}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        # Convert nested config to dict
        if self.coupling_network is not None:
            data["coupling_network"] = self.coupling_network.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowConfig":
        """Create config from dictionary with nested config handling."""
        data = data.copy()

        # Convert nested dict to CouplingNetworkConfig
        if "coupling_network" in data and isinstance(data["coupling_network"], dict):
            data["coupling_network"] = CouplingNetworkConfig.from_dict(data["coupling_network"])

        # Handle base_distribution_params
        if "base_distribution_params" not in data:
            data["base_distribution_params"] = {"loc": 0.0, "scale": 1.0}

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class RealNVPConfig(FlowConfig):
    """Configuration for RealNVP (Real-valued Non-Volume Preserving) flow.

    RealNVP uses affine coupling layers with alternating masks.

    Attributes:
        num_coupling_layers: Number of coupling layers in the flow
        mask_type: Type of mask pattern ("checkerboard", "channel-wise")
    """

    num_coupling_layers: int = 8
    mask_type: str = "checkerboard"

    def __post_init__(self) -> None:
        """Validate RealNVP configuration."""
        super().__post_init__()

        # Validate num_coupling_layers
        if self.num_coupling_layers <= 0:
            raise ValueError(
                f"num_coupling_layers must be positive, got {self.num_coupling_layers}"
            )

        # Validate mask_type
        if self.mask_type not in VALID_MASK_TYPES:
            raise ValueError(f"mask_type must be one of {VALID_MASK_TYPES}, got '{self.mask_type}'")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RealNVPConfig":
        """Create config from dictionary."""
        data = data.copy()

        # Convert nested dict to CouplingNetworkConfig
        if "coupling_network" in data and isinstance(data["coupling_network"], dict):
            data["coupling_network"] = CouplingNetworkConfig.from_dict(data["coupling_network"])

        # Handle base_distribution_params
        if "base_distribution_params" not in data:
            data["base_distribution_params"] = {"loc": 0.0, "scale": 1.0}

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class GlowConfig(FlowConfig):
    """Configuration for Glow (Generative Flow with Invertible 1x1 Convolutions).

    Glow uses a multi-scale architecture with ActNorm, invertible 1x1 convolutions,
    and affine coupling layers.

    Attributes:
        image_shape: Shape of input images (H, W, C)
        num_scales: Number of multi-scale levels
        blocks_per_scale: Number of flow blocks per scale level
    """

    image_shape: tuple[int, ...] | None = None
    num_scales: int = 3
    blocks_per_scale: int = 6

    def __post_init__(self) -> None:
        """Validate Glow configuration."""
        super().__post_init__()

        # Validate image_shape is provided
        if self.image_shape is None:
            raise ValueError("image_shape is required for GlowConfig")

        # Validate num_scales
        if self.num_scales <= 0:
            raise ValueError(f"num_scales must be positive, got {self.num_scales}")

        # Validate blocks_per_scale
        if self.blocks_per_scale <= 0:
            raise ValueError(f"blocks_per_scale must be positive, got {self.blocks_per_scale}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlowConfig":
        """Create config from dictionary."""
        data = data.copy()

        # Convert nested dict to CouplingNetworkConfig
        if "coupling_network" in data and isinstance(data["coupling_network"], dict):
            data["coupling_network"] = CouplingNetworkConfig.from_dict(data["coupling_network"])

        # Convert image_shape list to tuple
        if "image_shape" in data and isinstance(data["image_shape"], list):
            data["image_shape"] = tuple(data["image_shape"])

        # Handle base_distribution_params
        if "base_distribution_params" not in data:
            data["base_distribution_params"] = {"loc": 0.0, "scale": 1.0}

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class MAFConfig(FlowConfig):
    """Configuration for MAF (Masked Autoregressive Flow).

    MAF uses autoregressive transformations for fast density estimation
    but slow sampling.

    Attributes:
        num_layers: Number of MAF layers
        reverse_ordering: Whether to alternate variable ordering between layers
    """

    num_layers: int = 5
    reverse_ordering: bool = True

    def __post_init__(self) -> None:
        """Validate MAF configuration."""
        super().__post_init__()

        # Validate num_layers
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MAFConfig":
        """Create config from dictionary."""
        data = data.copy()

        # Convert nested dict to CouplingNetworkConfig
        if "coupling_network" in data and isinstance(data["coupling_network"], dict):
            data["coupling_network"] = CouplingNetworkConfig.from_dict(data["coupling_network"])

        # Handle base_distribution_params
        if "base_distribution_params" not in data:
            data["base_distribution_params"] = {"loc": 0.0, "scale": 1.0}

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class IAFConfig(FlowConfig):
    """Configuration for IAF (Inverse Autoregressive Flow).

    IAF uses inverse autoregressive transformations for fast sampling
    but slow density estimation.

    Attributes:
        num_layers: Number of IAF layers
        reverse_ordering: Whether to alternate variable ordering between layers
    """

    num_layers: int = 5
    reverse_ordering: bool = True

    def __post_init__(self) -> None:
        """Validate IAF configuration."""
        super().__post_init__()

        # Validate num_layers
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IAFConfig":
        """Create config from dictionary."""
        data = data.copy()

        # Convert nested dict to CouplingNetworkConfig
        if "coupling_network" in data and isinstance(data["coupling_network"], dict):
            data["coupling_network"] = CouplingNetworkConfig.from_dict(data["coupling_network"])

        # Handle base_distribution_params
        if "base_distribution_params" not in data:
            data["base_distribution_params"] = {"loc": 0.0, "scale": 1.0}

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class NeuralSplineConfig(FlowConfig):
    """Configuration for Neural Spline Flow.

    Neural Spline Flows use rational quadratic spline transformations
    for highly expressive yet tractable flows.

    Attributes:
        num_layers: Number of spline coupling layers
        num_bins: Number of spline bins/segments
        tail_bound: Spline domain bounds [-tail_bound, tail_bound]
    """

    num_layers: int = 8
    num_bins: int = 8
    tail_bound: float = 3.0

    def __post_init__(self) -> None:
        """Validate Neural Spline configuration."""
        super().__post_init__()

        # Validate num_layers
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        # Validate num_bins
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {self.num_bins}")

        # Validate tail_bound
        if self.tail_bound <= 0:
            raise ValueError(f"tail_bound must be positive, got {self.tail_bound}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuralSplineConfig":
        """Create config from dictionary."""
        data = data.copy()

        # Convert nested dict to CouplingNetworkConfig
        if "coupling_network" in data and isinstance(data["coupling_network"], dict):
            data["coupling_network"] = CouplingNetworkConfig.from_dict(data["coupling_network"])

        # Handle base_distribution_params
        if "base_distribution_params" not in data:
            data["base_distribution_params"] = {"loc": 0.0, "scale": 1.0}

        return cls(**data)
