"""Energy-Based Model configuration classes.

This module provides dataclass-based configuration for energy-based models
(EBM, DeepEBM), following the established pattern from other config modules.

Energy-based models define an energy function E(x) and use MCMC sampling
(typically Langevin dynamics) to generate samples from the learned distribution.
"""

import dataclasses
from typing import Any

from .base_dataclass import BaseConfig
from .base_network import BaseNetworkConfig
from .validation import (
    validate_positive_float,
    validate_positive_int,
    validate_probability,
)


# Valid options for validation
VALID_NETWORK_TYPES = ("mlp", "cnn")


@dataclasses.dataclass(frozen=True)
class EnergyNetworkConfig(BaseNetworkConfig):
    """Configuration for energy network architecture.

    This configures the neural network that computes the energy function E(x).
    It extends BaseNetworkConfig to inherit hidden_dims, activation, etc.

    Attributes:
        network_type: Type of network architecture ("mlp" or "cnn")
        use_bias: Whether to use bias in linear layers
        use_spectral_norm: Whether to apply spectral normalization
        use_residual: Whether to use residual connections
    """

    # Energy network-specific fields
    network_type: str = "mlp"
    use_bias: bool = True
    use_spectral_norm: bool = False
    use_residual: bool = False

    def __post_init__(self) -> None:
        """Validate energy network configuration."""
        super().__post_init__()

        # Validate network_type
        if self.network_type not in VALID_NETWORK_TYPES:
            raise ValueError(
                f"network_type must be one of {VALID_NETWORK_TYPES}, got '{self.network_type}'"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnergyNetworkConfig":
        """Create config from dictionary.

        Handles conversion of lists to tuples for hidden_dims.
        """
        data = data.copy()

        # Convert lists to tuples for immutability
        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class MCMCConfig(BaseConfig):
    """Configuration for MCMC sampling parameters.

    This configures the Langevin dynamics MCMC sampler used for
    generating samples during training and inference.

    Attributes:
        n_steps: Number of MCMC steps per sample
        step_size: Step size for Langevin dynamics
        noise_scale: Scale of noise added during sampling
        clip_value: Value to clip gradients at during sampling
        temperature: Temperature for energy scaling (higher = more exploration)
    """

    # MCMC parameters
    n_steps: int = 60
    step_size: float = 0.01
    noise_scale: float = 0.005
    clip_value: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate MCMC configuration."""
        super().__post_init__()

        # Validate n_steps
        validate_positive_int(self.n_steps, "n_steps")

        # Validate step_size
        validate_positive_float(self.step_size, "step_size")

        # Validate noise_scale (can be 0 for deterministic)
        if self.noise_scale < 0.0:
            raise ValueError(f"noise_scale must be non-negative, got {self.noise_scale}")

        # Validate clip_value
        validate_positive_float(self.clip_value, "clip_value")

        # Validate temperature
        validate_positive_float(self.temperature, "temperature")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCMCConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class SampleBufferConfig(BaseConfig):
    """Configuration for replay buffer (sample buffer).

    The sample buffer stores past generated samples to stabilize
    training and improve sample diversity.

    Attributes:
        capacity: Maximum number of samples to store in buffer
        reinit_prob: Probability of reinitializing a sample from scratch
    """

    # Buffer parameters
    capacity: int = 8192
    reinit_prob: float = 0.05

    def __post_init__(self) -> None:
        """Validate sample buffer configuration."""
        super().__post_init__()

        # Validate capacity
        validate_positive_int(self.capacity, "capacity")

        # Validate reinit_prob
        validate_probability(self.reinit_prob, "reinit_prob")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleBufferConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class EBMConfig(BaseConfig):
    """Configuration for Energy-Based Models.

    This is the main configuration class for standard EBMs operating
    on flattened input data (like MNIST vectors).

    Attributes:
        input_dim: Dimensionality of input data (flattened)
        energy_network: Configuration for the energy network
        mcmc: Configuration for MCMC sampling
        sample_buffer: Configuration for the replay buffer
        alpha: Regularization coefficient for energy magnitude
    """

    # Required fields
    input_dim: int = 0

    # Nested configurations (required)
    energy_network: EnergyNetworkConfig | None = None
    mcmc: MCMCConfig | None = None
    sample_buffer: SampleBufferConfig | None = None

    # Regularization
    alpha: float = 0.01

    def __post_init__(self) -> None:
        """Validate EBM configuration."""
        super().__post_init__()

        # Validate input_dim
        validate_positive_int(self.input_dim, "input_dim")

        # Validate required nested configs
        if self.energy_network is None:
            raise ValueError("energy_network is required and cannot be None")

        if self.mcmc is None:
            raise ValueError("mcmc is required and cannot be None")

        if self.sample_buffer is None:
            raise ValueError("sample_buffer is required and cannot be None")

        # Validate nested config types
        if not isinstance(self.energy_network, EnergyNetworkConfig):
            raise TypeError(
                f"energy_network must be EnergyNetworkConfig, "
                f"got {type(self.energy_network).__name__}"
            )

        if not isinstance(self.mcmc, MCMCConfig):
            raise TypeError(f"mcmc must be MCMCConfig, got {type(self.mcmc).__name__}")

        if not isinstance(self.sample_buffer, SampleBufferConfig):
            raise TypeError(
                f"sample_buffer must be SampleBufferConfig, got {type(self.sample_buffer).__name__}"
            )

        # Validate alpha
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        # Convert nested configs to dicts
        if self.energy_network is not None:
            data["energy_network"] = self.energy_network.to_dict()
        if self.mcmc is not None:
            data["mcmc"] = self.mcmc.to_dict()
        if self.sample_buffer is not None:
            data["sample_buffer"] = self.sample_buffer.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EBMConfig":
        """Create config from dictionary with nested config handling."""
        data = data.copy()

        # Convert nested dicts to config objects
        if "energy_network" in data and isinstance(data["energy_network"], dict):
            data["energy_network"] = EnergyNetworkConfig.from_dict(data["energy_network"])

        if "mcmc" in data and isinstance(data["mcmc"], dict):
            data["mcmc"] = MCMCConfig.from_dict(data["mcmc"])

        if "sample_buffer" in data and isinstance(data["sample_buffer"], dict):
            data["sample_buffer"] = SampleBufferConfig.from_dict(data["sample_buffer"])

        return cls(**data)


# =============================================================================
# Factory Function
# =============================================================================


def create_energy_function(
    config: EnergyNetworkConfig,
    *,
    input_dim: int | None = None,
    input_channels: int | None = None,
    rngs: "nnx.Rngs",
) -> "EnergyFunction":
    """Create an energy function from configuration.

    This factory function creates the appropriate energy function type
    (MLPEnergyFunction or CNNEnergyFunction) based on the network_type
    in the configuration.

    Args:
        config: EnergyNetworkConfig with network_type discriminator
        input_dim: Input dimension for MLP (required if network_type="mlp")
        input_channels: Input channels for CNN (required if network_type="cnn")
        rngs: Random number generators for initialization

    Returns:
        Initialized energy function (MLPEnergyFunction or CNNEnergyFunction)

    Raises:
        ValueError: If required parameters are missing for the network type
        ValueError: If network_type is not supported
    """
    # Import here to avoid circular imports
    from flax import nnx

    from artifex.generative_models.models.energy.base import (
        CNNEnergyFunction,
        MLPEnergyFunction,
    )

    # Map activation strings to functions
    activation_map = {
        "relu": nnx.relu,
        "tanh": nnx.tanh,
        "sigmoid": nnx.sigmoid,
        "gelu": nnx.gelu,
        "swish": nnx.swish,
        "silu": nnx.silu,
    }
    activation = activation_map.get(config.activation, nnx.gelu)

    match config.network_type:
        case "mlp":
            if input_dim is None:
                raise ValueError("input_dim is required for MLP energy function")

            return MLPEnergyFunction(
                hidden_dims=list(config.hidden_dims),
                input_dim=input_dim,
                activation=activation,
                use_bias=config.use_bias,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )

        case "cnn":
            if input_channels is None:
                raise ValueError("input_channels is required for CNN energy function")

            return CNNEnergyFunction(
                hidden_dims=list(config.hidden_dims),
                input_channels=input_channels,
                activation=activation,
                use_bias=config.use_bias,
                rngs=rngs,
            )

        case _:
            raise ValueError(
                f"Unsupported network_type: {config.network_type}. "
                f"Expected one of: {VALID_NETWORK_TYPES}"
            )


# Type annotation imports (for forward references)
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from flax import nnx

    from artifex.generative_models.models.energy.base import EnergyFunction


@dataclasses.dataclass(frozen=True)
class DeepEBMConfig(EBMConfig):
    """Configuration for Deep Energy-Based Models.

    DeepEBM is designed for high-dimensional structured data like images,
    using CNN architectures for the energy network.

    Attributes:
        input_shape: Shape of input data (H, W, C)
        alpha: Regularization coefficient (defaults lower than EBM)
    """

    # Image-specific fields
    input_shape: tuple[int, int, int] | None = None

    # Override alpha default for deep models (lower regularization)
    alpha: float = 0.001

    # Set input_dim to placeholder (will be derived from input_shape)
    input_dim: int = 1  # Placeholder, derived from input_shape

    def __post_init__(self) -> None:
        """Validate DeepEBM configuration."""
        # Validate input_shape before calling parent
        if self.input_shape is None:
            raise ValueError("input_shape is required and cannot be None")

        if len(self.input_shape) != 3:
            raise ValueError(
                f"input_shape must have 3 dimensions (H, W, C), "
                f"got {len(self.input_shape)} dimensions"
            )

        # Validate all dimensions are positive
        for i, dim in enumerate(self.input_shape):
            if dim <= 0:
                raise ValueError(
                    f"All input_shape dimensions must be positive, got {self.input_shape}"
                )

        # Set input_dim from input_shape for parent validation
        # We need to bypass frozen to set this derived value
        object.__setattr__(self, "input_dim", self.derived_input_dim)

        # Now call parent validation
        super().__post_init__()

    @property
    def derived_input_dim(self) -> int:
        """Get flattened input dimension from input_shape."""
        if self.input_shape is None:
            return 0
        h, w, c = self.input_shape
        return h * w * c

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with input_shape handling."""
        data = super().to_dict()
        # input_shape is already in data from parent
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeepEBMConfig":
        """Create config from dictionary with input_shape handling."""
        data = data.copy()

        # Convert input_shape list to tuple
        if "input_shape" in data and isinstance(data["input_shape"], list):
            data["input_shape"] = tuple(data["input_shape"])

        # Convert nested dicts to config objects
        if "energy_network" in data and isinstance(data["energy_network"], dict):
            data["energy_network"] = EnergyNetworkConfig.from_dict(data["energy_network"])

        if "mcmc" in data and isinstance(data["mcmc"], dict):
            data["mcmc"] = MCMCConfig.from_dict(data["mcmc"])

        if "sample_buffer" in data and isinstance(data["sample_buffer"], dict):
            data["sample_buffer"] = SampleBufferConfig.from_dict(data["sample_buffer"])

        return cls(**data)
