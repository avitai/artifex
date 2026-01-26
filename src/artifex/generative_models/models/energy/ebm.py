"""Energy-Based Model implementations.

This module provides concrete implementations of energy-based models
for different data types and use cases.

Uses dataclass-based configuration following Principle #4:
Methods Take Configs, NOT Individual Parameters.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    create_energy_function,
    DeepEBMConfig,
    EBMConfig,
)
from artifex.generative_models.models.energy.base import (
    EnergyBasedModel,
    EnergyFunction,
)
from artifex.generative_models.models.energy.mcmc import SampleBuffer


class EBM(EnergyBasedModel):
    """Standard Energy-Based Model implementation.

    This class provides a complete EBM implementation with support for
    different energy function types and training with persistent contrastive
    divergence.

    Uses nested EBMConfig with:
    - energy_network: EnergyNetworkConfig for the energy function
    - mcmc: MCMCConfig for MCMC sampling parameters
    - sample_buffer: SampleBufferConfig for the replay buffer
    - alpha: Regularization coefficient

    Energy function type is determined by energy_network.network_type discriminator.
    """

    def __init__(
        self,
        config: EBMConfig,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ) -> None:
        """Initialize EBM.

        Args:
            config: EBMConfig with nested energy_network, mcmc, and sample_buffer configs.
                    The energy function is created based on energy_network.network_type.
            rngs: Random number generators
            precision: Numerical precision

        Raises:
            TypeError: If config is not an EBMConfig
        """
        # Validate config type
        if not isinstance(config, EBMConfig):
            raise TypeError(
                f"config must be EBMConfig, got {type(config).__name__}. "
                "Use EBMConfig(...) to create the configuration."
            )

        # Store config
        self.config = config

        # Create energy function from config using factory
        energy_fn = create_energy_function(
            config.energy_network,
            input_dim=config.input_dim,
            rngs=rngs,
        )

        super().__init__(
            energy_fn=energy_fn,
            rngs=rngs,
            precision=precision,
        )

        # Initialize sample buffer from config
        self.sample_buffer = SampleBuffer(
            capacity=config.sample_buffer.capacity,
            reinit_prob=config.sample_buffer.reinit_prob,
        )

        # MCMC hyperparameters from config
        self.mcmc_steps = config.mcmc.n_steps
        self.mcmc_step_size = config.mcmc.step_size
        self.mcmc_noise_scale = config.mcmc.noise_scale
        self.mcmc_temperature = config.mcmc.temperature

        # Regularization from config
        self.alpha = config.alpha

    def train_step(
        self,
        batch: dict[str, jax.Array],
    ) -> dict[str, Any]:
        """Perform one training step with persistent contrastive divergence.

        Args:
            batch: Training batch

        Returns:
            Dictionary containing loss and metrics
        """
        from artifex.generative_models.models.energy.mcmc import persistent_contrastive_divergence

        # Extract real data
        real_data = self._extract_data_from_batch(batch)

        # Extract raw JAX keys from internal rngs BEFORE any traced operations
        # This is critical: nnx.Rngs mutation inside traced functions causes TraceContextError
        noise_key = self.rngs.noise()
        sample_key = self.rngs.sample()

        # Add small amount of noise to real data (training trick)
        small_noise = jax.random.normal(noise_key, real_data.shape) * 0.005
        real_data = real_data + small_noise
        real_data = jnp.clip(real_data, -1.0, 1.0)

        # Set energy function to eval mode to disable dropout during MCMC
        # This prevents dropout from trying to mutate its RNG inside nnx.vmap(nnx.grad(...))
        self.energy_fn.eval()

        # Generate samples using persistent contrastive divergence
        _, fake_data = persistent_contrastive_divergence(
            energy_fn=self.energy,
            real_samples=real_data,
            sample_buffer=self.sample_buffer,
            rng_key=sample_key,
            n_mcmc_steps=self.mcmc_steps,
            step_size=self.mcmc_step_size,
            noise_scale=self.mcmc_noise_scale,
            temperature=self.mcmc_temperature,
        )

        # Set back to train mode for loss computation (if needed)
        self.energy_fn.train()

        # Compute loss
        loss_dict = self.contrastive_divergence_loss(
            real_data=real_data,
            fake_data=fake_data,
            alpha=self.alpha,
        )

        return loss_dict

    def sample_from_buffer(self, n_samples: int) -> jax.Array:
        """Sample from the current sample buffer.

        Args:
            n_samples: Number of samples to return

        Returns:
            Samples from the buffer
        """
        if not self.sample_buffer.buffer:
            raise RuntimeError("Sample buffer is empty")

        # Extract raw JAX key from internal rngs
        sample_key = self.rngs.sample()

        # Get a representative shape from the buffer
        sample_shape = self.sample_buffer.buffer[0].shape[1:]

        return self.sample_buffer.sample_initial(
            batch_size=n_samples,
            rng_key=sample_key,
            sample_shape=sample_shape,
        )

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        base_config = super().get_config()
        base_config.update(
            {
                "sample_buffer_capacity": self.sample_buffer.capacity,
                "sample_buffer_reinit_prob": self.sample_buffer.reinit_prob,
                "mcmc_steps": self.mcmc_steps,
                "mcmc_step_size": self.mcmc_step_size,
                "mcmc_noise_scale": self.mcmc_noise_scale,
                "mcmc_temperature": self.mcmc_temperature,
                "alpha": self.alpha,
            }
        )
        return base_config


class DeepEBM(EBM):
    """Deep Energy-Based Model for complex data.

    A deeper EBM implementation suitable for complex datasets like images.
    Uses CNN energy function with residual connections and normalization.

    Uses nested DeepEBMConfig with:
    - input_shape: Shape of input images (H, W, C)
    - energy_network: EnergyNetworkConfig with network_type="cnn"
    - mcmc: MCMCConfig for MCMC sampling parameters
    - sample_buffer: SampleBufferConfig for the replay buffer
    """

    def __init__(
        self,
        config: DeepEBMConfig,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ) -> None:
        """Initialize Deep EBM.

        Args:
            config: DeepEBMConfig with nested energy_network, mcmc, sample_buffer configs.
                    Uses CNN energy function for image data.
            rngs: Random number generators
            precision: Numerical precision

        Raises:
            TypeError: If config is not a DeepEBMConfig
        """
        # Validate config type
        if not isinstance(config, DeepEBMConfig):
            raise TypeError(
                f"config must be DeepEBMConfig, got {type(config).__name__}. "
                "Use DeepEBMConfig(...) to create the configuration."
            )

        # Store config
        self.config = config

        # For DeepEBM, we use a special deep CNN energy function
        # Extract input channels from input_shape (H, W, C)
        input_channels = config.input_shape[2] if len(config.input_shape) == 3 else 1

        # Check if config specifies CNN, otherwise default to deep CNN
        if config.energy_network.network_type == "cnn":
            # Create DeepCNNEnergyFunction for image data
            energy_fn = DeepCNNEnergyFunction(
                hidden_dims=list(config.energy_network.hidden_dims),
                input_channels=input_channels,
                use_residual=config.energy_network.use_residual,
                use_spectral_norm=config.energy_network.use_spectral_norm,
                rngs=rngs,
            )
        else:
            # Fallback to standard factory
            energy_fn = create_energy_function(
                config.energy_network,
                input_channels=input_channels,
                rngs=rngs,
            )

        # Initialize base class (EnergyBasedModel, not EBM to avoid double init)
        EnergyBasedModel.__init__(
            self,
            energy_fn=energy_fn,
            rngs=rngs,
            precision=precision,
        )

        # Initialize sample buffer from config
        self.sample_buffer = SampleBuffer(
            capacity=config.sample_buffer.capacity,
            reinit_prob=config.sample_buffer.reinit_prob,
        )

        # MCMC hyperparameters from config
        self.mcmc_steps = config.mcmc.n_steps
        self.mcmc_step_size = config.mcmc.step_size
        self.mcmc_noise_scale = config.mcmc.noise_scale
        self.mcmc_temperature = config.mcmc.temperature

        # Regularization from config
        self.alpha = config.alpha


class DeepCNNEnergyFunction(EnergyFunction):
    """Deep CNN energy function with residual connections.

    A more sophisticated CNN energy function suitable for complex image data.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        *,
        input_channels: int = 1,
        use_residual: bool = True,
        use_spectral_norm: bool = True,
        activation: Callable = nnx.silu,
        kernel_size: int = 3,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize deep CNN energy function.

        Args:
            hidden_dims: Channel dimensions for conv layers
            input_channels: Number of input channels
            use_residual: Whether to use residual connections
            use_spectral_norm: Whether to use spectral normalization
            activation: Activation function
            kernel_size: Convolution kernel size
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)

        self.hidden_dims = hidden_dims
        self.input_channels = input_channels
        self.use_residual = use_residual
        self.use_spectral_norm = use_spectral_norm
        self.activation = activation
        self.kernel_size = kernel_size

        # Build blocks
        self.blocks = nnx.List([])
        in_channels = input_channels

        for i, out_channels in enumerate(hidden_dims):
            block = EnergyBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2 if i < len(hidden_dims) - 1 else 1,  # No stride on last layer
                use_residual=use_residual and in_channels == out_channels,
                activation=activation,
                rngs=rngs,
            )
            self.blocks.append(block)
            in_channels = out_channels

        # Global pooling and final layers
        self.global_pool = lambda x: jnp.mean(x, axis=(1, 2))

        self.final_layers = nnx.List([])
        final_dim = hidden_dims[-1]

        # Add some fully connected layers
        for intermediate_dim in [final_dim // 2, final_dim // 4]:
            self.final_layers.append(
                nnx.Linear(
                    in_features=final_dim,
                    out_features=intermediate_dim,
                    rngs=rngs,
                )
            )
            final_dim = intermediate_dim

        # Output layer
        self.output_layer = nnx.Linear(
            in_features=final_dim,
            out_features=1,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass through deep CNN energy function.

        Args:
            x: Input images of shape (batch_size, height, width, channels)
            deterministic: Whether to disable dropout (unused for CNN)

        Returns:
            Energy values of shape (batch_size,)
        """
        # Forward through conv blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)

        # Final layers
        for layer in self.final_layers:
            x = layer(x)
            x = self.activation(x)

        # Output
        x = self.output_layer(x)

        return x.squeeze(-1)


class EnergyBlock(nnx.Module):
    """A single block for the deep CNN energy function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        use_residual: bool = False,
        activation: Callable = nnx.silu,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize energy block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            use_residual: Whether to use residual connection
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_residual = use_residual and in_channels == out_channels and stride == 1
        self.activation = activation

        # Main convolution
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding="SAME",
            rngs=rngs,
        )

        # Normalization (group norm for better stability)
        self.norm = nnx.GroupNorm(
            num_groups=min(32, out_channels),
            num_features=out_channels,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through energy block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        residual = x

        # Main path
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Residual connection
        if self.use_residual:
            x = x + residual

        return x


# =============================================================================
# Factory Functions for Common Datasets
# =============================================================================


def create_mnist_ebm(*, rngs: nnx.Rngs, **kwargs: Any) -> EBM:
    """Create EBM configured for MNIST dataset.

    Args:
        rngs: Random number generators
        **kwargs: Override default config values

    Returns:
        EBM configured for MNIST (28x28x1 images)
    """
    from artifex.generative_models.core.configuration.energy_config import (
        EnergyNetworkConfig,
        MCMCConfig,
        SampleBufferConfig,
    )

    # Create nested configs with sensible defaults for MNIST
    energy_network = EnergyNetworkConfig(
        name="mnist_energy_network",
        hidden_dims=kwargs.pop("hidden_dims", (16, 32, 64)),
        activation=kwargs.pop("activation", "silu"),
        network_type="cnn",
        use_bias=True,
    )

    mcmc = MCMCConfig(
        name="mnist_mcmc",
        n_steps=kwargs.pop("mcmc_steps", 60),
        step_size=kwargs.pop("step_size", 0.01),
        noise_scale=kwargs.pop("noise_scale", 0.005),
    )

    sample_buffer = SampleBufferConfig(
        name="mnist_buffer",
        capacity=kwargs.pop("sample_buffer_capacity", 4096),
        reinit_prob=kwargs.pop("reinit_prob", 0.05),
    )

    # For CNN-based EBM on MNIST, we need to use DeepEBMConfig
    # since it handles image input shapes
    from artifex.generative_models.core.configuration.energy_config import DeepEBMConfig

    config = DeepEBMConfig(
        name="mnist_ebm",
        input_shape=(28, 28, 1),  # MNIST dimensions
        energy_network=energy_network,
        mcmc=mcmc,
        sample_buffer=sample_buffer,
        alpha=kwargs.pop("alpha", 0.01),
    )

    # Use DeepEBM for image data
    return DeepEBM(config=config, rngs=rngs)


def create_cifar_ebm(*, rngs: nnx.Rngs, **kwargs: Any) -> DeepEBM:
    """Create Deep EBM configured for CIFAR dataset.

    Args:
        rngs: Random number generators
        **kwargs: Override default config values

    Returns:
        DeepEBM configured for CIFAR (32x32x3 images)
    """
    from artifex.generative_models.core.configuration.energy_config import (
        EnergyNetworkConfig,
        MCMCConfig,
        SampleBufferConfig,
    )

    # Create nested configs with sensible defaults for CIFAR
    energy_network = EnergyNetworkConfig(
        name="cifar_energy_network",
        hidden_dims=kwargs.pop("hidden_dims", (32, 64, 128, 256)),
        activation=kwargs.pop("activation", "silu"),
        network_type="cnn",
        use_bias=True,
        use_residual=kwargs.pop("use_residual", True),
        use_spectral_norm=kwargs.pop("use_spectral_norm", True),
    )

    mcmc = MCMCConfig(
        name="cifar_mcmc",
        n_steps=kwargs.pop("mcmc_steps", 100),
        step_size=kwargs.pop("step_size", 0.005),
        noise_scale=kwargs.pop("noise_scale", 0.001),
    )

    sample_buffer = SampleBufferConfig(
        name="cifar_buffer",
        capacity=kwargs.pop("sample_buffer_capacity", 8192),
        reinit_prob=kwargs.pop("reinit_prob", 0.05),
    )

    from artifex.generative_models.core.configuration.energy_config import DeepEBMConfig

    config = DeepEBMConfig(
        name="cifar_ebm",
        input_shape=(32, 32, 3),  # CIFAR dimensions
        energy_network=energy_network,
        mcmc=mcmc,
        sample_buffer=sample_buffer,
        alpha=kwargs.pop("alpha", 0.001),  # Lower regularization for deeper models
    )

    return DeepEBM(config=config, rngs=rngs)


def create_simple_ebm(input_dim: int, *, rngs: nnx.Rngs, **kwargs: Any) -> EBM:
    """Create simple MLP EBM for tabular data.

    Args:
        input_dim: Dimensionality of input data
        rngs: Random number generators
        **kwargs: Override default config values

    Returns:
        EBM with MLP energy function for tabular data
    """
    from artifex.generative_models.core.configuration.energy_config import (
        EnergyNetworkConfig,
        MCMCConfig,
        SampleBufferConfig,
    )

    # Create nested configs with sensible defaults for tabular data
    energy_network = EnergyNetworkConfig(
        name="simple_energy_network",
        hidden_dims=kwargs.pop("hidden_dims", (128, 128, 64)),
        activation=kwargs.pop("activation", "gelu"),
        network_type="mlp",
        use_bias=True,
        dropout_rate=kwargs.pop("dropout_rate", 0.1),
    )

    mcmc = MCMCConfig(
        name="simple_mcmc",
        n_steps=kwargs.pop("mcmc_steps", 60),
        step_size=kwargs.pop("step_size", 0.01),
        noise_scale=kwargs.pop("noise_scale", 0.005),
    )

    sample_buffer = SampleBufferConfig(
        name="simple_buffer",
        capacity=kwargs.pop("sample_buffer_capacity", 2048),
        reinit_prob=kwargs.pop("reinit_prob", 0.05),
    )

    config = EBMConfig(
        name="simple_ebm",
        input_dim=input_dim,
        energy_network=energy_network,
        mcmc=mcmc,
        sample_buffer=sample_buffer,
        alpha=kwargs.pop("alpha", 0.01),
    )

    return EBM(config=config, rngs=rngs)
