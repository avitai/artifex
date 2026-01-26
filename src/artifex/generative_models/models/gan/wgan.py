"""Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation."""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.configuration.gan_config import WGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.models.gan.base import Discriminator, Generator


class WGANGenerator(Generator):
    """Wasserstein GAN Generator using convolutional architecture.

    Based on the PyTorch WGAN-GP reference implementation:
    - Uses ConvTranspose layers like DCGAN
    - BatchNorm is typically used in WGAN generators
    - Tanh activation at output
    """

    def __init__(
        self,
        config: ConvGeneratorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the WGAN generator.

        Args:
            config: ConvGeneratorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConvGeneratorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for WGANGenerator")

        if not isinstance(config, ConvGeneratorConfig):
            raise TypeError(f"config must be ConvGeneratorConfig, got {type(config).__name__}")

        hidden_dims_list = list(config.hidden_dims)

        # Call the parent class initialization with config object
        super().__init__(config=config, rngs=rngs)

        # Extract conv parameters from config
        kernel_size = config.kernel_size
        stride = config.stride
        padding = config.padding
        batch_norm_momentum = config.batch_norm_momentum
        batch_norm_use_running_avg = config.batch_norm_use_running_avg

        # Store WGAN-specific convolutional architecture attributes
        self.wgan_activation_fn = self._get_activation_fn(config.activation)

        # Calculate initial spatial dimensions
        channels, height, width = self.output_shape
        self.init_h, self.init_w = (
            height // (2 ** len(hidden_dims_list)),
            width // (2 ** len(hidden_dims_list)),
        )

        # DCGAN "project and reshape" - Linear layer for initial projection
        # Following DCGAN architecture: latent vector -> fully connected -> reshape
        self.initial_linear = nnx.Linear(
            in_features=config.latent_dim,
            out_features=self.init_h * self.init_w * hidden_dims_list[0],
            rngs=rngs,
        )

        # Initial batch norm for the projected features
        if config.batch_norm:
            self.initial_bn = nnx.BatchNorm(
                num_features=hidden_dims_list[0],
                use_running_average=batch_norm_use_running_avg,
                momentum=batch_norm_momentum,
                rngs=rngs,
            )

        # Transposed convolutions: 1024->512->256->channels
        self.conv_transpose_layers = nnx.List([])
        self.wgan_batch_norm_layers = nnx.List([])

        for i in range(len(hidden_dims_list) - 1):
            in_features = hidden_dims_list[i]
            out_features = hidden_dims_list[i + 1]

            self.conv_transpose_layers.append(
                nnx.ConvTranspose(
                    in_features=in_features,
                    out_features=out_features,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    rngs=rngs,
                )
            )

            if config.batch_norm:
                self.wgan_batch_norm_layers.append(
                    nnx.BatchNorm(
                        num_features=out_features,
                        use_running_average=batch_norm_use_running_avg,
                        momentum=batch_norm_momentum,
                        rngs=rngs,
                    )
                )

        # Final output layer: 256 -> channels
        self.output_conv = nnx.ConvTranspose(
            in_features=hidden_dims_list[-1],
            out_features=channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=rngs,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        """Generate samples from latent vectors.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            z: Latent vectors of shape (batch_size, latent_dim).

        Returns:
            Generated samples of shape (batch_size, channels, height, width).
        """
        # DCGAN "project and reshape": Linear projection then reshape to feature maps
        # (batch, latent_dim) -> (batch, init_h * init_w * channels)
        x = self.initial_linear(z)

        # Reshape to feature maps:
        # (batch, init_h * init_w * channels) -> (batch, init_h, init_w, channels)
        batch_size = z.shape[0]
        x = jnp.reshape(x, (batch_size, self.init_h, self.init_w, self.hidden_dims[0]))

        # Apply batch norm and activation
        if hasattr(self, "initial_bn"):
            x = self.initial_bn(x)  # Auto mode from model.train()/eval()
        x = self.wgan_activation_fn(x)

        # Transposed convolutions with progressive upsampling
        for i, conv_transpose in enumerate(self.conv_transpose_layers):
            x = conv_transpose(x)
            if i < len(self.wgan_batch_norm_layers):
                x = self.wgan_batch_norm_layers[i](x)  # Auto mode
            x = self.wgan_activation_fn(x)

        # Final output layer without batch norm
        x = self.output_conv(x)
        x = jnp.tanh(x)  # Tanh activation for image output

        # Convert from NHWC to NCHW format
        return jnp.transpose(x, (0, 3, 1, 2))


class WGANDiscriminator(Discriminator):
    """Wasserstein GAN Discriminator (Critic) using convolutional architecture.

    Key differences from standard discriminator:
    - Uses InstanceNorm instead of BatchNorm (as per WGAN-GP paper)
    - No sigmoid activation at the end (outputs raw scores)
    - LeakyReLU activation
    """

    def __init__(
        self,
        config: ConvDiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the WGAN discriminator.

        Args:
            config: ConvDiscriminatorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConvDiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for WGANDiscriminator")

        if not isinstance(config, ConvDiscriminatorConfig):
            raise TypeError(f"config must be ConvDiscriminatorConfig, got {type(config).__name__}")

        hidden_dims_list = list(config.hidden_dims)

        # Call parent constructor with config object
        super().__init__(config=config, rngs=rngs)

        # Extract WGAN-specific parameters from config
        use_instance_norm = config.use_instance_norm
        kernel_size = config.kernel_size
        stride = config.stride
        padding = config.padding

        channels, height, width = config.input_shape

        # Convolutional layers
        self.conv_layers = nnx.List([])
        self.norm_layers = nnx.List([])

        # First layer: channels -> 256 (no normalization as per WGAN-GP paper)
        self.conv_layers.append(
            nnx.Conv(
                in_features=channels,
                out_features=hidden_dims_list[0],  # 256
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                rngs=rngs,
            )
        )

        # Subsequent layers: 256->512->1024
        for i in range(len(hidden_dims_list) - 1):
            in_features = hidden_dims_list[i]
            out_features = hidden_dims_list[i + 1]

            self.conv_layers.append(
                nnx.Conv(
                    in_features=in_features,
                    out_features=out_features,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    rngs=rngs,
                )
            )

            # Instance normalization for all layers except first
            if use_instance_norm:
                # Note: JAX/Flax doesn't have built-in InstanceNorm,
                # so we'll use GroupNorm with groups=out_features (equivalent to InstanceNorm)
                self.norm_layers.append(
                    nnx.GroupNorm(
                        num_features=out_features,  # Total number of features
                        num_groups=out_features,  # One group per channel = InstanceNorm
                        rngs=rngs,
                    )
                )

        # Final output layer: 1024 -> 1
        self.output_conv = nnx.Conv(
            in_features=hidden_dims_list[-1],
            out_features=1,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="VALID",  # No padding for final layer
            rngs=rngs,
        )

        # Store for use in forward pass
        self.wgan_use_instance_norm = use_instance_norm

    def __call__(self, x: jax.Array) -> jax.Array:
        """Discriminate samples.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input samples of shape (batch_size, channels, height, width).

        Returns:
            Raw discrimination scores of shape (batch_size,).
        """
        # Convert from NCHW to NHWC format for JAX/Flax Conv layers
        x = jnp.transpose(x, (0, 2, 3, 1))

        # First conv layer (no normalization)
        x = self.conv_layers[0](x)
        x = jax.nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        # Subsequent conv layers with normalization
        norm_idx = 0
        for i in range(1, len(self.conv_layers)):
            x = self.conv_layers[i](x)

            if self.wgan_use_instance_norm and norm_idx < len(self.norm_layers):
                x = self.norm_layers[norm_idx](x)
                norm_idx += 1

            x = jax.nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        # Final output layer (no activation - raw scores for Wasserstein distance)
        x = self.output_conv(x)

        # Global average pooling to get scalar output per sample
        # This handles variable spatial sizes gracefully
        x = jnp.mean(x, axis=(1, 2))  # Average over spatial dimensions
        return jnp.squeeze(x, axis=-1)  # Remove channel dimension


def compute_gradient_penalty(
    discriminator: WGANDiscriminator,
    real_samples: jax.Array,
    fake_samples: jax.Array,
    rngs: nnx.Rngs,
    lambda_gp: float = 10.0,
) -> jax.Array:
    """Compute gradient penalty for WGAN-GP.

    The gradient penalty enforces the Lipschitz constraint by penalizing
    the discriminator when the gradient norm deviates from 1.

    Args:
        discriminator: The discriminator network.
        real_samples: Real samples from the dataset.
        fake_samples: Generated fake samples.
        rngs: Random number generators for interpolation.
        lambda_gp: Gradient penalty weight.

    Returns:
        Gradient penalty loss value.
    """
    batch_size = real_samples.shape[0]

    # Sample random interpolation weights
    # Shape: (batch_size, 1, 1, 1) for broadcasting over image dimensions
    alpha_shape = (batch_size, 1, 1, 1)
    alpha = jax.random.uniform(rngs.params(), shape=alpha_shape, minval=0.0, maxval=1.0)

    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    # Get discriminator output for interpolated samples
    def discriminator_fn(x):
        return jnp.sum(discriminator(x))

    # Compute gradients of discriminator output w.r.t. interpolated samples
    gradients = jax.grad(discriminator_fn)(interpolated)

    # Flatten gradients for each sample
    gradients = jnp.reshape(gradients, (batch_size, -1))

    # Compute gradient norm for each sample
    gradient_norms = jnp.sqrt(jnp.sum(gradients**2, axis=1) + 1e-12)

    # Gradient penalty: penalize deviation from unit norm
    gradient_penalty = jnp.mean((gradient_norms - 1.0) ** 2) * lambda_gp

    return gradient_penalty


class WGAN(nnx.Module):
    """Wasserstein GAN with Gradient Penalty (WGAN-GP) model.

    Based on the PyTorch reference implementation with proper convolutional architecture.
    """

    def __init__(
        self,
        config: WGANConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the WGAN-GP model.

        Args:
            config: WGANConfig with nested ConvGeneratorConfig and ConvDiscriminatorConfig
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not WGANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for WGAN")

        if not isinstance(config, WGANConfig):
            raise TypeError(f"config must be WGANConfig, got {type(config).__name__}")

        super().__init__()

        # Extract nested configs (already validated by WGANConfig.__post_init__)
        gen_config: ConvGeneratorConfig = config.generator  # type: ignore
        disc_config: ConvDiscriminatorConfig = config.discriminator  # type: ignore

        # Create generator with config
        self.generator = WGANGenerator(config=gen_config, rngs=rngs)

        # Create discriminator with config
        self.discriminator = WGANDiscriminator(config=disc_config, rngs=rngs)

        # Store RNGs for sampling
        self.rngs = rngs

        # Store WGAN-specific parameters from config
        self.lambda_gp = config.gradient_penalty_weight
        self.n_critic = config.critic_iterations
        self.latent_dim = gen_config.latent_dim
        self.config = config

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples from the generator.

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generator
            batch_size: Alternative way to specify number of samples (for compatibility)
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples
        """
        # Use batch_size if provided, otherwise use n_samples
        num_samples = batch_size if batch_size is not None else n_samples

        sample_rng = (rngs or self.rngs).sample()

        # Sample latent vectors
        z = jax.random.normal(sample_rng, (num_samples, self.latent_dim))

        # Generate samples
        return self.generator(z)

    def discriminator_loss(
        self,
        real_samples: jax.Array,
        fake_samples: jax.Array,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Compute WGAN-GP discriminator loss.

        Args:
            real_samples: Real samples from dataset.
            fake_samples: Generated fake samples.
            rngs: Random number generators.

        Returns:
            Discriminator loss value.
        """
        # WGAN discriminator loss: minimize -E[D(real)] + E[D(fake)]
        real_validity = self.discriminator(real_samples)
        fake_validity = self.discriminator(fake_samples)

        wasserstein_distance = jnp.mean(fake_validity) - jnp.mean(real_validity)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            self.discriminator, real_samples, fake_samples, rngs, self.lambda_gp
        )

        # Total discriminator loss
        disc_loss = wasserstein_distance + gradient_penalty

        return disc_loss

    def generator_loss(self, fake_samples: jax.Array) -> jax.Array:
        """Compute WGAN generator loss.

        Args:
            fake_samples: Generated fake samples.

        Returns:
            Generator loss value.
        """
        # WGAN generator loss: minimize -E[D(fake)]
        fake_validity = self.discriminator(fake_samples)
        gen_loss = -jnp.mean(fake_validity)

        return gen_loss
