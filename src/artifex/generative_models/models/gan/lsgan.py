"""Least Squares GAN (LSGAN) implementation.

Following Flax NNX best practices:
- All parameters in configuration objects
- No separate conv parameters (use ConvGeneratorConfig/ConvDiscriminatorConfig)
- Activation functions as strings
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.configuration.gan_config import LSGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.core.losses.adversarial import least_squares_discriminator_loss
from artifex.generative_models.models.gan.base import Discriminator, Generator


class LSGANGenerator(Generator):
    """Least Squares GAN Generator using convolutional architecture.

    LSGAN uses the same architecture as DCGAN but with least squares loss
    instead of the standard adversarial loss.
    """

    def __init__(
        self,
        config: ConvGeneratorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the LSGAN generator.

        Args:
            config: ConvGeneratorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConvGeneratorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for LSGANGenerator")

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

        # Get activation function from name
        self.lsgan_activation_fn = self._get_activation(config.activation)

        # Calculate initial spatial dimensions
        channels, height, width = self.output_shape
        self.init_h, self.init_w = (
            height // (2 ** len(hidden_dims_list)),
            width // (2 ** len(hidden_dims_list)),
        )

        # Initial linear layer to project latent vector to feature map
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

        # Transposed convolution layers
        self.conv_transpose_layers = nnx.List([])
        self.lsgan_batch_norm_layers = nnx.List([])

        for i, dim in enumerate(hidden_dims_list[1:]):
            self.conv_transpose_layers.append(
                nnx.ConvTranspose(
                    in_features=hidden_dims_list[i],
                    out_features=dim,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    rngs=rngs,
                )
            )

            if config.batch_norm and i < len(hidden_dims_list) - 2:
                self.lsgan_batch_norm_layers.append(
                    nnx.BatchNorm(
                        num_features=dim,
                        use_running_average=batch_norm_use_running_avg,
                        momentum=batch_norm_momentum,
                        rngs=rngs,
                    )
                )

        # Output layer to generate final image
        self.output_conv = nnx.ConvTranspose(
            in_features=hidden_dims_list[-1],
            out_features=channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=rngs,
        )

    def _get_activation(self, activation: str):
        """Get activation function from name.

        Args:
            activation: Activation function name.

        Returns:
            Activation function.
        """
        if activation == "relu":
            return nnx.relu
        elif activation == "leaky_relu":
            return lambda x: nnx.leaky_relu(x, negative_slope=0.2)
        elif activation == "gelu":
            return nnx.gelu
        elif activation == "silu":
            return nnx.silu
        elif activation == "tanh":
            return jnp.tanh
        elif activation == "sigmoid":
            return nnx.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def __call__(self, z: jax.Array) -> jax.Array:
        """Forward pass through LSGAN generator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            Generated images of shape (batch_size, channels, height, width)
        """
        # Project latent vector to initial feature map
        x = self.initial_linear(z)

        # Reshape to spatial dimensions
        x = jnp.reshape(x, (-1, self.init_h, self.init_w, self.hidden_dims[0]))

        # Apply initial batch norm and activation
        if self.batch_norm:
            x = self.initial_bn(x)  # Auto mode from model.train()/eval()
        x = self.lsgan_activation_fn(x)

        # Pass through transposed convolution layers
        for i, conv_layer in enumerate(self.conv_transpose_layers):
            x = conv_layer(x)

            # Apply batch norm if available and not the last layer
            if self.batch_norm and i < len(self.lsgan_batch_norm_layers):
                x = self.lsgan_batch_norm_layers[i](x)  # Auto mode

            # Apply activation
            x = self.lsgan_activation_fn(x)

        # Final output layer with tanh activation
        x = self.output_conv(x)
        x = jnp.tanh(x)

        # Convert from NHWC to NCHW format (JAX convention)
        x = jnp.transpose(x, (0, 3, 1, 2))

        return x


class LSGANDiscriminator(Discriminator):
    """Least Squares GAN Discriminator using convolutional architecture.

    LSGAN discriminator uses the same architecture as DCGAN discriminator
    but with least squares loss instead of sigmoid cross-entropy loss.
    """

    def __init__(
        self,
        config: ConvDiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the LSGAN discriminator.

        Args:
            config: ConvDiscriminatorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConvDiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for LSGANDiscriminator")

        if not isinstance(config, ConvDiscriminatorConfig):
            raise TypeError(f"config must be ConvDiscriminatorConfig, got {type(config).__name__}")

        hidden_dims_list = list(config.hidden_dims)

        # Call parent class initialization with config object
        super().__init__(config=config, rngs=rngs)

        # Extract conv parameters from config
        kernel_size = config.kernel_size
        stride = config.stride
        padding = config.padding
        batch_norm_momentum = config.batch_norm_momentum
        batch_norm_use_running_avg = config.batch_norm_use_running_avg

        channels, height, width = config.input_shape

        # Convolutional layers
        self.conv_layers = nnx.List([])
        self.lsgan_batch_norm_layers = nnx.List([])

        in_channels = channels
        for i, out_channels in enumerate(hidden_dims_list):
            # Use stride 2 for downsampling
            self.conv_layers.append(
                nnx.Conv(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    rngs=rngs,
                )
            )

            # Batch norm (typically not used in first layer)
            if config.batch_norm and i > 0:
                self.lsgan_batch_norm_layers.append(
                    nnx.BatchNorm(
                        num_features=out_channels,
                        use_running_average=batch_norm_use_running_avg,
                        momentum=batch_norm_momentum,
                        rngs=rngs,
                    )
                )

            in_channels = out_channels

        # Calculate the spatial dimensions after convolutions
        final_h = height // (2 ** len(hidden_dims_list))
        final_w = width // (2 ** len(hidden_dims_list))
        final_features = final_h * final_w * hidden_dims_list[-1]

        # Output layer (no sigmoid activation for LSGAN)
        self.output_layer: nnx.Linear = nnx.Linear(
            in_features=final_features, out_features=1, rngs=rngs
        )

        # Dropout layer if needed
        if config.dropout_rate > 0:
            self.lsgan_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        else:
            self.lsgan_dropout = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through LSGAN discriminator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input images of shape (batch_size, channels, height, width)

        Returns:
            Discriminator scores (raw logits, no sigmoid for LSGAN)
        """
        # Convert from NCHW to NHWC format for convolutions
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Pass through convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)

            # Apply batch norm if available (skip first layer typically)
            if self.batch_norm and i > 0 and i - 1 < len(self.lsgan_batch_norm_layers):
                x = self.lsgan_batch_norm_layers[i - 1](x)  # Auto mode

            # Apply activation (use inherited activation_fn from parent)
            x = self.activation_fn(x)

            # Apply dropout
            if self.lsgan_dropout is not None:
                x = self.lsgan_dropout(x)  # Auto mode from model.train()/eval()

        # Flatten for output layer
        x = jnp.reshape(x, (x.shape[0], -1))

        # Output layer (no activation for least squares loss)
        return self.output_layer(x)


class LSGAN(nnx.Module):
    """Least Squares GAN implementation.

    LSGAN replaces the log loss in the original GAN formulation with
    a least squares loss, which provides more stable training and
    better quality gradients for the generator.

    Reference:
        Mao et al. "Least Squares Generative Adversarial Networks" (2017)
    """

    def __init__(
        self,
        config: LSGANConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize LSGAN.

        Args:
            config: LSGANConfig with nested ConvGeneratorConfig and
                ConvDiscriminatorConfig
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not LSGANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for LSGAN")

        if not isinstance(config, LSGANConfig):
            raise TypeError(f"config must be LSGANConfig, got {type(config).__name__}")

        super().__init__()

        # Store loss type for later use
        self.loss_type = "least_squares"

        # Extract nested configs (already validated by LSGANConfig.__post_init__)
        gen_config: ConvGeneratorConfig = config.generator  # type: ignore
        disc_config: ConvDiscriminatorConfig = config.discriminator  # type: ignore

        # Create generator with config
        self.generator = LSGANGenerator(config=gen_config, rngs=rngs)

        # Create discriminator with config
        self.discriminator = LSGANDiscriminator(config=disc_config, rngs=rngs)

        # Store LSGAN-specific parameters from config
        self.a = config.a
        self.b = config.b
        self.c = config.c
        self.latent_dim = gen_config.latent_dim
        self.config = config

    def generator_loss(
        self,
        fake_scores: jax.Array,
        target_real: float = 1.0,
        reduction: str = "mean",
    ) -> jax.Array:
        """Compute LSGAN generator loss.

        Args:
            fake_scores: Discriminator scores for fake samples
            target_real: Target value for fake samples (usually 1.0)
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Generator loss
        """
        from artifex.generative_models.core.losses.adversarial import (
            least_squares_generator_loss,
        )

        return least_squares_generator_loss(fake_scores, target_real, reduction)

    def discriminator_loss(
        self,
        real_scores: jax.Array,
        fake_scores: jax.Array,
        target_real: float = 1.0,
        target_fake: float = 0.0,
        reduction: str = "mean",
    ) -> jax.Array:
        """Compute LSGAN discriminator loss.

        Args:
            real_scores: Discriminator scores for real samples
            fake_scores: Discriminator scores for fake samples
            target_real: Target value for real samples (usually 1.0)
            target_fake: Target value for fake samples (usually 0.0)
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Discriminator loss
        """
        return least_squares_discriminator_loss(
            real_scores, fake_scores, target_real, target_fake, reduction
        )

    def training_step(
        self,
        real_images: jax.Array,
        latent_vectors: jax.Array,
    ) -> dict[str, jax.Array]:
        """Perform a single training step.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            real_images: Batch of real images
            latent_vectors: Batch of latent vectors

        Returns:
            Dictionary with loss values and generated images
        """
        # Generate fake images
        fake_images = self.generator(latent_vectors)

        # Get discriminator scores
        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        # Compute losses
        gen_loss = self.generator_loss(fake_scores)
        disc_loss = self.discriminator_loss(real_scores, fake_scores)

        return {
            "generator_loss": gen_loss,
            "discriminator_loss": disc_loss,
            "real_scores": real_scores,
            "fake_scores": fake_scores,
            "fake_images": fake_images,
        }
