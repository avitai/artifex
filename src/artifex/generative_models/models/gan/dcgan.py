"""Deep Convolutional GAN (DCGAN) implementation.

DCGAN uses convolutional architecture for both generator (transposed conv)
and discriminator (strided conv), following the paper guidelines:
- Replace pooling with strided convolutions (discriminator) and
  fractional-strided convolutions (generator)
- Use BatchNorm in both generator and discriminator
- Use ReLU in generator (except output which uses tanh)
- Use LeakyReLU in discriminator
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.configuration.gan_config import DCGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.models.gan.base import Discriminator, GAN, Generator


class DCGANGenerator(Generator):
    """Deep Convolutional GAN Generator.

    Uses transposed convolutions for progressive upsampling from latent vector
    to output image. All configuration comes from ConvGeneratorConfig.
    """

    def __init__(
        self,
        config: ConvGeneratorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the DCGAN generator.

        Args:
            config: ConvGeneratorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConvGeneratorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for DCGANGenerator")

        if not isinstance(config, ConvGeneratorConfig):
            raise TypeError(f"config must be ConvGeneratorConfig, got {type(config).__name__}")

        hidden_dims_list = list(config.hidden_dims)
        if not hidden_dims_list:
            raise ValueError("hidden_dims must be a non-empty tuple")

        # Call parent class with config object (ConvGeneratorConfig inherits from GeneratorConfig)
        super().__init__(config=config, rngs=rngs)

        # Calculate initial spatial dimensions for DCGAN architecture
        channels, height, width = self.output_shape
        num_upsample_layers = len(hidden_dims_list)
        stride_factor = config.stride[0]  # Assuming square strides
        self.init_h = max(1, height // (stride_factor**num_upsample_layers))
        self.init_w = max(1, width // (stride_factor**num_upsample_layers))

        # Initial projection from latent vector to spatial feature map
        initial_features = self.init_h * self.init_w * hidden_dims_list[0]
        self.initial_linear = nnx.Linear(
            in_features=config.latent_dim,
            out_features=initial_features,
            rngs=rngs,
        )

        # Transposed convolutions for progressive upsampling
        self.conv_transpose_layers = nnx.List([])
        self.dcgan_batch_norms = nnx.List([])

        for i, dim in enumerate(hidden_dims_list[1:]):
            self.conv_transpose_layers.append(
                nnx.ConvTranspose(
                    in_features=hidden_dims_list[i],
                    out_features=dim,
                    kernel_size=config.kernel_size,
                    strides=config.stride,
                    padding=config.padding,
                    rngs=rngs,
                )
            )

            # Apply batch norm to all hidden layers except the last one
            if config.batch_norm and i < len(hidden_dims_list) - 2:
                self.dcgan_batch_norms.append(
                    nnx.BatchNorm(
                        num_features=dim,
                        use_running_average=config.batch_norm_use_running_avg,
                        momentum=config.batch_norm_momentum,
                        rngs=rngs,
                    )
                )

        # Final transpose conv to output image
        self.output_conv = nnx.ConvTranspose(
            in_features=hidden_dims_list[-1],
            out_features=channels,
            kernel_size=config.kernel_size,
            strides=config.stride,
            padding=config.padding,
            rngs=rngs,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        """Generate samples from latent vectors.

        Args:
            z: Latent vectors of shape (batch_size, latent_dim)

        Returns:
            Generated samples of shape (batch_size, channels, height, width)
        """
        # Initial projection and reshape
        x = self.initial_linear(z)
        x = jnp.reshape(x, (-1, self.init_h, self.init_w, self.hidden_dims[0]))

        # Transposed convolutions with batch norm and activation
        for i, conv_transpose in enumerate(self.conv_transpose_layers):
            x = conv_transpose(x)

            # Apply batch normalization if available
            if i < len(self.dcgan_batch_norms):
                x = self.dcgan_batch_norms[i](x)

            # Use parent's activation function
            x = self.activation_fn(x)

        # Output convolutional layer
        x = self.output_conv(x)

        # Apply tanh activation for bounded outputs [-1, 1]
        x = jnp.tanh(x)

        # Reshape to expected format (batch, channels, height, width)
        return jnp.transpose(x, (0, 3, 1, 2))


class DCGANDiscriminator(Discriminator):
    """Deep Convolutional GAN Discriminator.

    Uses strided convolutions for progressive downsampling from input image
    to binary classification. All configuration comes from ConvDiscriminatorConfig.
    """

    def __init__(
        self,
        config: ConvDiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the DCGAN discriminator.

        Args:
            config: ConvDiscriminatorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConvDiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for DCGANDiscriminator")

        if not isinstance(config, ConvDiscriminatorConfig):
            raise TypeError(f"config must be ConvDiscriminatorConfig, got {type(config).__name__}")

        hidden_dims_list = list(config.hidden_dims)
        if not hidden_dims_list:
            raise ValueError("hidden_dims must be a non-empty tuple")

        # Call parent class with config object
        super().__init__(config=config, rngs=rngs)

        # DCGAN-specific convolutional layers for progressive downsampling
        self.conv_layers = nnx.List([])
        self.dcgan_batch_norms = nnx.List([])

        for i, dim in enumerate(hidden_dims_list):
            in_channels = config.input_shape[0] if i == 0 else hidden_dims_list[i - 1]
            conv = nnx.Conv(
                in_features=in_channels,
                out_features=dim,
                kernel_size=config.kernel_size,
                strides=config.stride,
                padding=config.padding,
                rngs=rngs,
            )

            # Spectral norm placeholder (not yet implemented in Flax NNX)
            if config.use_spectral_norm:
                pass

            self.conv_layers.append(conv)

            # Apply batch norm to all layers except the first (following DCGAN paper)
            if config.batch_norm and i > 0:
                self.dcgan_batch_norms.append(
                    nnx.BatchNorm(
                        num_features=dim,
                        use_running_average=config.batch_norm_use_running_avg,
                        momentum=config.batch_norm_momentum,
                        rngs=rngs,
                    )
                )

        # Calculate final flattened feature dimension
        num_downsample_layers = len(hidden_dims_list)
        stride_factor = config.stride[0]
        final_spatial_size = max(1, config.input_shape[1] // (stride_factor**num_downsample_layers))
        final_features = hidden_dims_list[-1] * final_spatial_size * final_spatial_size

        # Final linear layer for binary real/fake classification
        self.final_linear = nnx.Linear(
            in_features=final_features,
            out_features=config.output_dim,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Discriminate samples.

        Args:
            x: Input samples of shape (batch_size, channels, height, width)

        Returns:
            Discrimination scores of shape (batch_size, output_dim)
        """
        # Reshape input to NHWC format expected by Flax
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Convolutional layers with batch norm and activation
        bn_idx = 0
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)

            # Apply batch normalization for all but the first layer
            if i > 0 and bn_idx < len(self.dcgan_batch_norms):
                x = self.dcgan_batch_norms[bn_idx](x)
                bn_idx += 1

            # Use parent's activation function
            x = self.activation_fn(x)

        # Flatten and apply final linear layer
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.final_linear(x)

        # Apply sigmoid activation for [0,1] range
        x = jax.nn.sigmoid(x)

        return x


class DCGAN(GAN):
    """Deep Convolutional GAN (DCGAN) model.

    Uses DCGANConfig which contains ConvGeneratorConfig and ConvDiscriminatorConfig
    for complete architecture specification.
    """

    def __init__(
        self,
        config: DCGANConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the DCGAN model.

        Args:
            config: DCGANConfig with nested ConvGeneratorConfig and ConvDiscriminatorConfig
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not DCGANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for DCGAN")

        if not isinstance(config, DCGANConfig):
            raise TypeError(f"config must be DCGANConfig, got {type(config).__name__}")

        # Bypass GAN.__init__ and call GenerativeModel directly
        # GAN expects flat config but DCGANConfig uses nested configs
        from artifex.generative_models.core.base import GenerativeModel

        GenerativeModel.__init__(self, rngs=rngs, precision=None)

        # Store config
        self.config = config

        # Store RNG for dynamic use in generate() and loss_fn()
        if "sample" not in rngs:
            raise ValueError(
                "rngs must contain 'sample' stream for DCGAN. "
                "Initialize with: nnx.Rngs(params=0, dropout=1, sample=2)"
            )
        self.rngs = rngs

        # Create generator from nested ConvGeneratorConfig
        self.generator = DCGANGenerator(
            config=config.generator,
            rngs=rngs,
        )

        # Create discriminator from nested ConvDiscriminatorConfig
        self.discriminator = DCGANDiscriminator(
            config=config.discriminator,
            rngs=rngs,
        )

        # Store hyperparameters for training (extracted from nested configs)
        self.latent_dim = config.generator.latent_dim
        self.loss_type = config.loss_type
        self.gradient_penalty_weight = config.gradient_penalty_weight
        self.generator_lr = config.generator_lr
        self.discriminator_lr = config.discriminator_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
