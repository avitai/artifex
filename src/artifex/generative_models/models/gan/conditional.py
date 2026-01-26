"""Conditional Generative Adversarial Network (CGAN) implementation.

Based on the paper "Conditional Generative Adversarial Nets" by Mirza & Osindero (2014)
and the reference implementation from https://github.com/Lornatang/conditional_gan
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.configuration.gan_config import ConditionalGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConditionalDiscriminatorConfig,
    ConditionalGeneratorConfig,
)
from artifex.generative_models.models.gan.base import Discriminator, Generator


class ConditionalGenerator(Generator):
    """Conditional GAN Generator using convolutional architecture.

    The generator is conditioned on class labels by concatenating the label
    embedding with the noise vector before passing through the network.
    """

    def __init__(
        self,
        config: ConditionalGeneratorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Conditional GAN generator.

        Args:
            config: ConditionalGeneratorConfig with network architecture and
                conditional parameters (num_classes, embedding_dim via config.conditional)
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConditionalGeneratorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for ConditionalGenerator")

        if not isinstance(config, ConditionalGeneratorConfig):
            raise TypeError(
                f"config must be ConditionalGeneratorConfig, got {type(config).__name__}"
            )

        hidden_dims_list = list(config.hidden_dims)

        # Call the parent class initialization with config object
        super().__init__(config=config, rngs=rngs)

        # Extract conditional params from nested config (composition pattern)
        num_classes = config.conditional.num_classes
        embedding_dim = config.conditional.embedding_dim

        # Extract conv params from config
        kernel_size = config.kernel_size
        stride = config.stride
        padding = config.padding
        batch_norm_momentum = config.batch_norm_momentum
        batch_norm_use_running_avg = config.batch_norm_use_running_avg

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # Get activation function from string using parent class method
        self.activation_fn = self._get_activation_fn(config.activation)

        channels, height, width = config.output_shape

        # Label embedding - project class labels into embedding space
        self.label_embedding = nnx.Linear(
            in_features=num_classes,
            out_features=embedding_dim,
            rngs=rngs,
        )

        # Combined input projection (noise + label embedding)
        combined_input_dim = config.latent_dim + embedding_dim  # noise + label embedding

        # Calculate initial spatial dimensions based on target output size
        # We need to work backwards from the target size to determine starting size
        # Each ConvTranspose with stride=2 doubles the spatial dimensions
        # Count: intermediate layers + final output layer
        total_upsample_layers = len(hidden_dims_list) - 1 + 1  # +1 for output layer

        # Start small and upsample to reach target size
        target_size = min(height, width)
        self.init_h = self.init_w = target_size // (2**total_upsample_layers)

        # Ensure minimum size of at least 1
        if self.init_h < 1:
            self.init_h = self.init_w = 1

        # Initial projection from combined input to feature map
        self.initial_projection = nnx.Linear(
            in_features=combined_input_dim,
            out_features=self.init_h * self.init_w * hidden_dims_list[0],
            rngs=rngs,
        )

        if config.batch_norm:
            self.initial_bn = nnx.BatchNorm(
                num_features=hidden_dims_list[0],
                use_running_average=batch_norm_use_running_avg,
                momentum=batch_norm_momentum,
                rngs=rngs,
            )

        # Transposed convolutions for upsampling
        self.conv_transpose_layers = nnx.List([])
        self.batch_norm_layers = nnx.List([])

        # Create upsampling layers
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
                self.batch_norm_layers.append(
                    nnx.BatchNorm(
                        num_features=out_features,
                        use_running_average=batch_norm_use_running_avg,
                        momentum=batch_norm_momentum,
                        rngs=rngs,
                    )
                )

        # Final output layer
        self.output_conv = nnx.ConvTranspose(
            in_features=hidden_dims_list[-1],
            out_features=channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=rngs,
        )

    def __call__(  # type: ignore[override]
        self, z: jax.Array, labels: jax.Array | None = None
    ) -> jax.Array:
        """Generate samples from noise and labels.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            z: Noise tensor of shape (batch_size, latent_dim).
            labels: One-hot encoded labels of shape (batch_size, num_classes).

        Returns:
            Generated samples of shape (batch_size, channels, height, width).
        """
        if z.ndim != 2:
            raise ValueError(f"Expected noise tensor 'z' to have 2 dimensions, but got {z.ndim}.")

        if labels is None:
            raise ValueError("Labels must be provided for conditional generation.")

        if labels.ndim != 2:
            raise ValueError(f"Expected labels tensor to have 2 dimensions, but got {labels.ndim}.")

        if labels.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected labels to have {self.num_classes} classes, but got {labels.shape[1]}."
            )

        # Project labels to latent space
        label_embedding = self.label_embedding(labels)

        # Combine noise and label embedding
        x = jnp.concatenate([z, label_embedding], axis=1)

        # Initial projection to spatial dimensions
        x = self.initial_projection(x)
        x = jnp.reshape(x, (x.shape[0], self.init_h, self.init_w, self.hidden_dims[0]))

        # Apply initial batch norm and activation
        if hasattr(self, "initial_bn"):
            x = self.initial_bn(x)  # Auto mode from model.train()/eval()
        x = self.activation_fn(x)

        # Progressive upsampling through transposed convolutions
        for i, conv_transpose in enumerate(self.conv_transpose_layers):
            x = conv_transpose(x)
            if i < len(self.batch_norm_layers):
                x = self.batch_norm_layers[i](x)  # Auto mode from model.train()/eval()
            x = self.activation_fn(x)

        # Final output layer
        x = self.output_conv(x)
        x = jnp.tanh(x)  # Tanh activation for image output

        # Convert from NHWC to NCHW format
        return jnp.transpose(x, (0, 3, 1, 2))


class ConditionalDiscriminator(Discriminator):
    """Conditional GAN Discriminator using convolutional architecture.

    The discriminator is conditioned on class labels by concatenating the label
    embedding with the input image before passing through the network.
    """

    def __init__(
        self,
        config: ConditionalDiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Conditional GAN discriminator.

        Args:
            config: ConditionalDiscriminatorConfig with network architecture and
                conditional parameters (num_classes, embedding_dim via config.conditional)
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConditionalDiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for ConditionalDiscriminator")

        if not isinstance(config, ConditionalDiscriminatorConfig):
            raise TypeError(
                f"config must be ConditionalDiscriminatorConfig, got {type(config).__name__}"
            )

        hidden_dims_list = list(config.hidden_dims)

        # Call parent constructor with config object
        super().__init__(config=config, rngs=rngs)

        # Extract conditional params from nested config (composition pattern)
        num_classes = config.conditional.num_classes
        embedding_dim = config.conditional.embedding_dim

        # Extract conv params from config
        kernel_size = config.kernel_size
        stride = config.stride
        stride_first = config.stride_first
        padding = config.padding

        self.input_shape = config.input_shape
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.leaky_relu_slope = config.leaky_relu_slope

        channels, height, width = config.input_shape

        # Store expected input dimensions for runtime validation and reshaping
        self.expected_height = height
        self.expected_width = width
        self.expected_channels = channels

        # Label embedding - project class labels to match spatial dimensions
        # We'll create one channel of label information that matches image spatial size
        self.label_embedding = nnx.Linear(
            in_features=num_classes,
            out_features=height * width,  # Just one channel worth of spatial data
            rngs=rngs,
        )

        # Convolutional layers
        self.conv_layers = nnx.List([])

        # First layer: (channels + 1) -> first hidden dim
        # +1 for the concatenated label embedding channel
        self.conv_layers.append(
            nnx.Conv(
                in_features=channels + 1,
                out_features=hidden_dims_list[0],
                kernel_size=kernel_size,
                strides=stride_first,
                padding=padding,
                rngs=rngs,
            )
        )

        # Subsequent layers with downsampling
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

        # Final output layer
        self.output_conv = nnx.Conv(
            in_features=hidden_dims_list[-1],
            out_features=1,
            kernel_size=kernel_size,
            strides=stride_first,
            padding=padding,
            rngs=rngs,
        )

    def __call__(  # type: ignore[override]
        self, x: jax.Array, labels: jax.Array | None = None
    ) -> jax.Array:
        """Discriminate samples with label conditioning.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input samples of shape (batch_size, channels, height, width).
            labels: One-hot encoded labels of shape (batch_size, num_classes).

        Returns:
            Discrimination scores of shape (batch_size,).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input tensor 'x' to have 4 dimensions, but got {x.ndim}.")

        if labels is None:
            raise ValueError("Labels must be provided for conditional discrimination.")

        if labels.ndim != 2:
            raise ValueError(f"Expected labels tensor to have 2 dimensions, but got {labels.ndim}.")

        if labels.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected labels to have {self.num_classes} classes, but got {labels.shape[1]}."
            )

        batch_size, channels, height, width = x.shape

        # Validate input shape matches expected dimensions
        if (channels, height, width) != (
            self.expected_channels,
            self.expected_height,
            self.expected_width,
        ):
            expected_shape = (self.expected_channels, self.expected_height, self.expected_width)
            actual_shape = (channels, height, width)
            raise ValueError(
                f"Input shape mismatch: expected {expected_shape}, "
                f"but got {actual_shape}. The discriminator was configured for "
                f"input_shape={self.input_shape}."
            )

        # Convert from NCHW to NHWC format for JAX/Flax Conv layers
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Project labels to spatial dimensions and reshape
        # Use expected dimensions (from config) not runtime dimensions
        label_embedding = self.label_embedding(
            labels
        )  # Shape: (batch_size, expected_height*expected_width)
        label_embedding = jnp.reshape(
            label_embedding, (batch_size, self.expected_height, self.expected_width, 1)
        )  # Add channel dim

        # Concatenate input image with label embedding (adds 1 extra channel)
        x = jnp.concatenate([x, label_embedding], axis=-1)

        # Pass through convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = jax.nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)
            # Use inherited dropout from base class (only if dropout was created)
            if hasattr(self, "dropout") and self.dropout is not None:
                x = self.dropout(x)  # Auto mode from model.train()/eval()

        # Final output layer
        x = self.output_conv(x)

        # Global average pooling and flatten
        x = jnp.mean(x, axis=(1, 2))  # Average over spatial dimensions
        return jnp.squeeze(x, axis=-1)  # Shape: (batch_size,)


class ConditionalGAN(nnx.Module):
    """Conditional Generative Adversarial Network (CGAN).

    Based on "Conditional Generative Adversarial Nets" by Mirza & Osindero (2014).
    The generator and discriminator are both conditioned on class labels.

    Uses composition pattern: conditional parameters (num_classes, embedding_dim)
    are embedded in the nested ConditionalGeneratorConfig and ConditionalDiscriminatorConfig
    via ConditionalParams.
    """

    def __init__(
        self,
        config: "ConditionalGANConfig",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Conditional GAN model.

        Args:
            config: ConditionalGANConfig with nested ConditionalGeneratorConfig
                and ConditionalDiscriminatorConfig. All parameters are specified
                in the config objects.
            rngs: Random number generators.

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not ConditionalGANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for ConditionalGAN")

        super().__init__()

        # Validate config type
        from artifex.generative_models.core.configuration import ConditionalGANConfig

        if not isinstance(config, ConditionalGANConfig):
            raise TypeError(f"config must be ConditionalGANConfig, got {type(config).__name__}")

        # Extract nested configs (already validated by ConditionalGANConfig)
        gen_config: ConditionalGeneratorConfig = config.generator
        disc_config: ConditionalDiscriminatorConfig = config.discriminator

        # Create conditional generator with config object
        self.generator = ConditionalGenerator(config=gen_config, rngs=rngs)

        # Create conditional discriminator with config object
        self.discriminator = ConditionalDiscriminator(config=disc_config, rngs=rngs)

        # Store RNGs for sampling
        self.rngs = rngs

        # Store configuration (extract from nested configs via composition)
        self.num_classes = gen_config.conditional.num_classes
        self.latent_dim = gen_config.latent_dim
        self.config = config

    def generate(
        self,
        n_samples: int = 1,
        labels: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate conditional samples from the generator.

        Args:
            n_samples: Number of samples to generate
            labels: One-hot encoded labels of shape (n_samples, num_classes)
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

        # Handle labels
        if labels is None:
            # Generate random labels if none provided
            label_indices = jax.random.randint(sample_rng, (num_samples,), 0, self.num_classes)
            labels = jax.nn.one_hot(label_indices, self.num_classes)
        elif labels.ndim == 1:
            # Convert label indices to one-hot
            labels = jax.nn.one_hot(labels, self.num_classes)

        # Generate samples
        return self.generator(z, labels)

    def discriminator_loss(
        self,
        real_samples: jax.Array,
        fake_samples: jax.Array,
        real_labels: jax.Array,
        fake_labels: jax.Array,
    ) -> jax.Array:
        """Compute conditional discriminator loss.

        Args:
            real_samples: Real samples from dataset.
            fake_samples: Generated fake samples.
            real_labels: Labels for real samples.
            fake_labels: Labels for fake samples.

        Returns:
            Discriminator loss value.
        """
        # Get discriminator outputs for real and fake samples with their labels
        real_validity = self.discriminator(real_samples, real_labels)
        fake_validity = self.discriminator(fake_samples, fake_labels)

        # Standard GAN discriminator loss
        real_loss = -jnp.mean(jnp.log(nnx.sigmoid(real_validity) + 1e-12))
        fake_loss = -jnp.mean(jnp.log(1 - nnx.sigmoid(fake_validity) + 1e-12))

        return real_loss + fake_loss

    def generator_loss(self, fake_samples: jax.Array, fake_labels: jax.Array) -> jax.Array:
        """Compute conditional generator loss.

        Args:
            fake_samples: Generated fake samples.
            fake_labels: Labels for fake samples.

        Returns:
            Generator loss value.
        """
        # Get discriminator output for fake samples with their labels
        fake_validity = self.discriminator(fake_samples, fake_labels)

        # Standard GAN generator loss (fool the discriminator)
        gen_loss = -jnp.mean(jnp.log(nnx.sigmoid(fake_validity) + 1e-12))

        return gen_loss
