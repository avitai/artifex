"""CycleGAN implementation for unpaired image-to-image translation.

Based on the paper "Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks" by Zhu et al. (2017).

CycleGAN learns mappings between two image domains X and Y without requiring
paired training examples. It uses cycle consistency loss to enforce that
translated images can be mapped back to the original domain.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration.gan_config import CycleGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    CycleGANGeneratorConfig,
    PatchGANDiscriminatorConfig,
)
from artifex.generative_models.core.layers.residual import ResidualBlock
from artifex.generative_models.core.losses.reconstruction import mae_loss


class CycleGANGenerator(nnx.Module):
    """CycleGAN Generator for image-to-image translation.

    Uses a ResNet-based architecture with reflection padding as described
    in the original CycleGAN paper. This follows the pytorch reference
    implementation more closely.
    """

    def __init__(
        self,
        config: CycleGANGeneratorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CycleGAN generator.

        Args:
            config: CycleGANGeneratorConfig with network architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not CycleGANGeneratorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for CycleGANGenerator")

        if not isinstance(config, CycleGANGeneratorConfig):
            raise TypeError(f"config must be CycleGANGeneratorConfig, got {type(config).__name__}")

        super().__init__()

        # Store config for reference
        self.config = config

        # Extract configuration values
        self.input_shape = config.input_shape
        self.output_shape = config.output_shape
        self.hidden_dims = list(config.hidden_dims)
        self.n_residual_blocks = config.n_residual_blocks
        self.batch_norm = config.batch_norm
        self.dropout_rate = config.dropout_rate
        self.use_skip_connections = config.use_skip_connections

        # Get activation function
        activation_name = config.activation
        if hasattr(nnx, activation_name):
            self.activation = getattr(nnx, activation_name)
        elif hasattr(jax.nn, activation_name):
            self.activation = getattr(jax.nn, activation_name)
        else:
            self.activation = jax.nn.relu

        input_height, input_width, input_channels = config.input_shape
        output_height, output_width, output_channels = config.output_shape

        # Initial convolution
        self.initial_conv = nnx.Conv(
            in_features=input_channels,
            out_features=self.hidden_dims[0],
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        # Only create batch norm if needed (don't initialize to None)
        if self.batch_norm:
            self.initial_norm = nnx.BatchNorm(num_features=self.hidden_dims[0], rngs=rngs)

        # Downsampling layers
        self.downsample_layers = nnx.List([])
        self.downsample_norms = nnx.List([])

        in_channels = self.hidden_dims[0]
        for i, out_channels in enumerate(self.hidden_dims[1:]):
            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
            )
            self.downsample_layers.append(conv)

            norm = None
            if self.batch_norm:
                norm = nnx.BatchNorm(num_features=out_channels, rngs=rngs)
            self.downsample_norms.append(norm)

            in_channels = out_channels

        # Residual blocks
        self.residual_blocks = nnx.List([])
        res_channels = self.hidden_dims[-1]
        for i in range(self.n_residual_blocks):
            block = ResidualBlock(
                channels=res_channels,
                kernel_size=3,
                activation=self.activation,
                rngs=rngs,
            )
            self.residual_blocks.append(block)

        # Upsampling layers
        self.upsample_layers = nnx.List([])
        self.upsample_norms = nnx.List([])

        reversed_dims = list(reversed(self.hidden_dims))
        for i, out_channels in enumerate(reversed_dims[1:]):
            conv_transpose = nnx.ConvTranspose(
                in_features=reversed_dims[i],
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
            )
            self.upsample_layers.append(conv_transpose)

            norm = None
            if self.batch_norm:
                norm = nnx.BatchNorm(num_features=out_channels, rngs=rngs)
            self.upsample_norms.append(norm)

        # Final output layer
        self.output_conv = nnx.Conv(
            in_features=self.hidden_dims[0],
            out_features=output_channels,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        # Dropout layer
        self.dropout = None
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        **kwargs,
    ) -> jax.Array:
        """Forward pass through generator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input image tensor.

        Returns:
            Generated output image.
        """
        # Initial convolution
        h = self.initial_conv(x)
        if hasattr(self, "initial_norm"):
            h = self.initial_norm(h)  # Auto mode from model.train()/eval()
        h = self.activation(h)

        # Downsampling
        for conv, norm in zip(self.downsample_layers, self.downsample_norms):
            h = conv(h)
            if norm is not None:
                h = norm(h)  # Auto mode from model.train()/eval()
            h = self.activation(h)

        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # Upsampling
        for conv_transpose, norm in zip(self.upsample_layers, self.upsample_norms):
            h = conv_transpose(h)
            if norm is not None:
                h = norm(h)  # Auto mode from model.train()/eval()
            h = self.activation(h)

        # Final output with tanh activation
        h = self.output_conv(h)
        h = jnp.tanh(h)

        return h


class CycleGANDiscriminator(nnx.Module):
    """CycleGAN Discriminator (PatchGAN-style).

    Uses a PatchGAN discriminator that classifies patches of the input
    as real or fake, rather than the entire image.
    """

    def __init__(
        self,
        config: PatchGANDiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CycleGAN discriminator.

        Args:
            config: PatchGANDiscriminatorConfig with network architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not PatchGANDiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for CycleGANDiscriminator")

        if not isinstance(config, PatchGANDiscriminatorConfig):
            raise TypeError(
                f"config must be PatchGANDiscriminatorConfig, got {type(config).__name__}"
            )

        super().__init__()

        # Store config for reference
        self.config = config

        # Extract configuration values
        self.input_shape = config.input_shape
        self.hidden_dims = list(config.hidden_dims)
        self.batch_norm = config.batch_norm
        self.dropout_rate = config.dropout_rate

        # Get activation function
        activation_name = config.activation
        if activation_name == "leaky_relu":
            self.activation = lambda x: jax.nn.leaky_relu(x, negative_slope=config.leaky_relu_slope)
        elif hasattr(nnx, activation_name):
            self.activation = getattr(nnx, activation_name)
        elif hasattr(jax.nn, activation_name):
            self.activation = getattr(jax.nn, activation_name)
        else:
            self.activation = jax.nn.leaky_relu

        input_channels = config.input_shape[-1]  # Assuming (H, W, C) format

        # Build convolutional layers
        self.conv_layers = nnx.List([])
        self.norm_layers = nnx.List([])

        in_channels = input_channels
        for i, out_channels in enumerate(self.hidden_dims):
            # Convolutional layer
            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=config.kernel_size,
                strides=config.stride,
                padding=config.padding,
                rngs=rngs,
            )
            self.conv_layers.append(conv)

            # Batch normalization (skip for first layer)
            norm = None
            if self.batch_norm and i > 0:
                norm = nnx.BatchNorm(num_features=out_channels, rngs=rngs)
            self.norm_layers.append(norm)

            in_channels = out_channels

        # Final classification layer
        self.final_conv = nnx.Conv(
            in_features=self.hidden_dims[-1],
            out_features=1,
            kernel_size=config.kernel_size,
            strides=(1, 1),
            padding=config.padding,
            rngs=rngs,
        )

        # Dropout layer
        self.dropout = None
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        **kwargs,
    ) -> jax.Array:
        """Forward pass through discriminator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input image tensor.

        Returns:
            Discriminator scores (patch-wise).
        """
        h = x

        # Pass through convolutional layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            h = conv(h)

            if norm is not None:
                h = norm(h)  # Auto mode from model.train()/eval()

            h = self.activation(h)

            if self.dropout is not None:
                h = self.dropout(h)  # Auto mode from model.train()/eval()

        # Final classification — returns patch-level predictions
        h = self.final_conv(h)  # Shape: (batch, H', W', 1)

        return h


class CycleGAN(GenerativeModel):
    """CycleGAN for unpaired image-to-image translation.

    Implements the complete CycleGAN architecture with two generators
    and two discriminators for bidirectional domain translation.
    """

    def __init__(
        self,
        config: CycleGANConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CycleGAN with CycleGANConfig.

        Args:
            config: CycleGANConfig with nested CycleGANGeneratorConfig and
                   PatchGANDiscriminatorConfig objects
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not CycleGANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for CycleGAN")

        super().__init__(rngs=rngs)

        if not isinstance(config, CycleGANConfig):
            raise TypeError(f"config must be CycleGANConfig, got {type(config).__name__}")

        # Store config for reference
        self.config = config

        # Extract CycleGAN-specific fields
        self.input_shape_a = config.input_shape_a
        self.input_shape_b = config.input_shape_b
        self.lambda_cycle = config.lambda_cycle
        self.lambda_identity = config.lambda_identity

        # Extract nested configs from dict
        # These must be CycleGANGeneratorConfig and PatchGANDiscriminatorConfig
        gen_a_to_b_config: CycleGANGeneratorConfig = config.generator["a_to_b"]
        gen_b_to_a_config: CycleGANGeneratorConfig = config.generator["b_to_a"]
        disc_a_config: PatchGANDiscriminatorConfig = config.discriminator["disc_a"]
        disc_b_config: PatchGANDiscriminatorConfig = config.discriminator["disc_b"]

        # Initialize generators with config objects
        self.generator_a_to_b = CycleGANGenerator(
            config=gen_a_to_b_config,
            rngs=rngs,
        )

        self.generator_b_to_a = CycleGANGenerator(
            config=gen_b_to_a_config,
            rngs=rngs,
        )

        # Initialize discriminators with config objects
        self.discriminator_a = CycleGANDiscriminator(
            config=disc_a_config,
            rngs=rngs,
        )

        self.discriminator_b = CycleGANDiscriminator(
            config=disc_b_config,
            rngs=rngs,
        )

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        batch_size: int | None = None,
        domain: str = "a_to_b",
        input_images: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate translated images.

        Args:
            n_samples: Number of samples to generate (ignored if input_images provided).
            rngs: Random number generators.
            batch_size: Alternative way to specify number of samples (for compatibility).
            domain: Translation direction ("a_to_b" or "b_to_a").
            input_images: Input images to translate. If None, generates random input.
            **kwargs: Additional keyword arguments.

        Returns:
            Translated images.
        """
        # Use batch_size if provided, otherwise use n_samples
        num_samples = batch_size if batch_size is not None else n_samples

        if input_images is None:
            # Generate random input images
            if domain == "a_to_b":
                shape = (num_samples, *self.input_shape_a)
            else:
                shape = (num_samples, *self.input_shape_b)

            sample_rng = (rngs or self.rngs).sample()

            input_images = jax.random.normal(sample_rng, shape)

        # Translate images
        if domain == "a_to_b":
            return self.generator_a_to_b(input_images)
        elif domain == "b_to_a":
            return self.generator_b_to_a(input_images)
        else:
            raise ValueError(f"Unknown domain: {domain}. Use 'a_to_b' or 'b_to_a'.")

    def compute_cycle_loss(
        self,
        real_a: jax.Array,
        real_b: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute cycle consistency losses.

        Args:
            real_a: Real images from domain A.
            real_b: Real images from domain B.

        Returns:
            Tuple of (cycle_loss_a, cycle_loss_b).
        """
        # Forward cycle: A -> B -> A
        fake_b = self.generator_a_to_b(real_a)
        reconstructed_a = self.generator_b_to_a(fake_b)
        cycle_loss_a = mae_loss(reconstructed_a, real_a)

        # Backward cycle: B -> A -> B
        fake_a = self.generator_b_to_a(real_b)
        reconstructed_b = self.generator_a_to_b(fake_a)
        cycle_loss_b = mae_loss(reconstructed_b, real_b)

        return cycle_loss_a, cycle_loss_b

    def compute_identity_loss(
        self,
        real_a: jax.Array,
        real_b: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute identity losses.

        Identity loss encourages generators to preserve color composition
        when translating images that already belong to the target domain.

        Args:
            real_a: Real images from domain A.
            real_b: Real images from domain B.

        Returns:
            Tuple of (identity_loss_a, identity_loss_b).
        """
        # Identity mappings
        identity_a = self.generator_b_to_a(real_a)  # A should stay A
        identity_b = self.generator_a_to_b(real_b)  # B should stay B

        identity_loss_a = mae_loss(identity_a, real_a)
        identity_loss_b = mae_loss(identity_b, real_b)

        return identity_loss_a, identity_loss_b

    def loss_fn(
        self,
        batch: dict[str, Any],
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute total CycleGAN loss.

        Args:
            batch: Batch containing real images from both domains.
            model_outputs: Model outputs (not used in basic implementation).
            rngs: Random number generators.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing loss and metrics.
        """
        real_a = batch["domain_a"]
        real_b = batch["domain_b"]

        # Compute cycle consistency losses
        cycle_loss_a, cycle_loss_b = self.compute_cycle_loss(real_a, real_b)
        total_cycle_loss = cycle_loss_a + cycle_loss_b

        # Compute identity losses
        identity_loss_a, identity_loss_b = self.compute_identity_loss(real_a, real_b)
        total_identity_loss = identity_loss_a + identity_loss_b

        # Total loss (adversarial loss would be added during training)
        total_loss = (
            self.lambda_cycle * total_cycle_loss + self.lambda_identity * total_identity_loss
        )

        return {
            "loss": total_loss,
            "cycle_loss": total_cycle_loss,
            "identity_loss": total_identity_loss,
        }
