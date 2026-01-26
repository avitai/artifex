"""PatchGAN Discriminator implementation.

Based on:
- Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks
- Pix2PixHD: High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    MultiScalePatchGANConfig,
    PatchGANDiscriminatorConfig,
)
from artifex.generative_models.models.gan.base import Discriminator


class PatchGANDiscriminator(Discriminator):
    """PatchGAN Discriminator for image-to-image translation.

    The PatchGAN discriminator classifies whether N×N patches in an image are real or fake,
    rather than classifying the entire image. This is particularly effective for image
    translation tasks where local texture and structure are important.

    Reference:
        Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
        Wang et al. "High-Resolution Image Synthesis and Semantic Manipulation" (2018)
    """

    def __init__(
        self,
        config: PatchGANDiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize PatchGAN discriminator.

        Args:
            config: PatchGANDiscriminatorConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not PatchGANDiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for PatchGANDiscriminator")

        if not isinstance(config, PatchGANDiscriminatorConfig):
            raise TypeError(
                f"config must be PatchGANDiscriminatorConfig, got {type(config).__name__}"
            )

        # Call parent constructor with config
        super().__init__(config=config, rngs=rngs)

        # Extract PatchGAN-specific parameters from config
        num_filters = config.num_filters
        num_layers = config.num_layers
        kernel_size = config.kernel_size
        stride = config.stride
        use_bias = config.use_bias
        last_kernel_size = config.last_kernel_size

        # Store for use in forward pass
        self.patchgan_num_filters = num_filters
        self.patchgan_num_layers = num_layers
        self.patchgan_use_bias = use_bias
        self.patchgan_kernel_size = kernel_size
        self.patchgan_stride = stride
        self.patchgan_last_kernel_size = last_kernel_size

        # Store activation function for forward pass
        if config.activation == "leaky_relu":
            self.patchgan_activation_fn = lambda x: jax.nn.leaky_relu(
                x, negative_slope=config.leaky_relu_slope
            )
        else:
            self.patchgan_activation_fn = getattr(jax.nn, config.activation)

        # Build the convolutional layers
        self.patchgan_conv_layers = nnx.List([])
        self.patchgan_batch_norm_layers = nnx.List([])

        # First layer (no batch norm)
        channels = config.input_shape[0]
        self.initial_conv = nnx.Conv(
            in_features=channels,
            out_features=num_filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="SAME",
            use_bias=True,  # First layer always uses bias
            rngs=rngs,
        )

        # Intermediate layers
        in_channels = num_filters
        for i in range(num_layers):
            out_channels = num_filters * (2 ** (i + 1))

            # Use stride=1 for the last layer to maintain spatial resolution
            layer_stride = (1, 1) if i == num_layers - 1 else stride

            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=kernel_size,
                strides=layer_stride,
                padding="SAME",
                use_bias=use_bias,
                rngs=rngs,
            )
            self.patchgan_conv_layers.append(conv)

            # Add batch norm (except for first layer)
            if config.batch_norm:
                bn = nnx.BatchNorm(
                    num_features=out_channels,
                    use_running_average=False,
                    momentum=0.9,
                    rngs=rngs,
                )
                self.patchgan_batch_norm_layers.append(bn)

            in_channels = out_channels

        # Final output layer
        self.final_conv = nnx.Conv(
            in_features=in_channels,
            out_features=1,
            kernel_size=last_kernel_size,
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            rngs=rngs,
        )

        # Dropout layer if needed
        if config.dropout_rate > 0:
            self.patchgan_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> list[jax.Array]:
        """Forward pass through PatchGAN discriminator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input image tensor of shape (batch, height, width, channels)

        Returns:
            List of intermediate features, with the last element being the final output.
            This enables feature matching loss computation.
        """
        # Convert from (B, C, H, W) to (B, H, W, C) for JAX convolutions
        if x.ndim == 4 and x.shape[1] in [1, 3]:  # Likely channel-first format
            x = jnp.transpose(x, (0, 2, 3, 1))

        features = [x]

        # Initial convolution (no batch norm)
        x = self.initial_conv(x)
        x = self.patchgan_activation_fn(x)
        features.append(x)

        # Intermediate layers
        for i, conv_layer in enumerate(self.patchgan_conv_layers):
            x = conv_layer(x)

            # Apply batch norm (if enabled)
            if self.batch_norm and i < len(self.patchgan_batch_norm_layers):
                x = self.patchgan_batch_norm_layers[i](x)  # Auto mode from model.train()/eval()

            x = self.patchgan_activation_fn(x)

            # Apply dropout if specified
            if hasattr(self, "patchgan_dropout"):
                x = self.patchgan_dropout(x)  # Auto mode from model.train()/eval()

            features.append(x)

        # Final output layer (no activation)
        x = self.final_conv(x)
        features.append(x)

        # Return all intermediate features for potential feature matching loss
        return features[1:]  # Exclude input image from features

    def _call_without_conversion(self, x: jax.Array) -> list[jax.Array]:
        """Forward pass without input format conversion (for multi-scale use).

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input image tensor already in (B, H, W, C) format

        Returns:
            List of intermediate features, with the last element being the final output.
        """
        features = [x]

        # Initial convolution (no batch norm)
        x = self.initial_conv(x)
        x = self.patchgan_activation_fn(x)
        features.append(x)

        # Intermediate layers
        for i, conv_layer in enumerate(self.patchgan_conv_layers):
            x = conv_layer(x)

            # Apply batch norm (if enabled)
            if self.batch_norm and i < len(self.patchgan_batch_norm_layers):
                x = self.patchgan_batch_norm_layers[i](x)  # Auto mode from model.train()/eval()

            x = self.patchgan_activation_fn(x)

            # Apply dropout if specified
            if hasattr(self, "patchgan_dropout"):
                x = self.patchgan_dropout(x)  # Auto mode from model.train()/eval()

            features.append(x)

        # Final output layer (no activation)
        x = self.final_conv(x)
        features.append(x)

        # Return all intermediate features for potential feature matching loss
        return features[1:]  # Exclude input image from features


class MultiScalePatchGANDiscriminator(nnx.Module):
    """Multi-scale PatchGAN discriminator.

    Processes images at multiple scales using several PatchGAN discriminators.
    This allows the discriminator to capture both fine-grained and coarse-grained
    features at different resolutions.

    Reference:
        Wang et al. "High-Resolution Image Synthesis and Semantic Manipulation" (2018)
    """

    def __init__(
        self,
        config: MultiScalePatchGANConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-scale PatchGAN discriminator.

        Args:
            config: MultiScalePatchGANConfig with all architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None or configuration is invalid
            TypeError: If config is not MultiScalePatchGANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for MultiScalePatchGANDiscriminator")

        if not isinstance(config, MultiScalePatchGANConfig):
            raise TypeError(f"config must be MultiScalePatchGANConfig, got {type(config).__name__}")

        super().__init__()

        # Store config for reference
        self.config = config
        base_disc_config = config.discriminator
        self.input_shape = base_disc_config.input_shape
        self.num_discriminators = config.num_discriminators

        # Determine number of layers per discriminator
        if config.num_layers_per_disc is not None:
            self.num_layers_per_disc = list(config.num_layers_per_disc)
        else:
            self.num_layers_per_disc = [base_disc_config.num_layers] * config.num_discriminators

        # Validate that we won't downsample too much
        for i, layers in enumerate(self.num_layers_per_disc):
            # Each discriminator processes an image downsampled by 2^i (input downsampling)
            # Initial conv downsamples by 2,
            # then layers-1 layers downsample by 2,
            # last layer stride=1
            # So total downsampling is 2^i (input) * 2 (initial) * 2^(layers-1) = 2^(i + layers)
            input_downsample = 2**i
            layer_downsample = 2**layers  # initial conv + layers-1 intermediate layers
            total_downsample = input_downsample * layer_downsample
            final_size = (
                min(base_disc_config.input_shape[1], base_disc_config.input_shape[2])
                // total_downsample
            )
            if final_size < 1:
                raise ValueError(
                    f"Discriminator {i} with {layers} layers would downsample "
                    f"to size {final_size}, which is below minimum_size {config.minimum_size}. "
                    f"Reduce num_layers or num_discriminators."
                )

        # Create pooling layer for downsampling
        if config.pooling_method == "avg":
            # We'll implement average pooling in the forward pass
            self.use_avg_pool = True
        else:
            # Default to average pooling
            self.use_avg_pool = True

        # Create discriminators with config for each
        self.discriminators = nnx.List([])
        for i in range(config.num_discriminators):
            # Create a config for this discriminator (may have different num_layers)
            disc_config = PatchGANDiscriminatorConfig(
                name=f"{base_disc_config.name}_scale{i}",
                input_shape=base_disc_config.input_shape,
                hidden_dims=base_disc_config.hidden_dims,
                activation=base_disc_config.activation,
                leaky_relu_slope=base_disc_config.leaky_relu_slope,
                batch_norm=base_disc_config.batch_norm,
                dropout_rate=base_disc_config.dropout_rate,
                use_spectral_norm=base_disc_config.use_spectral_norm,
                num_filters=base_disc_config.num_filters,
                num_layers=self.num_layers_per_disc[i],
                use_bias=base_disc_config.use_bias,
                last_kernel_size=base_disc_config.last_kernel_size,
                kernel_size=base_disc_config.kernel_size,
                stride=base_disc_config.stride,
                padding=base_disc_config.padding,
            )
            disc = PatchGANDiscriminator(config=disc_config, rngs=rngs)
            self.discriminators.append(disc)

    def downsample_image(self, x: jax.Array, factor: int) -> jax.Array:
        """Downsample image by given factor using average pooling.

        Args:
            x: Input image tensor (B, H, W, C)
            factor: Downsampling factor

        Returns:
            Downsampled image
        """
        if factor <= 1:
            return x

        # Apply average pooling
        return jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, factor, factor, 1),
            window_strides=(1, factor, factor, 1),
            padding="VALID",
        ) / (factor * factor)

    def __call__(self, x: jax.Array) -> tuple[list[jax.Array], list[list[jax.Array]]]:
        """Forward pass through multi-scale discriminator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input image tensor (either B,C,H,W or B,H,W,C)

        Returns:
            Tuple of (outputs, intermediate_features):
            - outputs: List of discriminator outputs (one per scale)
            - intermediate_features: List of intermediate feature lists (one per discriminator)
        """
        # Convert to (B, H, W, C) format once for all discriminators
        if x.ndim == 4 and x.shape[1] in [1, 3]:  # Likely channel-first format
            x = jnp.transpose(x, (0, 2, 3, 1))

        outputs = []
        all_features = []

        for i, discriminator in enumerate(self.discriminators):
            # Downsample input for this scale (except for the first discriminator)
            if i == 0:
                disc_input = x
            else:
                disc_input = self.downsample_image(x, 2**i)

            # Get features from this discriminator
            # Note: discriminator input is already in (B, H, W, C) format,
            # so we need to bypass its format conversion
            features = discriminator._call_without_conversion(disc_input)

            # Last feature is the output, others are intermediate
            outputs.append(features[-1])
            all_features.append(features[:-1])

        return outputs, all_features
