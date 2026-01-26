"""Spatial autoencoder for Stable Diffusion.

This module provides encoder and decoder that preserve spatial structure,
unlike traditional VAE encoders that flatten to vectors.
"""

from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)


def _get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "relu":
        return nnx.relu
    elif activation == "silu":
        return nnx.silu
    elif activation == "gelu":
        return nnx.gelu
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class SpatialEncoder(nnx.Module):
    """Spatial encoder that preserves spatial dimensions.

    Unlike CNNEncoder which flattens to a vector, this encoder maintains
    spatial structure: (B, H, W, C) -> (B, H//2^n, W//2^n, latent_channels)
    where n is len(hidden_dims).
    """

    def __init__(
        self,
        config: EncoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spatial encoder from EncoderConfig.

        Args:
            config: EncoderConfig with:
                - hidden_dims: Channel dimensions for each conv layer
                - latent_dim: Number of channels in latent representation
                - activation: Activation function name
                - input_shape: Input shape (H, W, C) to get input_channels
            rngs: Random number generators
        """
        super().__init__()

        hidden_dims = list(config.hidden_dims)
        latent_channels = config.latent_dim
        activation = config.activation
        # Extract input channels from input_shape (last dimension for NHWC)
        input_channels = config.input_shape[-1] if len(config.input_shape) >= 3 else 3

        self.activation_fn = _get_activation_fn(activation)

        # Create convolutional layers with stride 2 for downsampling
        conv_layers = []
        in_channels = input_channels

        for out_channels in hidden_dims:
            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
            )
            conv_layers.append(conv)
            in_channels = out_channels

        # Store as nnx.List for proper pytree handling
        self.conv_layers = nnx.List(conv_layers)

        # Final convolution to latent channels (no downsampling)
        # Mean and log_var projections
        self.mean_conv = nnx.Conv(
            in_features=hidden_dims[-1],
            out_features=latent_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.logvar_conv = nnx.Conv(
            in_features=hidden_dims[-1],
            out_features=latent_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x):
        """Encode input to spatial latent representation.

        Args:
            x: Input images [batch, height, width, channels]

        Returns:
            Tuple of (mean, log_var) with shape [batch, H//8, W//8, latent_channels]
        """
        h = x
        for conv in self.conv_layers:
            h = conv(h)
            h = self.activation_fn(h)

        # Project to mean and log_var
        mean = self.mean_conv(h)
        log_var = self.logvar_conv(h)

        return mean, log_var


class SpatialDecoder(nnx.Module):
    """Spatial decoder that upsamples from latent representation.

    Decodes from spatial latents: (B, H//2^n, W//2^n, latent_channels) -> (B, H, W, C)
    where n is len(hidden_dims).
    """

    def __init__(
        self,
        config: DecoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spatial decoder from DecoderConfig.

        Args:
            config: DecoderConfig with:
                - hidden_dims: Channel dimensions for each transpose conv layer
                - latent_dim: Number of channels in latent representation
                - output_shape: Output shape (H, W, C) to get output_channels
                - activation: Activation function name
            rngs: Random number generators
        """
        super().__init__()

        hidden_dims = list(config.hidden_dims)
        latent_channels = config.latent_dim
        # Extract output channels from output_shape (last dimension for NHWC)
        output_channels = config.output_shape[-1] if len(config.output_shape) >= 3 else 3
        activation = config.activation

        self.activation_fn = _get_activation_fn(activation)

        # Initial convolution from latent channels
        self.input_conv = nnx.Conv(
            in_features=latent_channels,
            out_features=hidden_dims[0],
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        # Create transpose convolutional layers with stride 2 for upsampling
        # We need len(hidden_dims) transpose convs to match encoder's downsampling
        conv_transpose_layers = []
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i + 1]

            conv_t = nnx.ConvTranspose(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
            )
            conv_transpose_layers.append(conv_t)

        # Store as nnx.List for proper pytree handling
        self.conv_transpose_layers = nnx.List(conv_transpose_layers)

        # Final transpose convolution to output channels (one more upsample)
        self.output_conv = nnx.ConvTranspose(
            in_features=hidden_dims[-1],
            out_features=output_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, z):
        """Decode latent representation to images.

        Args:
            z: Latent representation [batch, latent_height, latent_width, latent_channels]

        Returns:
            Decoded images [batch, height, width, channels]
        """
        h = self.input_conv(z)
        h = self.activation_fn(h)

        for conv_t in self.conv_transpose_layers:
            h = conv_t(h)
            h = self.activation_fn(h)

        # Final convolution with sigmoid activation
        output = self.output_conv(h)
        output = nnx.sigmoid(output)

        return output
