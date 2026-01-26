"""Decoder implementations for VAE models."""

import jax
from flax import nnx

from artifex.generative_models.core.base import CNN, MLP
from artifex.generative_models.core.configuration.network_configs import DecoderConfig


SigmoidActivation = nnx.sigmoid


class MLPDecoder(nnx.Module):
    """Simple MLP decoder for VAE."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the MLP decoder.

        Args:
            config: DecoderConfig with hidden_dims, output_shape, latent_dim, activation
            rngs: Random number generator
        """
        super().__init__()

        hidden_dims = list(config.hidden_dims)
        output_shape = config.output_shape
        latent_dim = config.latent_dim
        use_batch_norm = config.batch_norm

        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Reverse the hidden dimensions for decoder
        decoder_dims = list(reversed(hidden_dims))

        # Determine the flattened output size
        self.output_size = 1
        for dim in output_shape:
            self.output_size *= dim

        self.backbone = MLP(
            hidden_dims=decoder_dims,
            activation=config.activation,
            in_features=latent_dim,
            use_batch_norm=use_batch_norm,
            rngs=rngs,
        )
        self.output_layer = nnx.Linear(
            in_features=decoder_dims[-1], out_features=self.output_size, rngs=rngs
        )
        self.activation = SigmoidActivation

    def __call__(self, z: jax.Array) -> jax.Array:
        """Forward pass of the decoder.

        Args:
            z: Latent vector

        Returns:
            Reconstructed output
        """
        h = self.backbone(z)
        flat_output = self.output_layer(h)

        # Apply sigmoid to get values in [0, 1]
        flat_output = self.activation(flat_output)

        # Reshape to the output shape
        batch_size = z.shape[0]
        output = flat_output.reshape(batch_size, *self.output_shape)

        return output


class CNNDecoder(nnx.Module):
    """CNN-based decoder for VAE."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the CNN decoder.

        Args:
            config: DecoderConfig with hidden_dims, output_shape, latent_dim, activation
            rngs: Random number generator
        """
        super().__init__()

        hidden_dims = list(config.hidden_dims)
        output_shape = config.output_shape
        latent_dim = config.latent_dim
        activation = config.activation
        use_batch_norm = config.batch_norm

        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Calculate initial spatial dimensions
        # With SAME padding and stride 2, output = ceil(input/stride)
        # So input = output after num_layers of stride-2 upsampling
        h, w, _c = output_shape
        num_layers = len(hidden_dims)
        # Compute what the encoder output would be (to match dimensions)
        enc_h, enc_w = h, w
        for _ in range(num_layers):
            enc_h = (enc_h + 1) // 2  # ceil(h / 2)
            enc_w = (enc_w + 1) // 2  # ceil(w / 2)
        self.initial_h = max(enc_h, 1)
        self.initial_w = max(enc_w, 1)

        # Determine initial feature size
        self.initial_features = hidden_dims[0]

        # Project latent vector to initial features
        initial_size = self.initial_h * self.initial_w * self.initial_features
        self.project = nnx.Linear(in_features=latent_dim, out_features=initial_size, rngs=rngs)

        # Reverse hidden dimensions for decoder
        decoder_dims = list(reversed(hidden_dims))

        # Transpose CNN layers
        self.cnn = CNN(
            hidden_dims=decoder_dims,
            activation=activation,
            use_transpose=True,
            in_features=self.initial_features,
            use_batch_norm=use_batch_norm,
            rngs=rngs,
        )

        # Final layer to get the right number of channels
        final_channels = decoder_dims[-1] if decoder_dims else hidden_dims[-1]
        # Store output channels as instance attribute for JIT efficiency
        # Avoids shape[3] access during JIT compilation
        self.output_channels = output_shape[2]
        self.output_conv = nnx.Conv(
            in_features=final_channels,
            out_features=self.output_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.activation = SigmoidActivation

    def __call__(self, z: jax.Array) -> jax.Array:
        """Forward pass of the decoder.

        Args:
            z: Latent vector

        Returns:
            Reconstructed output
        """
        # Project and reshape to initial spatial dimensions
        batch_size = z.shape[0]
        h = self.project(z)
        h = h.reshape(batch_size, self.initial_h, self.initial_w, self.initial_features)

        # Apply transpose CNN
        h = self.cnn(h)

        # Final convolution to get the right number of channels
        output = self.output_conv(h)

        # Resize to match target output shape (handles non-power-of-2 sizes)
        # Uses stored output_channels for JIT efficiency (avoids shape access)
        target_h, target_w = self.output_shape[0], self.output_shape[1]
        output = jax.image.resize(
            output,
            shape=(batch_size, target_h, target_w, self.output_channels),
            method="bilinear",
        )

        # Apply sigmoid to get values in [0, 1]
        output = self.activation(output)

        return output


class ResNetDecoder(nnx.Module):
    """ResNet-based decoder for VAE.

    Note: This is a placeholder implementation that uses CNN internally.
    A full ResNet implementation would include proper residual blocks.
    """

    def __init__(
        self,
        config: DecoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize ResNet decoder.

        Args:
            config: DecoderConfig with hidden_dims, output_shape, latent_dim, activation
            rngs: Random number generators
        """
        super().__init__()

        self.latent_dim = config.latent_dim
        self.output_shape = config.output_shape
        # For now, use CNN decoder as a placeholder
        self.cnn_decoder = CNNDecoder(config=config, rngs=rngs)

    def __call__(self, z):
        """Forward pass through ResNet decoder.

        Args:
            z: Latent vector

        Returns:
            Reconstructed output
        """
        return self.cnn_decoder(z)


class ConditionalDecoder(nnx.Module):
    """Wrapper that adds conditioning to any decoder.

    This wrapper adds class conditioning to an existing decoder by
    converting labels to one-hot and concatenating them with the latent vector.
    This follows the standard CVAE pattern from popular implementations.
    """

    def __init__(
        self,
        decoder: nnx.Module,
        num_classes: int,
        *,
        rngs: nnx.Rngs,  # noqa: ARG002 - kept for API consistency
    ):
        """Initialize conditional decoder wrapper.

        Args:
            decoder: Base decoder to wrap
            num_classes: Number of classes for conditioning
            rngs: Random number generators (for API consistency)
        """
        super().__init__()
        self.decoder = decoder
        self.num_classes = num_classes

    def __call__(self, z, condition=None):
        """Forward pass with conditioning.

        Args:
            z: Latent vector
            condition: Class labels (integers) or one-hot encoded conditions

        Returns:
            Reconstructed output from the decoder
        """
        import jax
        import jax.numpy as jnp

        if condition is not None:
            # Convert integer labels to one-hot if needed
            if condition.dtype in [jnp.int32, jnp.int64]:
                condition = jax.nn.one_hot(condition, self.num_classes)

            # Concatenate conditioning with latent vector
            z = jnp.concatenate([z, condition], axis=-1)

        # Pass through the base decoder
        return self.decoder(z)


def create_decoder(
    config: DecoderConfig,
    decoder_type: str = "dense",
    *,
    conditional: bool = False,
    num_classes: int | None = None,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create a decoder from DecoderConfig.

    Args:
        config: DecoderConfig with network architecture settings
        decoder_type: Type of decoder (dense, cnn, resnet)
        conditional: Whether to create conditional decoder
        num_classes: Number of classes for conditional decoder (one-hot dimension)
        rngs: Random number generators

    Returns:
        Decoder module
    """
    # If conditional, adjust latent_dim to account for concatenated one-hot condition
    if conditional:
        if num_classes is None:
            raise ValueError("num_classes must be provided for conditional decoder")
        # Adjust latent_dim for the extra condition dimensions (one-hot = num_classes)
        adjusted_latent_dim = config.latent_dim + num_classes
        config = DecoderConfig(
            name=config.name,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            output_shape=config.output_shape,
            latent_dim=adjusted_latent_dim,
            batch_norm=config.batch_norm,
        )

    if decoder_type == "dense":
        decoder = MLPDecoder(config=config, rngs=rngs)
    elif decoder_type == "cnn":
        decoder = CNNDecoder(config=config, rngs=rngs)
    elif decoder_type == "resnet":
        decoder = ResNetDecoder(config=config, rngs=rngs)
    elif decoder_type == "spatial":
        from artifex.generative_models.models.vae.spatial_autoencoder import SpatialDecoder

        decoder = SpatialDecoder(config=config, rngs=rngs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    # Wrap in conditional decoder if needed
    if conditional:
        decoder = ConditionalDecoder(
            decoder=decoder,
            num_classes=num_classes,
            rngs=rngs,
        )

    return decoder
