"""Conditional VAE Implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.vae_config import ConditionalVAEConfig
from artifex.generative_models.models.vae.base import VAE
from artifex.generative_models.models.vae.decoders import create_decoder
from artifex.generative_models.models.vae.encoders import create_encoder


class ConditionalVAE(VAE):
    """Conditional Variational Autoencoder implementation.

    Extends the base VAE with conditioning capabilities by incorporating
    additional information at both encoding and decoding steps.
    This follows the standard CVAE pattern using one-hot concatenation.
    """

    def __init__(
        self,
        config: ConditionalVAEConfig,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize the Conditional VAE.

        Args:
            config: ConditionalVAEConfig with encoder, decoder, encoder_type, conditioning settings
            rngs: Random number generators
            precision: Numerical precision for computations
        """
        # Initialize base GenerativeModel (skip VAE's encoder/decoder creation)
        # We need to call the grandparent's __init__ directly
        nnx.Module.__init__(self)

        # Store rngs and precision (from GenerativeModel)
        self.rngs = rngs
        self.precision = precision

        # Store settings from config (from VAE)
        self.latent_dim = config.encoder.latent_dim
        self.kl_weight = config.kl_weight
        encoder_type = config.encoder_type

        # Store conditioning settings
        self.condition_dim = config.condition_dim
        self.condition_type = config.condition_type

        # Create conditional encoder and decoder
        self.encoder = create_encoder(
            config.encoder,
            encoder_type,
            conditional=True,
            num_classes=config.condition_dim,
            rngs=rngs,
        )
        self.decoder = create_decoder(
            config.decoder,
            encoder_type,
            conditional=True,
            num_classes=config.condition_dim,
            rngs=rngs,
        )

    def _prepare_condition(self, y: jax.Array | None, batch_size: int) -> jax.Array:
        """Prepare condition for encoding/decoding.

        Args:
            y: Conditioning information (optional)
            batch_size: Batch size for default condition

        Returns:
            Processed condition (one-hot encoded if needed)
        """
        if y is None:
            return jnp.zeros((batch_size, self.condition_dim))

        # Convert integer labels to one-hot if needed
        if jnp.issubdtype(y.dtype, jnp.integer) and y.ndim == 1:
            return jax.nn.one_hot(y, self.condition_dim)

        return y

    def _reshape_condition(self, condition: jax.Array, target_shape: tuple) -> jax.Array:
        """Reshape condition to match input dimensions for concatenation.

        For image-based CVAEs, this broadcasts the condition to spatial dimensions.

        Args:
            condition: Conditioning information (integers or one-hot)
            target_shape: Shape to match (must be static for JIT compatibility)

        Returns:
            Reshaped condition suitable for concatenation

        Note:
            When JIT-compiling, target_shape must be a static argument.
        """
        batch_size = condition.shape[0]

        # Convert integer labels to one-hot if needed
        if condition.ndim == 1 and jnp.issubdtype(condition.dtype, jnp.integer):
            condition = jax.nn.one_hot(condition, self.condition_dim)

        if len(target_shape) == 2:  # 1D input (batch_size, features)
            return condition  # No reshape needed

        if len(target_shape) == 3:  # 2D input (batch_size, height, width)
            # Expand to (batch_size, height, width, condition_dim)
            condition = condition.reshape(batch_size, 1, 1, -1)
            return jnp.broadcast_to(
                condition,
                (batch_size, target_shape[1], target_shape[2], condition.shape[-1]),
            )

        if len(target_shape) == 4:  # Image input (batch, height, width, channels)
            # Expand to (batch_size, height, width, condition_dim)
            condition = condition.reshape(batch_size, 1, 1, -1)
            return jnp.broadcast_to(
                condition,
                (batch_size, target_shape[1], target_shape[2], condition.shape[-1]),
            )

        # Default: return the original condition
        return condition

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass with conditioning.

        Args:
            x: Input data
            y: Conditioning information (optional)
            **kwargs: Additional arguments for compatibility

        Returns:
            Dictionary with model outputs
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # Prepare condition
        y = self._prepare_condition(y, batch_size)

        # Encode with conditioning
        mean, log_var = self.encode(x_flat, y=y)

        # Reparameterize (inherited from VAE)
        z = self.reparameterize(mean, log_var)

        # Decode with conditioning
        reconstructed = self.decode(z, y=y)

        return {
            "reconstructed": reconstructed,
            "mean": mean,
            "log_var": log_var,
            "z": z,
        }

    def encode(self, x: jax.Array, y: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        """Encode input to the latent space with conditioning.

        Args:
            x: Input data
            y: Conditioning information (optional)

        Returns:
            Tuple of (mean, log_var) of the latent distribution
        """
        batch_size = x.shape[0]
        y = self._prepare_condition(y, batch_size)

        # Call encoder with condition
        result = self.encoder(x, condition=y)

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            return result
        elif isinstance(result, dict):
            mean = result.get("mean", result.get("mu", None))
            log_var = result.get("log_var", result.get("logvar", None))
            if mean is not None and log_var is not None:
                return mean, log_var

        raise ValueError("Encoder output format not recognized")

    def decode(self, z: jax.Array, y: jax.Array | None = None) -> jax.Array:
        """Decode latent vectors with conditioning.

        Args:
            z: Latent vectors
            y: Conditioning information (optional)

        Returns:
            Reconstructed output
        """
        batch_size = z.shape[0]
        y = self._prepare_condition(y, batch_size)

        # Call decoder with condition
        result = self.decoder(z, condition=y)

        # Handle different return formats
        if isinstance(result, jax.Array):
            return result
        elif isinstance(result, dict):
            for key in ["reconstructed", "reconstruction", "output", "x_hat"]:
                if key in result:
                    return result[key]

        raise ValueError("Decoder output format not recognized")

    def sample(
        self,
        n_samples: int = 1,
        *,
        temperature: float = 1.0,
        y: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Sample from the model with conditioning.

        Args:
            n_samples: Number of samples to generate
            temperature: Temperature parameter controlling randomness
            y: Conditioning information (optional)
            **kwargs: Additional arguments for compatibility

        Returns:
            Generated samples
        """
        # Prepare condition
        if y is None:
            y = jnp.zeros((n_samples, self.condition_dim))
        elif y.shape[0] == 1 and n_samples > 1:
            # Broadcast single condition to n_samples
            y = jnp.broadcast_to(y, (n_samples, *y.shape[1:]))

        y = self._prepare_condition(y, n_samples)

        sample_key = self.rngs.sample()

        # Sample from prior
        z = jax.random.normal(sample_key, (n_samples, self.latent_dim)) * temperature

        # Decode with conditioning
        return self.decode(z, y=y)

    def generate(
        self,
        n_samples: int = 1,
        *,
        temperature: float = 1.0,
        y: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples from the model with conditioning.

        Args:
            n_samples: Number of samples to generate
            temperature: Temperature parameter controlling randomness
            y: Conditioning information (optional)
            **kwargs: Additional arguments

        Returns:
            Generated samples
        """
        return self.sample(n_samples=n_samples, temperature=temperature, y=y, **kwargs)

    def reconstruct(
        self, x: jax.Array, *, y: jax.Array | None = None, deterministic: bool = False
    ) -> jax.Array:
        """Reconstruct inputs with conditioning.

        Args:
            x: Input data
            y: Conditioning information (optional)
            deterministic: If True, use mean instead of sampling

        Returns:
            Reconstructed outputs
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        y = self._prepare_condition(y, batch_size)

        # Encode with conditioning
        mean, log_var = self.encode(x_flat, y=y)

        if deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)

        # Decode with conditioning
        return self.decode(z, y=y)
