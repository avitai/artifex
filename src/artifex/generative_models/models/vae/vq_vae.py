"""Vector Quantized VAE Implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.vae_config import VQVAEConfig
from artifex.generative_models.models.vae.base import VAE


class VQVAE(VAE):
    """Vector Quantized Variational Autoencoder implementation.

    Extends the base VAE with discrete latent variables using a codebook.
    """

    def __init__(
        self,
        config: VQVAEConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the VQ-VAE.

        Args:
            config: VQVAEConfig with encoder, decoder, encoder_type, and VQ settings
            rngs: Random number generators for initialization
        """
        super().__init__(config=config, rngs=rngs)

        # Store VQ-VAE specific parameters from config
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.commitment_cost = config.commitment_cost

        # Initialize embedding table
        self.embeddings = nnx.Embed(
            num_embeddings=self.num_embeddings, features=self.embedding_dim, rngs=rngs
        )

        # Store last quantization auxiliary data for loss computation
        self._last_quantize_aux = nnx.Dict({})

    def quantize(self, encoding: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Quantize the input using the codebook.

        Args:
            encoding: Continuous encoding from the encoder

        Returns:
            Tuple of (quantized encoding, auxiliary dict containing losses and indices)
        """

        encoding_shape = encoding.shape

        # Flatten the encoding to [batch_size * height * width, embedding_dim]
        flat_encoding = encoding.reshape((-1, self.embedding_dim))

        # Calculate distances to embeddings
        distances = (
            (flat_encoding**2).sum(axis=1, keepdims=True)
            + (self.embeddings.embedding**2).sum(axis=1)
            - 2 * jnp.matmul(flat_encoding, self.embeddings.embedding.T)
        )

        # Get nearest embedding indices
        encoding_indices = jnp.argmin(distances, axis=1)

        # Get the nearest embeddings
        quantized_flat = self.embeddings(encoding_indices)

        # Reshape back to original shape
        quantized = quantized_flat.reshape(encoding_shape)

        # Compute commitment and codebook loss
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - encoding) ** 2)
        codebook_loss = jnp.mean((quantized - jax.lax.stop_gradient(encoding)) ** 2)

        # Create auxiliary dict
        aux = {
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "encoding_indices": encoding_indices,
        }

        # Store auxiliary info for later use in __call__ and loss_fn
        self._last_quantize_aux = aux

        # Use straight-through estimator for gradients
        # In JAX we use stop_gradient for this
        quantized_st = encoding + jax.lax.stop_gradient(quantized - encoding)

        return quantized_st, aux

    def encode(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode input to continuous latent representation (before quantization).

        Args:
            x: Input data, shape [batch_size, ...]

        Returns:
            Tuple of (mean, log_var) for VAE interface compatibility.
            For VQ-VAE, log_var is zeros since encoding is deterministic.
        """
        # Use the encoder to get the continuous representation
        encoder_output = self.encoder(x)

        # Extract mean from encoder output (depending on encoder implementation)
        if isinstance(encoder_output, dict) and "mean" in encoder_output:
            encoding = encoder_output["mean"]
        elif isinstance(encoder_output, tuple) and len(encoder_output) > 0:
            encoding = encoder_output[0]
        else:
            # Direct output for simple encoders
            encoding = encoder_output

        # Return (mean, log_var) for VAE interface compatibility
        # VQ-VAE has deterministic encoding, so log_var is zeros
        log_var = jnp.zeros_like(encoding)
        return encoding, log_var

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode latent representation to reconstruction.

        Args:
            z: Quantized representation

        Returns:
            Reconstructed output
        """
        # Get decoder output depending on implementation
        decoder_output = self.decoder(z)

        # Extract reconstruction from decoder output
        if isinstance(decoder_output, dict) and "reconstructed" in decoder_output:
            return decoder_output["reconstructed"]
        elif isinstance(decoder_output, dict) and "reconstruction" in decoder_output:
            return decoder_output["reconstruction"]
        elif isinstance(decoder_output, jax.Array):
            return decoder_output
        else:
            raise ValueError("Unexpected decoder output format")

    def __call__(self, x: jax.Array) -> dict[str, Any]:
        """Forward pass through the VQ-VAE.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing model outputs
        """
        # Encode input (returns mean, log_var for VAE interface compatibility)
        encoding, log_var = self.encode(x)

        # Quantize the encoding (returns quantized and aux dict)
        quantized, aux = self.quantize(encoding)

        # Decode quantized vector to get reconstruction
        reconstruction = self.decode(quantized)

        # Return a dictionary with all outputs
        # Include 'z' and 'z_e' for compatibility with tests
        return {
            "reconstructed": reconstruction,
            "encoding": encoding,  # Pre-quantization encoding
            # (also called z_e in the literature)
            "z_e": encoding,  # Alternative name for pre-quantization encoding
            "quantized": quantized,  # Quantized encoding
            "z": quantized,  # For compatibility with test expectations
            "mean": encoding,  # For VAE interface compatibility
            "log_var": log_var,  # Include for VAE base class compatibility
            "commitment_loss": aux["commitment_loss"],
            "codebook_loss": aux["codebook_loss"],
            "encoding_indices": aux["encoding_indices"],
            "quantization_aux": aux,
        }

    def loss_fn(
        self, x: jax.Array, outputs: dict[str, jax.Array], **kwargs: Any
    ) -> dict[str, jax.Array]:
        """Compute the VQ-VAE loss.

        Args:
            x: Input data
            outputs: Dictionary of model outputs from forward pass
            **kwargs: Additional arguments

        Returns:
            Dictionary of loss components
        """
        # Get reconstruction and auxiliary data
        recon_x = outputs["reconstructed"]

        # Get losses from outputs if available, otherwise from aux
        if "commitment_loss" in outputs and "codebook_loss" in outputs:
            commitment_loss = outputs["commitment_loss"]
            codebook_loss = outputs["codebook_loss"]
        else:
            aux = outputs["quantization_aux"]
            commitment_loss = aux["commitment_loss"]
            codebook_loss = aux["codebook_loss"]

        # Compute reconstruction loss (binary cross entropy)
        if len(x.shape) > 2:  # Multi-dimensional input (like images)
            recon_loss = -jnp.sum(
                x * jnp.log(recon_x + 1e-8) + (1 - x) * jnp.log(1 - recon_x + 1e-8),
                axis=tuple(range(1, len(x.shape))),
            )
        else:  # 1D input
            recon_loss = -jnp.sum(
                x * jnp.log(recon_x + 1e-8) + (1 - x) * jnp.log(1 - recon_x + 1e-8), axis=1
            )
        recon_loss = jnp.mean(recon_loss)

        # Compute VQ loss (combination of commitment and codebook loss)
        vq_loss = self.commitment_cost * commitment_loss + codebook_loss

        # Compute the total loss
        total_loss = recon_loss + vq_loss

        # Return dictionary of losses
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "vq_loss": vq_loss,
        }

    def sample(
        self,
        n_samples: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> jax.Array:
        """Sample from the model.
        Args:
            n_samples: Number of samples to generate
            temperature: Temperature parameter controlling randomness
                (higher = more random)
            **kwargs: Additional keyword arguments for compatibility

        Returns:
            Generated samples
        """
        sample_key = self.rngs.sample()

        # For VQ-VAE, we sample from the discrete codebook
        # First, we randomly sample indices from the codebook
        if temperature > 0:
            # Higher temperature gives more uniform sampling
            logits = jnp.ones((n_samples, self.num_embeddings)) / temperature
            indices = jax.random.categorical(sample_key, logits, axis=-1)
        else:
            # Temperature 0 means deterministic sampling (just use first few embeddings)
            indices = jnp.arange(n_samples) % self.num_embeddings

        # Look up the embedding vectors for these indices
        z = self.embeddings(indices)

        # Decode the embedding vectors
        samples = self.decode(z)
        return samples

    def generate(
        self,
        n_samples: int = 1,
        *,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples from the model.

        Implements the required method from GenerativeModel base class.

        Args:
            n_samples: Number of samples to generate
            temperature: Temperature parameter controlling randomness
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples
        """
        return self.sample(n_samples=n_samples, temperature=temperature, **kwargs)
