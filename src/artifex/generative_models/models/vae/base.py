"""VAE base class definition."""

from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae.decoders import create_decoder
from artifex.generative_models.models.vae.encoders import create_encoder


class VAE(GenerativeModel):
    """Base class for Variational Autoencoders.

    This class provides a foundation for implementing various VAE models
    using Flax NNX. All VAE models should inherit from this class and
    implement the required methods.
    """

    def __init__(
        self,
        config: VAEConfig,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ) -> None:
        """Initialize a VAE.

        Args:
            config: VAEConfig with encoder, decoder, encoder_type, and kl_weight settings
            rngs: Random number generators
            precision: Numerical precision for computations
        """
        super().__init__(
            rngs=rngs,
            precision=precision,
        )

        # Extract settings from config
        self.latent_dim = config.encoder.latent_dim
        self.kl_weight = config.kl_weight
        encoder_type = config.encoder_type

        # Create encoder and decoder from nested configs
        self.encoder = create_encoder(config.encoder, encoder_type, rngs=rngs)
        self.decoder = create_decoder(config.decoder, encoder_type, rngs=rngs)

    def encode(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode input to the latent space.

        Args:
            x: Input data

        Returns:
            Tuple of (mean, log_var) of the latent distribution

        Raises:
            ValueError: If encoder output format is invalid
        """
        # Call the encoder and handle different possible return formats
        result = self.encoder(x)

        # Return format handling
        if isinstance(result, tuple) and len(result) == 2:
            return cast(tuple[jax.Array, jax.Array], result)
        elif isinstance(result, dict):
            # Try different common key naming conventions
            mean = result.get("mean", result.get("mu", None))
            log_var = result.get("log_var", result.get("logvar", result.get("log_variance", None)))
            if mean is not None and log_var is not None:
                return mean, log_var

        raise ValueError("Encoder must return a tuple of (mean, log_var) or a dict with these keys")

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode latent vectors to the output space.

        Args:
            z: Latent vectors

        Returns:
            Reconstructed outputs

        Raises:
            ValueError: If decoder output format is invalid
        """
        # Call the decoder and handle different possible return formats
        result = self.decoder(z)

        # Return format handling
        if isinstance(result, jax.Array):
            return result
        elif isinstance(result, dict):
            for key in ["reconstructed", "reconstruction", "output", "x_hat"]:
                if key in result:
                    return result[key]

        raise ValueError(
            "Decoder must return an array or a dict with reconstructed output under a standard key"
        )

    def reparameterize(self, mean: jax.Array, log_var: jax.Array) -> jax.Array:
        """Apply the reparameterization trick.

        Args:
            mean: Mean vectors of the latent distribution
            log_var: Log variance vectors of the latent distribution

        Returns:
            Sampled latent vectors
        """
        sample_key = self.rngs.sample()

        # Ensure numerical stability by clipping log_var
        log_var_clipped = jnp.clip(log_var, -20.0, 20.0)
        std = jnp.exp(0.5 * log_var_clipped)
        eps = jax.random.normal(sample_key, mean.shape)
        return mean + eps * std

    def __call__(self, x: jax.Array) -> dict[str, Any]:
        """Forward pass through the VAE.

        Args:
            x: Input data

        Returns:
            Dictionary containing model outputs (reconstructed data, latent vectors,
            and distribution parameters)
        """
        # Encode
        mean, log_var = self.encode(x)

        # Sample from the latent distribution
        z = self.reparameterize(mean, log_var)

        # Decode
        reconstructed = self.decode(z)

        return {
            "reconstructed": reconstructed,
            "mean": mean,
            "log_var": log_var,
            "z": z,
        }

    def loss_fn(
        self,
        params: dict | None = None,
        batch: dict | None = None,
        rng: jax.Array | None = None,
        x: jax.Array | None = None,
        outputs: dict[str, jax.Array] | None = None,
        beta: float | None = None,
        reconstruction_loss_fn: Callable | None = None,
        **kwargs: Any,
    ) -> dict[str, jax.Array]:
        """Calculate loss for VAE.

        Args:
            params: Model parameters (optional, for compatibility with Trainer)
            batch: Input batch (optional, for compatibility with Trainer)
            rng: Random number generator (optional, for compatibility with Trainer)
            x: Input data (if not provided in batch)
            outputs: Dictionary of model outputs from forward pass
            beta: Weight for KL divergence term
            reconstruction_loss_fn: Optional custom reconstruction loss function.
                Signature: fn(predictions, targets) -> loss (JAX/Optax convention)
            **kwargs: Additional arguments

        Returns:
            Dictionary of loss components
        """
        # Handle different input patterns for compatibility
        if batch is not None and x is None:
            if isinstance(batch, dict) and "inputs" in batch:
                x = batch["inputs"]
            else:
                x = batch

        # Ensure x is a proper JAX array, not a dictionary
        if isinstance(x, dict):
            if "inputs" in x:
                x = x["inputs"]
            elif "input" in x:
                x = x["input"]
            else:
                # If x is still a dictionary without recognized keys, raise error
                raise ValueError(
                    "Input 'x' is a dictionary without 'inputs' or 'input' keys. "
                    "Expected a JAX array."
                )

        if outputs is None:
            if hasattr(self, "apply") and params is not None:
                # For compatibility with Trainer
                rngs_dict = None
                if rng is not None:
                    rngs_dict = nnx.Rngs(dropout=rng)
                outputs = self.apply(params, x, rngs=rngs_dict)
            else:
                # Use direct call
                outputs = self(x)

        # Reconstruction loss (Mean Squared Error by default)
        # Try both keys for compatibility
        reconstructed = outputs.get("reconstructed", outputs.get("reconstruction", None))
        if reconstructed is None:
            raise KeyError("Missing required output key: 'reconstructed' or 'reconstruction'")

        if reconstruction_loss_fn is None:
            # Default to MSE
            reconstruction_loss = jnp.mean((x - reconstructed) ** 2)
        else:
            # Use custom loss function (JAX/Optax convention: predictions, targets)
            reconstruction_loss = reconstruction_loss_fn(reconstructed, x)

        # KL divergence loss
        # Try both keys for compatibility
        mean = outputs["mean"]
        log_var = outputs.get("log_var", outputs.get("logvar", None))
        if log_var is None:
            raise KeyError("Missing required output key: 'log_var' or 'logvar'")

        # Compute KL divergence analytically for Gaussian distributions
        kl_loss = -0.5 * jnp.sum(1 + log_var - mean**2 - jnp.exp(log_var), axis=-1).mean()

        # Use provided beta or default to instance kl_weight
        beta_value = beta if beta is not None else self.kl_weight

        # Total loss
        total_loss = reconstruction_loss + beta_value * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def sample(self, n_samples: int = 1, *, temperature: float = 1.0) -> jax.Array:
        """Sample from the prior distribution.

        Args:
            n_samples: Number of samples to generate. Must be a Python int (static value).
                When JIT-compiling, mark this as a static argument.
            temperature: Scaling factor for the standard deviation (higher = more diverse)

        Returns:
            Generated samples

        Note:
            When JIT-compiling functions that call this method, mark n_samples as static:
            Example: `@nnx.jit(static_argnums=(1,))` or `@jax.jit(static_argnums=(1,))`
        """
        sample_key = self.rngs.sample()

        # Sample from the prior (standard normal)
        # n_samples must be a static/concrete value for JIT compatibility
        z = jax.random.normal(sample_key, (n_samples, self.latent_dim))

        # Apply temperature scaling
        z = z * temperature

        # Decode
        return self.decode(z)

    def reconstruct(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Reconstruct inputs.

        Args:
            x: Input data
            deterministic: If True, use mean of the latent distribution instead of sampling

        Returns:
            Reconstructed outputs
        """
        # Encode
        mean, log_var = self.encode(x)

        if deterministic:
            # Use the mean for deterministic reconstruction (no randomness)
            z = mean
        else:
            # Sample from the latent distribution
            z = self.reparameterize(mean, log_var)

        # Decode
        return self.decode(z)

    def generate(
        self,
        n_samples: int = 1,
        *,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples from the model.

        This is an alias for the sample method to maintain consistency
        with the GenerativeModel interface.

        Args:
            n_samples: Number of samples to generate
            temperature: Scaling factor for the standard deviation (higher = more diverse)
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            Generated samples
        """
        return self.sample(n_samples, temperature=temperature)

    def interpolate(
        self,
        x1: jax.Array,
        x2: jax.Array,
        steps: int = 10,
    ) -> jax.Array:
        """Interpolate between two inputs in latent space.

        Args:
            x1: First input
            x2: Second input
            steps: Number of interpolation steps (including endpoints).
                Must be a Python int (static value) when JIT-compiling.

        Returns:
            Interpolated outputs

        Note:
            When JIT-compiling functions that call this method, mark steps as static:
            Example: `@nnx.jit(static_argnums=(3,))` for the third argument.
        """
        # Ensure inputs are batched
        if x1.ndim == 1:
            x1 = x1[None, ...]
        if x2.ndim == 1:
            x2 = x2[None, ...]

        # Encode both inputs (deterministic)
        mean1, _ = self.encode(x1)
        mean2, _ = self.encode(x2)

        # Create interpolation path in latent space
        # steps must be a static/concrete value for JIT compatibility
        alphas = jnp.linspace(0, 1, steps)

        # Vectorized interpolation: broadcast and compute all interpolations at once
        # alphas shape: (steps,) -> (steps, 1)
        # mean1[0] shape: (latent_dim,) -> (1, latent_dim)
        alphas_expanded = alphas[:, None]
        z_interp = mean1[0] * (1 - alphas_expanded) + mean2[0] * alphas_expanded

        # Decode interpolated latent vectors
        return self.decode(z_interp)

    def latent_traversal(
        self,
        x: jax.Array,
        dim: int,
        range_vals: tuple[float, float] = (-3.0, 3.0),
        steps: int = 10,
    ) -> jax.Array:
        """Traverse a single dimension of the latent space.

        Args:
            x: Input data
            dim: Dimension to traverse
            range_vals: Range of values for traversal
            steps: Number of steps in the traversal

        Returns:
            Decoded outputs from the traversal

        Raises:
            ValueError: If dim is out of range
        """
        if dim < 0 or dim >= self.latent_dim:
            raise ValueError(f"Dimension {dim} out of range [0, {self.latent_dim})")

        # Ensure input is batched
        if x.ndim == 1:
            x = x[None, ...]

        # Encode input (deterministic)
        mean, _ = self.encode(x)
        mean = mean[0]  # Use first example if batched

        # Create traversal values
        traversal_values = jnp.linspace(range_vals[0], range_vals[1], steps)

        # Create traversal vectors by modifying only the specified dimension
        z_traversal = jnp.tile(mean, (steps, 1))
        z_traversal = z_traversal.at[:, dim].set(traversal_values)

        # Decode traversal vectors
        return self.decode(z_traversal)
