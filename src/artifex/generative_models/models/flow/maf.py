"""Masked Autoregressive Flow (MAF) implementation.

Based on the paper "Masked Autoregressive Flow for Density Estimation" by Papamakarios et al.
Reference implementations from benchmark_VAE and other sources are used as guidance.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import MAFConfig
from artifex.generative_models.models.flow.base import FlowLayer, NormalizingFlow
from artifex.generative_models.models.flow.made import MADE


class MAFLayer(FlowLayer):
    """A single MAF layer using MADE for autoregressive transformation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        rngs: nnx.Rngs,
        order: jax.Array | None = None,
    ):
        """Initialize MAF layer.

        Args:
            input_dim: Dimension of input
            hidden_dims: Dimensions of hidden layers in MADE
            rngs: Random number generators
            order: Variable ordering (if None, uses natural ordering)
        """
        super().__init__(rngs=rngs)
        self.input_dim = input_dim

        # Create MADE for autoregressive transformation
        self.made = MADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_multiplier=2,  # For mu and log_alpha
            rngs=rngs,
            order=order,
        )

    def forward(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation (x -> z).

        Based on benchmark_VAE reference implementation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            rngs: Optional random number generators

        Returns:
            Tuple of (z, log_det_jacobian)
        """
        # Get autoregressive parameters from MADE
        mu, log_alpha = self.made(x)

        # Apply transformation: z = (x - mu) * exp(-log_alpha)
        # Following benchmark_VAE: x = (x - mu) * (-log_var).exp()
        z = (x - mu) * jnp.exp(-log_alpha)

        # Log determinant of Jacobian (negative log_alpha sum)
        log_det_jac = -jnp.sum(log_alpha, axis=1)

        # Flip dimensions as in benchmark_VAE reference
        z = jnp.flip(z, axis=-1)

        return z, log_det_jac

    def inverse(self, z: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation (z -> x).

        For MAF, this requires sequential computation following the autoregressive property.
        Based on benchmark_VAE reference implementation.

        Args:
            z: Input tensor of shape (batch_size, input_dim)
            rngs: Optional random number generators

        Returns:
            Tuple of (x, log_det_jacobian)
        """
        # Reverse the dimension flip from forward pass
        z = jnp.flip(z, axis=-1)

        batch_size = z.shape[0]
        x = jnp.zeros_like(z)
        log_det_jac = jnp.zeros(batch_size)

        # Sequential computation for inverse (autoregressive property)
        # Each x_i depends on previously computed x_{1...i-1}
        for i in range(self.input_dim):
            # Get parameters from MADE using current partial x
            mu, log_alpha = self.made(x)

            # Inverse transformation: x_i = z_i * exp(log_alpha_i) + mu_i
            # Following benchmark_VAE: x[:, i] = y[:, i] * (log_var[:, i]).exp() + mu[:, i]
            x_i = z[:, i] * jnp.exp(log_alpha[:, i]) + mu[:, i]

            # Update x at position i
            x = x.at[:, i].set(x_i)

            # Accumulate log determinant
            log_det_jac += log_alpha[:, i]

        return x, log_det_jac


class MAF(NormalizingFlow):
    """Masked Autoregressive Flow model.

    A normalizing flow that uses masked autoregressive transformations
    to model complex probability distributions.
    """

    def __init__(
        self,
        config: MAFConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize MAF model.

        Args:
            config: MAF configuration.
            rngs: Random number generators.

        Raises:
            TypeError: If config is not a MAFConfig
        """
        if not isinstance(config, MAFConfig):
            raise TypeError(f"config must be MAFConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Extract configuration from MAFConfig
        input_dim = config.input_dim
        if isinstance(input_dim, (tuple, list)):
            self.maf_input_dim = int(jnp.prod(jnp.array(input_dim)))
            self.original_shape = input_dim
            # Set input_dim for compatibility with base class and tests
            self.input_dim = self.maf_input_dim
        else:
            self.maf_input_dim = input_dim
            self.original_shape = (input_dim,)
            self.input_dim = input_dim

        # Extract configuration from MAFConfig
        self.hidden_dims = list(config.coupling_network.hidden_dims)
        self.num_layers = config.num_layers
        self.reverse_ordering = config.reverse_ordering

        # Build MAF layers
        self.flow_layers = nnx.List([])
        for i in range(self.num_layers):
            # Alternate ordering between layers if specified
            if self.reverse_ordering and i % 2 == 1:
                order = jnp.arange(self.maf_input_dim)[::-1]
            else:
                order = None

            layer = MAFLayer(
                input_dim=self.maf_input_dim,
                hidden_dims=self.hidden_dims,
                rngs=rngs,
                order=order,
            )
            self.flow_layers.append(layer)

    def __call__(
        self, x: jax.Array, *args, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Forward pass through MAF.

        Args:
            x: Input tensor
            rngs: Optional random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with model outputs including z, logdet, log_prob
        """
        # Flatten input if needed
        batch_size = x.shape[0]
        if len(x.shape) > 2:
            x_flat = x.reshape(batch_size, -1)
        else:
            x_flat = x

        z = x_flat
        total_logdet = jnp.zeros(batch_size)

        # Forward through all layers
        for layer in self.flow_layers:
            z, logdet = layer.forward(z)
            total_logdet += logdet

        # Reshape z back to original shape if needed
        if len(self.original_shape) > 1:
            z = z.reshape(batch_size, *self.original_shape)

        return {
            "z": z,
            "logdet": total_logdet,
            "log_prob": total_logdet,  # For compatibility
        }

    def inverse(self, z: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation from latent to data space.

        Args:
            z: Latent tensor
            rngs: Optional random number generators

        Returns:
            Tuple of (reconstructed_data, log_det_jacobian)
        """
        # Flatten input if needed
        batch_size = z.shape[0]
        if len(z.shape) > 2:
            z_flat = z.reshape(batch_size, -1)
        else:
            z_flat = z

        z = z_flat
        total_logdet = jnp.zeros(batch_size)

        # Inverse through all layers (in reverse order)
        for layer in reversed(self.flow_layers):
            z, logdet = layer.inverse(z)
            total_logdet += logdet

        # Reshape z back to original shape if needed
        if len(self.original_shape) > 1:
            z = z.reshape(batch_size, *self.original_shape)

        return z, total_logdet

    def sample(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Sample from the flow model.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generators

        Returns:
            Generated samples
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(42))

        # Sample from base distribution (standard Gaussian)
        if hasattr(rngs, "params") and callable(rngs.params):
            key = rngs.params()
        else:
            # Fallback to a default key if rngs is not properly set up
            key = jax.random.PRNGKey(42)

        z = jax.random.normal(key, (n_samples, self.maf_input_dim))

        # Transform through inverse flow
        x, _ = self.inverse(z, rngs=rngs)
        return x

    def log_prob(self, x: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Compute log probability of data.

        Args:
            x: Input data
            rngs: Optional random number generators

        Returns:
            Log probabilities
        """
        # Forward transformation
        result = self(x, rngs=rngs)
        z = result["z"]
        logdet = result["logdet"]

        # Log probability of base distribution (standard Gaussian)
        # For multidimensional inputs, we need to sum over all feature dimensions
        if len(z.shape) > 2:
            # Flatten z to compute log probability correctly
            batch_size = z.shape[0]
            z_flat = z.reshape(batch_size, -1)
            log_base = -0.5 * jnp.sum(z_flat**2, axis=-1) - 0.5 * self.maf_input_dim * jnp.log(
                2 * jnp.pi
            )
        else:
            log_base = -0.5 * jnp.sum(z**2, axis=-1) - 0.5 * self.maf_input_dim * jnp.log(
                2 * jnp.pi
            )

        # Total log probability
        return log_base + logdet
