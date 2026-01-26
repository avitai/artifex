"""RealNVP (Real-valued Non-Volume Preserving) normalizing flow implementation."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    RealNVPConfig,
)
from artifex.generative_models.models.flow.base import FlowLayer, NormalizingFlow


class CouplingLayer(FlowLayer):
    """Coupling layer for RealNVP.

    This implements the affine coupling layer from the RealNVP paper.
    """

    def __init__(
        self,
        mask: jax.Array,
        hidden_dims: list[int],
        scale_activation: Callable[[jax.Array], jax.Array] = nnx.tanh,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize coupling layer.

        Args:
            mask: Binary mask indicating which inputs to transform
            hidden_dims: Hidden layer dimensions for scale/translation networks
            scale_activation: Activation function for scaling factor
            rngs: Random number generators
        """
        if rngs is None:
            raise ValueError("rngs must be provided for CouplingLayer")

        super().__init__(rngs=rngs)
        self.mask = mask
        self.scale_activation = scale_activation

        # Get dimensions of masked and non-masked parts
        self.masked_dim = int(jnp.sum(mask))
        self.unmasked_dim = int(jnp.sum(1 - mask))

        # Pre-compute indices for JIT compatibility (masks are static)
        # Convert to Python tuples to avoid tracer issues
        self._masked_indices = tuple(int(i) for i in jnp.where(mask > 0)[0])
        self._unmasked_indices = tuple(int(i) for i in jnp.where(mask < 1)[0])

        # Build scale network (use nnx.List for proper Flax NNX handling)
        scale_layers_list = []
        input_dim = self.masked_dim
        for dim in hidden_dims:
            layer = nnx.Linear(in_features=input_dim, out_features=dim, rngs=rngs)
            scale_layers_list.append(layer)
            input_dim = dim
        self.scale_layers = nnx.List(scale_layers_list)

        # Output layer for scale function
        self.scale_out = nnx.Linear(
            in_features=input_dim, out_features=self.unmasked_dim, rngs=rngs
        )

        # Build translation network (use nnx.List for proper Flax NNX handling)
        translation_layers_list = []
        input_dim = self.masked_dim
        for dim in hidden_dims:
            layer = nnx.Linear(in_features=input_dim, out_features=dim, rngs=rngs)
            translation_layers_list.append(layer)
            input_dim = dim
        self.translation_layers = nnx.List(translation_layers_list)

        # Output layer for translation function
        self.translation_out = nnx.Linear(
            in_features=input_dim, out_features=self.unmasked_dim, rngs=rngs
        )

    def _scale_and_translate(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Compute scale and translation factors.

        Args:
            x: Input tensor
            rngs: Random number generators (unused but kept for API consistency)

        Returns:
            Tuple of (scale, translation) factors
        """
        # Use pre-computed indices for JIT compatibility
        masked_indices = jnp.array(self._masked_indices)

        if len(x.shape) > 2:
            # For higher dimensional input, flatten then extract
            batch_size = x.shape[0]
            x_flat = jnp.reshape(x, (batch_size, -1))
            nn_input = x_flat[:, masked_indices]
        else:
            # For 2D input, directly extract masked columns
            nn_input = x[:, masked_indices]

        # Scale network
        s = nn_input
        for layer in self.scale_layers:
            s = nnx.relu(layer(s))
        s = self.scale_activation(self.scale_out(s))

        # Translation network
        t = nn_input
        for layer in self.translation_layers:
            t = nnx.relu(layer(t))
        t = self.translation_out(t)

        return s, t

    def forward(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation (from data to latent).

        Args:
            x: Input tensor from data space
            rngs: Random number generators

        Returns:
            Tuple of (transformed_x, log_det_jacobian)
        """
        # Get scale and translation factors
        s, t = self._scale_and_translate(x, rngs=rngs)

        # Initialize output with copy of input
        y = x.copy()

        # Use pre-computed indices for JIT compatibility
        unmasked_indices = jnp.array(self._unmasked_indices)

        # Apply affine transformation to non-masked part
        if len(y.shape) > 2:
            # For higher dimensional data (images)
            batch_size = x.shape[0]
            y_flat = jnp.reshape(y, (batch_size, -1))

            # Apply transformation: y = x * exp(s) + t
            y_flat = y_flat.at[:, unmasked_indices].set(
                y_flat[:, unmasked_indices] * jnp.exp(s) + t
            )

            # Restore original shape
            y = jnp.reshape(y_flat, x.shape)
        else:
            # For 2D input
            y = y.at[:, unmasked_indices].set(y[:, unmasked_indices] * jnp.exp(s) + t)

        # Log determinant of Jacobian: sum(s)
        log_det = jnp.sum(s, axis=1)

        return y, log_det

    def inverse(self, y: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation (from latent to data).

        Args:
            y: Input tensor from latent space
            rngs: Random number generators

        Returns:
            Tuple of (transformed_y, log_det_jacobian)
        """
        # Get scale and translation based on masked part
        s, t = self._scale_and_translate(y, rngs=rngs)

        # Initialize output with copy of input
        x = y.copy()

        # Use pre-computed indices for JIT compatibility
        unmasked_indices = jnp.array(self._unmasked_indices)

        # Apply inverse transformation to non-masked part
        if len(y.shape) > 2:
            # For higher dimensional data (images)
            batch_size = y.shape[0]
            x_flat = jnp.reshape(x, (batch_size, -1))

            # Apply inverse transformation: x = (y - t) * exp(-s)
            x_flat = x_flat.at[:, unmasked_indices].set(
                (x_flat[:, unmasked_indices] - t) * jnp.exp(-s)
            )

            # Restore original shape
            x = jnp.reshape(x_flat, y.shape)
        else:
            # For 2D input
            x = x.at[:, unmasked_indices].set((x[:, unmasked_indices] - t) * jnp.exp(-s))

        # Log determinant of Jacobian: -sum(s)
        log_det = -jnp.sum(s, axis=1)

        return x, log_det


class RealNVP(NormalizingFlow):
    """Real-valued Non-Volume Preserving (RealNVP) flow implementation.

    This is a normalizing flow model based on affine coupling layers.
    """

    def __init__(self, config: RealNVPConfig, *, rngs: nnx.Rngs):
        """Initialize RealNVP model.

        Args:
            config: RealNVP configuration.
            rngs: Random number generators.

        Raises:
            TypeError: If config is not a RealNVPConfig
        """
        if not isinstance(config, RealNVPConfig):
            raise TypeError(f"config must be RealNVPConfig, got {type(config).__name__}")

        # Extract configuration from RealNVPConfig
        self.hidden_dims = list(config.coupling_network.hidden_dims)
        self.num_coupling_layers = config.num_coupling_layers
        self.mask_type = config.mask_type

        # Initialize parent class
        super().__init__(config, rngs=rngs)

        # Calculate total dimension for mask creation
        if isinstance(self.input_dim, (tuple, list)):
            self.total_dim = 1
            for dim in self.input_dim:
                self.total_dim *= dim
        else:
            self.total_dim = self.input_dim

        # Initialize coupling layers
        self._init_coupling_layers(rngs=rngs)

    def _create_mask(self, layer_index: int) -> jax.Array:
        """Create mask for coupling layer.

        Args:
            layer_index: Index of the coupling layer

        Returns:
            Binary mask array
        """
        if self.mask_type == "checkerboard":
            # Alternating 0s and 1s
            base_mask = jnp.arange(self.total_dim) % 2
        elif self.mask_type == "channel-wise":
            if not isinstance(self.input_dim, (tuple, list)) or len(self.input_dim) < 3:
                raise ValueError(
                    "Channel-wise masking requires input_dim to be a tuple with "
                    "at least 3 dimensions (height, width, channels)"
                )
            channels = self.input_dim[-1]
            mask_dim = channels // 2
            base_mask = jnp.concatenate([jnp.ones(mask_dim), jnp.zeros(channels - mask_dim)])

            # Broadcast to full spatial dimensions
            spatial_size = self.input_dim[0] * self.input_dim[1]
            base_mask = jnp.tile(base_mask, spatial_size)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        # Alternate the mask for each layer
        return base_mask if layer_index % 2 == 0 else 1 - base_mask

    def _init_coupling_layers(self, *, rngs: nnx.Rngs | None = None):
        """Initialize coupling layers.

        Args:
            rngs: Optional random number generators
        """
        # Base class initializes flow_layers as regular list
        # Keep it as regular Python list (not nnx.List)
        for i in range(self.num_coupling_layers):
            # Create mask for this layer
            mask = self._create_mask(i)

            # Create layer-specific RNG if provided
            layer_rngs = None
            if rngs is not None:
                if hasattr(rngs, "params"):
                    # Create derived key for this layer
                    layer_key = jax.random.fold_in(rngs.params(), i)
                    layer_rngs = nnx.Rngs(params=layer_key)
                else:
                    layer_rngs = rngs

            # Create coupling layer
            coupling_layer = CouplingLayer(mask=mask, hidden_dims=self.hidden_dims, rngs=layer_rngs)

            # Append to existing flow_layers list from base class
            self.flow_layers.append(coupling_layer)

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        shape: tuple | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate samples from the flow model.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generators
            shape: Optional shape override for samples
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples
        """
        # Get sampling key
        sample_key = (rngs or self.rngs).sample()

        # Sample from base distribution
        z = self.sample_fn(sample_key, n_samples)

        # Transform to data space
        x, _ = self.inverse(z, rngs=rngs)

        # Reshape if needed
        if shape is not None:
            x = jnp.reshape(x, (n_samples, *shape))
        elif isinstance(self.input_dim, (tuple, list)):
            x = jnp.reshape(x, (n_samples, *self.input_dim))

        return x

    def __call__(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Forward pass through the RealNVP model.

        Args:
            x: Input data
            rngs: Optional random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            dictionary with model outputs
        """
        # Forward transformation
        z, logdet = self.forward(x, rngs=rngs)

        # Compute log probability
        log_prob = self.log_prob_fn(z) + logdet

        # Return outputs
        return {
            "z": z,
            "logdet": logdet,
            "log_prob": log_prob,
            "log_prob_x": log_prob,  # Alias for convenience
        }

    @property
    def params(self):
        """Get model parameters for compatibility.

        Returns:
            Empty dictionary (NNX handles parameters internally)
        """
        return {}
