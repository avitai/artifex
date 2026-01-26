"""Conditional normalizing flow implementations."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.models.flow.base import FlowLayer, NormalizingFlow


class ConditionalCouplingLayer(FlowLayer):
    """Conditional coupling layer for conditional normalizing flows.

    This layer conditions the affine transformation on additional context information.
    """

    def __init__(
        self,
        mask: jax.Array,
        hidden_dims: list[int],
        condition_dim: int,
        scale_activation: str = "tanh",
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize conditional coupling layer.

        Args:
            mask: Binary mask indicating which inputs to transform
            hidden_dims: Hidden layer dimensions for scale/translation networks
            condition_dim: Dimension of conditioning information
            scale_activation: Activation function for scaling factor
            rngs: Random number generators
        """
        if rngs is None:
            raise ValueError("rngs must be provided")

        super().__init__(rngs=rngs)
        self.mask = mask
        self.condition_dim = condition_dim

        # Set activation function
        activation_map: dict[str, Callable[[jax.Array], jax.Array]] = {
            "tanh": jax.nn.tanh,
            "sigmoid": jax.nn.sigmoid,
            "identity": lambda x: x,
        }
        self.scale_activation = activation_map.get(scale_activation, jax.nn.tanh)

        # Pre-compute indices for JIT compatibility (masks are static)
        self._masked_indices = tuple(int(i) for i in jnp.where(mask > 0)[0])
        self._unmasked_indices = tuple(int(i) for i in jnp.where(mask < 1)[0])

        # Get dimensions
        self.masked_dim = int(jnp.sum(mask))
        self.unmasked_dim = int(jnp.sum(1 - mask))

        # Input dimension for networks: masked features + conditioning
        input_dim = self.masked_dim + condition_dim

        # Scale network
        self.scale_layers = nnx.List([])
        current_dim = input_dim
        for dim in hidden_dims:
            layer = nnx.Linear(current_dim, dim, rngs=rngs)
            self.scale_layers.append(layer)
            current_dim = dim

        # Output layer for scale
        self.scale_out = nnx.Linear(current_dim, self.unmasked_dim, rngs=rngs)

        # Translation network
        self.translation_layers = nnx.List([])
        current_dim = input_dim
        for dim in hidden_dims:
            layer = nnx.Linear(current_dim, dim, rngs=rngs)
            self.translation_layers.append(layer)
            current_dim = dim

        # Output layer for translation
        self.translation_out = nnx.Linear(current_dim, self.unmasked_dim, rngs=rngs)

    def _scale_and_translate(
        self,
        x: jax.Array,
        condition: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute scale and translation factors based on input and conditioning.

        Args:
            x: Input tensor
            condition: Conditioning information
            rngs: Random number generators

        Returns:
            Tuple of (scale, translation) factors
        """
        # Use pre-computed indices for JIT compatibility
        masked_indices = jnp.array(self._masked_indices)

        if len(x.shape) > 2:
            # For higher dimensional input, flatten then extract
            batch_size = x.shape[0]
            x_flat = jnp.reshape(x, (batch_size, -1))
            masked_features = x_flat[:, masked_indices]
        else:
            # For 2D input, directly extract
            masked_features = x[:, masked_indices]

        # Concatenate masked features with conditioning
        nn_input = jnp.concatenate([masked_features, condition], axis=-1)

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

    def forward(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        condition: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward transformation with conditioning.

        Args:
            x: Input tensor
            rngs: Random number generators
            condition: Conditioning information
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (transformed_x, log_det_jacobian)
        """
        if condition is None:
            raise ValueError("condition must be provided for conditional coupling layer")

        # Get scale and translation
        s, t = self._scale_and_translate(x, condition, rngs=rngs)

        # Initialize output
        y = x.copy()

        # Use pre-computed indices for JIT compatibility
        unmasked_indices = jnp.array(self._unmasked_indices)

        # Apply transformation to unmasked part
        if len(y.shape) > 2:
            # For higher dimensional data
            batch_size = x.shape[0]
            y_flat = jnp.reshape(y, (batch_size, -1))
            y_flat = y_flat.at[:, unmasked_indices].set(
                y_flat[:, unmasked_indices] * jnp.exp(s) + t
            )
            y = jnp.reshape(y_flat, x.shape)
        else:
            # For 2D input
            y = y.at[:, unmasked_indices].set(y[:, unmasked_indices] * jnp.exp(s) + t)

        # Log determinant
        log_det = jnp.sum(s, axis=1)

        return y, log_det

    def inverse(
        self,
        y: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        condition: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation with conditioning.

        Args:
            y: Input tensor
            rngs: Random number generators
            condition: Conditioning information
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (transformed_y, log_det_jacobian)
        """
        if condition is None:
            raise ValueError("condition must be provided for conditional coupling layer")

        # Get scale and translation
        s, t = self._scale_and_translate(y, condition, rngs=rngs)

        # Initialize output
        x = y.copy()

        # Use pre-computed indices for JIT compatibility
        unmasked_indices = jnp.array(self._unmasked_indices)

        # Apply inverse transformation to unmasked part
        if len(y.shape) > 2:
            # For higher dimensional data
            batch_size = y.shape[0]
            x_flat = jnp.reshape(x, (batch_size, -1))
            x_flat = x_flat.at[:, unmasked_indices].set(
                (x_flat[:, unmasked_indices] - t) * jnp.exp(-s)
            )
            x = jnp.reshape(x_flat, y.shape)
        else:
            # For 2D input
            x = x.at[:, unmasked_indices].set((x[:, unmasked_indices] - t) * jnp.exp(-s))

        # Log determinant
        log_det = -jnp.sum(s, axis=1)

        return x, log_det


class ConditionalNormalizingFlow(NormalizingFlow):
    """Conditional normalizing flow that can condition on additional context.

    This allows the flow to generate different outputs based on conditioning information.
    """

    def __init__(self, config, *, rngs: nnx.Rngs | None = None) -> None:
        """Initialize conditional normalizing flow.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        # Extract conditioning configuration
        self.condition_dim = getattr(config, "condition_dim", 10)
        self.hidden_dims = getattr(config, "hidden_dims", [64, 64])
        self.num_coupling_layers = getattr(config, "num_coupling_layers", 8)
        self.mask_type = getattr(config, "mask_type", "checkerboard")

        # Initialize parent
        if rngs is None:
            raise ValueError("rngs must be provided for ConditionalNormalizingFlow")

        super().__init__(config, rngs=rngs)

        # Calculate total dimension for mask creation
        if isinstance(self.input_dim, (tuple, list)):
            self.total_dim = 1
            for dim in self.input_dim:
                self.total_dim *= dim
        else:
            self.total_dim = self.input_dim

        # Initialize conditional coupling layers
        self._init_conditional_layers(rngs=rngs)

    def _create_mask(self, layer_index: int) -> jax.Array:
        """Create mask for coupling layer.

        Args:
            layer_index: Index of the coupling layer

        Returns:
            Binary mask array
        """
        if self.mask_type == "checkerboard":
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
            spatial_size = self.input_dim[0] * self.input_dim[1]
            base_mask = jnp.tile(base_mask, spatial_size)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        # Alternate mask for each layer
        return base_mask if layer_index % 2 == 0 else 1 - base_mask

    def _init_conditional_layers(self, *, rngs: nnx.Rngs | None = None) -> None:
        """Initialize conditional coupling layers.

        Args:
            rngs: Random number generators
        """
        self.flow_layers = nnx.List([])

        for i in range(self.num_coupling_layers):
            # Create mask
            mask = self._create_mask(i)

            # Create layer-specific RNG
            layer_rngs = None
            if rngs is not None:
                layer_key = jax.random.fold_in(rngs.params(), i)
                layer_rngs = nnx.Rngs(params=layer_key)

            # Create conditional coupling layer
            coupling_layer = ConditionalCouplingLayer(
                mask=mask,
                hidden_dims=self.hidden_dims,
                condition_dim=self.condition_dim,
                rngs=layer_rngs,
            )

            self.flow_layers.append(coupling_layer)

    def forward(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        condition: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward transformation with conditioning.

        Args:
            x: Input tensor from data space
            rngs: Random number generators
            condition: Conditioning information
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (transformed_x, total_log_det_jacobian)
        """
        if condition is None:
            batch_size = x.shape[0]
            condition = jnp.zeros((batch_size, self.condition_dim))

        y = x
        total_log_det = jnp.zeros(x.shape[0])

        # Pass through each conditional layer
        for layer in self.flow_layers:
            y, log_det = layer.forward(y, rngs=rngs, condition=condition)
            total_log_det += log_det

        return y, total_log_det

    def inverse(
        self,
        z: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        condition: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation with conditioning.

        Args:
            z: Input tensor from latent space
            rngs: Random number generators
            condition: Conditioning information
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (reconstructed_data, total_log_det_jacobian)
        """
        if condition is None:
            batch_size = z.shape[0]
            condition = jnp.zeros((batch_size, self.condition_dim))

        x = z
        total_log_det = jnp.zeros(z.shape[0])

        # Pass through each layer in reverse order
        for layer in reversed(self.flow_layers):
            x, log_det = layer.inverse(x, rngs=rngs, condition=condition)
            total_log_det += log_det

        return x, total_log_det

    def log_prob(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        condition: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute conditional log probability of data points.

        Args:
            x: Input data points
            rngs: Random number generators
            condition: Conditioning information
            **kwargs: Additional keyword arguments

        Returns:
            Log probability of each data point given the condition
        """
        if condition is None:
            batch_size = x.shape[0]
            condition = jnp.zeros((batch_size, self.condition_dim))

        # Transform x to z with conditioning
        z, logdet = self.forward(x, rngs=rngs, condition=condition)

        # Compute log probability of z under base distribution
        log_prob_z = self.log_prob_fn(z)

        # Conditional log probability
        return log_prob_z + logdet

    def generate(
        self,
        n_samples: int = 1,
        condition: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate conditional samples from the flow model.

        Args:
            n_samples: Number of samples to generate
            condition: Conditioning information for generation
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Generated conditional samples
        """
        # Get sampling key
        sample_key = (rngs or self.rngs).sample()

        # Sample from base distribution
        z = self.sample_fn(sample_key, n_samples)

        # If no condition provided, create default conditioning
        if condition is None:
            condition = jnp.zeros((n_samples, self.condition_dim))
        elif len(condition.shape) == 1:
            # Broadcast condition to match batch size
            condition = jnp.tile(condition[None, :], (n_samples, 1))

        # Transform through inverse flow with conditioning
        x, _ = self.inverse(z, rngs=rngs, condition=condition)

        return x

    def __call__(
        self,
        x: jax.Array,
        condition: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass through the conditional flow model.

        Args:
            x: Input data
            condition: Conditioning information
            rngs: Random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with model outputs
        """
        # Handle case where no conditioning is provided
        if condition is None:
            batch_size = x.shape[0]
            condition = jnp.zeros((batch_size, self.condition_dim))

        # Forward transformation
        z, logdet = self.forward(x, rngs=rngs, condition=condition)

        # Compute conditional log probability
        log_prob = self.log_prob_fn(z) + logdet

        return {
            "z": z,
            "logdet": logdet,
            "log_prob": log_prob,
            "log_prob_x": log_prob,
            "condition": condition,
        }

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute loss for conditional flow model.

        Args:
            batch: Input batch (should contain 'x' and 'condition')
            model_outputs: Model outputs from forward pass
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing loss and metrics
        """
        # Extract data and conditioning from batch
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("data"))
            condition = batch.get("condition", batch.get("c"))
        else:
            # If batch is just data, assume no conditioning
            x = batch
            condition = None

        # Handle missing conditioning
        if condition is None:
            batch_size = x.shape[0]
            condition = jnp.zeros((batch_size, self.condition_dim))

        # Compute conditional log probability
        log_prob = self.log_prob(x, rngs=rngs, condition=condition)

        # Loss is negative log likelihood
        loss = -jnp.mean(log_prob)

        return {
            "loss": loss,
            "nll_loss": loss,
            "conditional_log_prob": jnp.mean(log_prob),
            "avg_log_prob": jnp.mean(log_prob),
        }


class ConditionalRealNVP(ConditionalNormalizingFlow):
    """Conditional version of RealNVP with conditioning support.

    This model extends RealNVP to support conditional generation based on
    additional context information.
    """

    def __init__(self, config, *, rngs: nnx.Rngs | None = None) -> None:
        """Initialize Conditional RealNVP model.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        super().__init__(config, rngs=rngs)

    def sample_class_conditional(
        self,
        n_samples: int,
        class_labels: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate samples conditioned on class labels.

        Args:
            n_samples: Number of samples to generate
            class_labels: Class labels for conditioning
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Generated class-conditional samples
        """
        # Convert class labels to one-hot encoding if needed
        if len(class_labels.shape) == 1:
            # Assume these are class indices, convert to one-hot
            num_classes = int(jnp.max(class_labels)) + 1
            condition = jax.nn.one_hot(class_labels, num_classes)
        else:
            # Assume already one-hot encoded
            condition = class_labels

        return self.generate(n_samples, condition=condition, rngs=rngs, **kwargs)
