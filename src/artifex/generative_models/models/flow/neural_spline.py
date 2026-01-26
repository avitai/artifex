"""Neural Spline Flows implementation.

Based on the paper:
"Neural Spline Flows" by Durkan et al. (2019)
https://arxiv.org/abs/1906.04032

This implementation provides rational quadratic spline transformations
for normalizing flows, following the existing architecture patterns.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import NeuralSplineConfig

from .base import FlowLayer, NormalizingFlow


class RationalQuadraticSplineTransform(nnx.Module):
    """Rational quadratic spline transformation for a single dimension.

    Implements monotonic rational quadratic splines with automatic
    constraint satisfaction as described in Neural Spline Flows.
    """

    def __init__(
        self,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize rational quadratic spline transform.

        Args:
            num_bins: Number of spline bins
            tail_bound: Bounds for spline support [-tail_bound, tail_bound]
            min_bin_width: Minimum bin width for numerical stability
            min_bin_height: Minimum bin height for numerical stability
            min_derivative: Minimum derivative for numerical stability
            rngs: Random number generators
        """
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.rngs = rngs

    def _constrain_parameters(
        self,
        unnormalized_widths: jax.Array,
        unnormalized_heights: jax.Array,
        unnormalized_derivatives: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Constrain spline parameters to ensure valid monotonic spline."""
        # Constrain widths to sum to 2*tail_bound and be positive
        widths = nnx.softmax(unnormalized_widths, axis=-1)
        widths = (
            self.min_bin_width + (2 * self.tail_bound - self.num_bins * self.min_bin_width) * widths
        )

        # Constrain heights to sum to 2*tail_bound and be positive
        heights = nnx.softmax(unnormalized_heights, axis=-1)
        heights = (
            self.min_bin_height
            + (2 * self.tail_bound - self.num_bins * self.min_bin_height) * heights
        )

        # Constrain derivatives to be positive
        derivatives = self.min_derivative + nnx.softplus(unnormalized_derivatives)

        return widths, heights, derivatives

    def _compute_knots(self, widths: jax.Array, heights: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute knot positions from widths and heights."""
        # Cumulative sum to get knot positions
        cumulative_widths = jnp.cumsum(widths, axis=-1)
        cumulative_heights = jnp.cumsum(heights, axis=-1)

        # Add starting points
        knots_x = jnp.concatenate(
            [
                jnp.full((*cumulative_widths.shape[:-1], 1), -self.tail_bound),
                -self.tail_bound + cumulative_widths,
            ],
            axis=-1,
        )

        knots_y = jnp.concatenate(
            [
                jnp.full((*cumulative_heights.shape[:-1], 1), -self.tail_bound),
                -self.tail_bound + cumulative_heights,
            ],
            axis=-1,
        )

        return knots_x, knots_y

    def apply_spline(
        self,
        x: jax.Array,
        widths: jax.Array,
        heights: jax.Array,
        derivatives: jax.Array,
        inverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply rational quadratic spline transformation.

        Args:
            x: Input values of shape (..., dim)
            widths: Bin widths of shape (..., dim, num_bins)
            heights: Bin heights of shape (..., dim, num_bins)
            derivatives: Knot derivatives of shape (..., dim, num_bins+1)
            inverse: Whether to compute inverse transformation

        Returns:
            Tuple of (transformed_x, log_abs_det_jacobian)
            where log_abs_det_jacobian has shape (...,)
        """
        knots_x, knots_y = self._compute_knots(widths, heights)

        # Handle values outside spline domain with identity transformation
        outside_domain = (x <= -self.tail_bound) | (x >= self.tail_bound)

        # Choose search knots based on forward/inverse
        if inverse:
            search_knots = knots_y
            search_x = x
            # For inverse, we need to find which bin in y-space the input belongs to
            bin_indices = self._search_sorted(search_knots, search_x)
        else:
            search_knots = knots_x
            search_x = x
            # For forward, we find which bin in x-space the input belongs to
            bin_indices = self._search_sorted(search_knots, search_x)

        bin_indices = jnp.clip(bin_indices, 0, self.num_bins - 1)

        # Apply rational quadratic transformation within domain
        if inverse:
            transformed, log_det = self._rational_quadratic_spline_inverse(
                x, knots_x, knots_y, derivatives, bin_indices
            )
        else:
            transformed, log_det = self._rational_quadratic_spline_forward(
                x, knots_x, knots_y, derivatives, bin_indices
            )

        # Use identity transformation outside domain
        outputs = jnp.where(outside_domain, x, transformed)
        log_det_inside = jnp.where(outside_domain, 0.0, log_det)

        # Sum log determinants across the last dimension (dimension axis)
        log_abs_det_total = jnp.sum(log_det_inside, axis=-1)

        return outputs, log_abs_det_total

    def _search_sorted(self, knots: jax.Array, x: jax.Array) -> jax.Array:
        """Find bin indices using binary search on knots.

        Args:
            knots: Knot positions of shape (..., dim, num_bins+1)
            x: Input values of shape (..., dim)

        Returns:
            Bin indices of shape (..., dim)
        """
        # Add an extra dimension to x for broadcasting
        x_expanded = x[..., jnp.newaxis]  # (..., dim, 1)

        # Find the rightmost bin where knot <= x
        # This gives us the bin index where x belongs
        bin_indices = jnp.sum(knots[..., :-1] <= x_expanded, axis=-1) - 1

        return bin_indices

    def _rational_quadratic_spline_forward(
        self,
        x: jax.Array,
        knots_x: jax.Array,
        knots_y: jax.Array,
        derivatives: jax.Array,
        bin_indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward rational quadratic spline transformation."""
        # Get bin boundaries and derivatives
        x_k = jnp.take_along_axis(knots_x, bin_indices[..., jnp.newaxis], axis=-1)[..., 0]
        x_k1 = jnp.take_along_axis(knots_x, (bin_indices + 1)[..., jnp.newaxis], axis=-1)[..., 0]
        y_k = jnp.take_along_axis(knots_y, bin_indices[..., jnp.newaxis], axis=-1)[..., 0]
        y_k1 = jnp.take_along_axis(knots_y, (bin_indices + 1)[..., jnp.newaxis], axis=-1)[..., 0]

        delta_k = jnp.take_along_axis(derivatives, bin_indices[..., jnp.newaxis], axis=-1)[..., 0]
        delta_k1 = jnp.take_along_axis(derivatives, (bin_indices + 1)[..., jnp.newaxis], axis=-1)[
            ..., 0
        ]

        # Compute xi (normalized position within bin)
        xi = (x - x_k) / (x_k1 - x_k)

        # Compute slope
        s_k = (y_k1 - y_k) / (x_k1 - x_k)

        # Rational quadratic transformation (Gregory & Delbourgo method)
        numerator = (y_k1 - y_k) * (s_k * xi**2 + delta_k * xi * (1 - xi))
        denominator = s_k + (delta_k1 + delta_k - 2 * s_k) * xi * (1 - xi)

        y = y_k + numerator / denominator

        # Compute log determinant
        numerator_grad = s_k**2 * (
            delta_k1 * xi**2 + 2 * s_k * xi * (1 - xi) + delta_k * (1 - xi) ** 2
        )
        denominator_grad = (s_k + (delta_k1 + delta_k - 2 * s_k) * xi * (1 - xi)) ** 2

        # Scale by bin width
        dy_dx = numerator_grad / denominator_grad / (x_k1 - x_k)
        log_det = jnp.log(jnp.abs(dy_dx))

        return y, log_det

    def _rational_quadratic_spline_inverse(
        self,
        y: jax.Array,
        knots_x: jax.Array,
        knots_y: jax.Array,
        derivatives: jax.Array,
        bin_indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Inverse rational quadratic spline transformation."""
        # Get bin boundaries and derivatives
        x_k = jnp.take_along_axis(knots_x, bin_indices[..., jnp.newaxis], axis=-1)[..., 0]
        x_k1 = jnp.take_along_axis(knots_x, (bin_indices + 1)[..., jnp.newaxis], axis=-1)[..., 0]
        y_k = jnp.take_along_axis(knots_y, bin_indices[..., jnp.newaxis], axis=-1)[..., 0]
        y_k1 = jnp.take_along_axis(knots_y, (bin_indices + 1)[..., jnp.newaxis], axis=-1)[..., 0]

        delta_k = jnp.take_along_axis(derivatives, bin_indices[..., jnp.newaxis], axis=-1)[..., 0]
        delta_k1 = jnp.take_along_axis(derivatives, (bin_indices + 1)[..., jnp.newaxis], axis=-1)[
            ..., 0
        ]

        # Compute slope
        s_k = (y_k1 - y_k) / (x_k1 - x_k)

        # Solve quadratic equation for xi
        a = (y_k1 - y_k) * (s_k - delta_k) + (y - y_k) * (delta_k1 + delta_k - 2 * s_k)
        b = (y_k1 - y_k) * delta_k - (y - y_k) * (delta_k1 + delta_k - 2 * s_k)
        c = -s_k * (y - y_k)

        # Use numerically stable quadratic formula
        discriminant = b**2 - 4 * a * c
        xi = 2 * c / (-b - jnp.sqrt(discriminant))

        # Convert back to x
        x = x_k + xi * (x_k1 - x_k)

        # Compute log determinant (inverse of forward)
        numerator_grad = s_k**2 * (
            delta_k1 * xi**2 + 2 * s_k * xi * (1 - xi) + delta_k * (1 - xi) ** 2
        )
        denominator_grad = (s_k + (delta_k1 + delta_k - 2 * s_k) * xi * (1 - xi)) ** 2

        # Scale by bin width and take negative (for inverse)
        dy_dx = numerator_grad / denominator_grad / (x_k1 - x_k)
        log_det = -jnp.log(jnp.abs(dy_dx))

        return x, log_det


class SplineCouplingLayer(FlowLayer):
    """Coupling layer with rational quadratic spline transformation.

    Similar to RealNVP's CouplingLayer but uses spline transformations
    instead of affine transformations.
    """

    def __init__(
        self,
        mask: jax.Array,
        hidden_dims: list[int] = [128, 128],
        num_bins: int = 8,
        tail_bound: float = 3.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spline coupling layer.

        Args:
            mask: Binary mask indicating which inputs to transform
            hidden_dims: Hidden layer dimensions for conditioning network
            num_bins: Number of spline bins
            tail_bound: Bounds for spline support [-tail_bound, tail_bound]
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)
        self.mask = mask
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        # Pre-compute indices for JIT compatibility (masks are static)
        self._masked_indices = tuple(int(i) for i in jnp.where(mask > 0)[0])
        self._unmasked_indices = tuple(int(i) for i in jnp.where(mask == 0)[0])

        # Get dimensions
        self.masked_dim = int(jnp.sum(mask))
        self.unmasked_dim = int(jnp.sum(1 - mask))

        # Output dimension: 3 parameters per bin + 1 for each transformed dimension
        output_dim = self.unmasked_dim * (3 * num_bins + 1)

        # Build conditioning network similar to RealNVP
        self.conditioning_layers: list[nnx.Linear] = nnx.List([])
        input_dim = self.masked_dim

        for dim in hidden_dims:
            layer = nnx.Linear(in_features=input_dim, out_features=dim, rngs=rngs)
            self.conditioning_layers.append(layer)
            input_dim = dim

        # Output layer for spline parameters
        self.param_out = nnx.Linear(in_features=input_dim, out_features=output_dim, rngs=rngs)

        # Spline transform
        self.spline_transform = RationalQuadraticSplineTransform(
            num_bins=num_bins, tail_bound=tail_bound, rngs=rngs
        )

    def _get_spline_params(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Get spline parameters from conditioning network."""
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

        # Pass through conditioning network
        h = nn_input
        for layer in self.conditioning_layers:
            h = nnx.relu(layer(h))

        spline_params = self.param_out(h)

        # Reshape and split parameters
        batch_shape = h.shape[:-1]
        spline_params = spline_params.reshape(
            *batch_shape, self.unmasked_dim, 3 * self.num_bins + 1
        )

        # Split into widths, heights, derivatives
        widths = spline_params[..., : self.num_bins]
        heights = spline_params[..., self.num_bins : 2 * self.num_bins]
        derivatives = spline_params[..., 2 * self.num_bins :]

        # Constrain parameters
        return self.spline_transform._constrain_parameters(widths, heights, derivatives)

    def forward(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation."""
        # Use pre-computed indices for JIT compatibility
        unmasked_indices = jnp.array(self._unmasked_indices)

        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x_flat = jnp.reshape(x, (batch_size, -1))
            x_unmasked = x_flat[:, unmasked_indices]
        else:
            x_unmasked = x[:, unmasked_indices]

        # Get spline parameters (uses masked indices internally)
        widths, heights, derivatives = self._get_spline_params(x)

        # Apply spline transformation to unmasked dimensions
        x_transformed, log_det = self.spline_transform.apply_spline(
            x_unmasked, widths, heights, derivatives, inverse=False
        )

        # Reconstruct output (masked dimensions preserved automatically)
        if len(x.shape) > 2:
            x_flat_out = x_flat.copy()
            x_flat_out = x_flat_out.at[:, unmasked_indices].set(x_transformed)
            output = jnp.reshape(x_flat_out, x.shape)
        else:
            output = x.copy()
            output = output.at[:, unmasked_indices].set(x_transformed)

        # log_det from apply_spline is already summed across dimensions and has shape (batch_size,)
        return output, log_det

    def inverse(self, y: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation."""
        # Use pre-computed indices for JIT compatibility
        unmasked_indices = jnp.array(self._unmasked_indices)

        if len(y.shape) > 2:
            batch_size = y.shape[0]
            y_flat = jnp.reshape(y, (batch_size, -1))
            y_unmasked = y_flat[:, unmasked_indices]
        else:
            y_unmasked = y[:, unmasked_indices]

        # Get spline parameters (uses masked indices internally)
        widths, heights, derivatives = self._get_spline_params(y)

        # Apply inverse spline transformation to unmasked dimensions
        x_transformed, log_det = self.spline_transform.apply_spline(
            y_unmasked, widths, heights, derivatives, inverse=True
        )

        # Reconstruct output (masked dimensions preserved automatically)
        if len(y.shape) > 2:
            y_flat_out = y_flat.copy()
            y_flat_out = y_flat_out.at[:, unmasked_indices].set(x_transformed)
            output = jnp.reshape(y_flat_out, y.shape)
        else:
            output = y.copy()
            output = output.at[:, unmasked_indices].set(x_transformed)

        # log_det from apply_spline is already summed across dimensions and has shape (batch_size,)
        return output, log_det


class NeuralSplineFlow(NormalizingFlow):
    """Neural Spline Flow model.

    Implements a normalizing flow using rational quadratic spline transformations
    in coupling layers as described in "Neural Spline Flows" by Durkan et al.
    """

    def __init__(
        self,
        config: NeuralSplineConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Neural Spline Flow.

        Args:
            config: NeuralSpline configuration.
            rngs: Random number generators.

        Raises:
            TypeError: If config is not a NeuralSplineConfig
            ValueError: If rngs is None
        """
        if not isinstance(config, NeuralSplineConfig):
            raise TypeError(f"config must be NeuralSplineConfig, got {type(config).__name__}")

        if rngs is None:
            raise ValueError("rngs must be provided for NeuralSplineFlow initialization")

        # Extract configuration from NeuralSplineConfig
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.hidden_dims = list(config.coupling_network.hidden_dims)
        self.num_bins = config.num_bins
        self.tail_bound = config.tail_bound
        self.base_distribution = config.base_distribution

        # Create coupling layers - this is the attribute the tests expect
        self.coupling_layers = nnx.List([])
        for i in range(self.num_layers):
            # Alternate mask pattern
            mask = jnp.arange(self.input_dim) % 2 == (i % 2)

            layer = SplineCouplingLayer(
                mask=mask,
                hidden_dims=self.hidden_dims,
                num_bins=self.num_bins,
                tail_bound=self.tail_bound,
                rngs=rngs,
            )
            self.coupling_layers.append(layer)

        super().__init__(config, rngs=rngs)

    def forward(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation from data to latent space.

        Args:
            x: Input data of shape (batch_size, input_dim)

        Returns:
            Tuple of (latent_variables, log_abs_det_jacobian)
        """
        z = x
        total_log_det = jnp.zeros(x.shape[0])

        for layer in self.coupling_layers:
            z, log_det = layer.forward(z)
            total_log_det += log_det

        return z, total_log_det

    def inverse(self, z: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation from latent to data space.

        Args:
            z: Latent variables of shape (batch_size, input_dim)

        Returns:
            Tuple of (data_variables, log_abs_det_jacobian)
        """
        x = z
        total_log_det = jnp.zeros(z.shape[0])

        # Apply layers in reverse order
        for layer in reversed(self.coupling_layers):
            x, log_det = layer.inverse(x)
            total_log_det += log_det

        return x, total_log_det

    def log_prob(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Compute log probability of data under the model.

        Args:
            x: Input data of shape (batch_size, input_dim)

        Returns:
            Log probabilities of shape (batch_size,)
        """
        z, log_det = self.forward(x, rngs=rngs)

        # Base distribution log probability
        if self.base_distribution == "normal":
            base_log_prob = -0.5 * jnp.sum(z**2, axis=-1) - 0.5 * self.input_dim * jnp.log(
                2 * jnp.pi
            )
        else:
            raise ValueError(f"Unsupported base distribution: {self.base_distribution}")

        return base_log_prob + log_det

    def sample(self, n_samples: int, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate samples from the model.

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators

        Returns:
            Generated samples of shape (n_samples, input_dim)
        """
        # Sample from base distribution
        if self.base_distribution == "normal":
            z = jax.random.normal(rngs.params(), (n_samples, self.input_dim))
        else:
            raise ValueError(f"Unsupported base distribution: {self.base_distribution}")

        x, _ = self.inverse(z, rngs=rngs)
        return x

    def generate(self, n_samples: int, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate samples from the model (alias for sample).

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators

        Returns:
            Generated samples of shape (n_samples, input_dim)
        """
        return self.sample(n_samples, rngs=rngs)

    def loss_fn(
        self,
        batch: dict[str, jax.Array],
        model_outputs: dict[str, jax.Array],
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Compute loss for training.

        Args:
            batch: Batch dictionary containing 'x' key with data
            model_outputs: Model outputs containing 'z' and 'log_det'

        Returns:
            Scalar loss value (negative log-likelihood)
        """
        x = batch["x"]
        log_prob = self.log_prob(x, rngs=rngs)

        # Return scalar loss (negative log-likelihood)
        return -jnp.mean(log_prob)
