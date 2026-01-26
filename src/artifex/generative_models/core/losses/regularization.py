"""
Regularization losses module.

This module provides regularization terms that can be added to other losses
to improve model stability, generalization, and prevent overfitting.
All functions are JAX-compatible and work with NNX modules.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.losses.base import reduce_loss


def l1_regularization(
    params: jax.Array | list[jax.Array] | dict[str, jax.Array] | nnx.State,
    scale: float = 1.0,
    reduction: str = "sum",
    predicate: Callable[[str, jax.Array], bool] | None = None,
) -> jax.Array:
    """
    L1 regularization for model parameters.

    Applies L1 regularization (Lasso) to the parameters:
    loss = scale * sum(|params|)

    Args:
        params: Model parameters (array, list of arrays, nested dict, or NNX State)
        scale: Regularization strength
        reduction: Reduction method ('none', 'mean', 'sum')
        predicate: Optional function to filter which parameters to regularize
            Should accept (name: str, param: Array) and return a boolean

    Returns:
        L1 regularization loss

    Example:
        >>> # With NNX model
        >>> model = nnx.Linear(10, 5, rngs=nnx.Rngs(0))
        >>> params = nnx.state(model, nnx.Param)
        >>> l1_regularization(params, scale=0.1)

        >>> # With dictionary
        >>> params = {"w1": jnp.array([1.0, -2.0]), "w2": jnp.array([0.5, 1.5])}
        >>> l1_regularization(params, scale=0.1)
    """
    if predicate is None:
        predicate = lambda name, param: True

    def _calculate_l1(name, param):
        if predicate(name, param):
            return jnp.sum(jnp.abs(param))
        return jnp.array(0.0)

    # Handle NNX State
    if isinstance(params, nnx.State):
        l1_norm = sum(
            _calculate_l1(name, param.value if hasattr(param, "value") else param)
            for name, param in _flatten_nnx_state(params)
        )
    # Handle different param structures
    elif isinstance(params, jax.Array):
        # Single array
        l1_norm = jnp.sum(jnp.abs(params))
    elif isinstance(params, list):
        # list of arrays
        l1_norm = sum(jnp.sum(jnp.abs(p)) for p in params)
    else:
        # Nested dictionary
        l1_norm = sum(_calculate_l1(name, param) for name, param in _flatten_dict(params))

    if reduction == "mean":
        # For mean, we need parameter count
        param_count = _count_params(params)
        return scale * (l1_norm / param_count)
    elif reduction == "sum":
        return scale * l1_norm
    else:  # "none" - doesn't make much sense for regularization, but return as-is
        return scale * l1_norm


def l2_regularization(
    params: jax.Array | list[jax.Array] | dict[str, jax.Array] | nnx.State,
    scale: float = 1.0,
    reduction: str = "sum",
    predicate: Callable[[str, jax.Array], bool] | None = None,
) -> jax.Array:
    """
    L2 regularization for model parameters.

    Applies L2 regularization (Ridge) to the parameters:
    loss = scale * sum(params²)

    Args:
        params: Model parameters (array, list of arrays, nested dict, or NNX State)
        scale: Regularization strength
        reduction: Reduction method ('none', 'mean', 'sum')
        predicate: Optional function to filter which parameters to regularize
            Should accept (name: str, param: Array) and return a boolean

    Returns:
        L2 regularization loss

    Example:
        >>> # With NNX model
        >>> model = nnx.Linear(10, 5, rngs=nnx.Rngs(0))
        >>> params = nnx.state(model, nnx.Param)
        >>> l2_regularization(params, scale=0.01)
    """
    if predicate is None:
        predicate = lambda name, param: True

    def _calculate_l2(name, param):
        if predicate(name, param):
            return jnp.sum(jnp.square(param))
        return jnp.array(0.0)

    # Handle NNX State
    if isinstance(params, nnx.State):
        l2_norm = sum(
            _calculate_l2(name, param.value if hasattr(param, "value") else param)
            for name, param in _flatten_nnx_state(params)
        )
    # Handle different param structures
    elif isinstance(params, jax.Array):
        # Single array
        l2_norm = jnp.sum(jnp.square(params))
    elif isinstance(params, list):
        # list of arrays
        l2_norm = sum(jnp.sum(jnp.square(p)) for p in params)
    else:
        # Nested dictionary
        l2_norm = sum(_calculate_l2(name, param) for name, param in _flatten_dict(params))

    if reduction == "mean":
        # For mean, we need parameter count
        param_count = _count_params(params)
        return scale * (l2_norm / param_count)
    elif reduction == "sum":
        return scale * l2_norm
    else:  # "none"
        return scale * l2_norm


class SpectralNormRegularization(nnx.Module):
    """Spectral norm regularization using NNX for stateful computation."""

    def __init__(self, n_power_iterations: int = 1, eps: float = 1e-12):
        """Initialize spectral norm regularization.

        Args:
            n_power_iterations: Number of power iterations
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # State for storing u vectors (will be initialized on first call)
        self._u_states: dict[str, nnx.Variable] = {}

    def __call__(
        self, weight: jax.Array, weight_name: str = "weight", scale: float = 1.0
    ) -> jax.Array:
        """Compute spectral norm regularization for a weight matrix.

        Args:
            weight: Weight matrix to regularize
            weight_name: Unique name for this weight (for state management)
            scale: Regularization strength

        Returns:
            Spectral norm regularization loss
        """
        # Initialize u vector if not exists
        if weight_name not in self._u_states:
            u_shape = (weight.shape[0], 1)
            # Use deterministic initialization for reproducibility
            u_init = jnp.ones(u_shape) / jnp.sqrt(u_shape[0])
            self._u_states[weight_name] = nnx.Variable(u_init)

        u = self._u_states[weight_name].value

        # Power iteration
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v = jnp.matmul(weight.T, u)
            v = v / (jnp.linalg.norm(v) + self.eps)

            # u = W v / ||W v||
            u = jnp.matmul(weight, v)
            u = u / (jnp.linalg.norm(u) + self.eps)

        # Update stored u vector
        self._u_states[weight_name].value = u

        # Compute spectral norm: u^T W v
        spectral_norm = jnp.matmul(jnp.matmul(u.T, weight), v).squeeze()

        return scale * jnp.abs(spectral_norm)


def spectral_norm_regularization(
    weight: jax.Array,
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    scale: float = 1.0,
    u_vector: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Spectral norm regularization for weight matrices (functional version).

    Computes an approximation of the spectral norm (largest singular value)
    using power iteration. Returns both the loss and updated u vector.

    Args:
        weight: Weight matrix to regularize
        n_power_iterations: Number of power iterations
        eps: Small constant for numerical stability
        scale: Regularization strength
        u_vector: Previous u vector (if None, initializes randomly)

    Returns:
        Tuple of (spectral_norm_loss, updated_u_vector)

    Example:
        >>> w = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> loss, u_new = spectral_norm_regularization(w, scale=0.01)
    """
    # Initialize u vector if not provided
    if u_vector is None:
        u_shape = (weight.shape[0], 1)
        u = jnp.ones(u_shape) / jnp.sqrt(u_shape[0])
    else:
        u = u_vector

    # Power iteration
    for _ in range(n_power_iterations):
        # v = W^T u / ||W^T u||
        v = jnp.matmul(weight.T, u)
        v = v / (jnp.linalg.norm(v) + eps)

        # u = W v / ||W v||
        u = jnp.matmul(weight, v)
        u = u / (jnp.linalg.norm(u) + eps)

    # Compute spectral norm: u^T W v
    spectral_norm = jnp.matmul(jnp.matmul(u.T, weight), v).squeeze()

    return scale * jnp.abs(spectral_norm), u


def orthogonal_regularization(
    weight: jax.Array,
    scale: float = 1.0,
    mode: str = "symmetric",
) -> jax.Array:
    """
    Orthogonal regularization for weight matrices.

    Penalizes the difference between W^T W and the identity matrix,
    encouraging orthogonal weight matrices.

    Args:
        weight: Weight matrix to regularize
        scale: Regularization strength
        mode: Regularization mode ('symmetric', 'rows', 'cols')
            - 'symmetric': ||W^T W - I||_F^2
            - 'rows': ||W W^T - I||_F^2 (orthogonal rows)
            - 'cols': ||W^T W - I||_F^2 (orthogonal columns)

    Returns:
        Orthogonal regularization loss

    Example:
        >>> w = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> orthogonal_regularization(w, scale=0.01, mode='symmetric')
    """
    if mode == "symmetric" or mode == "cols":
        # Compute W^T W
        gram = jnp.matmul(weight.T, weight)
        identity_size = weight.shape[1]
    elif mode == "rows":
        # Compute W W^T
        gram = jnp.matmul(weight, weight.T)
        identity_size = weight.shape[0]
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'symmetric', 'rows', 'cols'")

    # Create identity matrix
    identity = jnp.eye(identity_size)

    # Compute the difference from identity
    diff = gram - identity

    # Compute Frobenius norm squared: ||gram - I||_F^2
    orth_loss = jnp.sum(jnp.square(diff))

    return scale * orth_loss


def total_variation_loss(
    images: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    norm_type: str = "l2",
) -> jax.Array:
    """
    Total Variation (TV) loss for images.

    Penalizes large differences between adjacent pixels, encouraging
    spatial smoothness in images.

    Args:
        images: Image tensor with shape (B, H, W, C) or (B, H, W)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each image
        norm_type: Type of norm to use ('l1', 'l2')

    Returns:
        TV loss after specified reduction

    Example:
        >>> img = jnp.ones((2, 10, 10, 3))
        >>> total_variation_loss(img)
    """
    # Handle different image shapes
    if images.ndim == 3:  # (B, H, W)
        images = images[..., jnp.newaxis]

    # Compute differences along height and width
    height_diff = images[:, 1:, :, :] - images[:, :-1, :, :]
    width_diff = images[:, :, 1:, :] - images[:, :, :-1, :]

    # Compute variations based on norm type
    if norm_type == "l1":
        height_var = jnp.sum(jnp.abs(height_diff), axis=(1, 2, 3))
        width_var = jnp.sum(jnp.abs(width_diff), axis=(1, 2, 3))
    elif norm_type == "l2":
        height_var = jnp.sum(jnp.square(height_diff), axis=(1, 2, 3))
        width_var = jnp.sum(jnp.square(width_diff), axis=(1, 2, 3))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Choose 'l1' or 'l2'")

    # Combine height and width variations
    tv_loss = height_var + width_var

    return reduce_loss(tv_loss, reduction, weights)


def gradient_penalty(
    real_samples: jax.Array,
    fake_samples: jax.Array,
    discriminator_fn: Callable,
    lambda_gp: float = 10.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    interpolation_mode: str = "random",
    key: jax.Array | None = None,
) -> jax.Array:
    """
    Gradient penalty for WGAN-GP.

    Penalizes the gradient norm of the critic to enforce Lipschitz constraint.

    Args:
        real_samples: Real data samples
        fake_samples: Generated data samples
        discriminator_fn: Discriminator/critic function
        lambda_gp: Gradient penalty weight
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for batch elements
        interpolation_mode: How to interpolate ('random', 'uniform')
        key: JAX random key (required if interpolation_mode='random')

    Returns:
        Gradient penalty loss after specified reduction

    Example:
        >>> real = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fake = jnp.array([[0.5, 1.5], [2.5, 3.5]])
        >>> def disc_fn(x): return jnp.sum(x, axis=1)
        >>> gradient_penalty(real, fake, disc_fn, key=jax.random.key(0))
    """
    batch_size = real_samples.shape[0]

    if interpolation_mode == "random":
        if key is None:
            raise ValueError("Random key required for random interpolation")
        # Generate random interpolation factor for each sample in the batch
        alpha = jax.random.uniform(key, (batch_size,))
    elif interpolation_mode == "uniform":
        # Use uniform interpolation (0.5)
        alpha = jnp.full((batch_size,), 0.5)
    else:
        raise ValueError(f"Unknown interpolation_mode: {interpolation_mode}")

    # Reshape alpha to match the dimensions of real_samples
    alpha_reshaped = alpha
    for _ in range(real_samples.ndim - 1):
        alpha_reshaped = alpha_reshaped[..., jnp.newaxis]

    # Create interpolated samples
    interpolated = alpha_reshaped * real_samples + (1 - alpha_reshaped) * fake_samples

    # Compute gradients of discriminator w.r.t. interpolated samples
    def disc_interpolated(x):
        outputs = discriminator_fn(x)
        # Ensure scalar output for gradient computation
        return jnp.sum(outputs)

    grads = jax.grad(disc_interpolated)(interpolated)

    # Compute gradient norm (flattening non-batch dimensions)
    grad_flat = jnp.reshape(grads, (batch_size, -1))
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_flat), axis=1) + 1e-12)

    # Compute penalty: (||grad|| - 1)^2
    penalty = jnp.square(grad_norm - 1.0)

    return lambda_gp * reduce_loss(penalty, reduction, weights)


class DropoutRegularization(nnx.Module):
    """Dropout regularization as an NNX module."""

    def __init__(self, rate: float = 0.1):
        """Initialize dropout regularization.

        Args:
            rate: Dropout rate (0.0 to 1.0)
        """
        super().__init__()
        self.rate = rate

    def __call__(
        self, activations: jax.Array, training: bool = True, key: jax.Array | None = None
    ) -> jax.Array:
        """Apply dropout regularization.

        Args:
            activations: Input activations
            training: Whether in training mode
            key: Random key for dropout

        Returns:
            Regularization penalty (usually 0 for dropout)
        """
        if not training or self.rate == 0.0:
            return jnp.array(0.0)

        # Dropout doesn't contribute to loss directly,
        # but we could add some regularization term here
        # For now, return 0 as dropout is applied during forward pass
        return jnp.array(0.0)


# Helper functions


def _flatten_nnx_state(state: nnx.State):
    """Flatten an NNX State into (name, value) pairs."""
    items = []

    def _extract_from_state(obj, prefix=""):
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                new_key = f"{prefix}/{k}" if prefix else k
                if hasattr(v, "value"):  # NNX Variable
                    items.append((new_key, v.value))
                elif isinstance(v, (dict, nnx.State)):
                    _extract_from_state(v, new_key)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}/{k}" if prefix else k
                if hasattr(v, "value"):  # NNX Variable
                    items.append((new_key, v.value))
                elif isinstance(v, (dict, nnx.State)):
                    _extract_from_state(v, new_key)

    _extract_from_state(state)
    return items


def _flatten_dict(d, parent_key=""):
    """Flatten a nested dictionary into (name, value) pairs."""
    items = []
    for k, v in d.items() if isinstance(d, dict) else enumerate(d):
        k = str(k)
        new_key = parent_key + "/" + k if parent_key else k
        if isinstance(v, (dict, list)):
            items.extend(_flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items


def _count_params(params):
    """Count the total number of parameters."""
    if isinstance(params, nnx.State):
        return sum(
            param.value.size if hasattr(param, "value") else param.size
            for _, param in _flatten_nnx_state(params)
        )
    elif isinstance(params, jax.Array):
        return params.size
    elif isinstance(params, list):
        return sum(p.size for p in params)
    else:
        return sum(param.size for _, param in _flatten_dict(params))


# Predicate functions for common use cases


def exclude_bias_predicate(name: str, param: jax.Array) -> bool:
    """Predicate to exclude bias parameters from regularization."""
    return "bias" not in name.lower()


def only_conv_predicate(name: str, param: jax.Array) -> bool:
    """Predicate to only regularize convolutional layer parameters."""
    return "conv" in name.lower() and param.ndim >= 3


def exclude_norm_predicate(name: str, param: jax.Array) -> bool:
    """Predicate to exclude normalization layer parameters."""
    excluded_terms = ["batch_norm", "layer_norm", "group_norm", "norm"]
    return not any(term in name.lower() for term in excluded_terms)
