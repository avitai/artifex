"""
Base loss functionality and utilities.

This module provides core functions and classes for implementing and
combining loss functions. It forms the foundation for all loss
implementations in the artifex library and integrates seamlessly
with Flax NNX.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


def reduce_loss(
    loss: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Apply reduction to a loss tensor with optional weighting.

    Args:
        loss: Raw loss values, shape [batch, ...]
        reduction: Reduction method:
            - 'none': No reduction
            - 'mean': Mean over all elements (or specified axis)
            - 'sum': Sum over all elements (or specified axis)
            - 'batch_sum': Sum over non-batch dims, mean over batch (standard for VAE ELBO)
        weights: Optional weighting factors for the loss values
        axis: Axis or axes over which to reduce (ignored for 'batch_sum')

    Returns:
        Reduced loss value(s) depending on reduction method

    Raises:
        ValueError: If reduction is not a valid option

    Example:
        >>> loss = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        >>> reduce_loss(loss, reduction="mean")      # Returns 2.5
        >>> reduce_loss(loss, reduction="sum")       # Returns 10.0
        >>> reduce_loss(loss, reduction="batch_sum") # Returns 5.0 (mean of [3, 7])
    """
    valid_reductions = ["none", "mean", "sum", "batch_sum"]
    if reduction not in valid_reductions:
        raise ValueError(f"Invalid reduction: {reduction}, expected one of {valid_reductions}")

    if weights is not None:
        loss = loss * weights

    if reduction == "mean":
        return jnp.mean(loss, axis=axis)
    elif reduction == "sum":
        return jnp.sum(loss, axis=axis)
    elif reduction == "batch_sum":
        # Sum over all dimensions except batch (dim 0), then mean over batch
        # This is the standard reduction for VAE ELBO loss
        if loss.ndim <= 1:
            return jnp.mean(loss)
        batch_size = loss.shape[0]
        spatial_sum = jnp.sum(loss.reshape(batch_size, -1), axis=-1)
        return jnp.mean(spatial_sum)
    else:  # reduction == "none"
        return loss


class LossCollection:
    """
    A collection of loss functions that can be combined with weights.

    This class provides a convenient way to combine multiple loss functions
    and track their individual contributions to the total loss. It uses a
    functional approach and is compatible with JAX transformations.

    Note: For NNX-style modular composition, consider using CompositeLoss instead.
    """

    def __init__(self):
        """Initialize an empty loss collection."""
        self.losses: list[tuple[Callable, float, str]] = []

    def add(
        self,
        loss_fn: Callable,
        weight: float = 1.0,
        name: str | None = None,
    ) -> "LossCollection":
        """
        Add a loss function to the collection.

        Args:
            loss_fn: The loss function to add
            weight: Weight for this loss function
            name: Name for this loss (used for logging)

        Returns:
            Self, for method chaining
        """
        name_value = name or f"loss_{len(self.losses)}"
        self.losses.append((loss_fn, weight, name_value))
        return self

    def __call__(self, *args, **kwargs) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        Compute all losses and return their weighted sum.

        Args:
            *args: Positional arguments passed to loss functions
            **kwargs: Keyword arguments passed to loss functions

        Returns:
            tuple of (total_loss, loss_dict) where loss_dict maps names to
            individual loss values
        """
        total_loss = jnp.array(0.0)
        loss_dict = {}

        for loss_fn, weight, name in self.losses:
            loss_value = loss_fn(*args, **kwargs)
            weighted_loss = weight * loss_value
            # Use + for JAX traceability
            total_loss = total_loss + weighted_loss
            loss_dict[name] = loss_value

        return total_loss, loss_dict

    def compute_individual(self, *args, **kwargs) -> dict[str, jax.Array]:
        """
        Compute individual losses without combining them.

        Args:
            *args: Positional arguments passed to loss functions
            **kwargs: Keyword arguments passed to loss functions

        Returns:
            dictionary mapping loss names to their values
        """
        loss_dict = {}
        for loss_fn, weight, name in self.losses:
            loss_value = loss_fn(*args, **kwargs)
            loss_dict[name] = loss_value
        return loss_dict

    def get_weights(self) -> dict[str, float]:
        """Get the weights for all losses in the collection."""
        return {name: weight for _, weight, name in self.losses}

    def set_weight(self, name: str, weight: float) -> None:
        """Set the weight for a specific loss by name."""
        for i, (loss_fn, _, loss_name) in enumerate(self.losses):
            if loss_name == name:
                self.losses[i] = (loss_fn, weight, loss_name)
                return
        raise ValueError(f"Loss with name '{name}' not found in collection")

    def remove(self, name: str) -> None:
        """Remove a loss from the collection by name."""
        self.losses = [(fn, w, n) for fn, w, n in self.losses if n != name]

    def clear(self) -> None:
        """Remove all losses from the collection."""
        self.losses.clear()

    def __len__(self) -> int:
        """Return the number of losses in the collection."""
        return len(self.losses)

    def __repr__(self) -> str:
        """Return string representation of the loss collection."""
        loss_info = [f"{name}: {weight}" for _, weight, name in self.losses]
        return f"LossCollection({', '.join(loss_info)})"


class LossMetrics(nnx.Module):
    """
    NNX module for tracking loss metrics during training.

    This module keeps running statistics of losses for monitoring purposes.
    """

    def __init__(self, momentum: float = 0.99):
        """Initialize loss metrics tracker.

        Args:
            momentum: Momentum factor for exponential moving average
        """
        super().__init__()
        self.momentum = momentum
        self.metrics: dict[str, nnx.Variable] = {}
        self.counts: dict[str, nnx.Variable] = {}

    def update(self, loss_dict: dict[str, jax.Array]) -> None:
        """Update metrics with new loss values.

        Args:
            loss_dict: dictionary of loss names to values
        """
        for name, value in loss_dict.items():
            if name not in self.metrics:
                self.metrics[name] = nnx.Variable(jnp.array(0.0))
                self.counts[name] = nnx.Variable(jnp.array(0.0))

            # Exponential moving average
            current_value = self.metrics[name].value
            self.metrics[name].value = self.momentum * current_value + (1 - self.momentum) * value
            self.counts[name].value += 1

    def get_metrics(self) -> dict[str, float]:
        """Get current metric values.

        Returns:
            dictionary of metric names to their current values
        """
        return {name: float(var.value) for name, var in self.metrics.items()}

    def reset(self) -> None:
        """Reset all metrics."""
        for var in self.metrics.values():
            var.value = jnp.array(0.0)
        for var in self.counts.values():
            var.value = jnp.array(0.0)


class LossScheduler:
    """
    Scheduler for dynamically adjusting loss weights during training.

    This class provides various scheduling strategies for loss weights,
    useful for curriculum learning, progressive training, etc.
    """

    def __init__(self, initial_weights: dict[str, float]):
        """Initialize the loss scheduler.

        Args:
            initial_weights: Initial weights for each loss component
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.schedules: dict[str, Callable[[int], float]] = {}

    def add_schedule(self, loss_name: str, schedule_fn: Callable[[int], float]) -> None:
        """Add a scheduling function for a specific loss.

        Args:
            loss_name: Name of the loss to schedule
            schedule_fn: Function that takes step number and returns weight multiplier
        """
        if loss_name not in self.initial_weights:
            raise ValueError(f"Loss '{loss_name}' not in initial weights")
        self.schedules[loss_name] = schedule_fn

    def update(self, step: int) -> dict[str, float]:
        """Update weights based on current step.

        Args:
            step: Current training step

        Returns:
            Updated weights dictionary
        """
        for name, schedule_fn in self.schedules.items():
            multiplier = schedule_fn(step)
            self.current_weights[name] = self.initial_weights[name] * multiplier

        return self.current_weights.copy()

    def get_weights(self) -> dict[str, float]:
        """Get current weights."""
        return self.current_weights.copy()

    @staticmethod
    def linear_warmup(warmup_steps: int, max_weight: float = 1.0) -> Callable[[int], float]:
        """Create a linear warmup schedule.

        Args:
            warmup_steps: Number of steps to warm up over
            max_weight: Maximum weight multiplier

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            return min(max_weight, (step / warmup_steps) * max_weight)

        return schedule

    @staticmethod
    def cosine_annealing(
        period: int, min_weight: float = 0.0, max_weight: float = 1.0
    ) -> Callable[[int], float]:
        """Create a cosine annealing schedule.

        Args:
            period: Period of the cosine cycle
            min_weight: Minimum weight multiplier
            max_weight: Maximum weight multiplier

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            cos_factor = 0.5 * (1 + jnp.cos(jnp.pi * (step % period) / period))
            return min_weight + (max_weight - min_weight) * cos_factor

        return schedule

    @staticmethod
    def exponential_decay(decay_steps: int, decay_rate: float = 0.9) -> Callable[[int], float]:
        """Create an exponential decay schedule.

        Args:
            decay_steps: Number of steps per decay
            decay_rate: Decay rate

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            return decay_rate ** (step // decay_steps)

        return schedule


def validate_loss_inputs(*args, **kwargs) -> None:
    """Validate common loss function inputs.

    JIT-compatible: uses jax.debug.callback for NaN/Inf checks so they
    work under tracing without ConcretizationTypeError. Checks are only
    executed at runtime (not during tracing), and only when not suppressed
    by JAX_DISABLE_JIT or similar.

    Args:
        *args: Positional arguments to validate
        **kwargs: Keyword arguments to validate

    Raises:
        ValueError: If inputs are not JAX arrays (checked eagerly, not traced).
    """

    def _check_finite(arg_index: int, arg: jax.Array) -> None:
        """Runtime callback to check for NaN/Inf values."""
        has_nan = jnp.any(jnp.isnan(arg))
        has_inf = jnp.any(jnp.isinf(arg))
        jax.debug.callback(
            lambda nan, inf, idx: None,  # No-op; use jax.debug.print for logging
            has_nan,
            has_inf,
            arg_index,
        )

    for i, arg in enumerate(args):
        if not isinstance(arg, jax.Array):
            raise ValueError(f"Argument {i} must be a JAX array, got {type(arg)}")
        # JIT-compatible NaN/Inf check via debug callback
        _check_finite(i, arg)


def check_shapes_compatible(
    pred: jax.Array, target: jax.Array, allow_broadcast: bool = True
) -> None:
    """
    Check if prediction and target shapes are compatible.

    Args:
        pred: Prediction array
        target: Target array
        allow_broadcast: Whether to allow broadcasting

    Raises:
        ValueError: If shapes are incompatible
    """
    if allow_broadcast:
        try:
            jnp.broadcast_shapes(pred.shape, target.shape)
        except ValueError as e:
            raise ValueError(
                f"Prediction and target shapes are not broadcast-compatible: "
                f"{pred.shape} vs {target.shape}"
            ) from e
    else:
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match exactly: {pred.shape} vs {target.shape}"
            )


def safe_log(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """
    Numerically stable logarithm.

    Args:
        x: Input array
        eps: Small constant to add for stability

    Returns:
        log(x + eps)
    """
    return jnp.log(jnp.clip(x, eps, None))


def safe_divide(numerator: jax.Array, denominator: jax.Array, eps: float = 1e-8) -> jax.Array:
    """
    Numerically stable division.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Small constant to add to denominator

    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def normalize_probabilities(logits: jax.Array, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    """
    Convert logits to normalized probabilities.

    Args:
        logits: Input logits
        axis: Axis along which to normalize
        eps: Small constant for numerical stability

    Returns:
        Normalized probabilities
    """
    probs = nnx.softmax(logits, axis=axis)
    return jnp.clip(probs, eps, 1.0 - eps)


class LossRegistry:
    """
    Global registry for loss functions.

    This allows for easy registration and retrieval of loss functions
    by name, useful for configuration-driven training.
    """

    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, loss_fn: Callable) -> None:
        """Register a loss function.

        Args:
            name: Name to register the loss function under
            loss_fn: Loss function to register
        """
        cls._registry[name] = loss_fn

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a registered loss function.

        Args:
            name: Name of the loss function

        Returns:
            The registered loss function

        Raises:
            KeyError: If loss function is not registered
        """
        if name not in cls._registry:
            raise KeyError(
                f"Loss function '{name}' not found in registry. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available loss functions."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry."""
        cls._registry.clear()


def stable_loss(eps: float = 1e-8) -> Callable:
    """
    Decorate loss functions with numerical stability.

    Args:
        eps: Small constant for stability

    Returns:
        Decorated loss function
    """

    def decorator(loss_fn):
        def wrapper(*args, **kwargs):
            # Add numerical stability to inputs
            stable_args = []
            for arg in args:
                if isinstance(arg, jax.Array) and jnp.issubdtype(arg.dtype, jnp.floating):
                    stable_args.append(jnp.clip(arg, eps, 1.0 - eps))
                else:
                    stable_args.append(arg)
            return loss_fn(*stable_args, **kwargs)

        return wrapper

    return decorator


def validate_inputs(check_shapes: bool = True, allow_broadcast: bool = True):
    """
    Decorate loss functions with input validation.

    Args:
        check_shapes: Whether to check shape compatibility
        allow_broadcast: Whether to allow broadcasting when checking shapes

    Returns:
        Decorated loss function
    """

    def decorator(loss_fn):
        def wrapper(*args, **kwargs):
            # Validate inputs
            validate_loss_inputs(*args, **kwargs)

            # Check shapes for first two arguments (typically pred, target)
            if check_shapes and len(args) >= 2:
                check_shapes_compatible(args[0], args[1], allow_broadcast)

            return loss_fn(*args, **kwargs)

        return wrapper

    return decorator
