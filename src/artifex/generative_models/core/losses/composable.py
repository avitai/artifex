"""Composable loss function framework using Flax NNX."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx


LossFn = Callable[[Any, dict[str, Any] | None], jax.Array]


class Loss(nnx.Module):
    """Base class for composable loss functions.

    This class provides a common interface for all loss functions in the
    framework, enabling easy composition and weighting.
    """

    def __init__(self, weight: float = 1.0, name: str | None = None):
        """Initialize the loss module.

        Args:
            weight: Weight to apply to the loss.
            name: Optional name for the loss (useful for logging).
        """
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__

    def __call__(self, *args, **kwargs) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        """Calculate the loss.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Loss value.
        """
        raise NotImplementedError("Must be implemented by subclass")


class CompositeLoss(Loss):
    """Composite loss that combines multiple loss functions.

    This module allows combining multiple loss functions with optional
    individual weights, and returns both the total loss and individual
    loss components for monitoring.
    """

    def __init__(
        self,
        losses: list[Loss],
        weights: list[float] | None = None,
        name: str | None = None,
        return_components: bool = False,
    ):
        """Initialize the composite loss.

        Args:
            losses: List of loss modules to combine.
            weights: Optional list of weights for each loss.
                If not provided, uses weights from individual loss modules.
            name: Optional name for the composite loss.
            return_components: Whether to return individual loss components
                along with the total loss.
        """
        super().__init__(name=name or "CompositeLoss")
        self.losses = losses
        self.return_components = return_components

        # Use provided weights or individual loss weights
        if weights is not None:
            if len(weights) != len(losses):
                raise ValueError(f"Expected {len(losses)} weights, got {len(weights)}")
            for loss, weight in zip(losses, weights):
                loss.weight = weight

    def __call__(self, *args, **kwargs) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        """Calculate the composite loss.

        Args:
            *args: Positional arguments passed to each loss function.
            **kwargs: Keyword arguments passed to each loss function.

        Returns:
            If return_components is False: Weighted sum of all loss components.
            If return_components is True: Tuple of (total_loss, loss_dict) where
                loss_dict maps loss names to individual unweighted loss values.
        """
        total_loss = jnp.array(0.0)
        loss_dict = {}

        for loss in self.losses:
            # Get the weighted loss value (this is what WeightedLoss.__call__ returns)
            weighted_loss_value = loss(*args, **kwargs)

            # Add to total
            total_loss = total_loss + weighted_loss_value

            # For the dictionary, we want to store the unweighted base loss value
            # We need to reverse the weighting to get the original loss value
            if hasattr(loss, "loss_fn") and hasattr(loss, "weight"):
                # This is a WeightedLoss, so compute the base loss
                base_loss_value = loss.loss_fn(*args, **kwargs)
                loss_dict[loss.name] = base_loss_value
            else:
                # For other loss types, store the computed value (might already be weighted)
                loss_dict[loss.name] = weighted_loss_value

        if self.return_components:
            return total_loss, loss_dict
        return total_loss


class WeightedLoss(Loss):
    """A wrapper that applies a weight to any loss function."""

    def __init__(self, loss_fn: LossFn, weight: float = 1.0, name: str | None = None):
        """Initialize the weighted loss.

        Args:
            loss_fn: Base loss function to wrap.
            weight: Weight to apply to the loss.
            name: Optional name for the loss.
        """
        super().__init__(weight=weight, name=name)
        self.loss_fn = loss_fn

    def __call__(self, *args, **kwargs) -> jax.Array:
        """Calculate the weighted loss.

        Args:
            *args: Positional arguments passed to the loss function.
            **kwargs: Keyword arguments passed to the loss function.

        Returns:
            Weighted loss value.
        """
        base_loss = self.loss_fn(*args, **kwargs)
        return self.weight * base_loss


class ScheduledLoss(Loss):
    """Loss with time-based scheduling (e.g., for curriculum learning)."""

    def __init__(
        self, loss_fn: LossFn, schedule_fn: Callable[[int], float], name: str | None = None
    ):
        """Initialize the scheduled loss.

        Args:
            loss_fn: Base loss function.
            schedule_fn: Function that takes step number and returns weight.
            name: Optional name for the loss.
        """
        super().__init__(name=name)
        self.loss_fn = loss_fn
        self.schedule_fn = schedule_fn
        self.step = nnx.Variable(jnp.array(0))

    def __call__(self, *args, step: int | None = None, **kwargs) -> jax.Array:
        """Calculate the scheduled loss.

        Args:
            *args: Positional arguments passed to the loss function.
            step: Optional step number. If None, uses internal counter.
            **kwargs: Keyword arguments passed to the loss function.

        Returns:
            Scheduled loss value.
        """
        if step is None:
            step = int(self.step.value)  # type: ignore
            self.step.value += 1

        current_weight = self.schedule_fn(step)
        base_loss = self.loss_fn(*args, **kwargs)
        return current_weight * base_loss


def create_weighted_loss(loss_fn: LossFn, weight: float = 1.0) -> Callable:
    """Create a weighted loss function from a standard loss function.

    This is a functional alternative to the WeightedLoss class for cases
    where you don't need the full NNX module functionality.

    Args:
        loss_fn: Base loss function.
        weight: Weight to apply to the loss.

    Returns:
        Weighted loss function.
    """

    def weighted_loss_fn(*args, **kwargs):
        base_loss = loss_fn(*args, **kwargs)
        return weight * base_loss

    return weighted_loss_fn


def create_loss_suite(*losses: Loss, return_components: bool = True) -> CompositeLoss:
    """Convenience function to create a composite loss from multiple losses.

    Args:
        *losses: Variable number of Loss instances.
        return_components: Whether to return individual loss components.

    Returns:
        CompositeLoss instance combining all provided losses.
    """
    return CompositeLoss(list(losses), return_components=return_components)
