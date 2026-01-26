"""Gradient accumulation and dynamic loss scaling for advanced training.

Provides utilities for:
- Gradient accumulation for larger effective batch sizes
- Dynamic loss scaling for mixed-precision training

These utilities integrate with the base Trainer and can be used independently
or combined for optimal training performance.

Example:
    ```python
    from artifex.generative_models.training.gradient_accumulation import (
        GradientAccumulator,
        GradientAccumulatorConfig,
        DynamicLossScaler,
        DynamicLossScalerConfig,
    )

    # Gradient accumulation for 4x effective batch size
    accumulator = GradientAccumulator(
        GradientAccumulatorConfig(accumulation_steps=4)
    )

    # Dynamic loss scaling for mixed precision
    scaler = DynamicLossScaler(
        DynamicLossScalerConfig(initial_scale=2**15)
    )

    for step, batch in enumerate(dataloader):
        # Scale loss for mixed precision
        loss = compute_loss(model, batch)
        scaled_loss = scaler.scale_loss(loss)

        # Compute gradients
        grads = compute_gradients(model, scaled_loss)

        # Unscale and check for overflow
        grads = scaler.unscale_gradients(grads)
        overflow = scaler.check_overflow(grads)
        scaler.update_scale(overflow)

        if overflow:
            continue  # Skip update on overflow

        # Accumulate gradients
        accumulator.accumulate(grads)

        if accumulator.should_update(step):
            final_grads = accumulator.get_gradients()
            optimizer.update(model, final_grads)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


# =============================================================================
# Gradient Accumulator
# =============================================================================


@dataclass(slots=True)
class GradientAccumulatorConfig:
    """Configuration for gradient accumulation.

    Attributes:
        accumulation_steps: Number of microbatches to accumulate before update.
            Effective batch size = batch_size * accumulation_steps.
        normalize_gradients: Whether to divide gradients by accumulation_steps.
            When True, gradients are averaged. When False, gradients are summed.
    """

    accumulation_steps: int = 1
    normalize_gradients: bool = True


class GradientAccumulator:
    """Accumulate gradients across multiple microbatches.

    Enables larger effective batch sizes than GPU memory allows by
    accumulating gradients across multiple forward/backward passes
    before performing an optimizer update.

    Features:
        - Configurable accumulation steps
        - Optional gradient normalization (averaging vs summing)
        - Automatic reset after gradient retrieval
        - Step-based update scheduling

    Example:
        ```python
        accumulator = GradientAccumulator(
            GradientAccumulatorConfig(accumulation_steps=4)
        )

        for step, batch in enumerate(dataloader):
            grads = compute_gradients(model, batch)
            accumulator.accumulate(grads)

            if accumulator.should_update(step):
                final_grads = accumulator.get_gradients()
                optimizer.update(model, final_grads)
        ```
    """

    __slots__ = ("config", "accumulated_grads", "_step_count")

    def __init__(self, config: GradientAccumulatorConfig) -> None:
        """Initialize gradient accumulator.

        Args:
            config: Accumulator configuration.
        """
        self.config = config
        self.accumulated_grads: Any = None
        self._step_count: int = 0

    @property
    def current_step(self) -> int:
        """Return current step count within accumulation window."""
        return self._step_count

    def reset(self) -> None:
        """Reset accumulated gradients and step count."""
        self.accumulated_grads = None
        self._step_count = 0

    def should_update(self, step: int) -> bool:
        """Check if optimizer should be updated at this step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            True if this step should trigger an optimizer update.
        """
        # Update every accumulation_steps steps
        return (step + 1) % self.config.accumulation_steps == 0

    def accumulate(self, grads: Any) -> None:
        """Add gradients to accumulation buffer.

        Args:
            grads: Gradient pytree from current microbatch.
        """
        if self.accumulated_grads is None:
            # First accumulation - initialize with copy
            self.accumulated_grads = jax.tree.map(lambda g: g, grads)
        else:
            # Add to existing accumulation
            self.accumulated_grads = jax.tree.map(
                lambda acc, g: acc + g,
                self.accumulated_grads,
                grads,
            )
        self._step_count += 1

    def get_gradients(self) -> Any:
        """Get accumulated gradients and reset accumulator.

        Optionally normalizes gradients by dividing by accumulation_steps
        if normalize_gradients is enabled.

        Returns:
            Accumulated (and optionally normalized) gradient pytree.

        Raises:
            RuntimeError: If no gradients have been accumulated.
        """
        if self.accumulated_grads is None:
            msg = "No gradients accumulated. Call accumulate() first."
            raise RuntimeError(msg)

        if self.config.normalize_gradients:
            # Average gradients
            factor = 1.0 / self.config.accumulation_steps
            result = jax.tree.map(
                lambda g: g * factor,
                self.accumulated_grads,
            )
        else:
            # Return summed gradients
            result = self.accumulated_grads

        # Reset for next accumulation window
        self.reset()

        return result


# =============================================================================
# Dynamic Loss Scaler
# =============================================================================


@dataclass(slots=True)
class DynamicLossScalerConfig:
    """Configuration for dynamic loss scaling.

    Attributes:
        initial_scale: Initial loss scale value (typically 2^15 = 32768).
        growth_factor: Factor to multiply scale by after successful steps.
        backoff_factor: Factor to multiply scale by after overflow.
        growth_interval: Number of successful steps before growing scale.
        min_scale: Minimum allowed loss scale.
        max_scale: Maximum allowed loss scale.
    """

    initial_scale: float = 2**15  # 32768
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    min_scale: float = 1.0
    max_scale: float = 2**24  # 16777216


class DynamicLossScaler:
    """Dynamic loss scaling for mixed-precision training.

    Automatically adjusts loss scale to prevent gradient underflow
    in reduced precision (float16/bfloat16) training while avoiding
    overflow that would corrupt gradients.

    Features:
        - Automatic scale growth after successful steps
        - Scale reduction on gradient overflow/NaN detection
        - Configurable min/max bounds
        - Compatible with JAX's mixed precision

    The typical workflow:
    1. Scale loss before backward pass
    2. Compute gradients
    3. Unscale gradients
    4. Check for overflow
    5. Update scale based on overflow status
    6. Skip optimizer update if overflow occurred

    Example:
        ```python
        scaler = DynamicLossScaler(
            DynamicLossScalerConfig(initial_scale=2**15)
        )

        for batch in dataloader:
            # Forward pass
            loss = compute_loss(model, batch)

            # Scale loss for mixed precision
            scaled_loss = scaler.scale_loss(loss)

            # Backward pass with scaled loss
            grads = jax.grad(lambda: scaled_loss)(model)

            # Unscale gradients
            grads = scaler.unscale_gradients(grads)

            # Check for overflow and update scale
            overflow = scaler.check_overflow(grads)
            scaler.update_scale(overflow)

            if not overflow:
                optimizer.update(model, grads)
        ```
    """

    __slots__ = ("config", "_scale", "_steps_since_growth")

    def __init__(self, config: DynamicLossScalerConfig | None = None) -> None:
        """Initialize dynamic loss scaler.

        Args:
            config: Scaler configuration. Uses defaults if not provided.
        """
        self.config = config or DynamicLossScalerConfig()
        self._scale: float = self.config.initial_scale
        self._steps_since_growth: int = 0

    @property
    def scale(self) -> float:
        """Current loss scale value."""
        return self._scale

    @property
    def steps_since_growth(self) -> int:
        """Number of steps since last scale growth."""
        return self._steps_since_growth

    def scale_loss(self, loss: jax.Array) -> jax.Array:
        """Scale loss for mixed-precision backward pass.

        Args:
            loss: Unscaled loss value.

        Returns:
            Loss multiplied by current scale.
        """
        return loss * self._scale

    def unscale_gradients(self, grads: Any) -> Any:
        """Unscale gradients after backward pass.

        Args:
            grads: Scaled gradient pytree.

        Returns:
            Gradients divided by current scale.
        """
        inv_scale = 1.0 / self._scale
        return jax.tree.map(lambda g: g * inv_scale, grads)

    def check_overflow(self, grads: Any) -> bool:
        """Check if gradients contain inf or nan values.

        This method is intended for use outside JIT boundaries (e.g., in
        Python training loops). It materializes JAX arrays to Python bools.

        Args:
            grads: Gradient pytree to check.

        Returns:
            True if any gradient contains inf or nan.
        """
        # Reduce all leaves to a single finite check
        all_finite = jnp.array(True)
        for leaf in jax.tree.leaves(grads):
            all_finite = all_finite & jnp.all(jnp.isfinite(leaf))

        return not bool(all_finite)

    def update_scale(self, overflow_detected: bool) -> None:
        """Update loss scale based on overflow status.

        On overflow:
            - Reduce scale by backoff_factor
            - Reset steps_since_growth counter

        Without overflow:
            - Increment steps_since_growth
            - Grow scale by growth_factor after growth_interval steps

        Args:
            overflow_detected: Whether gradient overflow was detected.
        """
        if overflow_detected:
            # Reduce scale on overflow
            new_scale = self._scale * self.config.backoff_factor
            self._scale = max(new_scale, self.config.min_scale)
            self._steps_since_growth = 0
        else:
            # Increment counter
            self._steps_since_growth += 1

            # Grow scale after interval
            if self._steps_since_growth >= self.config.growth_interval:
                new_scale = self._scale * self.config.growth_factor
                self._scale = min(new_scale, self.config.max_scale)
                self._steps_since_growth = 0


__all__ = [
    "GradientAccumulator",
    "GradientAccumulatorConfig",
    "DynamicLossScaler",
    "DynamicLossScalerConfig",
]
