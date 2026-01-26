"""Protocol definitions for training modules.

This module defines protocols used by trainers to ensure type safety
and consistent interfaces across different model types.
"""

from __future__ import annotations

from typing import Protocol

import jax


class NoiseScheduleProtocol(Protocol):
    """Protocol for noise schedules used by diffusion trainers.

    Defines the interface that noise schedules must implement to be
    compatible with the diffusion training infrastructure.

    Attributes:
        num_timesteps: Total number of diffusion timesteps.
        alphas_cumprod: Cumulative product of alpha values at each timestep.
            Shape (num_timesteps,).

    Example:
        >>> class MyNoiseSchedule:
        ...     num_timesteps: int = 1000
        ...     alphas_cumprod: jax.Array = ...
        ...
        ...     def add_noise(self, x, noise, t):
        ...         alpha = self.alphas_cumprod[t]
        ...         return jnp.sqrt(alpha) * x + jnp.sqrt(1 - alpha) * noise
    """

    num_timesteps: int
    alphas_cumprod: jax.Array

    def add_noise(
        self,
        x: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Add noise to data at given timesteps.

        Args:
            x: Clean data, shape (batch, ...).
            noise: Noise samples, same shape as x.
            t: Integer timesteps, shape (batch,).

        Returns:
            Noisy data at timestep t.
        """
        ...
