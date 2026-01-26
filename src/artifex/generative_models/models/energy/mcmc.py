"""MCMC sampling utilities for energy-based models.

This module provides sampling algorithms for energy-based models, including
Langevin dynamics and buffer-based sampling strategies.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


def langevin_dynamics(
    energy_fn: Callable[[jax.Array], jax.Array],
    initial_samples: jax.Array,
    n_steps: int,
    step_size: float = 0.01,
    noise_scale: float = 0.005,
    rng_key: jax.Array | None = None,
    clip_range: tuple[float, float] = (-1.0, 1.0),
    grad_clip: float = 0.03,
    temperature: float = 1.0,
) -> jax.Array:
    """Sample from energy-based model using Langevin dynamics.

    Langevin dynamics performs MCMC sampling by taking gradient steps
    on the energy function with added noise. With temperature T, we sample
    from p(x) ∝ exp(-E(x)/T):

    x_{t+1} = x_t - α * ∇_x [E(x_t)/T] + √(2α) * ε_t

    where α is the step size and ε_t is Gaussian noise.
    Higher temperature leads to smaller effective gradients and more exploration.

    Args:
        energy_fn: Energy function E(x)
        initial_samples: Initial samples of shape (batch_size, ...)
        n_steps: Number of MCMC steps
        step_size: Step size α for gradient updates
        noise_scale: Standard deviation of noise added at each step
        rng_key: Random key for noise generation
        clip_range: Range to clip samples to (min, max)
        grad_clip: Maximum magnitude for gradients (for stability)
        temperature: Temperature for Boltzmann distribution (higher = more exploration)

    Returns:
        Final samples after n_steps of Langevin dynamics
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Define energy function for a single sample (used with vmap+grad)
    def energy_single(x_single: jax.Array) -> jax.Array:
        return energy_fn(x_single[None])[0] / temperature

    grad_fn = nnx.vmap(nnx.grad(energy_single))

    def langevin_body(
        carry: tuple[jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        samples, rng = carry
        rng, step_key = jax.random.split(rng)

        # Add noise first (from EBM literature)
        noise = jax.random.normal(step_key, samples.shape) * noise_scale
        samples = samples + noise
        samples = jnp.clip(samples, clip_range[0], clip_range[1])

        # Compute gradients of energy/temperature w.r.t. samples
        grads = grad_fn(samples)

        # Clip gradients for stability
        grads = jnp.clip(grads, -grad_clip, grad_clip)

        # Update samples using gradient descent on energy
        samples = samples - step_size * grads

        # Clip samples to valid range
        samples = jnp.clip(samples, clip_range[0], clip_range[1])

        return (samples, rng), None

    (samples, _), _ = jax.lax.scan(langevin_body, (initial_samples, rng_key), None, length=n_steps)

    return samples


def langevin_dynamics_with_trajectory(
    energy_fn: Callable[[jax.Array], jax.Array],
    initial_samples: jax.Array,
    n_steps: int,
    step_size: float = 0.01,
    noise_scale: float = 0.005,
    rng_key: jax.Array | None = None,
    clip_range: tuple[float, float] = (-1.0, 1.0),
    grad_clip: float = 0.03,
    save_every: int = 1,
    temperature: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Langevin dynamics with trajectory recording.

    Same as langevin_dynamics but also returns the trajectory of samples.
    Samples from p(x) ∝ exp(-E(x)/T).

    Args:
        energy_fn: Energy function E(x)
        initial_samples: Initial samples of shape (batch_size, ...)
        n_steps: Number of MCMC steps
        step_size: Step size for gradient updates
        noise_scale: Standard deviation of noise
        rng_key: Random key for noise generation
        clip_range: Range to clip samples to
        grad_clip: Maximum magnitude for gradients
        save_every: Save trajectory every N steps
        temperature: Temperature for Boltzmann distribution (higher = more exploration)

    Returns:
        Tuple of (final_samples, trajectory) where trajectory has shape
        (n_trajectory_points, batch_size, ...)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Define energy function for a single sample (used with vmap+grad)
    def energy_single(x_single: jax.Array) -> jax.Array:
        return energy_fn(x_single[None])[0] / temperature

    grad_fn = nnx.vmap(nnx.grad(energy_single))

    def langevin_body(
        carry: tuple[jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        samples, rng = carry
        rng, step_key = jax.random.split(rng)

        # Add noise first
        noise = jax.random.normal(step_key, samples.shape) * noise_scale
        samples = samples + noise
        samples = jnp.clip(samples, clip_range[0], clip_range[1])

        # Compute gradients of energy/temperature
        grads = grad_fn(samples)
        grads = jnp.clip(grads, -grad_clip, grad_clip)

        # Update samples
        samples = samples - step_size * grads
        samples = jnp.clip(samples, clip_range[0], clip_range[1])

        return (samples, rng), samples

    (final_samples, _), all_samples = jax.lax.scan(
        langevin_body, (initial_samples, rng_key), None, length=n_steps
    )

    # Build trajectory: select every save_every-th step
    # all_samples shape: (n_steps, batch_size, ...)
    # Include initial samples at the start
    trajectory_indices = jnp.arange(save_every - 1, n_steps, save_every)
    trajectory_steps = all_samples[trajectory_indices]
    trajectory = jnp.concatenate([initial_samples[None], trajectory_steps], axis=0)

    return final_samples, trajectory


class SampleBuffer:
    """Buffer for storing and reusing MCMC samples.

    This buffer implements the sampling strategy from EBM literature where
    95% of initial samples come from previous iterations (stored in buffer)
    and 5% are initialized from scratch. This significantly reduces the
    number of MCMC steps needed for good samples.
    """

    def __init__(
        self,
        capacity: int = 8192,
        reinit_prob: float = 0.05,
        sample_shape: tuple[int, ...] | None = None,
    ):
        """Initialize sample buffer.

        Args:
            capacity: Maximum number of samples to store
            reinit_prob: Probability of reinitializing samples from scratch
            sample_shape: Shape of individual samples (excluding batch dim)
        """
        self.capacity = capacity
        self.reinit_prob = reinit_prob
        self.sample_shape = sample_shape
        self.buffer: list[jax.Array] = []

    def sample_initial(
        self,
        batch_size: int,
        rng_key: jax.Array,
        sample_shape: tuple[int, ...] | None = None,
    ) -> jax.Array:
        """Sample initial points for MCMC.

        Args:
            batch_size: Number of samples to generate
            rng_key: Random key
            sample_shape: Shape of individual samples

        Returns:
            Initial samples of shape (batch_size, *sample_shape)
        """
        if sample_shape is None:
            sample_shape = self.sample_shape
        if sample_shape is None:
            raise ValueError("sample_shape must be provided")

        key1, key2 = jax.random.split(rng_key)

        # If buffer is empty, just use random samples
        if len(self.buffer) == 0:
            return jax.random.uniform(key2, (batch_size, *sample_shape), minval=-1.0, maxval=1.0)

        # Determine how many samples to reinitialize
        n_reinit = jax.random.binomial(key1, n=batch_size, p=self.reinit_prob).astype(int)
        n_from_buffer = batch_size - n_reinit

        samples = []

        # Add reinitialized samples
        if n_reinit > 0:
            reinit_samples = jax.random.uniform(
                key2,
                (n_reinit, *sample_shape),
                minval=-1.0,
                maxval=1.0,
            )
            samples.append(reinit_samples)

        # Add samples from buffer
        if n_from_buffer > 0:
            # Randomly select from buffer
            buffer_array = jnp.concatenate(self.buffer, axis=0)
            n_available = buffer_array.shape[0]

            if n_available >= n_from_buffer:
                indices = jax.random.choice(key2, n_available, (n_from_buffer,), replace=False)
                buffer_samples = buffer_array[indices]
            else:
                # If not enough samples in buffer, use all and pad with random
                buffer_samples = buffer_array
                n_additional = n_from_buffer - n_available
                additional_samples = jax.random.uniform(
                    key2,
                    (n_additional, *sample_shape),
                    minval=-1.0,
                    maxval=1.0,
                )
                buffer_samples = jnp.concatenate([buffer_samples, additional_samples])

            samples.append(buffer_samples)

        # Should not be empty at this point, but just in case
        if not samples:
            return jax.random.uniform(key2, (batch_size, *sample_shape), minval=-1.0, maxval=1.0)

        # Concatenate all samples
        return jnp.concatenate(samples, axis=0)

    def update_buffer(self, new_samples: jax.Array) -> None:
        """Update buffer with new samples.

        Args:
            new_samples: New samples to add to buffer
        """
        # Add the entire batch as a single element
        self.buffer.append(new_samples)

        # Trim buffer if it exceeds capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity :]


def improved_langevin_dynamics(
    energy_fn: Callable[[jax.Array], jax.Array],
    initial_samples: jax.Array,
    n_steps: int,
    step_size: float = 0.01,
    noise_scale: float = 0.005,
    rng_key: jax.Array | None = None,
    clip_range: tuple[float, float] = (-1.0, 1.0),
    grad_clip: float = 0.03,
    adaptive_step_size: bool = True,
    target_acceptance: float = 0.574,  # Optimal for Langevin
) -> jax.Array:
    """Improved Langevin dynamics with adaptive step size.

    Args:
        energy_fn: Energy function E(x)
        initial_samples: Initial samples
        n_steps: Number of MCMC steps
        step_size: Initial step size
        noise_scale: Noise standard deviation
        rng_key: Random key
        clip_range: Clipping range
        grad_clip: Gradient clipping
        adaptive_step_size: Whether to adapt step size
        target_acceptance: Target acceptance rate for adaptation

    Returns:
        Final samples after MCMC
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    adaptation_window = min(50, max(1, n_steps // 10))

    # Define energy function for a single sample (used with vmap+grad)
    def energy_single(x_single: jax.Array) -> jax.Array:
        return energy_fn(x_single[None])[0]

    grad_fn = nnx.vmap(nnx.grad(energy_single))

    # carry: (samples, rng_key, current_step_size, n_accepted, step_counter)
    def langevin_body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
        samples, rng, current_step_size, n_accepted, step_counter = carry
        rng, step_key = jax.random.split(rng)

        # Store current energy for acceptance computation
        old_energy = energy_fn(samples)

        # Langevin step
        noise = jax.random.normal(step_key, samples.shape) * noise_scale
        noisy_samples = samples + noise
        noisy_samples = jnp.clip(noisy_samples, clip_range[0], clip_range[1])

        # Compute gradients
        grads = grad_fn(noisy_samples)
        grads = jnp.clip(grads, -grad_clip, grad_clip)

        # Propose new samples
        new_samples = noisy_samples - current_step_size * grads
        new_samples = jnp.clip(new_samples, clip_range[0], clip_range[1])

        # Adaptive step size tracking
        new_energy = energy_fn(new_samples)
        energy_diff = new_energy - old_energy
        accepted = jnp.mean(energy_diff < 0)
        new_n_accepted = n_accepted + accepted
        new_counter = step_counter + 1

        # Adapt step size every adaptation_window steps (using jnp.where for JIT compat)
        at_window = (new_counter % adaptation_window) == 0
        acceptance_rate = new_n_accepted / adaptation_window
        grow = acceptance_rate > target_acceptance
        scale_factor = jnp.where(grow, 1.05, 0.95)
        adapted_step_size = jnp.where(
            at_window & adaptive_step_size,
            current_step_size * scale_factor,
            current_step_size,
        )
        adapted_n_accepted = jnp.where(at_window, jnp.zeros_like(new_n_accepted), new_n_accepted)

        return (new_samples, rng, adapted_step_size, adapted_n_accepted, new_counter), None

    init_carry = (
        initial_samples,
        rng_key,
        jnp.array(step_size),
        jnp.array(0.0),
        jnp.array(0),
    )
    (final_samples, _, _, _, _), _ = jax.lax.scan(langevin_body, init_carry, None, length=n_steps)

    return final_samples


def persistent_contrastive_divergence(
    energy_fn: Callable[[jax.Array], jax.Array],
    real_samples: jax.Array,
    sample_buffer: SampleBuffer,
    rng_key: jax.Array,
    n_mcmc_steps: int = 60,
    step_size: float = 0.01,
    noise_scale: float = 0.005,
    temperature: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Persistent Contrastive Divergence with sample buffer.

    Args:
        energy_fn: Energy function
        real_samples: Real data samples
        sample_buffer: Buffer for storing generated samples
        rng_key: Random key
        n_mcmc_steps: Number of MCMC steps
        step_size: Step size for Langevin dynamics
        noise_scale: Noise scale for Langevin dynamics
        temperature: Temperature for Boltzmann distribution (higher = more exploration)

    Returns:
        Tuple of (real_samples, generated_samples)
    """
    batch_size = real_samples.shape[0]
    sample_shape = real_samples.shape[1:]

    # Get initial samples from buffer
    initial_samples = sample_buffer.sample_initial(
        batch_size=batch_size,
        rng_key=rng_key,
        sample_shape=sample_shape,
    )

    # Run MCMC
    generated_samples = langevin_dynamics(
        energy_fn=energy_fn,
        initial_samples=initial_samples,
        n_steps=n_mcmc_steps,
        step_size=step_size,
        noise_scale=noise_scale,
        rng_key=rng_key,
        temperature=temperature,
    )

    # Update buffer
    sample_buffer.update_buffer(generated_samples)

    return real_samples, generated_samples
