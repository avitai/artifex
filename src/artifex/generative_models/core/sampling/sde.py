"""SDE-based sampling utilities.

This module provides utilities for sampling using stochastic differential
equations (SDEs). SDE-based sampling is useful for score-based generative
models and stochastic processes like Brownian motion.
"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx


def euler_maruyama_step(
    state: jax.Array,
    t: float,
    dt: float,
    key: jax.Array,
    drift_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_fn: Callable[[jax.Array, float], jax.Array],
) -> jax.Array:
    """Perform a single Euler-Maruyama integration step for an SDE.

    Args:
        state: Current state.
        t: Current time.
        dt: Time step.
        key: JAX random key.
        drift_fn: Function defining the drift term.
        diffusion_fn: Function defining the diffusion term.

    Returns:
        New state after the Euler-Maruyama step.
    """
    # Compute the drift and diffusion at the current state and time
    drift = drift_fn(state, t)
    diffusion = diffusion_fn(state, t)

    # Generate Gaussian noise for the diffusion term
    noise = jax.random.normal(key, state.shape)

    # Euler-Maruyama step:
    # dx = f(x,t)dt + g(x,t)dW
    # where dW is Brownian motion increment ~ N(0, dt)
    return state + drift * dt + diffusion * noise * jnp.sqrt(dt)


def milstein_step(
    state: jax.Array,
    t: float,
    dt: float,
    key: jax.Array,
    drift_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_grad_fn: Callable[[jax.Array, float], jax.Array] | None = None,
) -> jax.Array:
    """Perform a single Milstein integration step for an SDE.

    Milstein method provides higher order accuracy than Euler-Maruyama.

    Args:
        state: Current state.
        t: Current time.
        dt: Time step.
        key: JAX random key.
        drift_fn: Function defining the drift term.
        diffusion_fn: Function defining the diffusion term.
        diffusion_grad_fn: Function for diffusion gradient with respect to state.
            If None, approximated using finite differences.

    Returns:
        New state after the Milstein step.
    """
    # Compute the drift and diffusion at the current state and time
    drift = drift_fn(state, t)
    diffusion = diffusion_fn(state, t)

    # Generate Gaussian noise for the diffusion term
    noise = jax.random.normal(key, state.shape)
    dw = noise * jnp.sqrt(dt)

    # Basic Euler-Maruyama step
    next_state = state + drift * dt + diffusion * dw

    # Milstein correction term
    if diffusion_grad_fn is not None:
        # Use provided gradient function
        diffusion_grad = diffusion_grad_fn(state, t)
        # Add Milstein correction term
        next_state = next_state + 0.5 * diffusion * diffusion_grad * (dw**2 - dt)
    else:
        # Approximate diffusion gradient using finite differences
        eps = 1e-6
        diffusion_plus = diffusion_fn(state + eps, t)
        diffusion_grad_approx = (diffusion_plus - diffusion) / eps
        # Add Milstein correction term
        next_state = next_state + 0.5 * diffusion * diffusion_grad_approx * (dw**2 - dt)

    return next_state


def _euler_maruyama_wrapper(
    state: jax.Array,
    t: float,
    dt: float,
    key: jax.Array,
    drift_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_fn: Callable[[jax.Array, float], jax.Array],
    *args,
) -> jax.Array:
    """Wrapper for euler_maruyama_step to match the same interface as milstein."""
    return euler_maruyama_step(state, t, dt, key, drift_fn, diffusion_fn)


def _milstein_wrapper(
    state: jax.Array,
    t: float,
    dt: float,
    key: jax.Array,
    drift_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_grad_fn: Callable[[jax.Array, float], jax.Array] | None = None,
) -> jax.Array:
    """Wrapper for milstein_step to have a uniform interface."""
    return milstein_step(state, t, dt, key, drift_fn, diffusion_fn, diffusion_grad_fn)


def sde_sampling(
    drift_fn: Callable[[jax.Array, float], jax.Array],
    diffusion_fn: Callable[[jax.Array, float], jax.Array],
    init_state: jax.Array,
    t_span: tuple[float, float],
    key: jax.Array | nnx.Rngs,
    n_steps: int = 100,
    method: Literal["euler_maruyama", "milstein"] = "euler_maruyama",
    return_trajectory: bool = False,
    diffusion_grad_fn: Callable[[jax.Array, float], jax.Array] | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample from an SDE using the specified numerical method.

    Args:
        drift_fn: Function defining the drift term.
        diffusion_fn: Function defining the diffusion term.
        init_state: Initial state.
        t_span: Time span as (t_start, t_end).
        key: JAX random key or nnx.Rngs object.
        n_steps: Number of integration steps.
        method: Integration method, either "euler_maruyama" or "milstein".
        return_trajectory: Whether to return the entire trajectory.
        diffusion_grad_fn: For Milstein method, function computing diffusion gradient.

    Returns:
        If return_trajectory is False, returns final state after integration.
        If return_trajectory is True, returns (trajectory, time_points) tuple.
    """
    # Handle nnx.Rngs if provided
    if isinstance(key, nnx.Rngs):
        if hasattr(key, "params"):
            key = key.params()
        elif hasattr(key, "default"):
            key = key.default()
        else:
            # Use the first available key
            for k in dir(key):
                if not k.startswith("_") and callable(getattr(key, k)):
                    key = getattr(key, k)()
                    break

    # Unpack the time span
    t_start, t_end = t_span

    # Time step
    dt = (t_end - t_start) / n_steps

    # Select the integration method
    if method == "euler_maruyama":
        step_fn = _euler_maruyama_wrapper
    elif method == "milstein":
        step_fn = _milstein_wrapper
    else:
        raise ValueError(f"Unknown method: {method}")

    # Initialize the state
    state = init_state
    t = t_start

    # For returning trajectory if needed
    if return_trajectory:
        time_points = jnp.linspace(t_start, t_end, n_steps + 1)
        trajectory = jnp.zeros((n_steps + 1, *init_state.shape))
        trajectory = trajectory.at[0].set(init_state)

    # Integrate
    for i in range(n_steps):
        # Get a new key for this step
        key, subkey = jax.random.split(key)

        # Take an integration step
        state = step_fn(state, t, dt, subkey, drift_fn, diffusion_fn, diffusion_grad_fn)

        # Update time
        t = t + dt

        # Store trajectory if needed
        if return_trajectory:
            trajectory = trajectory.at[i + 1].set(state)

    if return_trajectory:
        return trajectory, time_points
    return state
