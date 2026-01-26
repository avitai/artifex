"""ODE-based sampling utilities.

This module provides utilities for sampling using ordinary differential
equations (ODEs). ODE-based sampling is useful for score-based generative
models like diffusion models.
"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp


def euler_step(
    state: jax.Array,
    t: float,
    dt: float,
    vector_field_fn: Callable[[jax.Array, float], jax.Array],
) -> jax.Array:
    """Perform a single Euler integration step.

    Args:
        state: Current state.
        t: Current time.
        dt: Time step.
        vector_field_fn: Function defining the vector field.

    Returns:
        New state after the Euler step.
    """
    # Compute the vector field at the current state and time
    vector_field = vector_field_fn(state, t)

    # Euler step: x_{t+dt} = x_t + dt * f(x_t, t)
    return state + dt * vector_field


def rk4_step(
    state: jax.Array,
    t: float,
    dt: float,
    vector_field_fn: Callable[[jax.Array, float], jax.Array],
) -> jax.Array:
    """Perform a single 4th-order Runge-Kutta integration step.

    Args:
        state: Current state.
        t: Current time.
        dt: Time step.
        vector_field_fn: Function defining the vector field.

    Returns:
        New state after the RK4 step.
    """
    # Compute the four RK4 increments
    k1 = vector_field_fn(state, t)
    k2 = vector_field_fn(state + dt * k1 / 2, t + dt / 2)
    k3 = vector_field_fn(state + dt * k2 / 2, t + dt / 2)
    k4 = vector_field_fn(state + dt * k3, t + dt)

    # Combine the increments with appropriate weights
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def ode_sampling(
    vector_field_fn: Callable[[jax.Array, float], jax.Array],
    init_state: jax.Array,
    t_span: tuple[float, float],
    method: Literal["euler", "rk4"] = "rk4",
    n_steps: int = 100,
    return_trajectory: bool = False,
    use_adjoint: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample from a vector field using ODE integration.

    Args:
        vector_field_fn: Function defining the vector field.
        init_state: Initial state.
        t_span: Time span as (t_start, t_end).
        method: Integration method, either "euler" or "rk4".
        n_steps: Number of integration steps.
        return_trajectory: Whether to return the entire trajectory.
        use_adjoint: Whether to use the adjoint method for backpropagation.
            This is more memory-efficient but can be slower.

    Returns:
        If return_trajectory is False, returns final state after integration.
        If return_trajectory is True, returns (trajectory, time_points) tuple.
    """
    # Unpack the time span
    t_start, t_end = t_span

    # Time step
    dt = (t_end - t_start) / n_steps

    # Select the integration method
    if method == "euler":
        step_fn = euler_step
    elif method == "rk4":
        step_fn = rk4_step
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
        state = step_fn(state, t, dt, vector_field_fn)
        t = t + dt

        # Store trajectory if needed
        if return_trajectory:
            trajectory = trajectory.at[i + 1].set(state)

    if return_trajectory:
        return trajectory, time_points
    return state
