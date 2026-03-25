"""ODE-based sampling utilities.

This module provides utilities for sampling using ordinary differential
equations (ODEs). ODE-based sampling is useful for score-based generative
models like diffusion models.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp


def euler_step(
    state: jax.Array,
    t: float,
    dt: float,
    vector_field_fn: Callable[[jax.Array, float], jax.Array],
) -> jax.Array:
    vector_field = vector_field_fn(state, t)
    return state + dt * vector_field


def rk4_step(
    state: jax.Array,
    t: float,
    dt: float,
    vector_field_fn: Callable[[jax.Array, float], jax.Array],
) -> jax.Array:
    k1 = vector_field_fn(state, t)
    k2 = vector_field_fn(state + dt * k1 / 2, t + dt / 2)
    k3 = vector_field_fn(state + dt * k2 / 2, t + dt / 2)
    k4 = vector_field_fn(state + dt * k3, t + dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def ode_sampling(
    vector_field_fn: Callable[[jax.Array, float], jax.Array],
    init_state: jax.Array,
    t_span: tuple[float, float],
    method: Literal["euler", "rk4"] = "rk4",
    n_steps: int = 100,
    return_trajectory: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample from a vector field using ODE integration."""
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps

    if method == "euler":
        step_fn = euler_step
    elif method == "rk4":
        step_fn = rk4_step
    else:
        raise ValueError(f"Unknown method: {method}")

    state = init_state
    t = t_start

    if return_trajectory:
        time_points = jnp.linspace(t_start, t_end, n_steps + 1)
        trajectory = jnp.zeros((n_steps + 1, *init_state.shape))
        trajectory = trajectory.at[0].set(init_state)

    for i in range(n_steps):
        state = step_fn(state, t, dt, vector_field_fn)
        t = t + dt
        if return_trajectory:
            trajectory = trajectory.at[i + 1].set(state)

    if return_trajectory:
        return trajectory, time_points
    return state
