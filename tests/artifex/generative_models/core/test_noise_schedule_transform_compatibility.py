"""JAX transform contracts for diffusion noise schedules."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import NoiseScheduleConfig
from artifex.generative_models.core.noise_schedule import (
    create_noise_schedule,
    extract_timesteps_into_tensor,
)


def test_extract_timesteps_into_tensor_is_jittable_and_differentiable() -> None:
    schedule_values = jnp.linspace(0.1, 0.9, 8, dtype=jnp.float32)
    timesteps = jnp.array([0, 3], dtype=jnp.int32)
    values = jnp.ones((4, 2, 3), dtype=jnp.float32)

    def scalar_loss(inputs):
        coefficients = extract_timesteps_into_tensor(schedule_values, timesteps, inputs.shape)
        return jnp.sum(coefficients * inputs)

    compiled_value = jax.jit(scalar_loss)(values)
    gradients = jax.grad(scalar_loss)(values)

    assert compiled_value.shape == ()
    assert jnp.isfinite(compiled_value)
    assert gradients.shape == values.shape
    assert jnp.all(jnp.isfinite(gradients))


@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic", "sqrt"])
def test_noise_schedule_methods_are_nnx_jittable_and_differentiable(
    schedule_type: str,
) -> None:
    config = NoiseScheduleConfig(
        name=f"{schedule_type}_schedule",
        schedule_type=schedule_type,
        num_timesteps=16,
        beta_start=1e-4,
        beta_end=0.02,
    )
    schedule = create_noise_schedule(config)
    values = jnp.linspace(-1.0, 1.0, 4 * 3 * 2, dtype=jnp.float32).reshape(4, 3, 2)
    noise = jnp.full_like(values, 0.15)
    timesteps = jnp.array([0, 4, 8, 15], dtype=jnp.int32)

    def scalar_schedule_path(module, inputs):
        noisy = module.q_sample(inputs, timesteps, noise)
        alias_noisy = module.add_noise(inputs, noise, timesteps)
        predicted_start = module.predict_start_from_noise(noisy, timesteps, noise)
        posterior_mean, posterior_variance, posterior_log_variance = (
            module.q_posterior_mean_variance(predicted_start, noisy, timesteps)
        )
        return (
            jnp.sum(noisy)
            + 0.5 * jnp.sum(alias_noisy)
            + 0.25 * jnp.sum(predicted_start)
            + 0.1 * jnp.sum(posterior_mean)
            + 0.01 * jnp.sum(posterior_variance)
            + 0.01 * jnp.sum(posterior_log_variance)
        )

    compiled_value = nnx.jit(scalar_schedule_path)(schedule, values)
    gradients = nnx.grad(scalar_schedule_path, argnums=1)(schedule, values)

    assert compiled_value.shape == ()
    assert jnp.isfinite(compiled_value)
    assert gradients.shape == values.shape
    assert jnp.all(jnp.isfinite(gradients))
