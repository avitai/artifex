"""Diffusion-based sampling algorithms."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from artifex.generative_models.core.configuration import NoiseScheduleConfig
from artifex.generative_models.core.noise_schedule import create_noise_schedule
from artifex.generative_models.core.sampling.base import SamplingAlgorithm


# Beta schedules this sampler accepts, mapped onto NoiseScheduleConfig types.
SUPPORTED_BETA_SCHEDULES = ("linear", "quadratic")


class DiffusionSampler(SamplingAlgorithm):
    """Diffusion-based stepper and wrapper around model-owned sampling."""

    def __init__(
        self,
        predict_noise_fn: Callable | None = None,
        model=None,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """Initialize diffusion sampling helpers."""
        if model is None and predict_noise_fn is None:
            raise ValueError("DiffusionSampler requires a model or predict_noise_fn")
        if model is not None:
            self.predict_noise_fn = lambda x, t, **kwargs: model(x, t, **kwargs)
        else:
            self.predict_noise_fn = predict_noise_fn

        self.model = model
        self.num_timesteps = num_timesteps

        if beta_schedule not in SUPPORTED_BETA_SCHEDULES:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # The schedule and every quantity derived from it come from the shared
        # NoiseSchedule. Re-deriving them here is what let this sampler's reverse
        # step drift away from the canonical posterior.
        self.noise_schedule = create_noise_schedule(
            NoiseScheduleConfig(
                name=f"diffusion_sampler_{beta_schedule}",
                schedule_type=beta_schedule,
                num_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
            )
        )

        self.betas = self.noise_schedule.betas
        self.alphas = self.noise_schedule.alphas
        self.alphas_cumprod = self.noise_schedule.alphas_cumprod
        self.alphas_cumprod_prev = self.noise_schedule.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = self.noise_schedule.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = self.noise_schedule.sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.noise_schedule.posterior_variance

    def init(self, x: jax.Array, key: jax.Array) -> dict:
        """Initialize sampler state."""
        return {
            "x": x,
            "key": key,
            "t": self.num_timesteps - 1,
        }

    def step(self, state: dict) -> tuple[dict, dict]:
        """Advance the sampler by one step."""
        x, key, t = state["x"], state["key"], state["t"]
        predict_noise_fn = self.predict_noise_fn
        if predict_noise_fn is None:
            raise RuntimeError("DiffusionSampler is missing a predict_noise_fn")
        predicted_noise = predict_noise_fn(x, t)

        timesteps = jnp.asarray(t, dtype=jnp.int32)
        x0_pred = self.noise_schedule.predict_start_from_noise(x, timesteps, predicted_noise)
        model_mean, posterior_variance_t, _ = self.noise_schedule.q_posterior_mean_variance(
            x0_pred, x, timesteps
        )

        noise_key, new_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, x.shape)
        next_x = model_mean + jnp.sqrt(posterior_variance_t) * noise * (t > 0)

        next_state = {
            "x": next_x,
            "key": new_key,
            "t": t - 1,
        }
        aux_info = {
            "x0_prediction": x0_pred,
            "mean": model_mean,
            "variance": posterior_variance_t,
        }
        return next_state, aux_info

    def sample(self, n_samples, scheduler="ddpm", steps=None, *, rngs=None):
        """Generate samples with the configured diffusion model."""
        if self.model is not None and hasattr(self.model, "sample"):
            kwargs = {"scheduler": scheduler}
            if steps is not None:
                kwargs["steps"] = steps
            if rngs is not None:
                kwargs["rngs"] = rngs
            return self.model.sample(n_samples, **kwargs)

        raise NotImplementedError(
            "DiffusionSampler.sample is wrapper-only; initialize it with a model "
            "that implements sample(...)."
        )
