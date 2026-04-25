"""Diffusion-based sampling algorithms."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from artifex.generative_models.core.sampling.base import SamplingAlgorithm


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

        if beta_schedule == "linear":
            self.betas = jnp.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "quadratic":
            self.betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.alphas_cumprod_prev = jnp.append(jnp.array([1.0]), self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

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

        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        x0_pred = (x - predicted_noise * jnp.sqrt(1 - alpha_cumprod_t)) / jnp.sqrt(alpha_cumprod_t)
        model_mean = (
            jnp.sqrt(alpha_t)
            * (1 - alpha_cumprod_t / alpha_t)
            / jnp.sqrt(1 - alpha_cumprod_t)
            * x0_pred
            + jnp.sqrt(alpha_cumprod_t / alpha_t)
            * (1 - alpha_t)
            / jnp.sqrt(1 - alpha_cumprod_t)
            * x
        )
        posterior_variance_t = self.posterior_variance[t]

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
