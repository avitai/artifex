"""Diffusion-based sampling algorithms."""

from typing import Callable

import jax
import jax.numpy as jnp

from artifex.generative_models.core.sampling.base import SamplingAlgorithm


class DiffusionSampler(SamplingAlgorithm):
    """Diffusion-based sampling algorithm for diffusion models."""

    def __init__(
        self,
        predict_noise_fn: Callable | None = None,
        model=None,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """Initialize the diffusion sampler.

        Args:
            predict_noise_fn: Function that predicts noise from a noisy input.
            model: Diffusion model object to use for noise prediction.
            num_timesteps: Number of diffusion timesteps.
            beta_schedule: Schedule for the noise variance beta.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
        """
        if model is not None:
            self.predict_noise_fn = lambda x, t, **kwargs: model(x, t, **kwargs)
        else:
            self.predict_noise_fn = predict_noise_fn

        self.model = model
        self.num_timesteps = num_timesteps

        # Set up noise schedule
        if beta_schedule == "linear":
            self.betas = jnp.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "quadratic":
            self.betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Calculate alphas and related terms for the diffusion process
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
        """Initialize the sampler state.

        Args:
            x: Initial position (pure noise).
            key: Random key.

        Returns:
            Initial state.
        """
        return {
            "x": x,
            "key": key,
            "t": self.num_timesteps - 1,
        }

    def step(self, state: dict) -> tuple[dict, dict]:
        """Perform one sampling step.

        Args:
            state: Current state.

        Returns:
            New state and auxiliary information.
        """
        x, key, t = state["x"], state["key"], state["t"]

        # Get noise prediction
        predicted_noise = self.predict_noise_fn(x, t)

        # Calculate mean for sampling
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]

        # Calculate mean and variance for the posterior q(x_{t-1} | x_t, x_0)
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

        # No noise for final step
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
        """Sample from the diffusion model.

        Args:
            n_samples: Number of samples to generate.
            scheduler: Sampling scheduler to use.
            steps: Number of sampling steps.
            rngs: Random number generator keys.

        Returns:
            Generated samples.
        """
        # Use model.sample if available
        if self.model is not None and hasattr(self.model, "sample"):
            return self.model.sample(n_samples, scheduler=scheduler, steps=steps, rngs=rngs)

        # Set default steps if not provided
        if steps is None:
            steps = self.num_timesteps

        # Get sampling key
        if rngs is not None and "params" in rngs:
            # We would use the key here when implementing sampling
            pass

        # Create initial noise
        # TODO: Add proper shape handling

        # Run sampling loop
        # TODO: Implement sampling

        raise NotImplementedError("Direct sampling not yet implemented")
