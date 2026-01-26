"""Score-based diffusion models.

Implementation of score-based diffusion model for generating samples from the
data distribution by solving the reverse SDE.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ScoreDiffusionConfig
from artifex.generative_models.models.diffusion.base import DiffusionModel


class ScoreDiffusionModel(DiffusionModel):
    """Score-based diffusion model.

    This model is based on score matching principles where the model
    predicts the score (gradient of log-likelihood) instead of noise directly.

    Uses nested ScoreDiffusionConfig with:
    - backbone: BackboneConfig (polymorphic) for the denoising network
    - noise_schedule: NoiseScheduleConfig for the diffusion schedule
    - sigma_min: Minimum noise level
    - sigma_max: Maximum noise level
    - score_scaling: Score function scaling factor

    Backbone type is determined by config.backbone.backbone_type discriminator.
    """

    def __init__(
        self,
        config: ScoreDiffusionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the score diffusion model.

        Args:
            config: ScoreDiffusionConfig with nested backbone and noise_schedule configs.
                    The backbone field accepts any BackboneConfig type and the
                    appropriate backbone is created based on backbone_type.
            rngs: Random number generators for initialization.
        """
        # Initialize parent class - handles backbone and noise schedule
        super().__init__(config, rngs=rngs)

        # Extract score-specific parameters from config
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.score_scaling = config.score_scaling

        # Use input_shape from config
        self.input_dim = config.input_shape

    def _expand_sigma(self, sigma: jax.Array, x: jax.Array) -> jax.Array:
        """Expand sigma to broadcast with x.

        Args:
            sigma: Noise level with shape (batch,)
            x: Input tensor with shape (batch, ...)

        Returns:
            Sigma expanded to shape (batch, 1, 1, ...) to broadcast with x
        """
        # Expand sigma to match the number of dimensions in x
        # From (batch,) to (batch, 1, 1, ...) with len(x.shape)-1 trailing 1s
        for _ in range(len(x.shape) - 1):
            sigma = sigma[..., None]
        return sigma

    def score(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Compute the score function.

        Args:
            x: Input samples
            t: Time steps

        Returns:
            Score values
        """
        # Convert time to noise level
        sigma = self._get_sigma(t)

        # Expand sigma for proper broadcasting with input dimensions
        sigma_expanded = self._expand_sigma(sigma, x)

        # Scale input by sigma
        scaled_x = x / sigma_expanded

        # Get score from model
        score_pred = self.denoise(scaled_x, t)

        # Scale score appropriately
        return score_pred * self.score_scaling / sigma_expanded

    def _get_sigma(self, t: jax.Array) -> jax.Array:
        """Get noise level sigma for time t.

        Args:
            t: Time steps in [0, 1]

        Returns:
            Noise levels
        """
        # Log-linear interpolation between sigma_min and sigma_max
        log_sigma_min = jnp.log(self.sigma_min)
        log_sigma_max = jnp.log(self.sigma_max)

        log_sigma = log_sigma_min + t * (log_sigma_max - log_sigma_min)
        return jnp.exp(log_sigma)

    def loss(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Compute the score matching loss.

        Args:
            x: Input samples
            rngs: Random number generators

        Returns:
            Loss value
        """
        if rngs is None:
            rngs = self.rngs

        batch_size = x.shape[0]

        # Sample random time steps
        t = jax.random.uniform(rngs.time(), (batch_size,))

        # Get noise level and expand for broadcasting
        sigma = self._get_sigma(t)
        sigma_expanded = self._expand_sigma(sigma, x)

        # Add noise to data
        noise = jax.random.normal(rngs.noise(), x.shape)
        noisy_x = x + sigma_expanded * noise

        # Predict score
        score_pred = self.score(noisy_x, t)

        # Score matching loss
        target_score = -noise / sigma_expanded
        return jnp.mean((score_pred - target_score) ** 2)

    def sample(
        self,
        num_samples: int,
        *,
        rngs: nnx.Rngs | None = None,
        num_steps: int = 1000,
        return_trajectory: bool = False,
    ) -> jax.Array | list[jax.Array]:
        """Generate samples using the reverse SDE.

        Args:
            num_samples: Number of samples to generate
            rngs: Random number generators
            num_steps: Number of integration steps
            return_trajectory: If True, return full trajectory

        Returns:
            Generated samples or trajectory
        """
        if rngs is None:
            rngs = self.rngs

        # Initialize from noise
        x = jax.random.normal(rngs.sample(), (num_samples, *self.input_dim))
        x = x * self.sigma_max

        trajectory = []
        dt = 1.0 / num_steps

        # Reverse SDE integration
        for i in range(num_steps):
            t = jnp.full((num_samples,), 1.0 - i * dt)

            # Get score
            score_val = self.score(x, t)

            # Get noise level and its derivative
            sigma = self._get_sigma(t)
            sigma_next = self._get_sigma(t - dt)
            d_sigma = (sigma_next - sigma) / dt

            # Expand sigma and d_sigma for proper broadcasting
            sigma_expanded = self._expand_sigma(sigma, x)
            d_sigma_expanded = self._expand_sigma(d_sigma, x)

            # Update using Euler-Maruyama
            drift = -0.5 * d_sigma_expanded * score_val
            diffusion = jnp.sqrt(2 * sigma_expanded)

            noise = jax.random.normal(rngs.noise(), x.shape)
            x = x + drift * dt + diffusion * jnp.sqrt(dt) * noise

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return trajectory
        else:
            return x

    def denoise(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Predict denoised output.

        Args:
            x: Noisy input
            t: Time steps

        Returns:
            Denoised output
        """
        # Use the backbone network to predict score
        return self.backbone(x, t)
