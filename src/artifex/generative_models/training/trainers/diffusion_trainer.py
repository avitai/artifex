"""Diffusion-specific trainer with SOTA training techniques.

Provides specialized training utilities for diffusion models including:
- Multiple prediction types (epsilon, v-prediction, x-prediction)
- Timestep sampling strategies (uniform, logit-normal, mode)
- Loss weighting schemes (uniform, SNR, min-SNR, EDM)
- EMA model updates

References:
    - DDPM: https://arxiv.org/abs/2006.11239
    - v-prediction: https://arxiv.org/abs/2202.00512
    - min-SNR: https://arxiv.org/abs/2303.09556
    - EDM: https://arxiv.org/abs/2206.00364
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.protocols import NoiseScheduleProtocol
from artifex.generative_models.training.utils import (
    expand_dims_to_match,
    extract_batch_data,
    extract_model_prediction,
    sample_logit_normal,
)


@dataclass(slots=True)
class DiffusionTrainingConfig:
    """Configuration for diffusion model training.

    Attributes:
        prediction_type: What the model predicts.
            - "epsilon": Predicts the added noise
            - "v_prediction": Predicts v = sqrt(alpha)*noise - sqrt(1-alpha)*x0
            - "x_start": Predicts the original clean data
        timestep_sampling: How to sample timesteps during training.
            - "uniform": Uniform random sampling
            - "logit_normal": Logit-normal distribution (favors middle timesteps)
            - "mode": Mode-seeking (favors high-noise timesteps)
        loss_weighting: How to weight the loss across timesteps.
            - "uniform": Equal weighting
            - "snr": Weight by signal-to-noise ratio
            - "min_snr": Min-SNR-gamma weighting (3.4x faster convergence)
            - "edm": EDM-style weighting
        snr_gamma: Gamma parameter for min-SNR weighting (5.0 typical).
        logit_normal_loc: Location parameter for logit-normal sampling.
        logit_normal_scale: Scale parameter for logit-normal sampling.
        ema_decay: EMA decay rate for model weights.
        ema_update_every: Update EMA every N training steps.
    """

    prediction_type: Literal["epsilon", "v_prediction", "x_start"] = "epsilon"
    timestep_sampling: Literal["uniform", "logit_normal", "mode"] = "uniform"
    loss_weighting: Literal["uniform", "snr", "min_snr", "edm"] = "uniform"
    snr_gamma: float = 5.0
    logit_normal_loc: float = -0.5
    logit_normal_scale: float = 1.0
    ema_decay: float = 0.9999
    ema_update_every: int = 10


class DiffusionTrainer:
    """Diffusion model trainer with modern training techniques.

    This trainer provides a JIT-compatible interface for training diffusion models
    with state-of-the-art techniques. The train_step method takes model and optimizer
    as explicit arguments, allowing it to be wrapped with nnx.jit for performance.

    Features:
        - Multiple prediction types (epsilon, v, x0)
        - Non-uniform timestep sampling (logit-normal, mode-seeking)
        - Loss weighting (SNR, min-SNR, EDM)
        - EMA model updates (call update_ema separately, outside JIT)

    Example (non-JIT):
        ```python
        from artifex.generative_models.training.trainers import (
            DiffusionTrainer,
            DiffusionTrainingConfig,
        )

        # Create trainer with noise schedule and config
        config = DiffusionTrainingConfig(
            prediction_type="v_prediction",
            timestep_sampling="logit_normal",
            loss_weighting="min_snr",
        )
        trainer = DiffusionTrainer(noise_schedule, config)

        # Create model and optimizer separately
        model = DDPMModel(model_config, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adamw(1e-4))

        # Training loop
        for batch in data:
            rng, step_rng = jax.random.split(rng)
            loss, metrics = trainer.train_step(model, optimizer, batch, step_rng)
            trainer.update_ema(model)  # EMA updates outside train_step
        ```

    Example (JIT-compiled):
        ```python
        trainer = DiffusionTrainer(noise_schedule, config)
        jit_step = nnx.jit(trainer.train_step)

        for batch in data:
            rng, step_rng = jax.random.split(rng)
            loss, metrics = jit_step(model, optimizer, batch, step_rng)
            trainer.update_ema(model)  # Outside JIT
        ```
    """

    __slots__ = (
        "noise_schedule",
        "config",
        "_ema_params",
        "_step_count",
    )

    def __init__(
        self,
        noise_schedule: NoiseScheduleProtocol,
        config: DiffusionTrainingConfig | None = None,
    ) -> None:
        """Initialize diffusion trainer.

        Args:
            noise_schedule: Noise schedule with alphas_cumprod and add_noise.
            config: Diffusion training configuration.
        """
        self.noise_schedule = noise_schedule
        self.config = config or DiffusionTrainingConfig()
        self._ema_params: Any = None
        self._step_count = 0

    def sample_timesteps(
        self,
        batch_size: int,
        key: jax.Array,
    ) -> jax.Array:
        """Sample timesteps according to configured strategy.

        Args:
            batch_size: Number of timesteps to sample.
            key: PRNG key for random sampling.

        Returns:
            Integer timesteps array of shape (batch_size,).
        """
        num_timesteps = self.noise_schedule.num_timesteps

        if self.config.timestep_sampling == "uniform":
            return jax.random.randint(key, (batch_size,), 0, num_timesteps)

        if self.config.timestep_sampling == "logit_normal":
            # Use shared utility for logit-normal sampling
            t = sample_logit_normal(
                key,
                (batch_size,),
                loc=self.config.logit_normal_loc,
                scale=self.config.logit_normal_scale,
            )
            return jnp.clip((t * num_timesteps).astype(jnp.int32), 0, num_timesteps - 1)

        if self.config.timestep_sampling == "mode":
            # Mode-seeking: favor high-noise timesteps
            u = jax.random.uniform(key, (batch_size,))
            t = u**2  # Quadratic bias toward t=0 (in normalized space)
            # Map to favor high-noise (large timesteps)
            return jnp.clip(((1 - t) * num_timesteps).astype(jnp.int32), 0, num_timesteps - 1)

        return jax.random.randint(key, (batch_size,), 0, num_timesteps)

    def get_loss_weight(self, t: jax.Array) -> jax.Array:
        """Compute loss weight for given timesteps.

        Args:
            t: Integer timesteps array.

        Returns:
            Loss weights for each timestep.
        """
        if self.config.loss_weighting == "uniform":
            return jnp.ones_like(t, dtype=jnp.float32)

        # Get SNR at timestep: SNR = alpha / (1 - alpha)
        alpha = self.noise_schedule.alphas_cumprod[t]
        # Add small epsilon to avoid division by zero
        snr = alpha / jnp.maximum(1.0 - alpha, 1e-8)

        if self.config.loss_weighting == "snr":
            return snr

        if self.config.loss_weighting == "min_snr":
            # min-SNR-gamma weighting: weight = min(SNR, gamma) / SNR
            # This down-weights low-noise timesteps where SNR is high
            return jnp.minimum(snr, self.config.snr_gamma) / jnp.maximum(snr, 1e-8)

        if self.config.loss_weighting == "edm":
            # EDM-style weighting: weight = 1 / (sigma^2 + 1)
            # where sigma = sqrt((1 - alpha) / alpha)
            sigma = jnp.sqrt(jnp.maximum(1.0 - alpha, 1e-8) / jnp.maximum(alpha, 1e-8))
            return 1.0 / (sigma**2 + 1.0)

        return jnp.ones_like(t, dtype=jnp.float32)

    def compute_target(
        self,
        x0: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Compute prediction target based on prediction type.

        Args:
            x0: Original clean data.
            noise: Added noise.
            t: Timesteps.

        Returns:
            Target for the model prediction.
        """
        if self.config.prediction_type == "epsilon":
            return noise

        if self.config.prediction_type == "v_prediction":
            # v = sqrt(alpha) * noise - sqrt(1-alpha) * x0
            alpha = self.noise_schedule.alphas_cumprod[t]
            # Use shared utility to expand alpha for broadcasting
            alpha = expand_dims_to_match(alpha, x0.ndim)
            return jnp.sqrt(alpha) * noise - jnp.sqrt(1.0 - alpha) * x0

        if self.config.prediction_type == "x_start":
            return x0

        return noise

    def compute_loss(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute diffusion training loss.

        Args:
            model: Diffusion model to evaluate.
            batch: Batch dictionary with "image" or "data" key.
            key: PRNG key for sampling noise and timesteps.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Get input data using shared utility
        x = extract_batch_data(batch)

        batch_size = x.shape[0]

        # Split PRNG key
        t_key, noise_key = jax.random.split(key)

        # Sample timesteps and noise
        t = self.sample_timesteps(batch_size, t_key)
        noise = jax.random.normal(noise_key, x.shape)

        # Add noise to data
        x_noisy = self.noise_schedule.add_noise(x, noise, t)

        # Model prediction
        pred = model(x_noisy, t)
        pred = extract_model_prediction(pred)

        # Compute target based on prediction type
        target = self.compute_target(x, noise, t)

        # Compute per-sample MSE loss
        # Mean over all dimensions except batch
        loss_per_sample = jnp.mean((pred - target) ** 2, axis=tuple(range(1, pred.ndim)))

        # Apply loss weighting
        weights = self.get_loss_weight(t)
        weighted_loss = jnp.mean(weights * loss_per_sample)
        unweighted_loss = jnp.mean(loss_per_sample)

        metrics = {
            "loss": weighted_loss,
            "loss_unweighted": unweighted_loss,
        }

        return weighted_loss, metrics

    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, Any],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Execute a single training step.

        This method can be wrapped with nnx.jit for performance:
            jit_step = nnx.jit(trainer.train_step)
            loss, metrics = jit_step(model, optimizer, batch, key)

        Note: Call update_ema() separately after train_step for EMA updates.

        Args:
            model: Diffusion model to train.
            optimizer: NNX optimizer for parameter updates.
            batch: Batch dictionary with "image" or "data" key.
            key: PRNG key for sampling.

        Returns:
            Tuple of (loss, metrics_dict).
        """

        def loss_fn(m: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            return self.compute_loss(m, batch, key)

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)

        return loss, metrics

    def update_ema(self, model: nnx.Module) -> None:
        """Update EMA parameters.

        Call this method separately after train_step, outside of JIT.

        Args:
            model: The model whose parameters to use for EMA update.
        """
        self._step_count += 1
        if self._step_count % self.config.ema_update_every != 0:
            return

        if self._ema_params is None:
            self._ema_params = nnx.state(model)
            return

        model_state = nnx.state(model)
        decay = self.config.ema_decay

        def ema_update(ema: Any, new: Any) -> Any:
            """Apply EMA update only to numeric arrays, pass through others."""
            if hasattr(ema, "dtype") and jnp.issubdtype(ema.dtype, jnp.floating):
                return decay * ema + (1 - decay) * new
            return new

        self._ema_params = jax.tree.map(ema_update, self._ema_params, model_state)

    def get_ema_params(self, model: nnx.Module) -> Any:
        """Get EMA parameters for inference.

        Args:
            model: The model to get fallback state from if EMA not initialized.

        Returns:
            EMA parameters, or current model state if EMA not initialized.
        """
        if self._ema_params is None:
            return nnx.state(model)
        return self._ema_params

    def create_loss_fn(
        self,
    ) -> Callable[[nnx.Module, dict[str, Any], jax.Array], tuple[jax.Array, dict[str, Any]]]:
        """Create loss function compatible with base Trainer.

        This enables integration with the base Trainer for callbacks,
        checkpointing, logging, and other training infrastructure.

        Returns:
            Function with signature: (model, batch, rng) -> (loss, metrics)
        """

        def loss_fn(
            model: nnx.Module,
            batch: dict[str, Any],
            rng: jax.Array,
        ) -> tuple[jax.Array, dict[str, Any]]:
            return self.compute_loss(model, batch, rng)

        return loss_fn
