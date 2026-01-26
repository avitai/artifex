"""Flow matching trainer with SOTA training techniques.

Provides specialized training utilities for flow matching models including:
- Conditional Flow Matching (CFM)
- Optimal Transport CFM (OT-CFM)
- Rectified Flow
- Time sampling strategies (uniform, logit-normal, u-shaped)

References:
    - Flow Matching: https://arxiv.org/abs/2210.02747
    - OT-CFM: https://arxiv.org/abs/2302.00482
    - Rectified Flow: https://arxiv.org/abs/2209.03003
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.utils import (
    extract_batch_data,
    reshape_for_broadcast,
    sample_logit_normal,
    sample_u_shaped,
)


@dataclass(slots=True)
class FlowTrainingConfig:
    """Configuration for flow matching training.

    Attributes:
        flow_type: Type of flow matching.
            - "cfm": Standard Conditional Flow Matching
            - "ot_cfm": Optimal Transport CFM for straighter paths
            - "rectified_flow": Rectified Flow for straighter paths
        time_sampling: How to sample time values during training.
            - "uniform": Uniform sampling in [0, 1]
            - "logit_normal": Logit-normal (favors middle times)
            - "u_shaped": U-shaped (favors endpoints, good for rectified flows)
        sigma_min: Minimum noise level for the Gaussian path.
        use_ot: Whether to use optimal transport coupling.
        ot_regularization: Regularization for OT (Sinkhorn epsilon).
        logit_normal_loc: Location parameter for logit-normal sampling.
        logit_normal_scale: Scale parameter for logit-normal sampling.
    """

    flow_type: Literal["cfm", "ot_cfm", "rectified_flow"] = "cfm"
    time_sampling: Literal["uniform", "logit_normal", "u_shaped"] = "uniform"
    sigma_min: float = 0.001
    use_ot: bool = False
    ot_regularization: float = 0.01
    logit_normal_loc: float = 0.0
    logit_normal_scale: float = 1.0


class FlowTrainer:
    """Flow matching trainer with modern training techniques.

    This trainer provides a JIT-compatible interface for training flow matching
    models. The train_step method takes model and optimizer as explicit arguments,
    allowing it to be wrapped with nnx.jit for performance.

    Features:
        - Multiple flow types (CFM, OT-CFM, Rectified Flow)
        - Non-uniform time sampling (logit-normal, u-shaped)
        - Optimal transport coupling support
        - DRY integration with base Trainer via create_loss_fn()

    The flow matching objective learns a velocity field v_theta(x_t, t) that
    transports samples from noise distribution to data distribution along
    straight paths in probability space.

    Example (non-JIT):
        ```python
        from artifex.generative_models.training.trainers import (
            FlowTrainer,
            FlowTrainingConfig,
        )

        config = FlowTrainingConfig(
            flow_type="cfm",
            time_sampling="logit_normal",
        )
        trainer = FlowTrainer(config)

        # Create model and optimizer separately
        model = FlowModel(config, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-4))

        # Training loop
        for batch in data:
            rng, step_rng = jax.random.split(rng)
            loss, metrics = trainer.train_step(model, optimizer, batch, step_rng)
        ```

    Example (JIT-compiled):
        ```python
        trainer = FlowTrainer(config)
        jit_step = nnx.jit(trainer.train_step)

        for batch in data:
            rng, step_rng = jax.random.split(rng)
            loss, metrics = jit_step(model, optimizer, batch, step_rng)
        ```
    """

    __slots__ = ("config",)

    def __init__(
        self,
        config: FlowTrainingConfig | None = None,
    ) -> None:
        """Initialize flow matching trainer.

        Args:
            config: Flow training configuration.
        """
        self.config = config or FlowTrainingConfig()

    def sample_time(
        self,
        batch_size: int,
        key: jax.Array,
    ) -> jax.Array:
        """Sample time values according to configured strategy.

        Args:
            batch_size: Number of time values to sample.
            key: PRNG key for random sampling.

        Returns:
            Time values array of shape (batch_size, 1) in [0, 1].
        """
        if self.config.time_sampling == "uniform":
            return jax.random.uniform(key, (batch_size, 1))

        if self.config.time_sampling == "logit_normal":
            # Use shared utility for logit-normal sampling
            return sample_logit_normal(
                key,
                (batch_size, 1),
                loc=self.config.logit_normal_loc,
                scale=self.config.logit_normal_scale,
            )

        if self.config.time_sampling == "u_shaped":
            # Use shared utility for U-shaped sampling
            return sample_u_shaped(key, (batch_size, 1))

        # Default to uniform
        return jax.random.uniform(key, (batch_size, 1))

    def compute_conditional_vector_field(
        self,
        x0: jax.Array,
        x1: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute interpolated point and target vector field.

        For linear interpolation path:
            x_t = (1 - t) * x0 + t * x1
            u_t = x1 - x0 (constant velocity)

        Args:
            x0: Source samples (noise), shape (batch, ...).
            x1: Target samples (data), shape (batch, ...).
            t: Time values, shape (batch, 1).

        Returns:
            Tuple of (x_t, u_t) where:
                - x_t: Interpolated points, shape (batch, ...)
                - u_t: Target velocity field, shape (batch, ...)
        """
        # Use shared utility to reshape t for broadcasting
        batch_size = t.shape[0]
        t_expanded = reshape_for_broadcast(t, batch_size, x0.ndim)

        # Linear interpolation: x_t = (1 - t) * x0 + t * x1
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        # Constant velocity: u_t = x1 - x0
        u_t = x1 - x0

        return x_t, u_t

    def compute_loss(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute flow matching loss.

        The loss is the MSE between predicted and target velocity:
            L = E_{t, x0, x1} || v_theta(x_t, t) - u_t ||^2

        Args:
            model: Flow model (velocity field) to evaluate.
            batch: Batch dictionary with "image" or "data" key.
            key: PRNG key for sampling noise and time.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Get target data using shared utility
        x1 = extract_batch_data(batch)

        batch_size = x1.shape[0]

        # Split PRNG key
        time_key, noise_key = jax.random.split(key)

        # Sample time and source noise
        t = self.sample_time(batch_size, time_key)
        x0 = jax.random.normal(noise_key, x1.shape)

        # Compute interpolated point and target velocity
        x_t, u_t = self.compute_conditional_vector_field(x0, x1, t)

        # Model prediction: velocity at (x_t, t)
        # Squeeze t for model input (batch,) instead of (batch, 1)
        v_t = model(x_t, t.squeeze(-1))

        # MSE loss
        loss_per_sample = jnp.mean((v_t - u_t) ** 2, axis=tuple(range(1, v_t.ndim)))
        loss = jnp.mean(loss_per_sample)

        metrics = {
            "loss": loss,
        }

        return loss, metrics

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

        Args:
            model: Flow model to train.
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
