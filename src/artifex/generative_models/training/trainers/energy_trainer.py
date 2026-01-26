"""Energy-based model trainer with Contrastive Divergence and MCMC sampling.

Provides specialized training utilities for Energy-Based Models (EBMs) including:
- Contrastive Divergence (CD-k) training
- Langevin dynamics MCMC sampling
- Persistent Contrastive Divergence (PCD) with replay buffer
- Spectral normalization and gradient penalty for stability

References:
    - CD: https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
    - Langevin Dynamics: https://arxiv.org/abs/1903.08689
    - IGEBM: https://arxiv.org/abs/1903.08689
    - Score Matching: https://arxiv.org/abs/1907.05600
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.utils import extract_batch_data


@dataclass(slots=True)
class EnergyTrainingConfig:
    """Configuration for energy-based model training.

    Attributes:
        training_method: Training method for the energy model.
            - "cd": Contrastive Divergence (initialize chains from data)
            - "pcd": Persistent Contrastive Divergence (persistent chains)
            - "score_matching": Denoising score matching (gradient-based)
        mcmc_sampler: MCMC sampler for negative samples.
            - "langevin": Langevin dynamics (gradient + noise)
            - "hmc": Hamiltonian Monte Carlo (for complex distributions)
            - "mala": Metropolis-Adjusted Langevin Algorithm
        mcmc_steps: Number of MCMC steps for sampling negatives.
        step_size: Step size for MCMC updates.
        noise_scale: Scale of noise injection in Langevin dynamics.
        gradient_clipping: Max gradient norm for MCMC updates.
        replay_buffer_size: Size of replay buffer for PCD (0 = no buffer).
        replay_buffer_init_prob: Probability of initializing from buffer vs noise.
        energy_regularization: L2 regularization on energy values.
        gradient_penalty_weight: Weight for gradient penalty regularization.
    """

    training_method: Literal["cd", "pcd", "score_matching"] = "cd"
    mcmc_sampler: Literal["langevin", "hmc", "mala"] = "langevin"
    mcmc_steps: int = 20
    step_size: float = 0.01
    noise_scale: float = 0.005
    gradient_clipping: float = 1.0
    replay_buffer_size: int = 10000
    replay_buffer_init_prob: float = 0.95
    energy_regularization: float = 0.0
    gradient_penalty_weight: float = 0.0


class ReplayBuffer:
    """Replay buffer for Persistent Contrastive Divergence.

    Stores samples from previous MCMC chains to enable persistent chains
    that continue from where they left off, improving mixing.
    """

    __slots__ = ("_buffer", "_max_size", "_position", "_size")

    def __init__(self, max_size: int) -> None:
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of samples to store.
        """
        self._buffer: list[jax.Array] = []
        self._max_size = max_size
        self._position = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return self._size

    def add(self, samples: jax.Array) -> None:
        """Add samples to the buffer.

        Args:
            samples: Batch of samples to add, shape (batch, ...).
        """
        batch_size = samples.shape[0]
        for i in range(batch_size):
            sample = samples[i]
            if self._size < self._max_size:
                self._buffer.append(sample)
                self._size += 1
            else:
                self._buffer[self._position] = sample
            self._position = (self._position + 1) % self._max_size

    def sample(self, batch_size: int, key: jax.Array) -> jax.Array:
        """Sample from the buffer.

        Args:
            batch_size: Number of samples to retrieve.
            key: PRNG key for random selection.

        Returns:
            Batch of samples from buffer.
        """
        if self._size == 0:
            msg = "Cannot sample from empty buffer"
            raise ValueError(msg)

        indices = jax.random.randint(key, (batch_size,), 0, self._size)
        # Stack buffer into a single array and use jnp.take for efficient indexing
        buffer_array = jnp.stack(self._buffer)
        return jnp.take(buffer_array, indices, axis=0)


class EnergyTrainer:
    """Energy-based model trainer with MCMC sampling.

    This trainer provides a JIT-compatible interface for training energy-based
    models with Contrastive Divergence and related methods. The train_step
    method takes model and optimizer as explicit arguments, allowing it to
    be wrapped with nnx.jit for performance.

    The loss function is:
        L = E_{x ~ data}[E(x)] - E_{x ~ model}[E(x)]

    where E(x) is the energy function (model output) and negative samples
    are generated via MCMC starting from either data (CD) or a persistent
    chain (PCD).

    Features:
        - Contrastive Divergence (CD-k) training
        - Persistent Contrastive Divergence (PCD) with replay buffer
        - Langevin dynamics with configurable step size
        - Gradient clipping for stable MCMC
        - Energy regularization for bounded outputs
        - Gradient penalty for Lipschitz continuity

    Example (non-JIT):
        ```python
        from artifex.generative_models.training.trainers import (
            EnergyTrainer,
            EnergyTrainingConfig,
        )

        config = EnergyTrainingConfig(
            training_method="pcd",
            mcmc_sampler="langevin",
            mcmc_steps=20,
            step_size=0.01,
        )
        trainer = EnergyTrainer(config)

        # Create model and optimizer separately
        model = EnergyModel(config, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-4))

        # Training loop
        for batch in data:
            rng, step_rng = jax.random.split(rng)
            loss, metrics = trainer.train_step(model, optimizer, batch, step_rng)
        ```

    Example (JIT-compiled):
        ```python
        trainer = EnergyTrainer(config)
        jit_step = nnx.jit(trainer.train_step)

        for batch in data:
            rng, step_rng = jax.random.split(rng)
            loss, metrics = jit_step(model, optimizer, batch, step_rng)
        ```

    Note:
        The model should output a scalar energy for each input sample.
        For images, this is typically done via a ConvNet followed by
        global pooling and a linear layer to a single output.
    """

    __slots__ = ("config", "_replay_buffer")

    def __init__(
        self,
        config: EnergyTrainingConfig | None = None,
    ) -> None:
        """Initialize energy model trainer.

        Args:
            config: Energy training configuration.
        """
        self.config = config or EnergyTrainingConfig()

        # Initialize replay buffer for PCD
        if self.config.training_method == "pcd" and self.config.replay_buffer_size > 0:
            self._replay_buffer: ReplayBuffer | None = ReplayBuffer(self.config.replay_buffer_size)
        else:
            self._replay_buffer = None

    def compute_energy(self, model: nnx.Module, x: jax.Array) -> jax.Array:
        """Compute energy for a batch of samples.

        Args:
            model: Energy model.
            x: Input samples, shape (batch, ...).

        Returns:
            Energy values, shape (batch,).
        """
        energies = model(x)
        # Ensure output is (batch,) by squeezing extra dimensions
        if energies.ndim > 1:
            energies = energies.squeeze(axis=tuple(range(1, energies.ndim)))
        return energies

    def langevin_step(
        self,
        model: nnx.Module,
        x: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Perform one step of Langevin dynamics.

        Langevin update:
            x' = x - (step_size/2) * grad_x(E(x)) + sqrt(step_size) * noise

        Args:
            model: Energy model.
            x: Current samples, shape (batch, ...).
            key: PRNG key for noise.

        Returns:
            Updated samples.
        """

        # Compute energy gradient w.r.t. input
        def energy_fn(samples: jax.Array) -> jax.Array:
            return jnp.sum(self.compute_energy(model, samples))

        grad = jax.grad(energy_fn)(x)

        # Clip gradients for stability
        grad_norm = jnp.sqrt(jnp.sum(grad**2, axis=tuple(range(1, grad.ndim)), keepdims=True))
        grad = (
            grad
            / jnp.maximum(grad_norm, self.config.gradient_clipping)
            * jnp.minimum(grad_norm, self.config.gradient_clipping)
        )

        # Langevin update: move in direction of lower energy + noise
        noise = jax.random.normal(key, x.shape)
        step_size = self.config.step_size
        x_new = x - (step_size / 2) * grad + jnp.sqrt(step_size) * self.config.noise_scale * noise

        return x_new

    def run_mcmc_chain(
        self,
        model: nnx.Module,
        x_init: jax.Array,
        key: jax.Array,
        num_steps: int | None = None,
    ) -> jax.Array:
        """Run MCMC chain to generate negative samples.

        Args:
            model: Energy model.
            x_init: Initial samples for the chain.
            key: PRNG key for MCMC noise.
            num_steps: Number of MCMC steps (uses config if None).

        Returns:
            Final samples from the MCMC chain.
        """
        num_steps = num_steps or self.config.mcmc_steps

        def mcmc_body(
            carry: tuple[jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            x, rng = carry
            rng, step_key = jax.random.split(rng)
            x = self.langevin_step(model, x, step_key)
            return (x, rng), None

        (x_final, _), _ = jax.lax.scan(mcmc_body, (x_init, key), None, length=num_steps)

        return x_final

    def sample_negatives(
        self,
        model: nnx.Module,
        x_data: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Sample negative samples for contrastive learning.

        For CD: Initialize from data + small noise.
        For PCD: Initialize from replay buffer or noise.

        Args:
            model: Energy model.
            x_data: Data samples (used for CD initialization).
            key: PRNG key for sampling.

        Returns:
            Negative samples generated via MCMC.
        """
        batch_size = x_data.shape[0]
        init_key, mcmc_key = jax.random.split(key)

        if self.config.training_method == "cd":
            # CD: Initialize from data with small noise
            noise = jax.random.normal(init_key, x_data.shape) * self.config.noise_scale
            x_init = x_data + noise

        elif self.config.training_method == "pcd" and self._replay_buffer is not None:
            # PCD: Initialize from replay buffer or noise
            if self._replay_buffer.size >= batch_size:
                buffer_key, noise_key = jax.random.split(init_key)
                use_buffer = jax.random.uniform(noise_key) < self.config.replay_buffer_init_prob
                if use_buffer:
                    x_init = self._replay_buffer.sample(batch_size, buffer_key)
                else:
                    x_init = jax.random.uniform(init_key, x_data.shape, minval=-1.0, maxval=1.0)
            else:
                # Buffer not full yet, initialize from noise
                x_init = jax.random.uniform(init_key, x_data.shape, minval=-1.0, maxval=1.0)
        else:
            # Default: Initialize from noise
            x_init = jax.random.uniform(init_key, x_data.shape, minval=-1.0, maxval=1.0)

        # Run MCMC chain
        x_neg = self.run_mcmc_chain(model, x_init, mcmc_key)

        # Update replay buffer for PCD
        if self.config.training_method == "pcd" and self._replay_buffer is not None:
            self._replay_buffer.add(x_neg)

        return x_neg

    def compute_gradient_penalty(
        self,
        model: nnx.Module,
        x: jax.Array,
    ) -> jax.Array:
        """Compute gradient penalty for Lipschitz regularization.

        Args:
            model: Energy model.
            x: Input samples.

        Returns:
            Gradient penalty loss.
        """

        def energy_sum(samples: jax.Array) -> jax.Array:
            return jnp.sum(self.compute_energy(model, samples))

        grad = jax.grad(energy_sum)(x)
        grad_norm = jnp.sqrt(jnp.sum(grad**2, axis=tuple(range(1, grad.ndim))) + 1e-8)
        return jnp.mean((grad_norm - 1.0) ** 2)

    def compute_score_matching_loss(
        self,
        model: nnx.Module,
        x: jax.Array,
        key: jax.Array,
        noise_std: float = 0.1,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute denoising score matching loss.

        Score matching learns the gradient of the log probability density,
        which is the negative energy gradient: s(x) = -grad_x(E(x)).

        Args:
            model: Energy model.
            x: Clean data samples.
            key: PRNG key for noise.
            noise_std: Standard deviation of noise to add.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Add noise to data
        noise = jax.random.normal(key, x.shape)
        x_noisy = x + noise_std * noise

        # Compute score (negative energy gradient)
        def energy_sum(samples: jax.Array) -> jax.Array:
            return jnp.sum(self.compute_energy(model, samples))

        score = -jax.grad(energy_sum)(x_noisy) / x.shape[0]

        # Target score: -noise / noise_std (for denoising score matching)
        target_score = -noise / noise_std

        # MSE loss between predicted and target score
        loss = jnp.mean((score - target_score) ** 2)

        metrics = {
            "loss": loss,
            "score_norm": jnp.mean(jnp.sqrt(jnp.sum(score**2, axis=tuple(range(1, score.ndim))))),
        }

        return loss, metrics

    def compute_loss(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute energy-based model training loss.

        For CD/PCD: L = E_data[E(x)] - E_neg[E(x)]
        The model learns to assign low energy to data and high energy to
        MCMC samples.

        Args:
            model: Energy model to evaluate.
            batch: Batch dictionary with "image" or "data" key.
            key: PRNG key for MCMC sampling.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        x_data = extract_batch_data(batch)

        if self.config.training_method == "score_matching":
            return self.compute_score_matching_loss(model, x_data, key)

        # Sample negatives via MCMC
        x_neg = self.sample_negatives(model, x_data, key)

        # Compute energies
        e_data = self.compute_energy(model, x_data)
        e_neg = self.compute_energy(model, x_neg)

        # Contrastive divergence loss: E[E(x_data)] - E[E(x_neg)]
        # We want low energy for data, high energy for negatives
        cd_loss = jnp.mean(e_data) - jnp.mean(e_neg)

        # Total loss
        total_loss = cd_loss

        metrics: dict[str, Any] = {
            "loss": cd_loss,
            "energy_data": jnp.mean(e_data),
            "energy_neg": jnp.mean(e_neg),
            "energy_gap": jnp.mean(e_neg - e_data),
        }

        # Energy regularization
        if self.config.energy_regularization > 0:
            e_reg = self.config.energy_regularization * (jnp.mean(e_data**2) + jnp.mean(e_neg**2))
            total_loss = total_loss + e_reg
            metrics["energy_reg"] = e_reg

        # Gradient penalty
        if self.config.gradient_penalty_weight > 0:
            gp = self.compute_gradient_penalty(model, x_data)
            total_loss = total_loss + self.config.gradient_penalty_weight * gp
            metrics["gradient_penalty"] = gp

        metrics["total_loss"] = total_loss

        return total_loss, metrics

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
            model: Energy model to train.
            optimizer: NNX optimizer for parameter updates.
            batch: Batch dictionary with "image" or "data" key.
            key: PRNG key for MCMC sampling.

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

    def generate_samples(
        self,
        model: nnx.Module,
        batch_size: int,
        key: jax.Array,
        num_steps: int | None = None,
        shape: tuple[int, ...] | None = None,
        x_init: jax.Array | None = None,
    ) -> jax.Array:
        """Generate samples from the energy model via MCMC.

        Args:
            model: Energy model to use for sampling.
            batch_size: Number of samples to generate.
            key: PRNG key for sampling.
            num_steps: Number of MCMC steps (defaults to 10x config steps).
            shape: Shape of each sample (required if x_init not provided).
            x_init: Initial samples (randomly initialized if None).

        Returns:
            Generated samples.
        """
        init_key, mcmc_key = jax.random.split(key)

        if x_init is None:
            if shape is None:
                msg = "Either x_init or shape must be provided"
                raise ValueError(msg)
            x_init = jax.random.uniform(init_key, (batch_size, *shape), minval=-1.0, maxval=1.0)

        num_steps = num_steps or self.config.mcmc_steps * 10

        return self.run_mcmc_chain(model, x_init, mcmc_key, num_steps)
