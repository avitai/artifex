"""Noise schedule implementations for diffusion models.

This module provides different noise schedule strategies that can be used
with diffusion models. Each schedule is implemented as an nnx.Module to allow
for potential trainable parameters (e.g., learned schedules).

Classes:
    NoiseSchedule: Abstract base class for noise schedules (nnx.Module)
    LinearNoiseSchedule: Linear beta schedule
    CosineNoiseSchedule: Cosine beta schedule (improved diffusion)
    QuadraticNoiseSchedule: Quadratic beta schedule
    SqrtNoiseSchedule: Square root beta schedule

Functions:
    create_noise_schedule: Factory function to create schedule from config
"""

import abc

import jax.numpy as jnp
import numpy as np
from flax import nnx

from artifex.generative_models.core.configuration import NoiseScheduleConfig


def derive_schedule_quantities(
    betas: jnp.ndarray,
    clip_min: float,
) -> dict[str, jnp.ndarray]:
    """Derive every schedule quantity from betas in float64, then narrow to float32.

    `alphas_cumprod` approaches one at low timesteps, so forming `1 - alphas_cumprod`
    in float32 cancels away most of the significand: with beta_start=1e-4 the first
    entry keeps barely four correct digits, and that error propagates into the
    posterior variance and both posterior mean coefficients. The damage lands in the
    low-noise regime, which is where reverse sampling takes its final,
    quality-determining steps.

    The reference implementations resolve this the same way, computing the schedule
    in float64 before use: openai/improved-diffusion annotates the step "Use float64
    for accuracy", and FlaxDiff builds its schedules with `np.float64`. Doing the
    derivation here in float64 and storing float32 lowers the relative error of
    `1 - alphas_cumprod` at the first timestep from 1.7e-4 to 2.5e-8.

    NumPy is used rather than `jax.numpy` because float64 in JAX requires enabling
    x64 globally. This function is deliberately module level so that no NumPy runs
    inside an `nnx.Module`, and it evaluates once at construction, never under trace.

    Args:
        betas: Beta values for each timestep.
        clip_min: Lower clip applied before taking logs of (1 - alphas_cumprod).

    Returns:
        Mapping of schedule attribute name to its float32 array.
    """
    betas_64 = np.asarray(betas, dtype=np.float64)

    alphas = 1.0 - betas_64
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # Carried as first-class quantities so no consumer re-derives them in float32.
    one_minus_alphas_cumprod = 1.0 - alphas_cumprod
    one_minus_alphas_cumprod_prev = 1.0 - alphas_cumprod_prev

    posterior_variance = betas_64 * one_minus_alphas_cumprod_prev / one_minus_alphas_cumprod

    # The posterior variance is zero at t=0, so its log reuses the t=1 entry. A
    # single-timestep schedule has no t=1; JAX used to clamp this index silently,
    # NumPy raises, so the clamp is now explicit.
    log_variance_head = posterior_variance[min(1, posterior_variance.shape[0] - 1)]

    quantities = {
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "one_minus_alphas_cumprod": one_minus_alphas_cumprod,
        "one_minus_alphas_cumprod_prev": one_minus_alphas_cumprod_prev,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(one_minus_alphas_cumprod),
        "log_one_minus_alphas_cumprod": np.log(np.clip(one_minus_alphas_cumprod, clip_min, None)),
        "sqrt_recip_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod - 1.0),
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": np.log(
            np.append(log_variance_head, posterior_variance[1:])
        ),
        "posterior_mean_coef1": (
            betas_64 * np.sqrt(alphas_cumprod_prev) / one_minus_alphas_cumprod
        ),
        "posterior_mean_coef2": (
            one_minus_alphas_cumprod_prev * np.sqrt(alphas) / one_minus_alphas_cumprod
        ),
    }

    return {name: jnp.asarray(value, dtype=jnp.float32) for name, value in quantities.items()}


def extract_timesteps_into_tensor(
    arr: jnp.ndarray,
    timesteps: jnp.ndarray,
    broadcast_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Extract schedule values with one explicit broadcast rule.

    Timesteps may either match the batch size exactly, contain one element that
    is broadcast across the batch, or define a shorter schedule that repeats
    evenly across the batch.
    """
    timesteps = jnp.asarray(timesteps, dtype=jnp.int32)
    if timesteps.ndim == 0:
        timesteps = timesteps.reshape(1)

    batch_size = broadcast_shape[0]
    if timesteps.shape[0] == 1:
        timesteps = jnp.broadcast_to(timesteps, (batch_size,))
    elif timesteps.shape[0] == batch_size:
        pass
    elif batch_size % timesteps.shape[0] == 0:
        timesteps = jnp.repeat(timesteps, batch_size // timesteps.shape[0])
    else:
        raise ValueError(
            "timesteps batch must match the data batch size, contain one element, "
            "or divide the batch size evenly"
        )

    res = arr[timesteps]
    target_shape = (batch_size,) + (1,) * (len(broadcast_shape) - 1)
    return jnp.broadcast_to(res.reshape(target_shape), broadcast_shape)


class NoiseSchedule(nnx.Module, abc.ABC):
    """Abstract base class for noise schedules.

    Implemented as an nnx.Module to allow for potential trainable parameters
    in learned noise schedules. Standard schedules (linear, cosine, quadratic)
    have fixed parameters but use the same interface.

    Attributes:
        num_timesteps: Number of diffusion timesteps
        betas: Beta values for each timestep
        alphas: Alpha values (1 - beta) for each timestep
        alphas_cumprod: Cumulative product of alphas
        alphas_cumprod_prev: Shifted cumulative product of alphas
        sqrt_alphas_cumprod: Square root of cumulative alphas
        sqrt_one_minus_alphas_cumprod: Square root of (1 - cumulative alphas)
        posterior_variance: Variance for posterior q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1: First coefficient for posterior mean
        posterior_mean_coef2: Second coefficient for posterior mean
    """

    def __init__(self, config: NoiseScheduleConfig):
        """Initialize noise schedule from config.

        Args:
            config: NoiseScheduleConfig with schedule parameters
        """
        super().__init__()
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.clip_min = config.clip_min

        # Compute betas using schedule-specific implementation
        self.betas = self._compute_betas(
            config.beta_start,
            config.beta_end,
            config.num_timesteps,
        )

        # Compute derived values
        self._compute_derived_values()

    @abc.abstractmethod
    def _compute_betas(
        self,
        beta_start: float,
        beta_end: float,
        num_timesteps: int,
    ) -> jnp.ndarray:
        """Compute beta values for the schedule.

        Args:
            beta_start: Starting beta value
            beta_end: Ending beta value
            num_timesteps: Number of timesteps

        Returns:
            Array of beta values
        """
        pass

    def _compute_derived_values(self) -> None:
        """Compute derived values from betas.

        The derivation runs in float64 and is stored as float32; see
        `derive_schedule_quantities` for why the precision matters.
        """
        derived = derive_schedule_quantities(self.betas, self.clip_min)

        # Assigned explicitly rather than through setattr so the attributes stay
        # visible to the type checker and to nnx's attribute tracking.
        self.alphas = derived["alphas"]
        self.alphas_cumprod = derived["alphas_cumprod"]
        self.alphas_cumprod_prev = derived["alphas_cumprod_prev"]
        self.one_minus_alphas_cumprod = derived["one_minus_alphas_cumprod"]
        self.one_minus_alphas_cumprod_prev = derived["one_minus_alphas_cumprod_prev"]

        # Values used for diffusion process
        self.sqrt_alphas_cumprod = derived["sqrt_alphas_cumprod"]
        self.sqrt_one_minus_alphas_cumprod = derived["sqrt_one_minus_alphas_cumprod"]
        self.log_one_minus_alphas_cumprod = derived["log_one_minus_alphas_cumprod"]
        self.sqrt_recip_alphas_cumprod = derived["sqrt_recip_alphas_cumprod"]
        self.sqrt_recipm1_alphas_cumprod = derived["sqrt_recipm1_alphas_cumprod"]

        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = derived["posterior_variance"]
        self.posterior_log_variance_clipped = derived["posterior_log_variance_clipped"]
        self.posterior_mean_coef1 = derived["posterior_mean_coef1"]
        self.posterior_mean_coef2 = derived["posterior_mean_coef2"]

    def _extract_into_tensor(
        self,
        arr: jnp.ndarray,
        timesteps: jnp.ndarray,
        broadcast_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Extract schedule values under the shared timestep contract."""
        return extract_timesteps_into_tensor(arr, timesteps, broadcast_shape)

    def q_sample(
        self,
        x_start: jnp.ndarray,
        t: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample from the forward diffusion process q(x_t | x_0).

        Implements the forward diffusion equation:
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            x_start: Starting clean data (x_0)
            t: Timesteps (indices into the schedule)
            noise: Noise to add (same shape as x_start)

        Returns:
            Noisy samples x_t
        """
        sqrt_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def add_noise(
        self,
        x: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Add noise to data at given timesteps.

        This is an alias for q_sample that conforms to the NoiseScheduleProtocol.
        The parameter order matches the protocol: (x, noise, t).

        Args:
            x: Clean data (x_0), shape (batch, ...).
            noise: Noise samples, same shape as x.
            t: Integer timesteps, shape (batch,).

        Returns:
            Noisy data at timestep t.
        """
        return self.q_sample(x, t, noise)

    def predict_start_from_noise(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict x_0 from noise model output.

        Args:
            x_t: Noisy input at timestep t
            t: Timesteps
            noise: Predicted noise

        Returns:
            Predicted x_0
        """
        sqrt_recip_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(
        self,
        x_start: jnp.ndarray,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Clean data (x_0)
            x_t: Noisy data (x_t)
            t: Timesteps

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean_coef1_t = self._extract_into_tensor(
            self.posterior_mean_coef1, t, x_start.shape
        )
        posterior_mean_coef2_t = self._extract_into_tensor(
            self.posterior_mean_coef2, t, x_start.shape
        )

        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t

        posterior_variance_t = self._extract_into_tensor(self.posterior_variance, t, x_start.shape)
        posterior_log_variance_clipped_t = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_start.shape
        )

        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t


class LinearNoiseSchedule(NoiseSchedule):
    """Linear beta noise schedule.

    The simplest schedule where beta increases linearly from beta_start to beta_end.
    This is the original schedule used in the DDPM paper.
    """

    def _compute_betas(
        self,
        beta_start: float,
        beta_end: float,
        num_timesteps: int,
    ) -> jnp.ndarray:
        """Compute linear beta schedule.

        Args:
            beta_start: Starting beta value
            beta_end: Ending beta value
            num_timesteps: Number of timesteps

        Returns:
            Array of beta values with linear interpolation
        """
        return jnp.linspace(beta_start, beta_end, num_timesteps)


class CosineNoiseSchedule(NoiseSchedule):
    """Cosine beta noise schedule.

    Improved schedule from "Improved Denoising Diffusion Probabilistic Models"
    that provides better sample quality, especially for high-resolution images.
    """

    def _compute_betas(
        self,
        beta_start: float,  # Not used for cosine schedule
        beta_end: float,  # Not used for cosine schedule
        num_timesteps: int,
    ) -> jnp.ndarray:
        """Compute cosine beta schedule.

        The cosine schedule is defined in terms of alpha_bar, which follows a
        cosine curve. Betas are derived from the alpha_bar values.

        Args:
            beta_start: Not used (cosine schedule is parameter-free)
            beta_end: Not used (cosine schedule is parameter-free)
            num_timesteps: Number of timesteps

        Returns:
            Array of beta values derived from cosine alpha schedule
        """
        # Small offset to prevent division by zero
        s = 0.008

        steps = jnp.linspace(0, num_timesteps, num_timesteps + 1)
        alpha_bar = jnp.cos((steps / num_timesteps + s) / (1 + s) * jnp.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return jnp.clip(betas, 0.0001, 0.9999)


class QuadraticNoiseSchedule(NoiseSchedule):
    """Quadratic beta noise schedule.

    Beta values follow a quadratic curve, which can provide different noise
    characteristics compared to linear schedules.
    """

    def _compute_betas(
        self,
        beta_start: float,
        beta_end: float,
        num_timesteps: int,
    ) -> jnp.ndarray:
        """Compute quadratic beta schedule.

        Args:
            beta_start: Starting beta value
            beta_end: Ending beta value
            num_timesteps: Number of timesteps

        Returns:
            Array of beta values with quadratic interpolation
        """
        return jnp.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2


class SqrtNoiseSchedule(NoiseSchedule):
    """Square root beta noise schedule.

    Beta values follow a square root curve, providing a gentler increase
    in noise at the beginning compared to linear schedules.
    """

    def _compute_betas(
        self,
        beta_start: float,
        beta_end: float,
        num_timesteps: int,
    ) -> jnp.ndarray:
        """Compute sqrt beta schedule.

        Args:
            beta_start: Starting beta value
            beta_end: Ending beta value
            num_timesteps: Number of timesteps

        Returns:
            Array of beta values with sqrt interpolation
        """
        # Square root schedule: beta increases faster at the end
        t = jnp.linspace(0, 1, num_timesteps)
        return beta_start + (beta_end - beta_start) * jnp.sqrt(t)


def create_noise_schedule(config: NoiseScheduleConfig) -> NoiseSchedule:
    """Factory function to create a noise schedule from config.

    Args:
        config: NoiseScheduleConfig specifying the schedule type and parameters

    Returns:
        Initialized noise schedule instance

    Raises:
        ValueError: If schedule_type is unknown

    Example:
        config = NoiseScheduleConfig(
            name="cosine_schedule",
            schedule_type="cosine",
            num_timesteps=1000,
        )
        schedule = create_noise_schedule(config)
    """
    schedule_classes: dict[str, type[NoiseSchedule]] = {
        "linear": LinearNoiseSchedule,
        "cosine": CosineNoiseSchedule,
        "quadratic": QuadraticNoiseSchedule,
        "sqrt": SqrtNoiseSchedule,
    }

    schedule_class = schedule_classes.get(config.schedule_type)
    if schedule_class is None:
        raise ValueError(
            f"Unknown schedule_type: '{config.schedule_type}'. "
            f"Valid options are: {list(schedule_classes.keys())}"
        )

    return schedule_class(config)
