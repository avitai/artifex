"""DDIM (Denoising Diffusion Implicit Models) implementation."""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import DDIMConfig
from artifex.generative_models.models.diffusion.ddpm import DDPMModel


class DDIMModel(DDPMModel):
    """DDIM (Denoising Diffusion Implicit Models) implementation.

    This model implements deterministic sampling from diffusion models
    as described in the DDIM paper by Song et al. DDIM enables faster
    sampling with fewer steps while maintaining high quality.

    Uses nested DDIMConfig with:
    - backbone: BackboneConfig (polymorphic) for the denoising network
    - noise_schedule: NoiseScheduleConfig for the diffusion schedule
    - eta: Stochasticity parameter (0=deterministic, 1=DDPM)
    - num_inference_steps: Number of sampling steps
    - skip_type: Timestep skip strategy
    """

    def __init__(
        self,
        config: DDIMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the DDIM model.

        Args:
            config: DDIMConfig with nested backbone and noise_schedule configs.
                    The backbone field accepts any BackboneConfig type and the
                    appropriate backbone is created based on backbone_type.
            rngs: Random number generators
        """
        # Initialize parent DDPM model
        super().__init__(config, rngs=rngs)

        # DDIM-specific parameters from config
        self.eta: float = float(config.eta)  # 0 = deterministic, 1 = stochastic (DDPM)
        self.ddim_steps: int = int(config.num_inference_steps)
        self.skip_type = config.skip_type  # "uniform" or "quadratic"

    def get_ddim_timesteps(self, ddim_steps: int) -> jax.Array:
        """Get timesteps for DDIM sampling.

        Args:
            ddim_steps: Number of DDIM sampling steps

        Returns:
            Array of timesteps for DDIM sampling
        """
        if self.skip_type == "uniform":
            # Uniform spacing
            timesteps = jnp.linspace(0, self.noise_steps - 1, ddim_steps, dtype=jnp.int32)
        elif self.skip_type == "quadratic":
            # Quadratic spacing (more steps near the end)
            seq = jnp.linspace(0, jnp.sqrt(self.noise_steps * 0.8), ddim_steps) ** 2
            timesteps = jnp.asarray(seq, dtype=jnp.int32)
        else:
            raise ValueError(f"Unknown skip_type: {self.skip_type}")

        return jnp.flip(timesteps)  # Reverse order for sampling

    def ddim_step(
        self,
        x_t: jax.Array,
        t: jax.Array,
        t_prev: jax.Array,
        predicted_noise: jax.Array,
        eta: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Perform a single DDIM sampling step.

        Args:
            x_t: Current sample at timestep t
            t: Current timestep
            t_prev: Previous timestep
            predicted_noise: Predicted noise from the model
            eta: DDIM interpolation parameter (0=deterministic, 1=DDPM)
            rngs: Optional RNG streams overriding the model-owned RNGs

        Returns:
            Sample at timestep t_prev
        """
        eta_value = self.eta if eta is None else eta
        active_rngs = self.rngs if rngs is None else rngs

        # Get alpha values
        alpha_t = self._extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = self._extract_into_tensor(self.alphas_cumprod, t_prev, x_t.shape)

        # Predict x_0
        pred_x0 = self.predict_start_from_noise(x_t, t, predicted_noise)
        pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        # DDIM variance
        sigma_t = (
            eta_value
            * jnp.sqrt((1 - alpha_t_prev) / (1 - alpha_t))
            * jnp.sqrt(1 - alpha_t / alpha_t_prev)
        )

        # Add noise if eta > 0 using the active RNG stream
        if eta_value > 0:
            noise = jax.random.normal(active_rngs.sample(), x_t.shape)
            random_noise = sigma_t * noise
        else:
            random_noise = 0.0

        # Compute x_{t-1}
        x_prev = (
            jnp.sqrt(alpha_t_prev) * pred_x0
            + jnp.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise
            + random_noise
        )

        return x_prev

    def ddim_sample(
        self,
        n_samples: int,
        steps: int | None = None,
        eta: float | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Generate samples using DDIM.

        Args:
            n_samples: Number of samples to generate
            steps: Number of DDIM steps (default: self.ddim_steps)
            eta: DDIM interpolation parameter
            rngs: Optional RNG streams overriding the model-owned RNGs

        Returns:
            Generated samples
        """
        steps_value = self.ddim_steps if steps is None else steps
        eta_value = self.eta if eta is None else eta
        active_rngs = self.rngs if rngs is None else rngs

        # Get sample shape
        input_shape = self._get_sample_shape()

        # Initialize with noise using the active RNG stream
        x = jax.random.normal(active_rngs.sample(), (n_samples, *input_shape))

        # Get DDIM timesteps
        timesteps = self.get_ddim_timesteps(steps_value)

        # DDIM sampling loop
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = jnp.full((n_samples,), t, dtype=jnp.int32)

            # Get model prediction
            model_output = self(x, t_batch)

            # Extract predicted noise
            if isinstance(model_output, dict):
                if "predicted_noise" in model_output:
                    predicted_noise = model_output["predicted_noise"]
                else:
                    predicted_noise = model_output.get("noise", next(iter(model_output.values())))
            else:
                predicted_noise = model_output

            # Determine previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = jnp.array(0, dtype=jnp.int32)

            t_prev_batch = jnp.full((n_samples,), t_prev, dtype=jnp.int32)

            # DDIM step
            x = self.ddim_step(
                x,
                t_batch,
                t_prev_batch,
                predicted_noise,
                eta=eta_value,
                rngs=active_rngs,
            )

        return jnp.clip(x, -1.0, 1.0)

    def generate(
        self,
        n_samples: int = 1,
        *,
        shape: tuple[int, ...] | None = None,
        clip_denoised: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Generate samples using the retained DDIM fast-sampling path.

        Args:
            n_samples: Number of samples to generate
            shape: Optional sample shape override; must match the configured shape
            clip_denoised: Unused compatibility parameter kept for the diffusion interface
            rngs: Optional RNG streams overriding the model-owned RNGs

        Returns:
            Generated samples
        """
        del clip_denoised

        configured_shape = self._get_sample_shape()
        if shape is not None and tuple(shape) != tuple(configured_shape):
            raise ValueError(
                "DDIMModel.generate(...) only supports the configured sample shape; "
                f"expected {configured_shape}, got {shape}"
            )

        return self.ddim_sample(
            n_samples,
            steps=self.ddim_steps,
            eta=self.eta,
            rngs=rngs,
        )

    def sample(
        self,
        n_samples: int,
        scheduler: str = "ddim",
        steps: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Sample from the model.

        Args:
            n_samples: Number of samples to generate
            scheduler: Sampling scheduler ("ddim" or "ddpm")
            steps: Number of sampling steps
            rngs: Optional RNG streams overriding the model-owned RNGs

        Returns:
            Generated samples
        """
        if scheduler == "ddim":
            return self.ddim_sample(n_samples, steps=steps, rngs=rngs)
        elif scheduler == "ddpm":
            # Use parent DDPM sampling
            return super().sample(n_samples, scheduler="ddpm", steps=steps, rngs=rngs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    def ddim_reverse(self, x0: jax.Array, ddim_steps: int) -> jax.Array:
        """DDIM reverse process (encoding) from x_0 to noise.

        This is useful for image editing applications where you want to
        encode a real image into the noise space and then decode it.

        Args:
            x0: Clean image to encode
            ddim_steps: Number of DDIM steps

        Returns:
            Encoded noise
        """
        # Get DDIM timesteps (reversed for forward process)
        timesteps = jnp.flip(self.get_ddim_timesteps(ddim_steps))

        x = x0
        batch_size = x.shape[0]

        # Forward DDIM process
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = jnp.full((batch_size,), t, dtype=jnp.int32)
            t_next_batch = jnp.full((batch_size,), t_next, dtype=jnp.int32)

            # Get model prediction at current state
            model_output = self(x, t_batch)

            if isinstance(model_output, dict):
                if "predicted_noise" in model_output:
                    predicted_noise = model_output["predicted_noise"]
                else:
                    predicted_noise = next(iter(model_output.values()))
            else:
                predicted_noise = model_output

            # Reverse DDIM step (forward diffusion)
            x = self._ddim_reverse_step(x, t_batch, t_next_batch, predicted_noise)

        return x

    def _ddim_reverse_step(
        self,
        x_t: jax.Array,
        t: jax.Array,
        t_next: jax.Array,
        predicted_noise: jax.Array,
    ) -> jax.Array:
        """Reverse DDIM step for encoding.

        Args:
            x_t: Current sample
            t: Current timestep
            t_next: Next timestep
            predicted_noise: Predicted noise

        Returns:
            Next sample in forward process
        """
        # Get alpha values
        self._extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_t_next = self._extract_into_tensor(self.alphas_cumprod, t_next, x_t.shape)

        # Predict x_0
        pred_x0 = self.predict_start_from_noise(x_t, t, predicted_noise)
        pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        # Forward diffusion step
        eps = predicted_noise
        x_next = jnp.sqrt(alpha_t_next) * pred_x0 + jnp.sqrt(1 - alpha_t_next) * eps

        return x_next
