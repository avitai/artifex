# Sampling Methods

A comprehensive guide to sampling techniques for generating high-quality outputs from VAE, GAN, Diffusion, and Flow models.

## Overview

Sampling is the process of generating new data from trained generative models. Different model architectures require different sampling strategies, each with unique trade-offs between quality, diversity, and computational cost.

!!! tip "Key Concepts"
    - **Deterministic Sampling**: Same latent code produces same output
    - **Stochastic Sampling**: Introduces randomness for diversity
    - **Temperature**: Controls sampling randomness
    - **Truncation**: Trades diversity for quality

<div class="grid cards" markdown>

- :material-brain:{ .lg .middle } **VAE Sampling**

    ---

    Sample from learned latent distributions with interpolation and conditional generation

    [:octicons-arrow-right-24: VAE Methods](#vae-sampling-methods)

- :material-image-multiple:{ .lg .middle } **GAN Sampling**

    ---

    Generate from latent codes with truncation trick and style mixing

    [:octicons-arrow-right-24: GAN Methods](#gan-sampling-methods)

- :material-gradient-horizontal:{ .lg .middle } **Diffusion Sampling**

    ---

    Iterative denoising with DDPM, DDIM, and DPM-Solver for speed/quality tradeoffs

    [:octicons-arrow-right-24: Diffusion Methods](#diffusion-sampling-methods)

- :material-water-outline:{ .lg .middle } **Flow Sampling**

    ---

    Invertible transformations with temperature scaling and rejection sampling

    [:octicons-arrow-right-24: Flow Methods](#flow-sampling-methods)

</div>

---

## Prerequisites

Before using sampling methods, ensure you have:

```python
from flax import nnx
import jax
import jax.numpy as jnp
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.models.gan import GAN
from artifex.generative_models.models.diffusion import DiffusionModel
from artifex.generative_models.models.flow import FlowModel
```

!!! note "Model Checkpoint"
    All sampling examples assume you have a trained model checkpoint loaded.
    See [Inference Overview](overview.md) for loading instructions.

---

## VAE Sampling Methods

### Latent Space Sampling

VAEs learn a probabilistic latent space, typically modeled as a Gaussian distribution.

```python
class VAESampler:
    """Sampling utilities for Variational Autoencoders."""

    def __init__(self, vae: VAE):
        self.vae = vae

    def sample_prior(
        self,
        num_samples: int,
        temperature: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample from the prior distribution.

        Args:
            num_samples: Number of samples to generate
            temperature: Controls randomness (default: 1.0)
            rngs: Random number generators

        Returns:
            Generated samples with shape (num_samples, *image_shape)
        """
        # Sample from standard normal
        z = jax.random.normal(
            rngs.sample(),
            (num_samples, self.vae.latent_dim)
        )

        # Apply temperature scaling
        z = z * temperature

        # Decode to image space
        samples = self.vae.decode(z)

        return samples

    def reconstruct(self, images: jax.Array) -> jax.Array:
        """Reconstruct images through the VAE.

        Args:
            images: Input images with shape (batch_size, *image_shape)

        Returns:
            Reconstructed images
        """
        # Encode to latent space
        z_mean, z_logvar = self.vae.encode(images)

        # Decode (using mean for reconstruction)
        reconstructed = self.vae.decode(z_mean)

        return reconstructed
```

### Conditional VAE Sampling

Generate samples conditioned on class labels or other attributes.

```python
class ConditionalVAESampler:
    """Sampling for conditional VAEs."""

    def __init__(self, vae: VAE):
        self.vae = vae

    def sample_conditional(
        self,
        labels: jax.Array,
        temperature: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample conditioned on labels.

        Args:
            labels: Class labels with shape (batch_size,)
            temperature: Sampling temperature
            rngs: Random number generators

        Returns:
            Conditional samples
        """
        batch_size = labels.shape[0]

        # Sample latent codes
        z = jax.random.normal(
            rngs.sample(),
            (batch_size, self.vae.latent_dim)
        ) * temperature

        # Decode with conditioning
        samples = self.vae.decode(z, labels=labels)

        return samples
```

### Latent Space Interpolation

Create smooth transitions between samples by interpolating in latent space.

```python
def interpolate_latent(
    sampler: VAESampler,
    start_image: jax.Array,
    end_image: jax.Array,
    num_steps: int = 10,
) -> jax.Array:
    """Interpolate between two images in latent space.

    Args:
        sampler: VAE sampler instance
        start_image: Starting image (1, *image_shape)
        end_image: Ending image (1, *image_shape)
        num_steps: Number of interpolation steps

    Returns:
        Interpolated images (num_steps, *image_shape)
    """
    # Encode both images
    z_start, _ = sampler.vae.encode(start_image)
    z_end, _ = sampler.vae.encode(end_image)

    # Linear interpolation in latent space
    alphas = jnp.linspace(0, 1, num_steps)[:, None]
    z_interp = z_start * (1 - alphas) + z_end * alphas

    # Decode interpolated latents
    interpolated = sampler.vae.decode(z_interp)

    return interpolated
```

!!! example "Spherical Interpolation (SLERP)"
    For better interpolation in high-dimensional spaces:

    ```python
    def slerp(v0, v1, t):
        """Spherical linear interpolation."""
        # Normalize vectors
        v0_norm = v0 / jnp.linalg.norm(v0)
        v1_norm = v1 / jnp.linalg.norm(v1)

        # Calculate angle
        omega = jnp.arccos(jnp.clip(jnp.dot(v0_norm, v1_norm), -1, 1))

        # Interpolate
        sin_omega = jnp.sin(omega)
        return (jnp.sin((1 - t) * omega) / sin_omega * v0 +
                jnp.sin(t * omega) / sin_omega * v1)

    # Use in interpolation
    z_interp = jax.vmap(lambda alpha: slerp(z_start[0], z_end[0], alpha))(alphas)
    ```

---

## GAN Sampling Methods

### Basic Latent Code Sampling

GANs generate samples by mapping random latent codes through the generator.

```python
class GANSampler:
    """Sampling utilities for Generative Adversarial Networks."""

    def __init__(self, gan: GAN):
        self.gan = gan

    def sample(
        self,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Generate samples from random latent codes.

        Args:
            num_samples: Number of samples to generate
            rngs: Random number generators

        Returns:
            Generated images
        """
        # Sample latent codes from standard normal
        z = jax.random.normal(
            rngs.sample(),
            (num_samples, self.gan.latent_dim)
        )

        # Generate images
        samples = self.gan.generator(z)

        return samples
```

### Truncation Trick

Improve sample quality at the cost of diversity by truncating the latent distribution.

```python
class TruncatedGANSampler:
    """GAN sampling with truncation trick."""

    def __init__(self, gan: GAN):
        self.gan = gan

    def sample_truncated(
        self,
        num_samples: int,
        truncation: float = 0.7,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample with truncation trick.

        Args:
            num_samples: Number of samples
            truncation: Truncation factor (0 < t <= 1)
                       Lower values = higher quality, less diversity
            rngs: Random number generators

        Returns:
            High-quality samples
        """
        # Sample latent codes
        z = jax.random.normal(
            rngs.sample(),
            (num_samples, self.gan.latent_dim)
        )

        # Truncate to reduce diversity
        z = z * truncation

        # Generate images
        samples = self.gan.generator(z)

        return samples

    def sample_adaptive_truncation(
        self,
        num_samples: int,
        threshold: float = 2.0,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample with adaptive truncation.

        Resamples latent dimensions that exceed threshold.

        Args:
            num_samples: Number of samples
            threshold: Threshold for resampling (typically 2-3)
            rngs: Random number generators

        Returns:
            Adaptively truncated samples
        """
        z = jax.random.normal(
            rngs.sample(),
            (num_samples, self.gan.latent_dim)
        )

        # Resample dimensions exceeding threshold
        mask = jnp.abs(z) > threshold

        while jnp.any(mask):
            z_new = jax.random.normal(rngs.sample(), z.shape)
            z = jnp.where(mask, z_new, z)
            mask = jnp.abs(z) > threshold

        samples = self.gan.generator(z)

        return samples
```

### Style Mixing (StyleGAN)

Mix styles from different latent codes for creative generation.

```python
class StyleGANSampler:
    """Advanced sampling for StyleGAN architectures."""

    def __init__(self, stylegan):
        self.stylegan = stylegan

    def mix_styles(
        self,
        num_samples: int,
        mix_layer: int,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Generate samples with style mixing.

        Args:
            num_samples: Number of samples
            mix_layer: Layer index to switch styles (0 to num_layers)
            rngs: Random number generators

        Returns:
            Style-mixed samples
        """
        # Sample two sets of latent codes
        z1 = jax.random.normal(
            rngs.sample(),
            (num_samples, self.stylegan.latent_dim)
        )
        z2 = jax.random.normal(
            rngs.sample(),
            (num_samples, self.stylegan.latent_dim)
        )

        # Map to W space
        w1 = self.stylegan.mapping_network(z1)
        w2 = self.stylegan.mapping_network(z2)

        # Generate with style mixing
        samples = self.stylegan.synthesis_network(
            w1, w2, mix_layer=mix_layer
        )

        return samples
```

---

## Diffusion Sampling Methods

### DDPM Sampling

Standard denoising diffusion probabilistic model sampling.

```python
class DDPMSampler:
    """DDPM sampling with full denoising process."""

    def __init__(self, diffusion: DiffusionModel):
        self.diffusion = diffusion
        self.num_timesteps = diffusion.num_timesteps

    def sample(
        self,
        num_samples: int,
        image_shape: tuple,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample using full DDPM process.

        Args:
            num_samples: Number of samples
            image_shape: Shape of output images (C, H, W)
            rngs: Random number generators

        Returns:
            Generated samples
        """
        # Start from pure noise
        x = jax.random.normal(
            rngs.sample(),
            (num_samples, *image_shape)
        )

        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            # Create timestep array
            timesteps = jnp.full((num_samples,), t)

            # Predict noise
            noise_pred = self.diffusion(x, timesteps)

            # Get schedule parameters
            alpha_t = self.diffusion.alpha_schedule[t]
            alpha_prev = self.diffusion.alpha_schedule[t - 1] if t > 0 else 1.0
            beta_t = 1 - alpha_t / alpha_prev

            # Compute mean
            x_mean = (x - beta_t * noise_pred / jnp.sqrt(1 - alpha_t))
            x_mean = x_mean / jnp.sqrt(1 - beta_t)

            # Add noise (except final step)
            if t > 0:
                noise = jax.random.normal(rngs.sample(), x.shape)
                sigma_t = jnp.sqrt(beta_t)
                x = x_mean + sigma_t * noise
            else:
                x = x_mean

        return x
```

### DDIM Sampling

Deterministic sampling for faster generation with fewer steps.

```python
class DDIMSampler:
    """DDIM sampling for fast inference."""

    def __init__(self, diffusion: DiffusionModel):
        self.diffusion = diffusion

    def sample(
        self,
        num_samples: int,
        image_shape: tuple,
        num_steps: int = 50,
        eta: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample using DDIM with configurable steps.

        Args:
            num_samples: Number of samples
            image_shape: Output shape
            num_steps: Number of denoising steps (< num_timesteps)
            eta: Stochasticity parameter (0 = deterministic)
            rngs: Random number generators

        Returns:
            Generated samples
        """
        # Create timestep schedule
        timesteps = jnp.linspace(
            self.diffusion.num_timesteps - 1,
            0,
            num_steps,
            dtype=jnp.int32
        )

        # Start from noise
        x = jax.random.normal(
            rngs.sample(),
            (num_samples, *image_shape)
        )

        # DDIM sampling loop
        for i, t in enumerate(timesteps):
            t_batch = jnp.full((num_samples,), t)

            # Predict noise
            noise_pred = self.diffusion(x, t_batch)

            # Get alphas
            alpha_t = self.diffusion.alpha_schedule[t]
            if i < len(timesteps) - 1:
                alpha_prev = self.diffusion.alpha_schedule[timesteps[i + 1]]
            else:
                alpha_prev = 1.0

            # Predict x0
            x0_pred = (x - jnp.sqrt(1 - alpha_t) * noise_pred) / jnp.sqrt(alpha_t)
            x0_pred = jnp.clip(x0_pred, -1, 1)

            # Direction pointing to x_t
            dir_xt = jnp.sqrt(1 - alpha_prev - eta**2) * noise_pred

            # Compute x_{t-1}
            x = jnp.sqrt(alpha_prev) * x0_pred + dir_xt

            # Add stochasticity
            if eta > 0 and i < len(timesteps) - 1:
                noise = jax.random.normal(rngs.sample(), x.shape)
                sigma_t = eta * jnp.sqrt((1 - alpha_prev) / (1 - alpha_t))
                sigma_t = sigma_t * jnp.sqrt(1 - alpha_t / alpha_prev)
                x = x + sigma_t * noise

        return x
```

### DPM-Solver

Advanced solver for high-quality samples with very few steps.

```python
class DPMSolver:
    """DPM-Solver for efficient diffusion sampling."""

    def __init__(self, diffusion: DiffusionModel):
        self.diffusion = diffusion

    def sample(
        self,
        num_samples: int,
        image_shape: tuple,
        num_steps: int = 20,
        order: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample using DPM-Solver.

        Args:
            num_samples: Number of samples
            image_shape: Output shape
            num_steps: Sampling steps (10-20 often sufficient)
            order: Solver order (1, 2, or 3)
            rngs: Random number generators

        Returns:
            High-quality samples
        """
        # Timestep schedule
        timesteps = jnp.linspace(
            self.diffusion.num_timesteps - 1,
            0,
            num_steps + 1,
            dtype=jnp.int32
        )

        # Start from noise
        x = jax.random.normal(
            rngs.sample(),
            (num_samples, *image_shape)
        )

        # DPM-Solver iterations
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # First-order update
            noise_pred = self.diffusion(x, jnp.full((num_samples,), t))
            x = self._dpm_solver_step(x, noise_pred, t, t_next, order)

        return x

    def _dpm_solver_step(
        self,
        x: jax.Array,
        noise: jax.Array,
        t: int,
        t_next: int,
        order: int,
    ) -> jax.Array:
        """Single DPM-Solver step."""
        lambda_t = self._log_snr(t)
        lambda_next = self._log_snr(t_next)

        h = lambda_next - lambda_t
        alpha_t = self.diffusion.alpha_schedule[t]
        alpha_next = self.diffusion.alpha_schedule[t_next]

        # First-order exponential integrator
        x_next = (
            jnp.sqrt(alpha_next / alpha_t) * x
            - jnp.sqrt(alpha_next) * jnp.expm1(h) * noise
        )

        return x_next

    def _log_snr(self, t: int) -> float:
        """Compute log signal-to-noise ratio."""
        alpha_t = self.diffusion.alpha_schedule[t]
        return jnp.log(alpha_t / (1 - alpha_t))
```

### Classifier-Free Guidance

Improve sample quality with guidance.

```python
def sample_with_guidance(
    diffusion: DiffusionModel,
    num_samples: int,
    labels: jax.Array,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    *,
    rngs: nnx.Rngs,
) -> jax.Array:
    """Sample with classifier-free guidance.

    Args:
        diffusion: Conditional diffusion model
        num_samples: Number of samples
        labels: Class labels for conditioning
        guidance_scale: Guidance strength (1.0 = no guidance, 7.5 = strong)
        num_steps: Sampling steps
        rngs: Random number generators

    Returns:
        Guided samples
    """
    # Create null labels for unconditional prediction
    null_labels = jnp.full_like(labels, -1)

    # Sampling loop (using DDIM)
    timesteps = jnp.linspace(diffusion.num_timesteps - 1, 0, num_steps, dtype=jnp.int32)
    x = jax.random.normal(rngs.sample(), (num_samples, *diffusion.image_shape))

    for t in timesteps:
        t_batch = jnp.full((num_samples,), t)

        # Conditional prediction
        noise_cond = diffusion(x, t_batch, labels)

        # Unconditional prediction
        noise_uncond = diffusion(x, t_batch, null_labels)

        # Apply guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Update (simplified DDIM step)
        alpha_t = diffusion.alpha_schedule[t]
        x = (x - jnp.sqrt(1 - alpha_t) * noise_pred) / jnp.sqrt(alpha_t)

    return x
```

---

## Flow Sampling Methods

### Inverse Transformation

Flows sample by inverting the learned transformation.

```python
class FlowSampler:
    """Sampling for normalizing flows."""

    def __init__(self, flow: FlowModel):
        self.flow = flow

    def sample(
        self,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample by inverting the flow.

        Args:
            num_samples: Number of samples
            rngs: Random number generators

        Returns:
            Generated samples
        """
        # Sample from base distribution (standard normal)
        z = jax.random.normal(
            rngs.sample(),
            (num_samples, self.flow.data_dim)
        )

        # Invert flow: z -> x
        x = self.flow.inverse(z)

        return x
```

### Temperature Scaling

Control sample diversity with temperature.

```python
class TemperatureFlowSampler:
    """Flow sampling with temperature control."""

    def __init__(self, flow: FlowModel):
        self.flow = flow

    def sample_with_temperature(
        self,
        num_samples: int,
        temperature: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample with temperature scaling.

        Args:
            num_samples: Number of samples
            temperature: Controls diversity (< 1 = less diverse, > 1 = more diverse)
            rngs: Random number generators

        Returns:
            Temperature-scaled samples
        """
        # Sample from scaled base distribution
        z = jax.random.normal(
            rngs.sample(),
            (num_samples, self.flow.data_dim)
        ) * temperature

        # Invert to data space
        x = self.flow.inverse(z)

        return x
```

### Rejection Sampling

Improve quality by rejecting low-probability samples.

```python
class RejectionFlowSampler:
    """Flow sampling with rejection for quality control."""

    def __init__(self, flow: FlowModel):
        self.flow = flow

    def sample_with_rejection(
        self,
        num_samples: int,
        threshold: float = -10.0,
        max_attempts: int = 1000,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample with rejection based on log-likelihood.

        Args:
            num_samples: Number of samples to generate
            threshold: Minimum log-likelihood threshold
            max_attempts: Maximum sampling attempts
            rngs: Random number generators

        Returns:
            High-quality samples (may be fewer than num_samples)
        """
        accepted_samples = []
        attempts = 0

        while len(accepted_samples) < num_samples and attempts < max_attempts:
            # Generate candidates
            z = jax.random.normal(rngs.sample(), (num_samples, self.flow.data_dim))
            x = self.flow.inverse(z)

            # Compute log-likelihood
            log_prob = self.flow.log_prob(x)

            # Accept samples above threshold
            mask = log_prob > threshold
            accepted_samples.append(x[mask])

            attempts += 1

        # Concatenate and return
        if accepted_samples:
            samples = jnp.concatenate(accepted_samples, axis=0)
            return samples[:num_samples]
        else:
            # Fallback: return best samples
            return x[jnp.argsort(log_prob)[-num_samples:]]
```

---

## Quality vs Speed Tradeoffs

### Sampling Steps Comparison

Different methods offer different speed/quality tradeoffs.

```python
def compare_sampling_methods(
    diffusion: DiffusionModel,
    image_shape: tuple,
    *,
    rngs: nnx.Rngs,
) -> dict:
    """Compare sampling methods on speed and quality.

    Returns:
        Dictionary with timing and sample results
    """
    import time

    results = {}

    # DDPM (1000 steps) - Highest quality, slowest
    start = time.time()
    ddpm_sampler = DDPMSampler(diffusion)
    samples_ddpm = ddpm_sampler.sample(1, image_shape, rngs=rngs)
    results['ddpm'] = {
        'time': time.time() - start,
        'steps': 1000,
        'samples': samples_ddpm,
    }

    # DDIM (50 steps) - Good quality, fast
    start = time.time()
    ddim_sampler = DDIMSampler(diffusion)
    samples_ddim = ddim_sampler.sample(1, image_shape, num_steps=50, rngs=rngs)
    results['ddim_50'] = {
        'time': time.time() - start,
        'steps': 50,
        'samples': samples_ddim,
    }

    # DDIM (20 steps) - Lower quality, faster
    start = time.time()
    samples_ddim_fast = ddim_sampler.sample(1, image_shape, num_steps=20, rngs=rngs)
    results['ddim_20'] = {
        'time': time.time() - start,
        'steps': 20,
        'samples': samples_ddim_fast,
    }

    # DPM-Solver (20 steps) - Best quality/speed tradeoff
    start = time.time()
    dpm_sampler = DPMSolver(diffusion)
    samples_dpm = dpm_sampler.sample(1, image_shape, num_steps=20, rngs=rngs)
    results['dpm_20'] = {
        'time': time.time() - start,
        'steps': 20,
        'samples': samples_dpm,
    }

    return results
```

### Guidance Scale Effects

Higher guidance improves quality but reduces diversity.

| Guidance Scale | Quality | Diversity | Use Case |
|----------------|---------|-----------|----------|
| 1.0 | Low | High | Exploration |
| 3.0 | Medium | Medium | Balanced generation |
| 7.5 | High | Low | Production quality |
| 15.0 | Very High | Very Low | Maximum quality |

---

## Best Practices

### DO

!!! success "Recommended Practices"
    ✅ **Use DDIM or DPM-Solver** for production (10-20x faster than DDPM)

    ✅ **Apply truncation** to GANs for higher quality samples

    ✅ **Cache compiled functions** with `@jax.jit` for repeated sampling

    ✅ **Batch samples** to maximize GPU utilization

    ✅ **Use temperature < 1.0** for conservative, high-quality samples

    ✅ **Validate samples** with quality metrics (FID, IS)

### DON'T

!!! danger "Avoid These Mistakes"
    ❌ **Don't use DDPM** for interactive applications (too slow)

    ❌ **Don't use guidance > 10** (over-saturated, artifacts)

    ❌ **Don't ignore batch size** (single samples waste GPU)

    ❌ **Don't skip warmup compilation** on first call

    ❌ **Don't use temperature > 2.0** (unstable, poor quality)

    ❌ **Don't mix sampling strategies** without understanding tradeoffs

---

## Common Patterns

### Production Sampling Pipeline

```python
class ProductionSampler:
    """Production-ready sampling with caching and batching."""

    def __init__(self, model, sampler_type: str = 'ddim'):
        self.model = model

        # Select sampler
        if sampler_type == 'ddim':
            self.sampler = DDIMSampler(model)
            self.sample_fn = self.sampler.sample
        elif sampler_type == 'dpm':
            self.sampler = DPMSolver(model)
            self.sample_fn = self.sampler.sample
        else:
            raise ValueError(f"Unknown sampler: {sampler_type}")

        # JIT compile for speed
        self.sample_fn_jit = jax.jit(self.sample_fn)

        # Warmup compilation
        self._warmup()

    def _warmup(self):
        """Warmup JIT compilation."""
        dummy_rngs = nnx.Rngs(0)
        _ = self.sample_fn_jit(1, self.model.image_shape, rngs=dummy_rngs)

    def generate_batch(
        self,
        batch_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Generate a batch of samples efficiently."""
        return self.sample_fn_jit(
            batch_size,
            self.model.image_shape,
            rngs=rngs
        )
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Blurry samples** | Too many DDIM steps or low temperature | Reduce steps to 20-50, increase temperature |
| **Mode collapse** | GAN training issue or too much truncation | Retrain GAN, reduce truncation to 0.7-0.8 |
| **Slow sampling** | Using DDPM with 1000 steps | Switch to DDIM (50 steps) or DPM (20 steps) |
| **OOM during sampling** | Batch size too large | Reduce batch size or use gradient checkpointing |
| **Artifacts with guidance** | Guidance scale too high | Reduce to 5.0-7.5 range |
| **Low diversity** | Truncation too aggressive | Increase truncation or use adaptive method |
| **Unstable flows** | Temperature too high | Reduce temperature to 0.8-1.2 range |

---

## Summary

Effective sampling is crucial for generating high-quality outputs:

- **VAEs**: Sample from learned distributions with temperature and interpolation
- **GANs**: Use truncation trick and style mixing for quality and creativity
- **Diffusion**: Trade off quality and speed with DDPM, DDIM, or DPM-Solver
- **Flows**: Control diversity with temperature and rejection sampling
- **Production**: Use JIT compilation and batching for efficiency

Choose your sampling method based on your quality requirements and computational budget.

---

## Next Steps

<div class="grid cards" markdown>

- :material-speedometer:{ .lg .middle } **Optimization**

    ---

    Learn advanced optimization techniques for faster inference

    [:octicons-arrow-right-24: Inference Optimization](optimization.md)

- :material-video-box:{ .lg .middle } **Inference Overview**

    ---

    Return to model loading and batch processing

    [:octicons-arrow-right-24: Inference Overview](overview.md)

- :material-brain:{ .lg .middle } **Model Training**

    ---

    Improve sampling quality by training better models

    [:octicons-arrow-right-24: Training Guide](../training/training-guide.md)

- :material-chart-line:{ .lg .middle } **Benchmarking**

    ---

    Evaluate sample quality with metrics like FID and IS

    [:octicons-arrow-right-24: Evaluation Framework](../../benchmarks/index.md)

</div>
