# Diffusion Models API Reference

Complete API documentation for all diffusion model classes in Artifex.

## Base Classes

### DiffusionModel

::: artifex.generative_models.models.diffusion.base.DiffusionModel

Base class for all diffusion models, implementing the core diffusion process.

**Purpose**: Provides the foundational diffusion framework including forward diffusion (adding noise), reverse diffusion (denoising), and noise scheduling.

#### Initialization

```python
DiffusionModel(
    config: ModelConfig,
    backbone_fn: Callable,
    *,
    rngs: nnx.Rngs
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Model configuration with input dimensions and parameters |
| `backbone_fn` | `Callable` | Function that creates the backbone network (e.g., U-Net) |
| `rngs` | `nnx.Rngs` | Random number generators for initialization |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_steps` | `int` | 1000 | Number of diffusion timesteps |
| `beta_start` | `float` | 1e-4 | Initial noise variance |
| `beta_end` | `float` | 0.02 | Final noise variance |

#### Methods

##### `__call__(x, timesteps, *, rngs=None, training=False, **kwargs)`

Forward pass through the diffusion model.

**Parameters:**

- `x` (`jax.Array`): Input data `(batch, *input_dim)`
- `timesteps` (`jax.Array`): Timestep indices `(batch,)`
- `rngs` (`nnx.Rngs | None`): Random number generators
- `training` (`bool`): Whether in training mode
- `**kwargs`: Additional arguments passed to backbone

**Returns:**

- `dict[str, Any]`: Dictionary containing `"predicted_noise"` and potentially other outputs

**Example:**

```python
# Create model
model = DiffusionModel(config, backbone_fn, rngs=rngs)

# Forward pass
x = jax.random.normal(rngs.sample(), (4, 32, 32, 3))
t = jnp.array([100, 200, 300, 400])
outputs = model(x, t, rngs=rngs, training=True)

print(outputs["predicted_noise"].shape)  # (4, 32, 32, 3)
```

##### `setup_noise_schedule()`

Set up the noise schedule for the diffusion process.

**Description:**

Computes the noise schedule (betas) and derived quantities (alphas, alpha_cumprod, etc.) used throughout the diffusion process. Default implementation uses a linear schedule.

**Computed Attributes:**

- `betas`: Noise variances at each timestep
- `alphas`: $\alpha_t = 1 - \beta_t$
- `alphas_cumprod`: $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$
- `sqrt_alphas_cumprod`: $\sqrt{\bar{\alpha}_t}$
- `sqrt_one_minus_alphas_cumprod`: $\sqrt{1 - \bar{\alpha}_t}$
- `posterior_variance`: Variance of $q(x_{t-1} | x_t, x_0)$

##### `q_sample(x_start, t, noise=None, *, rngs=None)`

Sample from the forward diffusion process $q(x_t | x_0)$.

**Parameters:**

- `x_start` (`jax.Array`): Clean data $x_0$
- `t` (`jax.Array`): Timesteps `(batch,)`
- `noise` (`jax.Array | None`): Optional pre-generated noise
- `rngs` (`nnx.Rngs | None`): Random number generators

**Returns:**

- `jax.Array`: Noisy samples $x_t$

**Mathematical Formula:**

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Example:**

```python
# Add noise to clean images
x_clean = jax.random.normal(rngs.sample(), (8, 32, 32, 3))
t = jnp.array([100] * 8)

x_noisy = model.q_sample(x_clean, t, rngs=rngs)
print(f"Clean: {x_clean.shape}, Noisy: {x_noisy.shape}")
```

##### `p_sample(model_output, x_t, t, *, rngs=None, clip_denoised=True)`

Sample from the denoising process $p(x_{t-1} | x_t)$.

**Parameters:**

- `model_output` (`jax.Array`): Predicted noise from model
- `x_t` (`jax.Array`): Noisy input at timestep $t$
- `t` (`jax.Array`): Current timesteps
- `rngs` (`nnx.Rngs | None`): Random number generators
- `clip_denoised` (`bool`): Whether to clip to [-1, 1]

**Returns:**

- `jax.Array`: Denoised sample $x_{t-1}$

**Example:**

```python
# Single denoising step
x_t = noisy_sample
t = jnp.array([500])

# Get model prediction
outputs = model(x_t, t, rngs=rngs)
predicted_noise = outputs["predicted_noise"]

# Denoise one step
x_t_minus_1 = model.p_sample(predicted_noise, x_t, t, rngs=rngs)
```

##### `generate(n_samples=1, *, rngs=None, shape=None, clip_denoised=True, **kwargs)`

Generate samples from random noise.

**Parameters:**

- `n_samples` (`int`): Number of samples to generate
- `rngs` (`nnx.Rngs | None`): Random number generators
- `shape` (`tuple[int, ...] | None`): Sample shape (excluding batch)
- `clip_denoised` (`bool`): Whether to clip to [-1, 1]
- `**kwargs`: Additional model arguments

**Returns:**

- `jax.Array`: Generated samples `(n_samples, *shape)`

**Example:**

```python
# Generate 16 samples
samples = model.generate(n_samples=16, rngs=rngs)
print(f"Generated: {samples.shape}")  # (16, 32, 32, 3)
```

##### `predict_start_from_noise(x_t, t, noise)`

Predict $x_0$ from $x_t$ and predicted noise.

**Parameters:**

- `x_t` (`jax.Array`): Noisy sample at timestep $t$
- `t` (`jax.Array`): Timesteps
- `noise` (`jax.Array`): Predicted noise

**Returns:**

- `jax.Array`: Predicted $x_0$

**Mathematical Formula:**

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} x_t - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \epsilon
$$

##### `loss_fn(batch, model_outputs, *, rngs=None, **kwargs)`

Compute the diffusion loss.

**Parameters:**

- `batch` (`Any`): Input batch (dict with `'x'` key or array)
- `model_outputs` (`dict[str, Any]`): Model predictions
- `rngs` (`nnx.Rngs | None`): Random number generators
- `**kwargs`: Additional arguments

**Returns:**

- `dict[str, Any]`: Dictionary with `'loss'` and metrics

**Example:**

```python
# Training loop
@nnx.jit
def train_step(model, optimizer, batch, rngs):
    def loss_fn_wrapper(model):
        # Add noise
        t = jax.random.randint(rngs.timestep(), (batch.shape[0],), 0, 1000)
        noise = jax.random.normal(rngs.noise(), batch.shape)
        x_noisy = model.q_sample(batch, t, noise, rngs=rngs)

        # Predict
        outputs = model(x_noisy, t, training=True, rngs=rngs)

        # Compute loss
        loss_dict = model.loss_fn({"x": batch}, outputs, rngs=rngs)
        return loss_dict["loss"]

    loss, grads = nnx.value_and_grad(loss_fn_wrapper)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API
    return {"loss": loss}
```

---

## DDPM (Denoising Diffusion Probabilistic Models)

### DDPMModel

::: artifex.generative_models.models.diffusion.ddpm.DDPMModel

Standard DDPM implementation with support for both DDPM and DDIM sampling.

**Purpose**: Implements the foundational denoising diffusion probabilistic model with standard training and sampling procedures.

#### Initialization

```python
DDPMModel(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Model configuration |
| `rngs` | `nnx.Rngs` | Random number generators |
| `**kwargs` | `dict` | Additional arguments (e.g., `backbone_fn`) |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_steps` | `int` | 1000 | Number of diffusion timesteps |
| `beta_start` | `float` | 1e-4 | Initial noise variance |
| `beta_end` | `float` | 0.02 | Final noise variance |
| `beta_schedule` | `str` | "linear" | Schedule type ("linear" or "cosine") |

#### Methods

##### `forward_diffusion(x, t, *, rngs=None)`

Forward diffusion process $q(x_t | x_0)$.

**Parameters:**

- `x` (`jax.Array`): Clean input data
- `t` (`jax.Array`): Timestep indices
- `rngs` (`nnx.Rngs | None`): Random number generators

**Returns:**

- `tuple[jax.Array, jax.Array]`: `(noisy_x, noise)` tuple

**Example:**

```python
model = DDPMModel(config, rngs=rngs)

x_clean = jnp.ones((4, 32, 32, 3))
t = jnp.array([100, 200, 300, 400])

x_noisy, noise = model.forward_diffusion(x_clean, t, rngs=rngs)
```

##### `denoise_step(x_t, t, predicted_noise, clip_denoised=True)`

Single denoising step.

**Parameters:**

- `x_t` (`jax.Array`): Noisy input at timestep $t$
- `t` (`jax.Array`): Current timesteps
- `predicted_noise` (`jax.Array`): Predicted noise
- `clip_denoised` (`bool`): Whether to clip to [-1, 1]

**Returns:**

- `jax.Array`: Denoised $x_{t-1}$

##### `sample(n_samples_or_shape, scheduler="ddpm", steps=None, *, rngs=None, **kwargs)`

Sample from the diffusion model.

**Parameters:**

- `n_samples_or_shape` (`int | tuple`): Number of samples or full shape
- `scheduler` (`str`): Sampling scheduler (`"ddpm"` or `"ddim"`)
- `steps` (`int | None`): Number of sampling steps
- `rngs` (`nnx.Rngs | None`): Random number generators
- `**kwargs`: Additional arguments

**Returns:**

- `jax.Array`: Generated samples

**Example:**

```python
# DDPM sampling (slow but high quality)
samples_ddpm = model.sample(16, scheduler="ddpm", rngs=rngs)

# DDIM sampling (fast)
samples_ddim = model.sample(16, scheduler="ddim", steps=50, rngs=rngs)
```

---

## DDIM (Denoising Diffusion Implicit Models)

### DDIMModel

::: artifex.generative_models.models.diffusion.ddim.DDIMModel

DDIM implementation with deterministic sampling and fast inference.

**Purpose**: Enables much faster sampling (10-20x) than DDPM while maintaining quality, and supports deterministic generation for image editing.

#### Initialization

```python
DDIMModel(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta` | `float` | 0.0 | Stochasticity (0=deterministic, 1=DDPM) |
| `ddim_steps` | `int` | 50 | Number of sampling steps |
| `skip_type` | `str` | "uniform" | Timestep selection ("uniform" or "quadratic") |
| `noise_steps` | `int` | 1000 | Training timesteps |

#### Methods

##### `get_ddim_timesteps(ddim_steps)`

Get timesteps for DDIM sampling.

**Parameters:**

- `ddim_steps` (`int`): Number of sampling steps

**Returns:**

- `jax.Array`: Timestep indices for DDIM

**Example:**

```python
model = DDIMModel(config, rngs=rngs)

# Get 50 uniformly spaced timesteps
timesteps = model.get_ddim_timesteps(50)
print(timesteps)  # [999, 979, 959, ..., 19, 0]
```

##### `ddim_step(x_t, t, t_prev, predicted_noise, eta=None, *, rngs=None)`

Single DDIM sampling step.

**Parameters:**

- `x_t` (`jax.Array`): Current sample at timestep $t$
- `t` (`jax.Array`): Current timestep
- `t_prev` (`jax.Array`): Previous timestep
- `predicted_noise` (`jax.Array`): Predicted noise
- `eta` (`float | None`): DDIM parameter (0=deterministic)
- `rngs` (`nnx.Rngs | None`): Random number generators

**Returns:**

- `jax.Array`: Sample at timestep $t_{prev}$

**Mathematical Formula:**

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t \epsilon
$$

Where $\hat{x}_0$ is the predicted clean sample and $\sigma_t = \eta \sqrt{(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)}\sqrt{1-\bar{\alpha}_t/\bar{\alpha}_{t-1}}$

##### `ddim_sample(n_samples, steps=None, eta=None, *, rngs=None, **kwargs)`

Generate samples using DDIM.

**Parameters:**

- `n_samples` (`int`): Number of samples
- `steps` (`int | None`): Number of DDIM steps
- `eta` (`float | None`): Stochasticity parameter
- `rngs` (`nnx.Rngs | None`): Random number generators
- `**kwargs`: Additional arguments

**Returns:**

- `jax.Array`: Generated samples

**Example:**

```python
# Deterministic generation (50 steps)
samples = model.ddim_sample(16, steps=50, eta=0.0, rngs=rngs)

# Stochastic generation (more diversity)
samples = model.ddim_sample(16, steps=50, eta=0.5, rngs=rngs)
```

##### `ddim_reverse(x0, ddim_steps, *, rngs=None, **kwargs)`

DDIM reverse process (encoding) from $x_0$ to noise.

**Purpose**: Encode a real image into the noise space for image editing.

**Parameters:**

- `x0` (`jax.Array`): Clean image to encode
- `ddim_steps` (`int`): Number of reverse steps
- `rngs` (`nnx.Rngs | None`): Random number generators
- `**kwargs`: Additional arguments

**Returns:**

- `jax.Array`: Encoded noise

**Example:**

```python
# Encode real image to noise
real_image = load_image("photo.png")
noise_code = model.ddim_reverse(real_image, ddim_steps=50, rngs=rngs)

# Edit in noise space
edited_noise = noise_code + modification

# Decode back to image
edited_image = model.ddim_sample(1, steps=50, rngs=rngs)
```

---

## Score-Based Models

### ScoreDiffusionModel

::: artifex.generative_models.models.diffusion.score.ScoreDiffusionModel

Score-based diffusion model using score matching.

**Purpose**: Implements score-based generative modeling where the model predicts the score function (gradient of log-likelihood) instead of noise directly.

#### Initialization

```python
ScoreDiffusionModel(
    *,
    config: ModelConfig,
    rngs: nnx.Rngs,
    **kwargs
)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma_min` | `float` | 0.01 | Minimum noise level |
| `sigma_max` | `float` | 1.0 | Maximum noise level |
| `score_scaling` | `float` | 1.0 | Score scaling factor |

#### Methods

##### `score(x, t)`

Compute the score function $\nabla_x \log p_t(x)$.

**Parameters:**

- `x` (`jax.Array`): Input samples
- `t` (`jax.Array`): Time steps in [0, 1]

**Returns:**

- `jax.Array`: Score values

**Mathematical Formula:**

$$
\nabla_x \log p_t(x) = -\frac{\epsilon}{\sigma_t}
$$

##### `sample(num_samples, *, rngs=None, num_steps=1000, return_trajectory=False)`

Generate samples using reverse SDE.

**Parameters:**

- `num_samples` (`int`): Number of samples
- `rngs` (`nnx.Rngs | None`): Random number generators
- `num_steps` (`int`): Number of integration steps
- `return_trajectory` (`bool`): Return full trajectory

**Returns:**

- `jax.Array | list[jax.Array]`: Samples or trajectory

**Example:**

```python
model = ScoreDiffusionModel(config=config, rngs=rngs)

# Generate samples
samples = model.sample(16, num_steps=1000, rngs=rngs)

# Get full trajectory
trajectory = model.sample(4, num_steps=1000, return_trajectory=True, rngs=rngs)
print(f"Trajectory length: {len(trajectory)}")  # 1000 steps
```

---

## Latent Diffusion Models

### LDMModel

::: artifex.generative_models.models.diffusion.latent.LDMModel

Latent Diffusion Model combining VAE and diffusion in latent space.

**Purpose**: Applies diffusion in a compressed latent space for efficient high-resolution generation. Foundation of Stable Diffusion.

#### Initialization

```python
LDMModel(
    *,
    config: ModelConfig,
    rngs: nnx.Rngs,
    **kwargs
)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_dim` | `int` | 8 | Latent space dimension |
| `encoder_hidden_dims` | `list[int]` | [32, 64] | Encoder layer sizes |
| `decoder_hidden_dims` | `list[int]` | [64, 32] | Decoder layer sizes |
| `encoder_type` | `str` | "simple" | Encoder type ("simple" or "vae") |
| `decoder_type` | `str` | "simple" | Decoder type |
| `scale_factor` | `float` | 0.18215 | Latent scaling factor |

#### Methods

##### `encode(x)`

Encode input to latent space.

**Parameters:**

- `x` (`jax.Array`): Input images

**Returns:**

- `tuple[jax.Array, jax.Array]`: `(mean, logvar)` of latent distribution

**Example:**

```python
model = LDMModel(config=config, rngs=rngs)

# Encode images to latent space
images = jax.random.normal(rngs.sample(), (8, 64, 64, 3))
mean, logvar = model.encode(images)

print(f"Latent mean: {mean.shape}")      # (8, 16)
print(f"Latent logvar: {logvar.shape}")  # (8, 16)
```

##### `decode(z)`

Decode latent code to image space.

**Parameters:**

- `z` (`jax.Array`): Latent codes

**Returns:**

- `jax.Array`: Decoded images

**Example:**

```python
# Sample latent code
z = jax.random.normal(rngs.sample(), (8, 16))

# Decode to images
images = model.decode(z)
print(f"Decoded images: {images.shape}")  # (8, 64, 64, 3)
```

##### `reparameterize(mean, logvar, *, rngs)`

Reparameterization trick for sampling.

**Parameters:**

- `mean` (`jax.Array`): Mean of latent distribution
- `logvar` (`jax.Array`): Log variance of latent distribution
- `rngs` (`nnx.Rngs`): Random number generators

**Returns:**

- `jax.Array`: Sampled latent code

**Mathematical Formula:**

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

##### `sample(num_samples, *, rngs=None, return_trajectory=False)`

Generate samples (automatically encoded/decoded).

**Parameters:**

- `num_samples` (`int`): Number of samples
- `rngs` (`nnx.Rngs | None`): Random number generators
- `return_trajectory` (`bool`): Return full trajectory

**Returns:**

- `jax.Array | list[jax.Array]`: Generated images

**Example:**

```python
# Generate high-resolution images efficiently
samples = model.sample(16, rngs=rngs)
print(f"Generated: {samples.shape}")  # (16, 64, 64, 3)

# Diffusion happens in compressed 16D latent space!
# 8x faster than pixel-space diffusion
```

---

## Diffusion Transformers

### DiTModel

::: artifex.generative_models.models.diffusion.dit.DiTModel

Diffusion model using Vision Transformer backbone.

**Purpose**: Replaces U-Net with Transformer for better scalability and state-of-the-art quality at large model sizes.

#### Initialization

```python
DiTModel(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs,
    backbone_fn: Optional[Callable] = None,
    **kwargs
)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_size` | `int` | 32 | Image size |
| `patch_size` | `int` | 2 | Patch size for Vision Transformer |
| `hidden_size` | `int` | 512 | Transformer hidden dimension |
| `depth` | `int` | 12 | Number of transformer layers |
| `num_heads` | `int` | 8 | Number of attention heads |
| `mlp_ratio` | `float` | 4.0 | MLP expansion ratio |
| `num_classes` | `int | None` | None | Number of classes for conditioning |
| `dropout_rate` | `float` | 0.0 | Dropout rate |
| `learn_sigma` | `bool` | False | Learn variance |
| `cfg_scale` | `float` | 1.0 | Classifier-free guidance scale |

#### Methods

##### `__call__(x, t, y=None, *, deterministic=False, cfg_scale=None)`

Forward pass through DiT model.

**Parameters:**

- `x` (`jax.Array`): Input images `(batch, H, W, C)`
- `t` (`jax.Array`): Timesteps `(batch,)`
- `y` (`jax.Array | None`): Optional class labels
- `deterministic` (`bool`): Whether to apply dropout
- `cfg_scale` (`float | None`): Classifier-free guidance scale

**Returns:**

- `jax.Array`: Predicted noise

**Example:**

```python
model = DiTModel(config, rngs=rngs)

# Forward pass
x = jax.random.normal(rngs.sample(), (4, 32, 32, 3))
t = jnp.array([100, 200, 300, 400])
y = jnp.array([0, 1, 2, 3])  # Class labels

noise_pred = model(x, t, y=y, deterministic=False)
```

##### `generate(n_samples=1, *, rngs, num_steps=1000, y=None, cfg_scale=None, img_size=None, **kwargs)`

Generate samples using DiT.

**Parameters:**

- `n_samples` (`int`): Number of samples
- `rngs` (`nnx.Rngs`): Random number generators
- `num_steps` (`int`): Number of diffusion steps
- `y` (`jax.Array | None`): Class labels for conditional generation
- `cfg_scale` (`float | None`): Classifier-free guidance scale
- `img_size` (`int | None`): Image size
- `**kwargs`: Additional arguments

**Returns:**

- `jax.Array`: Generated samples

**Example:**

```python
# Unconditional generation
samples = model.generate(n_samples=16, rngs=rngs, num_steps=1000)

# Conditional generation with classifier-free guidance
class_labels = jnp.array([i % 10 for i in range(16)])
samples = model.generate(
    n_samples=16,
    y=class_labels,
    cfg_scale=4.0,  # Strong conditioning
    rngs=rngs,
    num_steps=1000
)
```

---

## Guidance Techniques

### ClassifierFreeGuidance

::: artifex.generative_models.models.diffusion.guidance.ClassifierFreeGuidance

Classifier-free guidance for conditional generation.

**Purpose**: Enables strong conditioning without needing a separate classifier by training a single model to handle both conditional and unconditional generation.

#### Initialization

```python
ClassifierFreeGuidance(
    guidance_scale: float = 7.5,
    unconditional_conditioning: Any | None = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `guidance_scale` | `float` | 7.5 | Guidance strength (higher = stronger conditioning) |
| `unconditional_conditioning` | `Any | None` | None | Unconditional token/embedding |

#### Methods

##### `__call__(model, x, t, conditioning, *, rngs=None, **kwargs)`

Apply classifier-free guidance.

**Parameters:**

- `model` (`DiffusionModel`): Diffusion model
- `x` (`jax.Array`): Noisy input
- `t` (`jax.Array`): Timesteps
- `conditioning` (`Any`): Conditioning information
- `rngs` (`nnx.Rngs | None`): Random number generators
- `**kwargs`: Additional arguments

**Returns:**

- `jax.Array`: Guided noise prediction

**Mathematical Formula:**

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
$$

**Example:**

```python
from artifex.generative_models.models.diffusion.guidance import ClassifierFreeGuidance

# Create guidance
cfg = ClassifierFreeGuidance(guidance_scale=7.5)

# Use during sampling
x_t = noisy_sample
t = timesteps
conditioning = class_labels

guided_noise = cfg(model, x_t, t, conditioning, rngs=rngs)
```

### ClassifierGuidance

::: artifex.generative_models.models.diffusion.guidance.ClassifierGuidance

Classifier guidance using a pre-trained classifier.

**Purpose**: Uses gradients from a pre-trained classifier to guide generation towards desired classes.

#### Initialization

```python
ClassifierGuidance(
    classifier: nnx.Module,
    guidance_scale: float = 1.0,
    class_label: int | None = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `classifier` | `nnx.Module` | Pre-trained classifier model |
| `guidance_scale` | `float` | Guidance strength |
| `class_label` | `int | None` | Target class for guidance |

#### Methods

##### `__call__(model, x, t, *, rngs=None, class_label=None, **kwargs)`

Apply classifier guidance.

**Mathematical Formula:**

$$
\tilde{\epsilon} = \epsilon_\theta(x_t) - w \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y | x_t)
$$

**Example:**

```python
from artifex.generative_models.models.diffusion.guidance import ClassifierGuidance

# Load pre-trained classifier
classifier = load_classifier()

# Create classifier guidance
cg = ClassifierGuidance(
    classifier=classifier,
    guidance_scale=1.0,
    class_label=5  # Generate class 5
)

# Use during sampling
guided_noise = cg(model, x_t, t, rngs=rngs)
```

### GuidedDiffusionModel

::: artifex.generative_models.models.diffusion.guidance.GuidedDiffusionModel

Diffusion model with built-in guidance support.

**Purpose**: Extends base diffusion model to support various guidance techniques during generation.

#### Initialization

```python
GuidedDiffusionModel(
    config: ModelConfig,
    backbone_fn: Callable,
    *,
    rngs: nnx.Rngs,
    guidance_method: str | None = None,
    guidance_scale: float = 7.5,
    classifier: nnx.Module | None = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `guidance_method` | `str | None` | Guidance type ("classifier_free", "classifier", None) |
| `guidance_scale` | `float` | Guidance strength |
| `classifier` | `nnx.Module | None` | Classifier for classifier guidance |

**Example:**

```python
from artifex.generative_models.models.diffusion.guidance import GuidedDiffusionModel

# Create model with classifier-free guidance
model = GuidedDiffusionModel(
    config,
    backbone_fn,
    rngs=rngs,
    guidance_method="classifier_free",
    guidance_scale=7.5
)

# Generate with conditioning
samples = model.generate(
    n_samples=16,
    conditioning=class_labels,
    rngs=rngs
)
```

### Guidance Utility Functions

#### `apply_guidance(noise_pred_cond, noise_pred_uncond, guidance_scale)`

Apply classifier-free guidance formula.

**Parameters:**

- `noise_pred_cond` (`jax.Array`): Conditional noise prediction
- `noise_pred_uncond` (`jax.Array`): Unconditional noise prediction
- `guidance_scale` (`float`): Guidance strength

**Returns:**

- `jax.Array`: Guided noise prediction

**Example:**

```python
from artifex.generative_models.models.diffusion.guidance import apply_guidance

# Get predictions
noise_cond = model(x_t, t, conditioning=labels, rngs=rngs)
noise_uncond = model(x_t, t, conditioning=None, rngs=rngs)

# Apply guidance
guided = apply_guidance(noise_cond, noise_uncond, guidance_scale=7.5)
```

#### `linear_guidance_schedule(step, total_steps, start_scale=1.0, end_scale=7.5)`

Linear guidance scale schedule.

**Parameters:**

- `step` (`int`): Current step
- `total_steps` (`int`): Total number of steps
- `start_scale` (`float`): Starting guidance scale
- `end_scale` (`float`): Ending guidance scale

**Returns:**

- `float`: Guidance scale for current step

**Example:**

```python
from artifex.generative_models.models.diffusion.guidance import linear_guidance_schedule

# Gradually increase guidance during sampling
for step in range(total_steps):
    scale = linear_guidance_schedule(step, total_steps, start_scale=1.0, end_scale=7.5)
    # Use scale for this step
```

#### `cosine_guidance_schedule(step, total_steps, start_scale=1.0, end_scale=7.5)`

Cosine guidance scale schedule.

**Example:**

```python
from artifex.generative_models.models.diffusion.guidance import cosine_guidance_schedule

# Use cosine schedule (higher guidance at beginning and end)
for step in range(total_steps):
    scale = cosine_guidance_schedule(step, total_steps)
    # Use scale for this step
```

---

## Auxiliary Classes

### SimpleEncoder

::: artifex.generative_models.models.diffusion.latent.SimpleEncoder

Simple MLP encoder for Latent Diffusion Models.

**Purpose**: Encodes images to latent space with mean and log variance.

#### Initialization

```python
SimpleEncoder(
    input_dim: tuple[int, ...],
    latent_dim: int,
    hidden_dims: list | None = None,
    *,
    rngs: nnx.Rngs
)
```

### SimpleDecoder

::: artifex.generative_models.models.diffusion.latent.SimpleDecoder

Simple MLP decoder for Latent Diffusion Models.

**Purpose**: Decodes latent codes back to image space.

#### Initialization

```python
SimpleDecoder(
    latent_dim: int,
    output_dim: tuple[int, ...],
    hidden_dims: list | None = None,
    *,
    rngs: nnx.Rngs
)
```

---

## Configuration Reference

### ModelConfig for Diffusion Models

Complete reference of configuration parameters for all diffusion models.

#### Base Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Model name |
| `model_class` | `str` | Yes | Model class name |
| `input_dim` | `tuple[int, ...]` | Yes | Input dimensions (H, W, C) |
| `hidden_dims` | `list[int]` | No | Hidden layer dimensions |
| `output_dim` | `int | tuple` | No | Output dimensions |
| `activation` | `str` | No | Activation function |
| `parameters` | `dict` | No | Model-specific parameters |

#### DDPM Parameters

```python
{
    "noise_steps": 1000,      # Number of timesteps
    "beta_start": 1e-4,       # Initial noise level
    "beta_end": 0.02,         # Final noise level
    "beta_schedule": "linear" # Noise schedule
}
```

#### DDIM Parameters

```python
{
    "noise_steps": 1000,      # Training steps
    "ddim_steps": 50,         # Sampling steps
    "eta": 0.0,               # Stochasticity
    "skip_type": "uniform",   # Timestep selection
    "beta_start": 1e-4,
    "beta_end": 0.02
}
```

#### Score-Based Parameters

```python
{
    "sigma_min": 0.01,        # Minimum noise level
    "sigma_max": 1.0,         # Maximum noise level
    "score_scaling": 1.0,     # Score scaling factor
    "noise_steps": 1000
}
```

#### Latent Diffusion Parameters

```python
{
    "latent_dim": 16,                    # Latent space dimension
    "encoder_hidden_dims": [64, 128],    # Encoder architecture
    "decoder_hidden_dims": [128, 64],    # Decoder architecture
    "encoder_type": "simple",            # Encoder type
    "scale_factor": 0.18215,             # Latent scaling
    "noise_steps": 1000
}
```

#### DiT Parameters

```python
{
    "img_size": 32,           # Image size
    "patch_size": 4,          # Patch size
    "hidden_size": 512,       # Transformer dimension
    "depth": 12,              # Number of layers
    "num_heads": 8,           # Attention heads
    "mlp_ratio": 4.0,         # MLP expansion
    "num_classes": 10,        # Number of classes
    "dropout_rate": 0.1,      # Dropout rate
    "learn_sigma": False,     # Learn variance
    "cfg_scale": 2.0,         # Guidance scale
    "noise_steps": 1000
}
```

---

## Quick Reference

### Model Selection Guide

| Model | Best For | Sampling Speed | Memory | Quality |
|-------|----------|----------------|--------|---------|
| **DDPMModel** | Standard use, learning | Slow (1000 steps) | High | ⭐⭐⭐⭐⭐ |
| **DDIMModel** | Fast inference | Fast (50 steps) | High | ⭐⭐⭐⭐ |
| **ScoreDiffusionModel** | Research, flexibility | Medium | High | ⭐⭐⭐⭐ |
| **LDMModel** | High-res, efficiency | Fast | Medium | ⭐⭐⭐⭐ |
| **DiTModel** | Scalability, SOTA | Medium | Very High | ⭐⭐⭐⭐⭐ |

### Common Usage Patterns

```python
# Basic DDPM
model = DDPMModel(config, rngs=rngs)
samples = model.generate(16, rngs=rngs)

# Fast DDIM
model = DDIMModel(config, rngs=rngs)
samples = model.ddim_sample(16, steps=50, rngs=rngs)

# Latent Diffusion
model = LDMModel(config=config, rngs=rngs)
samples = model.sample(16, rngs=rngs)

# DiT with conditioning
model = DiTModel(config, rngs=rngs)
samples = model.generate(16, y=labels, cfg_scale=4.0, rngs=rngs)
```

## See Also

- [Diffusion Concepts](../../user-guide/concepts/diffusion-explained.md): Theory and mathematical foundations
- [User Guide](../../user-guide/models/diffusion-guide.md): Practical usage examples
- [MNIST Tutorial](../../examples/basic/diffusion-mnist.md): Complete working example
- [Core API](../core/base.md): Base generative model classes
- [Configuration API](../core/configuration.md): Configuration system
