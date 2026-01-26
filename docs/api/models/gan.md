# GAN API Reference

Complete API reference for all GAN model classes in Artifex.

## Overview

The GAN module provides implementations of various Generative Adversarial Network architectures:

- **Base GAN**: Standard generator and discriminator
- **DCGAN**: Deep convolutional architecture
- **WGAN**: Wasserstein distance with gradient penalty
- **LSGAN**: Least squares loss
- **Conditional GAN**: Class-conditioned generation
- **CycleGAN**: Unpaired image-to-image translation
- **PatchGAN**: Patch-based discrimination

## Base Classes

### Generator

::: artifex.generative_models.models.gan.Generator

Basic generator network that transforms latent vectors into data samples.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | Required | Hidden layer dimensions |
| `output_shape` | `tuple` | Required | Shape of generated samples (batch, C, H, W) |
| `latent_dim` | `int` | Required | Dimension of latent space |
| `activation` | `str` | `"relu"` | Activation function name |
| `batch_norm` | `bool` | `True` | Whether to use batch normalization |
| `dropout_rate` | `float` | `0.0` | Dropout rate |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(z, training=False)`

Generate samples from latent vectors.

**Parameters**:

- `z` (`jax.Array`): Latent vectors of shape `(batch_size, latent_dim)`
- `training` (`bool`): Whether in training mode (affects batch norm and dropout)

**Returns**:

- `jax.Array`: Generated samples of shape `(batch_size, *output_shape[1:])`

**Example**:

```python
from artifex.generative_models.models.gan import Generator
from flax import nnx
import jax.numpy as jnp

# Create generator
generator = Generator(
    hidden_dims=[256, 512, 1024],
    output_shape=(1, 1, 28, 28),  # MNIST
    latent_dim=100,
    activation="relu",
    batch_norm=True,
    rngs=nnx.Rngs(params=0),
)

# Generate samples
z = jnp.ones((32, 100))  # Batch of latent vectors
samples = generator(z, training=False)
print(samples.shape)  # (32, 1, 28, 28)
```

---

### Discriminator

::: artifex.generative_models.models.gan.Discriminator

Basic discriminator network that classifies samples as real or fake.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | Required | Hidden layer dimensions |
| `activation` | `str` | `"leaky_relu"` | Activation function name |
| `leaky_relu_slope` | `float` | `0.2` | Negative slope for LeakyReLU |
| `batch_norm` | `bool` | `False` | Whether to use batch normalization |
| `dropout_rate` | `float` | `0.3` | Dropout rate |
| `use_spectral_norm` | `bool` | `False` | Whether to use spectral normalization |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `initialize_layers(input_shape, rngs=None)`

Initialize layers based on input shape.

**Parameters**:

- `input_shape` (`tuple`): Shape of input data `(batch, C, H, W)`
- `rngs` (`nnx.Rngs`, optional): Random number generators

#### `__call__(x, training=False)`

Classify samples as real or fake.

**Parameters**:

- `x` (`jax.Array`): Input samples of shape `(batch_size, C, H, W)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Discrimination scores of shape `(batch_size, 1)`, values in `[0, 1]`

**Example**:

```python
from artifex.generative_models.models.gan import Discriminator
from flax import nnx
import jax.numpy as jnp

# Create discriminator
discriminator = Discriminator(
    hidden_dims=[512, 256, 128],
    activation="leaky_relu",
    leaky_relu_slope=0.2,
    batch_norm=False,
    dropout_rate=0.3,
    rngs=nnx.Rngs(params=0, dropout=1),
)

# Classify samples
samples = jnp.ones((32, 1, 28, 28))
scores = discriminator(samples, training=True)
print(scores.shape)  # (32, 1)
print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
```

---

### GAN

::: artifex.generative_models.models.gan.GAN

Complete GAN model combining generator and discriminator.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `object` | Required | Model configuration object |
| `rngs` | `nnx.Rngs` | Required | Random number generators |
| `precision` | `jax.lax.Precision` | `None` | Numerical precision |

**Configuration Object**:

The `config` must have the following structure:

```python
class GANConfig:
    latent_dim: int = 100                  # Latent space dimension
    loss_type: str = "vanilla"             # Loss type: "vanilla", "wasserstein", "least_squares", "hinge"
    gradient_penalty_weight: float = 0.0   # Weight for gradient penalty (WGAN-GP)

    class generator:
        hidden_dims: list[int]             # Generator hidden dimensions
        output_shape: tuple                # Output shape (batch, C, H, W)
        activation: str = "relu"
        batch_norm: bool = True
        dropout_rate: float = 0.0

    class discriminator:
        hidden_dims: list[int]             # Discriminator hidden dimensions
        activation: str = "leaky_relu"
        leaky_relu_slope: float = 0.2
        batch_norm: bool = False
        dropout_rate: float = 0.3
        use_spectral_norm: bool = False
```

**Methods**:

#### `__call__(x, rngs=None, training=False, **kwargs)`

Forward pass through the GAN (runs discriminator only).

**Parameters**:

- `x` (`jax.Array`): Input data
- `rngs` (`nnx.Rngs`, optional): Random number generators
- `training` (`bool`): Whether in training mode

**Returns**:

- `dict`: Dictionary with keys:
  - `"real_scores"`: Discriminator scores for real data
  - `"fake_scores"`: `None` (computed in `loss_fn`)
  - `"fake_samples"`: `None` (computed in `loss_fn`)

#### `generate(n_samples=1, rngs=None, batch_size=None, **kwargs)`

Generate samples from the generator.

**Parameters**:

- `n_samples` (`int`): Number of samples to generate
- `rngs` (`nnx.Rngs`, optional): Random number generators
- `batch_size` (`int`, optional): Alternative to `n_samples`

**Returns**:

- `jax.Array`: Generated samples

#### `loss_fn(batch, model_outputs, rngs=None, **kwargs)`

Compute GAN loss for training.

**Parameters**:

- `batch` (`dict` or `jax.Array`): Input batch (real data)
- `model_outputs` (`dict`): Model outputs (unused for GAN)
- `rngs` (`nnx.Rngs`, optional): Random number generators

**Returns**:

- `dict`: Dictionary with losses:
  - `"loss"`: Total loss (generator + discriminator)
  - `"generator_loss"`: Generator loss
  - `"discriminator_loss"`: Discriminator loss
  - `"real_scores_mean"`: Mean discriminator score for real samples
  - `"fake_scores_mean"`: Mean discriminator score for fake samples

**Example**:

```python
from artifex.generative_models.models.gan import GAN
from flax import nnx

# Create configuration
class GANConfig:
    latent_dim = 100
    loss_type = "vanilla"

    class generator:
        hidden_dims = [256, 512]
        output_shape = (1, 1, 28, 28)
        activation = "relu"
        batch_norm = True
        dropout_rate = 0.0

    class discriminator:
        hidden_dims = [512, 256]
        activation = "leaky_relu"
        leaky_relu_slope = 0.2
        batch_norm = False
        dropout_rate = 0.3
        use_spectral_norm = False

# Create GAN
gan = GAN(GANConfig(), rngs=nnx.Rngs(params=0, dropout=1, sample=2))

# Generate samples
samples = gan.generate(n_samples=16, rngs=nnx.Rngs(sample=0))

# Compute loss
import jax.numpy as jnp
batch = jnp.ones((32, 1, 28, 28))
losses = gan.loss_fn(batch, None, rngs=nnx.Rngs(sample=0))
print(f"Generator Loss: {losses['generator_loss']:.4f}")
print(f"Discriminator Loss: {losses['discriminator_loss']:.4f}")
```

---

## DCGAN

### DCGANGenerator

::: artifex.generative_models.models.gan.DCGANGenerator

Deep Convolutional GAN generator using transposed convolutions.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_shape` | `tuple[int, ...]` | Required | Output image shape `(C, H, W)` |
| `latent_dim` | `int` | `100` | Latent space dimension |
| `hidden_dims` | `tuple[int, ...]` | `(256, 128, 64, 32)` | Channel dimensions per layer |
| `activation` | `callable` | `jax.nn.relu` | Activation function |
| `batch_norm` | `bool` | `True` | Use batch normalization |
| `dropout_rate` | `float` | `0.0` | Dropout rate |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(z, training=True)`

Generate images from latent vectors.

**Parameters**:

- `z` (`jax.Array`): Latent vectors of shape `(batch_size, latent_dim)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Generated images of shape `(batch_size, C, H, W)`

**Example**:

```python
from artifex.generative_models.models.gan import DCGANGenerator
from flax import nnx
import jax
import jax.numpy as jnp

generator = DCGANGenerator(
    output_shape=(3, 64, 64),            # RGB 64×64 images
    latent_dim=100,
    hidden_dims=(256, 128, 64, 32),
    activation=jax.nn.relu,
    batch_norm=True,
    rngs=nnx.Rngs(params=0),
)

# Generate samples
z = jax.random.normal(jax.random.key(0), (16, 100))
images = generator(z, training=False)
print(images.shape)  # (16, 3, 64, 64)
```

---

### DCGANDiscriminator

::: artifex.generative_models.models.gan.DCGANDiscriminator

Deep Convolutional GAN discriminator using strided convolutions.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `tuple[int, ...]` | Required | Input image shape `(C, H, W)` |
| `hidden_dims` | `tuple[int, ...]` | `(32, 64, 128, 256)` | Channel dimensions per layer |
| `activation` | `callable` | `jax.nn.leaky_relu` | Activation function |
| `leaky_relu_slope` | `float` | `0.2` | Negative slope for LeakyReLU |
| `batch_norm` | `bool` | `False` | Use batch normalization |
| `dropout_rate` | `float` | `0.3` | Dropout rate |
| `use_spectral_norm` | `bool` | `True` | Use spectral normalization |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(x, training=True)`

Classify images as real or fake.

**Parameters**:

- `x` (`jax.Array`): Input images of shape `(batch_size, C, H, W)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Discrimination scores of shape `(batch_size, 1)`

**Example**:

```python
from artifex.generative_models.models.gan import DCGANDiscriminator
from flax import nnx
import jax.numpy as jnp

discriminator = DCGANDiscriminator(
    input_shape=(3, 64, 64),
    hidden_dims=(32, 64, 128, 256),
    activation=jax.nn.leaky_relu,
    leaky_relu_slope=0.2,
    batch_norm=False,
    dropout_rate=0.3,
    use_spectral_norm=True,
    rngs=nnx.Rngs(params=0, dropout=1),
)

# Classify images
images = jnp.ones((16, 3, 64, 64))
scores = discriminator(images, training=True)
print(scores.shape)  # (16, 1)
```

---

### DCGAN

::: artifex.generative_models.models.gan.DCGAN

Complete Deep Convolutional GAN model.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `DCGANConfiguration` | Required | DCGAN configuration |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Configuration**:

Use `DCGANConfiguration` from `artifex.generative_models.core.configuration.gan`:

```python
from artifex.generative_models.core.configuration.gan import DCGANConfiguration

config = DCGANConfiguration(
    image_size=64,                        # Image size (H=W)
    channels=3,                           # Number of channels
    latent_dim=100,                       # Latent dimension
    gen_hidden_dims=(256, 128, 64, 32),  # Generator channels
    disc_hidden_dims=(32, 64, 128, 256), # Discriminator channels
    loss_type="vanilla",                  # Loss type
    generator_lr=0.0002,                  # Generator learning rate
    discriminator_lr=0.0002,              # Discriminator learning rate
    beta1=0.5,                            # Adam β1
    beta2=0.999,                          # Adam β2
)
```

**Example**:

```python
from artifex.generative_models.models.gan import DCGAN
from artifex.generative_models.core.configuration.gan import DCGANConfiguration
from flax import nnx

config = DCGANConfiguration(
    image_size=64,
    channels=3,
    latent_dim=100,
    gen_hidden_dims=(256, 128, 64, 32),
    disc_hidden_dims=(32, 64, 128, 256),
    loss_type="vanilla",
)

dcgan = DCGAN(config, rngs=nnx.Rngs(params=0, dropout=1, sample=2))

# Generate samples
samples = dcgan.generate(n_samples=16, rngs=nnx.Rngs(sample=0))
print(samples.shape)  # (16, 3, 64, 64)
```

---

## WGAN

### WGANGenerator

::: artifex.generative_models.models.gan.WGANGenerator

Wasserstein GAN generator with convolutional architecture.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_shape` | `tuple[int, ...]` | Required | Output image shape `(C, H, W)` |
| `latent_dim` | `int` | `100` | Latent space dimension |
| `hidden_dims` | `tuple[int, ...]` | `(1024, 512, 256)` | Channel dimensions |
| `activation` | `callable` | `jax.nn.relu` | Activation function |
| `batch_norm` | `bool` | `True` | Use batch normalization |
| `dropout_rate` | `float` | `0.0` | Dropout rate |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(z, training=True)`

Generate images from latent vectors.

**Parameters**:

- `z` (`jax.Array`): Latent vectors of shape `(batch_size, latent_dim)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Generated images of shape `(batch_size, C, H, W)`

**Example**:

```python
from artifex.generative_models.models.gan import WGANGenerator
from flax import nnx
import jax

generator = WGANGenerator(
    output_shape=(3, 64, 64),
    latent_dim=100,
    hidden_dims=(1024, 512, 256),
    activation=jax.nn.relu,
    batch_norm=True,
    rngs=nnx.Rngs(params=0),
)

z = jax.random.normal(jax.random.key(0), (16, 100))
images = generator(z, training=False)
print(images.shape)  # (16, 3, 64, 64)
```

---

### WGANDiscriminator

::: artifex.generative_models.models.gan.WGANDiscriminator

Wasserstein GAN discriminator (critic) with instance normalization.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `tuple[int, ...]` | Required | Input image shape `(C, H, W)` |
| `hidden_dims` | `tuple[int, ...]` | `(256, 512, 1024)` | Channel dimensions |
| `activation` | `callable` | `jax.nn.leaky_relu` | Activation function |
| `leaky_relu_slope` | `float` | `0.2` | Negative slope for LeakyReLU |
| `use_instance_norm` | `bool` | `True` | Use instance normalization |
| `dropout_rate` | `float` | `0.0` | Dropout rate |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(x, training=True)`

Compute critic scores (no sigmoid activation).

**Parameters**:

- `x` (`jax.Array`): Input images of shape `(batch_size, C, H, W)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Raw critic scores (no sigmoid) of shape `(batch_size,)`

**Example**:

```python
from artifex.generative_models.models.gan import WGANDiscriminator
from flax import nnx
import jax.numpy as jnp

discriminator = WGANDiscriminator(
    input_shape=(3, 64, 64),
    hidden_dims=(256, 512, 1024),
    activation=jax.nn.leaky_relu,
    use_instance_norm=True,
    rngs=nnx.Rngs(params=0),
)

images = jnp.ones((16, 3, 64, 64))
scores = discriminator(images, training=True)
print(scores.shape)  # (16,)
# Note: No sigmoid, scores can be any real number
```

---

### WGAN

::: artifex.generative_models.models.gan.WGAN

Complete Wasserstein GAN with Gradient Penalty (WGAN-GP).

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ModelConfig` | Required | Model configuration |
| `rngs` | `nnx.Rngs` | Required | Random number generators |
| `precision` | `jax.lax.Precision` | `None` | Numerical precision |

**Configuration**:

```python
from artifex.generative_models.core.configuration import ModelConfig

config = ModelConfig(
    input_dim=100,                        # Latent dimension
    output_dim=(3, 64, 64),               # Output image shape
    hidden_dims=None,                     # Use defaults
    metadata={
        "gan_params": {
            "gen_hidden_dims": (1024, 512, 256),
            "disc_hidden_dims": (256, 512, 1024),
            "gradient_penalty_weight": 10.0,     # Lambda for GP
            "critic_iterations": 5,               # Critic updates per generator
        }
    }
)
```

**Methods**:

#### `generate(n_samples=1, rngs=None, batch_size=None, **kwargs)`

Generate samples from generator.

**Parameters**:

- `n_samples` (`int`): Number of samples
- `rngs` (`nnx.Rngs`, optional): Random number generators
- `batch_size` (`int`, optional): Alternative to `n_samples`

**Returns**:

- `jax.Array`: Generated samples

#### `discriminator_loss(real_samples, fake_samples, rngs)`

Compute WGAN-GP discriminator loss with gradient penalty.

**Parameters**:

- `real_samples` (`jax.Array`): Real images
- `fake_samples` (`jax.Array`): Generated images
- `rngs` (`nnx.Rngs`): Random number generators

**Returns**:

- `jax.Array`: Discriminator loss (scalar)

**Loss Formula**:
$$
\mathcal{L}_D = \mathbb{E}[D(G(z))] - \mathbb{E}[D(x)] + \lambda \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

#### `generator_loss(fake_samples)`

Compute WGAN generator loss.

**Parameters**:

- `fake_samples` (`jax.Array`): Generated images

**Returns**:

- `jax.Array`: Generator loss (scalar)

**Loss Formula**:
$$
\mathcal{L}_G = -\mathbb{E}[D(G(z))]
$$

**Example**:

```python
from artifex.generative_models.models.gan import WGAN
from artifex.generative_models.core.configuration import ModelConfig
from flax import nnx

config = ModelConfig(
    input_dim=100,
    output_dim=(3, 64, 64),
    metadata={
        "gan_params": {
            "gen_hidden_dims": (1024, 512, 256),
            "disc_hidden_dims": (256, 512, 1024),
            "gradient_penalty_weight": 10.0,
            "critic_iterations": 5,
        }
    }
)

wgan = WGAN(config, rngs=nnx.Rngs(params=0, sample=1))

# Generate samples
samples = wgan.generate(n_samples=16, rngs=nnx.Rngs(sample=0))
print(samples.shape)  # (16, 3, 64, 64)

# Training step
import jax
real_samples = jax.random.normal(jax.random.key(0), (32, 3, 64, 64))
z = jax.random.normal(jax.random.key(1), (32, 100))
fake_samples = wgan.generator(z, training=True)

disc_loss = wgan.discriminator_loss(real_samples, fake_samples, rngs=nnx.Rngs(params=2))
gen_loss = wgan.generator_loss(fake_samples)
```

---

### compute_gradient_penalty

::: artifex.generative_models.models.gan.compute_gradient_penalty

Compute gradient penalty for WGAN-GP.

**Module**: `artifex.generative_models.models.gan`

**Function Signature**:

```python
def compute_gradient_penalty(
    discriminator: WGANDiscriminator,
    real_samples: jax.Array,
    fake_samples: jax.Array,
    rngs: nnx.Rngs,
    lambda_gp: float = 10.0,
) -> jax.Array
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `discriminator` | `WGANDiscriminator` | Required | Discriminator network |
| `real_samples` | `jax.Array` | Required | Real images |
| `fake_samples` | `jax.Array` | Required | Generated images |
| `rngs` | `nnx.Rngs` | Required | Random number generators |
| `lambda_gp` | `float` | `10.0` | Gradient penalty weight |

**Returns**:

- `jax.Array`: Gradient penalty loss (scalar)

**Formula**:
$$
\text{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

where $\hat{x} = \epsilon x + (1-\epsilon)G(z)$ is a random interpolation.

**Example**:

```python
from artifex.generative_models.models.gan import compute_gradient_penalty, WGANDiscriminator
from flax import nnx
import jax

discriminator = WGANDiscriminator(
    input_shape=(3, 64, 64),
    rngs=nnx.Rngs(params=0),
)

real_samples = jax.random.normal(jax.random.key(0), (32, 3, 64, 64))
fake_samples = jax.random.normal(jax.random.key(1), (32, 3, 64, 64))

gp = compute_gradient_penalty(
    discriminator,
    real_samples,
    fake_samples,
    rngs=nnx.Rngs(params=2),
    lambda_gp=10.0,
)
print(f"Gradient Penalty: {gp:.4f}")
```

---

## LSGAN

### LSGANGenerator

::: artifex.generative_models.models.gan.LSGANGenerator

Least Squares GAN generator (same architecture as DCGAN).

**Module**: `artifex.generative_models.models.gan`

**Parameters**: Same as `DCGANGenerator`

**Methods**: Same as `DCGANGenerator`

**Example**:

```python
from artifex.generative_models.models.gan import LSGANGenerator
from flax import nnx
import jax

generator = LSGANGenerator(
    output_shape=(3, 64, 64),
    latent_dim=100,
    hidden_dims=(512, 256, 128, 64),
    rngs=nnx.Rngs(params=0),
)

z = jax.random.normal(jax.random.key(0), (16, 100))
images = generator(z, training=False)
```

---

### LSGANDiscriminator

::: artifex.generative_models.models.gan.LSGANDiscriminator

Least Squares GAN discriminator (no sigmoid activation).

**Module**: `artifex.generative_models.models.gan`

**Parameters**: Same as `DCGANDiscriminator`

**Key Difference**: Output layer has no sigmoid activation (outputs raw logits for least squares loss).

**Example**:

```python
from artifex.generative_models.models.gan import LSGANDiscriminator
from flax import nnx
import jax.numpy as jnp

discriminator = LSGANDiscriminator(
    input_shape=(3, 64, 64),
    hidden_dims=(64, 128, 256, 512),
    rngs=nnx.Rngs(params=0, dropout=1),
)

images = jnp.ones((16, 3, 64, 64))
scores = discriminator(images, training=True)
# Note: Scores are raw logits, not in [0, 1]
```

---

### LSGAN

::: artifex.generative_models.models.gan.LSGAN

Complete Least Squares GAN model.

**Module**: `artifex.generative_models.models.gan`

**Parameters**: Same as base `GAN`

**Methods**:

#### `generator_loss(fake_scores, target_real=1.0, reduction="mean")`

Compute LSGAN generator loss.

**Formula**:
$$
\mathcal{L}_G = \frac{1}{2}\mathbb{E}[(D(G(z)) - c)^2]
$$

where $c$ is the target value for fake samples (usually 1.0).

#### `discriminator_loss(real_scores, fake_scores, target_real=1.0, target_fake=0.0, reduction="mean")`

Compute LSGAN discriminator loss.

**Formula**:
$$
\mathcal{L}_D = \frac{1}{2}\mathbb{E}[(D(x) - b)^2] + \frac{1}{2}\mathbb{E}[D(G(z))^2]
$$

where $b$ is the target for real samples (usually 1.0).

**Example**:

```python
from artifex.generative_models.models.gan import LSGAN
from artifex.generative_models.core.configuration import ModelConfig
from flax import nnx

config = ModelConfig(
    input_dim=100,
    output_dim=(3, 64, 64),
    hidden_dims=[512, 256, 128, 64],
)

lsgan = LSGAN(config, rngs=nnx.Rngs(params=0, dropout=1, sample=2))

# Generate samples
samples = lsgan.generate(n_samples=16, rngs=nnx.Rngs(sample=0))

# Compute losses
import jax
real_images = jax.random.normal(jax.random.key(0), (32, 3, 64, 64))
z = jax.random.normal(jax.random.key(1), (32, 100))
fake_images = lsgan.generator(z, training=True)

real_scores = lsgan.discriminator(real_images, training=True)
fake_scores = lsgan.discriminator(fake_images, training=True)

gen_loss = lsgan.generator_loss(fake_scores)
disc_loss = lsgan.discriminator_loss(real_scores, fake_scores)
```

---

## Conditional GAN

### ConditionalGenerator

::: artifex.generative_models.models.gan.ConditionalGenerator

Conditional GAN generator that takes class labels as input.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_shape` | `tuple[int, ...]` | Required | Output image shape `(C, H, W)` |
| `num_classes` | `int` | Required | Number of classes |
| `latent_dim` | `int` | `100` | Latent space dimension |
| `hidden_dims` | `tuple[int, ...]` | `(512, 256, 128, 64)` | Channel dimensions |
| `activation` | `callable` | `jax.nn.relu` | Activation function |
| `batch_norm` | `bool` | `True` | Use batch normalization |
| `dropout_rate` | `float` | `0.0` | Dropout rate |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(z, labels, training=True)`

Generate samples conditioned on labels.

**Parameters**:

- `z` (`jax.Array`): Latent vectors of shape `(batch_size, latent_dim)`
- `labels` (`jax.Array`): One-hot encoded labels of shape `(batch_size, num_classes)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Generated images of shape `(batch_size, C, H, W)`

**Example**:

```python
from artifex.generative_models.models.gan import ConditionalGenerator
from flax import nnx
import jax
import jax.numpy as jnp

generator = ConditionalGenerator(
    output_shape=(1, 28, 28),           # MNIST
    num_classes=10,
    latent_dim=100,
    hidden_dims=(512, 256, 128, 64),
    rngs=nnx.Rngs(params=0),
)

# Generate specific digits
z = jax.random.normal(jax.random.key(0), (10, 100))
labels = jax.nn.one_hot(jnp.arange(10), 10)  # One of each digit
images = generator(z, labels, training=False)
print(images.shape)  # (10, 1, 28, 28)
```

---

### ConditionalDiscriminator

::: artifex.generative_models.models.gan.ConditionalDiscriminator

Conditional GAN discriminator that takes class labels as input.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `tuple[int, ...]` | Required | Input image shape `(C, H, W)` |
| `num_classes` | `int` | Required | Number of classes |
| `hidden_dims` | `tuple[int, ...]` | `(64, 128, 256, 512)` | Channel dimensions |
| `activation` | `callable` | `jax.nn.leaky_relu` | Activation function |
| `leaky_relu_slope` | `float` | `0.2` | Negative slope for LeakyReLU |
| `batch_norm` | `bool` | `False` | Use batch normalization |
| `dropout_rate` | `float` | `0.0` | Dropout rate |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Methods**:

#### `__call__(x, labels, training=True)`

Classify samples conditioned on labels.

**Parameters**:

- `x` (`jax.Array`): Input images of shape `(batch_size, C, H, W)`
- `labels` (`jax.Array`): One-hot encoded labels of shape `(batch_size, num_classes)`
- `training` (`bool`): Whether in training mode

**Returns**:

- `jax.Array`: Discrimination scores of shape `(batch_size,)`

**Example**:

```python
from artifex.generative_models.models.gan import ConditionalDiscriminator
from flax import nnx
import jax
import jax.numpy as jnp

discriminator = ConditionalDiscriminator(
    input_shape=(1, 28, 28),
    num_classes=10,
    hidden_dims=(64, 128, 256, 512),
    rngs=nnx.Rngs(params=0, dropout=1),
)

# Classify samples with labels
images = jax.random.normal(jax.random.key(0), (32, 1, 28, 28))
labels = jax.nn.one_hot(jnp.zeros(32, dtype=int), 10)  # All zeros
scores = discriminator(images, labels, training=True)
print(scores.shape)  # (32,)
```

---

### ConditionalGAN

::: artifex.generative_models.models.gan.ConditionalGAN

Complete Conditional GAN model.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ModelConfig` | Required | Model configuration |
| `rngs` | `nnx.Rngs` | Required | Random number generators |
| `precision` | `jax.lax.Precision` | `None` | Numerical precision |

**Configuration**:

```python
from artifex.generative_models.core.configuration import ModelConfig

config = ModelConfig(
    input_dim=100,                        # Latent dimension
    output_dim=(1, 28, 28),               # MNIST shape
    metadata={
        "gan_params": {
            "num_classes": 10,                      # Number of classes
            "gen_hidden_dims": (512, 256, 128, 64),
            "discriminator_features": [64, 128, 256, 512],
        }
    }
)
```

**Methods**:

#### `generate(n_samples=1, labels=None, rngs=None, batch_size=None, **kwargs)`

Generate conditional samples.

**Parameters**:

- `n_samples` (`int`): Number of samples
- `labels` (`jax.Array`, optional): One-hot encoded labels. If `None`, random labels are used.
- `rngs` (`nnx.Rngs`, optional): Random number generators
- `batch_size` (`int`, optional): Alternative to `n_samples`

**Returns**:

- `jax.Array`: Generated samples

**Example**:

```python
from artifex.generative_models.models.gan import ConditionalGAN
from artifex.generative_models.core.configuration import ModelConfig
from flax import nnx
import jax
import jax.numpy as jnp

config = ModelConfig(
    input_dim=100,
    output_dim=(1, 28, 28),
    metadata={
        "gan_params": {
            "num_classes": 10,
            "gen_hidden_dims": (512, 256, 128, 64),
            "discriminator_features": [64, 128, 256, 512],
        }
    }
)

cgan = ConditionalGAN(config, rngs=nnx.Rngs(params=0, dropout=1, sample=2))

# Generate specific digits
labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 10)
samples = cgan.generate(n_samples=10, labels=labels, rngs=nnx.Rngs(sample=0))
print(samples.shape)  # (10, 1, 28, 28)
```

---

## CycleGAN

### CycleGANGenerator

::: artifex.generative_models.models.gan.CycleGANGenerator

CycleGAN generator for image-to-image translation.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_shape` | `tuple[int, ...]` | Required | Output image shape `(C, H, W)` |
| `hidden_dims` | `tuple[int, ...]` | `(64, 128, 256)` | Channel dimensions |
| `num_residual_blocks` | `int` | `9` | Number of ResNet blocks |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

---

### CycleGANDiscriminator

::: artifex.generative_models.models.gan.CycleGANDiscriminator

CycleGAN PatchGAN discriminator.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `tuple[int, ...]` | Required | Input image shape `(C, H, W)` |
| `hidden_dims` | `tuple[int, ...]` | `(64, 128, 256, 512)` | Channel dimensions |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

---

### CycleGAN

::: artifex.generative_models.models.gan.CycleGAN

Complete CycleGAN model for unpaired image-to-image translation.

**Module**: `artifex.generative_models.models.gan`

**Key Features**:

- Two generators: `G: X → Y` and `F: Y → X`
- Two discriminators: `D_X` and `D_Y`
- Cycle consistency loss: `x → G(x) → F(G(x)) ≈ x`
- Identity loss (optional): `F(x) ≈ x` if `x ∈ X`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape_x` | `tuple` | Required | Domain X image shape `(C, H, W)` |
| `input_shape_y` | `tuple` | Required | Domain Y image shape `(C, H, W)` |
| `gen_hidden_dims` | `tuple` | `(64, 128, 256)` | Generator channels |
| `disc_hidden_dims` | `tuple` | `(64, 128, 256)` | Discriminator channels |
| `cycle_weight` | `float` | `10.0` | Cycle consistency weight |
| `identity_weight` | `float` | `0.5` | Identity loss weight |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Example**:

```python
from artifex.generative_models.models.gan import CycleGAN
from flax import nnx
import jax

cyclegan = CycleGAN(
    input_shape_x=(3, 256, 256),         # Horses
    input_shape_y=(3, 256, 256),         # Zebras
    gen_hidden_dims=(64, 128, 256),
    disc_hidden_dims=(64, 128, 256),
    cycle_weight=10.0,
    identity_weight=0.5,
    rngs=nnx.Rngs(params=0, dropout=1),
)

# Translate horse to zebra
horse_images = jax.random.normal(jax.random.key(0), (4, 3, 256, 256))
zebra_images = cyclegan.generator_g(horse_images, training=False)
print(zebra_images.shape)  # (4, 3, 256, 256)

# Translate zebra back to horse (cycle consistency)
reconstructed_horses = cyclegan.generator_f(zebra_images, training=False)
print(reconstructed_horses.shape)  # (4, 3, 256, 256)
```

---

## PatchGAN

### PatchGANDiscriminator

::: artifex.generative_models.models.gan.PatchGANDiscriminator

PatchGAN discriminator that outputs N×N array of patch predictions.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `tuple[int, ...]` | Required | Input image shape `(C, H, W)` |
| `hidden_dims` | `tuple[int, ...]` | `(64, 128, 256, 512)` | Channel dimensions |
| `kernel_size` | `int` | `4` | Convolution kernel size |
| `stride` | `int` | `2` | Convolution stride |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Returns**: N×N array of patch classifications instead of single scalar

**Example**:

```python
from artifex.generative_models.models.gan import PatchGANDiscriminator
from flax import nnx
import jax.numpy as jnp

discriminator = PatchGANDiscriminator(
    input_shape=(3, 256, 256),
    hidden_dims=(64, 128, 256, 512),
    kernel_size=4,
    stride=2,
    rngs=nnx.Rngs(params=0, dropout=1),
)

images = jnp.ones((16, 3, 256, 256))
patch_scores = discriminator(images, training=True)
print(patch_scores.shape)  # (16, H', W', 1) - array of patch predictions
```

---

### MultiScalePatchGANDiscriminator

::: artifex.generative_models.models.gan.MultiScalePatchGANDiscriminator

Multi-scale PatchGAN discriminator operating at multiple resolutions.

**Module**: `artifex.generative_models.models.gan`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `tuple[int, ...]` | Required | Input image shape `(C, H, W)` |
| `hidden_dims` | `tuple[int, ...]` | `(64, 128, 256)` | Channel dimensions |
| `num_scales` | `int` | `3` | Number of scales |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

**Returns**: List of predictions at different scales

**Example**:

```python
from artifex.generative_models.models.gan import MultiScalePatchGANDiscriminator
from flax import nnx
import jax.numpy as jnp

discriminator = MultiScalePatchGANDiscriminator(
    input_shape=(3, 256, 256),
    hidden_dims=(64, 128, 256),
    num_scales=3,
    rngs=nnx.Rngs(params=0, dropout=1),
)

images = jnp.ones((16, 3, 256, 256))
predictions = discriminator(images, training=True)
# predictions is a list of 3 arrays at different scales
for i, pred in enumerate(predictions):
    print(f"Scale {i}: {pred.shape}")
```

---

## Loss Functions

See [Adversarial Loss Functions](../core/losses.md#adversarial-losses) for detailed documentation of:

- `vanilla_generator_loss`
- `vanilla_discriminator_loss`
- `least_squares_generator_loss`
- `least_squares_discriminator_loss`
- `wasserstein_generator_loss`
- `wasserstein_discriminator_loss`
- `hinge_generator_loss`
- `hinge_discriminator_loss`

---

## Summary

This API reference covered all GAN model classes:

- **Base Classes**: `Generator`, `Discriminator`, `GAN`
- **DCGAN**: `DCGANGenerator`, `DCGANDiscriminator`, `DCGAN`
- **WGAN**: `WGANGenerator`, `WGANDiscriminator`, `WGAN`, `compute_gradient_penalty`
- **LSGAN**: `LSGANGenerator`, `LSGANDiscriminator`, `LSGAN`
- **Conditional GAN**: `ConditionalGenerator`, `ConditionalDiscriminator`, `ConditionalGAN`
- **CycleGAN**: `CycleGANGenerator`, `CycleGANDiscriminator`, `CycleGAN`
- **PatchGAN**: `PatchGANDiscriminator`, `MultiScalePatchGANDiscriminator`

## See Also

- [GAN Concepts](../../user-guide/concepts/gan-explained.md): Theory and mathematical foundations
- [GAN User Guide](../../user-guide/models/gan-guide.md): Practical usage examples
- [GAN MNIST Example](../../examples/basic/simple-gan.md): Complete training tutorial
- [Adversarial Losses](../core/losses.md#adversarial-losses): Loss function reference
