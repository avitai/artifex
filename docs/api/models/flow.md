# Flow Models API Reference

Complete API documentation for normalizing flow models in Artifex.

## Overview

This document provides detailed API reference for all flow-based model classes, including:

- **Base Classes**: `NormalizingFlow`, `FlowLayer`
- **RealNVP**: Coupling-based flows with affine transformations
- **Glow**: Multi-scale flow with ActNorm and invertible convolutions
- **MAF**: Masked Autoregressive Flow for fast density estimation
- **IAF**: Inverse Autoregressive Flow for fast sampling
- **Neural Spline Flows**: Spline-based transformations for higher expressiveness

## Base Classes

### `NormalizingFlow`

Base class for all normalizing flow models.

```python
from artifex.generative_models.models.flow.base import NormalizingFlow
```

**Initialization:**

```python
model = NormalizingFlow(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs,
    precision: jax.lax.Precision | None = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Model configuration object |
| `rngs` | `nnx.Rngs` | Random number generators (required) |
| `precision` | `jax.lax.Precision \| None` | JAX operation precision (optional) |

**Configuration Parameters** (in `config.parameters`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_distribution` | `str` | `"normal"` | Base distribution type (`"normal"` or `"uniform"`) |
| `base_distribution_params` | `dict` | `{}` | Base distribution parameters |

**Attributes:**

- `input_dim`: Input dimension (from config)
- `latent_dim`: Latent dimension (defaults to input_dim)
- `flow_layers`: List of flow layers
- `log_prob_fn`: Base distribution log probability function
- `sample_fn`: Base distribution sampling function

**Methods:**

#### `forward`

Transform data to latent space.

```python
z, log_det = model.forward(
    x: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, jax.Array]
```

**Parameters:**

- `x`: Input data of shape `(batch_size, ...)`
- `rngs`: Optional random number generators

**Returns:**

- `z`: Latent representation
- `log_det`: Log-determinant of Jacobian

**Example:**

```python
import jax.numpy as jnp
from flax import nnx

# Forward transformation
x = jnp.ones((16, 64))
z, log_det = model.forward(x, rngs=rngs)

print(f"Latent shape: {z.shape}")  # (16, 64)
print(f"Log-det shape: {log_det.shape}")  # (16,)
```

#### `inverse`

Transform latent to data space.

```python
x, log_det = model.inverse(
    z: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, jax.Array]
```

**Parameters:**

- `z`: Latent code of shape `(batch_size, ...)`
- `rngs`: Optional random number generators

**Returns:**

- `x`: Reconstructed data
- `log_det`: Log-determinant of Jacobian

**Example:**

```python
# Sample from base distribution
z = jax.random.normal(rngs.sample(), (16, 64))

# Transform to data space
x, log_det = model.inverse(z, rngs=rngs)
print(f"Generated data shape: {x.shape}")  # (16, 64)
```

#### `log_prob`

Compute exact log probability of data.

```python
log_prob = model.log_prob(
    x: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> jax.Array
```

**Parameters:**

- `x`: Input data of shape `(batch_size, ...)`
- `rngs`: Optional random number generators

**Returns:**

- `log_prob`: Log probability for each sample, shape `(batch_size,)`

**Example:**

```python
# Compute log probability
x = jnp.ones((16, 64))
log_prob = model.log_prob(x, rngs=rngs)

print(f"Mean log-likelihood: {jnp.mean(log_prob):.3f}")
```

#### `generate` / `sample`

Generate samples from the model.

```python
samples = model.generate(
    n_samples: int = 1,
    *,
    rngs: nnx.Rngs | None = None,
    **kwargs
) -> jax.Array
```

**Parameters:**

- `n_samples`: Number of samples to generate
- `rngs`: Optional random number generators
- `**kwargs`: Additional keyword arguments

**Returns:**

- `samples`: Generated samples of shape `(n_samples, ...)`

**Example:**

```python
# Generate 16 samples
samples = model.generate(n_samples=16, rngs=rngs)
print(f"Samples shape: {samples.shape}")
```

#### `__call__`

Forward pass returning dictionary of outputs.

```python
outputs = model(
    x: jax.Array,
    *,
    rngs: nnx.Rngs | None = None,
    training: bool = False,
    **kwargs
) -> dict[str, Any]
```

**Returns:**

Dictionary containing:

- `z`: Latent representation
- `logdet`: Log-determinant
- `log_prob`: Log probability of data
- `log_prob_x`: Alias for `log_prob`

**Example:**

```python
outputs = model(x, rngs=rngs, training=True)
loss = -jnp.mean(outputs["log_prob"])
```

#### `loss_fn`

Compute negative log-likelihood loss.

```python
loss_dict = model.loss_fn(
    batch: Any,
    model_outputs: dict[str, Any],
    *,
    rngs: nnx.Rngs | None = None,
    **kwargs
) -> dict[str, Any]
```

**Parameters:**

- `batch`: Input batch data
- `model_outputs`: Outputs from forward pass
- `rngs`: Optional random number generators

**Returns:**

Dictionary containing:

- `loss`: Negative log-likelihood loss
- `nll_loss`: Same as `loss`
- `log_prob`: Mean log probability
- `avg_log_prob`: Same as `log_prob`

---

### `FlowLayer`

Base class for flow layer transformations.

```python
from artifex.generative_models.models.flow.base import FlowLayer
```

**Initialization:**

```python
layer = FlowLayer(
    *,
    rngs: nnx.Rngs
)
```

**Abstract Methods:**

#### `forward`

Forward transformation.

```python
y, log_det = layer.forward(
    x: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, jax.Array]
```

#### `inverse`

Inverse transformation.

```python
x, log_det = layer.inverse(
    y: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, jax.Array]
```

---

## RealNVP

### `RealNVP`

Real-valued Non-Volume Preserving flow using affine coupling layers.

```python
from artifex.generative_models.models.flow import RealNVP
```

**Initialization:**

```python
model = RealNVP(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Configuration Parameters** (in `config.parameters`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_coupling_layers` | `int` | `8` | Number of coupling transformations |
| `mask_type` | `str` | `"checkerboard"` | Masking pattern (`"checkerboard"` or `"channel-wise"`) |
| `base_distribution` | `str` | `"normal"` | Base distribution type |
| `base_distribution_params` | `dict` | `{}` | Base distribution parameters |

**Example:**

```python
from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.models.flow import RealNVP
from flax import nnx

# Configure RealNVP
config = ModelConfig(
    name="realnvp",
    model_class="artifex.generative_models.models.flow.RealNVP",
    input_dim=784,
    output_dim=784,
    hidden_dims=[512, 512],
    parameters={
        "num_coupling_layers": 8,
        "mask_type": "checkerboard",
    }
)

# Create model
rngs = nnx.Rngs(params=0, sample=1)
model = RealNVP(config, rngs=rngs)

# Use model
import jax.numpy as jnp
x = jax.random.normal(rngs.sample(), (32, 784))

# Density estimation
log_prob = model.log_prob(x, rngs=rngs)
print(f"Log probability: {jnp.mean(log_prob):.3f}")

# Generation
samples = model.generate(n_samples=16, rngs=rngs)
print(f"Generated shape: {samples.shape}")
```

**Methods:**

Inherits all methods from `NormalizingFlow` base class.

---

### `CouplingLayer`

Affine coupling layer for RealNVP.

```python
from artifex.generative_models.models.flow.real_nvp import CouplingLayer
```

**Initialization:**

```python
layer = CouplingLayer(
    mask: jax.Array,
    hidden_dims: list[int],
    scale_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
    *,
    rngs: nnx.Rngs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask` | `jax.Array` | - | Binary mask (1 = unchanged, 0 = transformed) |
| `hidden_dims` | `list[int]` | - | Hidden dimensions for scale/translation networks |
| `scale_activation` | `Callable` | `jax.nn.tanh` | Activation for scale factor |
| `rngs` | `nnx.Rngs` | - | Random number generators (required) |

**Example:**

```python
import jax.numpy as jnp

# Create alternating mask
mask = jnp.arange(64) % 2  # [0, 1, 0, 1, ...]

# Create coupling layer
layer = CouplingLayer(
    mask=mask,
    hidden_dims=[256, 256],
    rngs=rngs
)

# Forward transformation
x = jax.random.normal(rngs.sample(), (16, 64))
y, log_det = layer.forward(x, rngs=rngs)

# Inverse transformation
x_recon, log_det_inv = layer.inverse(y, rngs=rngs)

# Verify invertibility
error = jnp.max(jnp.abs(x - x_recon))
print(f"Reconstruction error: {error:.6f}")
```

---

## Glow

### `Glow`

Multi-scale flow with ActNorm, invertible 1×1 convolutions, and coupling.

```python
from artifex.generative_models.models.flow import Glow
```

**Initialization:**

```python
model = Glow(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Configuration Parameters** (in `config.parameters`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_shape` | `tuple[int, int, int]` | `(32, 32, 3)` | Input image shape (H, W, C) |
| `num_scales` | `int` | `3` | Number of multi-scale levels |
| `blocks_per_scale` | `int` | `6` | Number of flow blocks per scale |
| `hidden_dims` | `list[int]` | `[512, 512]` | Hidden dimensions for coupling |

**Example:**

```python
from artifex.generative_models.models.flow import Glow

# Configure Glow for 32x32 RGB images
config = ModelConfig(
    name="glow",
    model_class="artifex.generative_models.models.flow.Glow",
    input_dim=(32, 32, 3),
    hidden_dims=[512, 512],
    parameters={
        "image_shape": (32, 32, 3),
        "num_scales": 3,
        "blocks_per_scale": 6,
    }
)

# Create Glow model
rngs = nnx.Rngs(params=0, sample=1)
model = Glow(config, rngs=rngs)

# Training
images = jax.random.normal(rngs.sample(), (16, 32, 32, 3))
outputs = model(images, rngs=rngs)
loss = -jnp.mean(outputs["log_prob"])

# Generation
samples = model.generate(n_samples=16, rngs=rngs)
```

**Methods:**

Inherits from `NormalizingFlow` with image-specific generation.

---

### `GlowBlock`

Single Glow block: ActNorm → 1×1 Conv → Coupling.

```python
from artifex.generative_models.models.flow.glow import GlowBlock
```

**Initialization:**

```python
block = GlowBlock(
    num_channels: int,
    hidden_dims: list[int] | None = None,
    *,
    rngs: nnx.Rngs
)
```

**Parameters:**

- `num_channels`: Number of channels in input
- `hidden_dims`: Hidden dimensions for coupling layer
- `rngs`: Random number generators

**Example:**

```python
from artifex.generative_models.models.flow.glow import GlowBlock

# Create Glow block for 32-channel input
block = GlowBlock(
    num_channels=32,
    hidden_dims=[512, 512],
    rngs=rngs
)

# Forward pass
x = jax.random.normal(rngs.sample(), (16, 8, 8, 32))
y, log_det = block.forward(x, rngs=rngs)

# Inverse pass
x_recon, log_det_inv = block.inverse(y, rngs=rngs)
```

---

### `ActNormLayer`

Activation normalization with learnable scale and bias.

```python
from artifex.generative_models.models.flow.glow import ActNormLayer
```

**Initialization:**

```python
layer = ActNormLayer(
    num_channels: int,
    *,
    rngs: nnx.Rngs | None = None
)
```

**Parameters:**

- `num_channels`: Number of channels to normalize
- `rngs`: Optional random number generators

**Example:**

```python
from artifex.generative_models.models.flow.glow import ActNormLayer

# Create ActNorm layer
layer = ActNormLayer(num_channels=32, rngs=rngs)

# Forward (initializes from data on first call)
x = jax.random.normal(rngs.sample(), (16, 8, 8, 32))
y, log_det = layer.forward(x, rngs=rngs)

# After initialization, parameters are learned
print(f"Scale: {layer.logs.value.shape}")  # (1, 1, 32)
print(f"Bias: {layer.bias.value.shape}")   # (1, 1, 32)
```

**Features:**

- Data-dependent initialization on first forward pass
- Learnable per-channel scale and bias
- Efficient Jacobian computation

---

### `InvertibleConv1x1`

Invertible 1×1 convolution for channel mixing.

```python
from artifex.generative_models.models.flow.glow import InvertibleConv1x1
```

**Initialization:**

```python
layer = InvertibleConv1x1(
    num_channels: int,
    *,
    rngs: nnx.Rngs | None = None
)
```

**Parameters:**

- `num_channels`: Number of channels
- `rngs`: Optional random number generators

**Example:**

```python
from artifex.generative_models.models.flow.glow import InvertibleConv1x1

# Create invertible 1x1 conv
layer = InvertibleConv1x1(num_channels=32, rngs=rngs)

# Forward
x = jax.random.normal(rngs.sample(), (16, 8, 8, 32))
y, log_det = layer.forward(x, rngs=rngs)

# Inverse
x_recon, log_det_inv = layer.inverse(y, rngs=rngs)

# Verify invertibility
error = jnp.max(jnp.abs(x - x_recon))
print(f"Reconstruction error: {error:.6f}")  # Should be ~0
```

**Features:**

- Initialized as orthogonal matrix (via QR decomposition)
- Efficient Jacobian: $h \cdot w \cdot \log|\det(W)|$
- Fully invertible

---

### `AffineCouplingLayer`

Affine coupling layer for Glow (similar to RealNVP but channel-split).

```python
from artifex.generative_models.models.flow.glow import AffineCouplingLayer
```

**Initialization:**

```python
layer = AffineCouplingLayer(
    num_channels: int,
    hidden_dims: list[int] | None = None,
    *,
    rngs: nnx.Rngs | None = None
)
```

**Parameters:**

- `num_channels`: Number of input channels
- `hidden_dims`: Hidden dimensions for conditioning network
- `rngs`: Optional random number generators

**Example:**

```python
from artifex.generative_models.models.flow.glow import AffineCouplingLayer

# Create affine coupling layer
layer = AffineCouplingLayer(
    num_channels=32,
    hidden_dims=[512, 512],
    rngs=rngs
)

# Forward
x = jax.random.normal(rngs.sample(), (16, 8, 8, 32))
y, log_det = layer.forward(x, rngs=rngs)
```

---

## MAF (Masked Autoregressive Flow)

### `MAF`

Masked Autoregressive Flow for fast density estimation.

```python
from artifex.generative_models.models.flow import MAF
```

**Initialization:**

```python
model = MAF(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Configuration Parameters** (in `config.parameters`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | `int` | `5` | Number of MAF layers |
| `reverse_ordering` | `bool` | `True` | Alternate variable ordering between layers |

**Example:**

```python
from artifex.generative_models.models.flow import MAF

# Configure MAF
config = ModelConfig(
    name="maf",
    model_class="artifex.generative_models.models.flow.MAF",
    input_dim=64,
    output_dim=64,
    hidden_dims=[512],
    parameters={
        "num_layers": 5,
        "reverse_ordering": True,
    }
)

# Create MAF
rngs = nnx.Rngs(params=0, sample=1)
model = MAF(config, rngs=rngs)

# Fast density estimation (parallel)
x = jax.random.normal(rngs.sample(), (100, 64))
log_prob = model.log_prob(x, rngs=rngs)  # Fast!

# Slow sampling (sequential)
samples = model.sample(n_samples=10, rngs=rngs)  # Slower
```

**Trade-offs:**

- **Fast Forward**: $O(1)$ passes for density estimation
- **Slow Inverse**: $O(d)$ sequential passes for sampling
- Best for applications where density estimation is primary

---

### `MAFLayer`

Single MAF transformation layer.

```python
from artifex.generative_models.models.flow.maf import MAFLayer
```

**Initialization:**

```python
layer = MAFLayer(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    rngs: nnx.Rngs,
    order: jax.Array | None = None
)
```

**Parameters:**

- `input_dim`: Input dimension
- `hidden_dims`: Hidden dimensions for MADE network
- `rngs`: Random number generators (required)
- `order`: Variable ordering (None for natural ordering)

---

### `MADE`

Masked Autoencoder for Distribution Estimation.

```python
from artifex.generative_models.models.flow.made import MADE
```

**Initialization:**

```python
made = MADE(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_multiplier: int = 2,
    *,
    rngs: nnx.Rngs,
    order: jax.Array | None = None
)
```

**Parameters:**

- `input_dim`: Input dimension
- `hidden_dims`: Hidden layer dimensions
- `output_multiplier`: Output dim multiplier (2 for mean and scale)
- `rngs`: Random number generators (required)
- `order`: Variable ordering

**Example:**

```python
from artifex.generative_models.models.flow.made import MADE

# Create MADE network
made = MADE(
    input_dim=64,
    hidden_dims=[512, 512],
    output_multiplier=2,  # For mean and log_scale
    rngs=rngs
)

# Forward pass
x = jax.random.normal(rngs.sample(), (16, 64))
mu, log_scale = made(x)  # Returns (16, 64) for each

print(f"Mean shape: {mu.shape}")       # (16, 64)
print(f"Log scale shape: {log_scale.shape}")  # (16, 64)
```

---

## IAF (Inverse Autoregressive Flow)

### `IAF`

Inverse Autoregressive Flow for fast sampling.

```python
from artifex.generative_models.models.flow import IAF
```

**Initialization:**

```python
model = IAF(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Configuration Parameters** (in `config.parameters`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | `int` | `5` | Number of IAF layers |
| `reverse_ordering` | `bool` | `True` | Alternate variable ordering |

**Example:**

```python
from artifex.generative_models.models.flow import IAF

# Configure IAF
config = ModelConfig(
    name="iaf",
    model_class="artifex.generative_models.models.flow.IAF",
    input_dim=64,
    output_dim=64,
    hidden_dims=[512],
    parameters={
        "num_layers": 5,
        "reverse_ordering": True,
    }
)

# Create IAF
rngs = nnx.Rngs(params=0, sample=1)
model = IAF(config, rngs=rngs)

# Fast sampling (parallel)
samples = model.sample(n_samples=100, rngs=rngs)  # Fast!

# Slow density estimation (sequential)
log_prob = model.log_prob(samples, rngs=rngs)  # Slower
```

**Trade-offs:**

- **Fast Inverse**: $O(1)$ passes for sampling/generation
- **Slow Forward**: $O(d)$ sequential passes for density estimation
- Best for applications where sampling is primary (e.g., variational inference)

---

### `IAFLayer`

Single IAF transformation layer.

```python
from artifex.generative_models.models.flow.iaf import IAFLayer
```

**Initialization:**

```python
layer = IAFLayer(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    rngs: nnx.Rngs,
    order: jax.Array | None = None
)
```

**Parameters:**

- `input_dim`: Input dimension
- `hidden_dims`: Hidden dimensions for MADE network
- `rngs`: Random number generators (required)
- `order`: Variable ordering

---

## Neural Spline Flows

### `NeuralSplineFlow`

Flow using rational quadratic spline transformations.

```python
from artifex.generative_models.models.flow import NeuralSplineFlow
```

**Initialization:**

```python
model = NeuralSplineFlow(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Configuration Parameters** (in `config.metadata["flow_params"]`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | `int` | `8` | Number of spline coupling layers |
| `num_bins` | `int` | `8` | Number of spline bins/segments |
| `tail_bound` | `float` | `3.0` | Spline domain bounds |
| `base_distribution` | `str` | `"normal"` | Base distribution type |

**Example:**

```python
from artifex.generative_models.models.flow import NeuralSplineFlow

# Configure Neural Spline Flow
config = ModelConfig(
    name="spline_flow",
    model_class="artifex.generative_models.models.flow.NeuralSplineFlow",
    input_dim=64,
    hidden_dims=[128, 128],
    metadata={
        "flow_params": {
            "num_layers": 8,
            "num_bins": 8,
            "tail_bound": 3.0,
        }
    }
)

# Create model
rngs = nnx.Rngs(params=0, sample=1)
model = NeuralSplineFlow(config, rngs=rngs)

# More expressive transformations
x = jax.random.normal(rngs.sample(), (32, 64))
log_prob = model.log_prob(x, rngs=rngs)

# Generate samples
samples = model.generate(n_samples=16, rngs=rngs)
```

**Features:**

- More expressive than affine transformations
- Monotonic by construction (ensures invertibility)
- Smooth with controlled derivatives
- Bounded domain with identity outside bounds

---

### `SplineCouplingLayer`

Coupling layer with spline transformations.

```python
from artifex.generative_models.models.flow.neural_spline import SplineCouplingLayer
```

**Initialization:**

```python
layer = SplineCouplingLayer(
    mask: jax.Array,
    hidden_dims: list[int] = [128, 128],
    num_bins: int = 8,
    tail_bound: float = 3.0,
    *,
    rngs: nnx.Rngs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask` | `jax.Array` | - | Binary mask for coupling |
| `hidden_dims` | `list[int]` | `[128, 128]` | Hidden dimensions for conditioning network |
| `num_bins` | `int` | `8` | Number of spline bins |
| `tail_bound` | `float` | `3.0` | Spline domain bounds $[-b, b]$ |
| `rngs` | `nnx.Rngs` | - | Random number generators (required) |

**Example:**

```python
from artifex.generative_models.models.flow.neural_spline import SplineCouplingLayer
import jax.numpy as jnp

# Create alternating mask
mask = jnp.arange(64) % 2

# Create spline coupling layer
layer = SplineCouplingLayer(
    mask=mask,
    hidden_dims=[256, 256],
    num_bins=8,
    tail_bound=3.0,
    rngs=rngs
)

# Forward
x = jax.random.normal(rngs.sample(), (16, 64))
y, log_det = layer.forward(x, rngs=rngs)

# Inverse
x_recon, log_det_inv = layer.inverse(y, rngs=rngs)
```

---

### `RationalQuadraticSplineTransform`

Rational quadratic spline transformation for single dimension.

```python
from artifex.generative_models.models.flow.neural_spline import (
    RationalQuadraticSplineTransform
)
```

**Initialization:**

```python
transform = RationalQuadraticSplineTransform(
    num_bins: int = 8,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
    *,
    rngs: nnx.Rngs
)
```

**Parameters:**

- `num_bins`: Number of spline bins
- `tail_bound`: Domain bounds $[-b, b]$
- `min_bin_width`: Minimum bin width (for numerical stability)
- `min_bin_height`: Minimum bin height
- `min_derivative`: Minimum derivative
- `rngs`: Random number generators

**Method:**

```python
y, log_det = transform.apply_spline(
    x: jax.Array,
    widths: jax.Array,
    heights: jax.Array,
    derivatives: jax.Array,
    inverse: bool = False
) -> tuple[jax.Array, jax.Array]
```

**Example:**

```python
from artifex.generative_models.models.flow.neural_spline import (
    RationalQuadraticSplineTransform
)

# Create spline transform
spline = RationalQuadraticSplineTransform(
    num_bins=8,
    tail_bound=3.0,
    rngs=rngs
)

# Generate spline parameters (typically from neural network)
batch_size, dim = 16, 64
widths = jax.random.uniform(rngs.sample(), (batch_size, dim, 8))
heights = jax.random.uniform(rngs.sample(), (batch_size, dim, 8))
derivatives = jax.random.uniform(rngs.sample(), (batch_size, dim, 9))

# Constrain parameters
widths, heights, derivatives = spline._constrain_parameters(
    widths, heights, derivatives
)

# Apply spline transformation
x = jax.random.normal(rngs.sample(), (batch_size, dim))
y, log_det = spline.apply_spline(
    x, widths, heights, derivatives, inverse=False
)

print(f"Transformed shape: {y.shape}")  # (16, 64)
print(f"Log-det shape: {log_det.shape}")  # (16,)
```

---

## Conditional Flows

### `ConditionalNormalizingFlow`

Base class for conditional normalizing flows.

```python
from artifex.generative_models.models.flow.conditional import ConditionalNormalizingFlow
```

**Initialization:**

```python
model = ConditionalNormalizingFlow(
    config: ModelConfig,
    *,
    rngs: nnx.Rngs
)
```

**Additional Methods:**

```python
# Conditional forward pass
z, log_det = model.forward(x, condition=c, rngs=rngs)

# Conditional generation
samples = model.generate(n_samples=16, condition=c, rngs=rngs)

# Conditional log probability
log_prob = model.log_prob(x, condition=c, rngs=rngs)
```

---

### `ConditionalRealNVP`

RealNVP with conditional generation.

```python
from artifex.generative_models.models.flow.conditional import ConditionalRealNVP
```

**Configuration:**

Add `condition_dim` to parameters:

```python
config = ModelConfig(
    name="conditional_realnvp",
    model_class="artifex.generative_models.models.flow.ConditionalRealNVP",
    input_dim=784,
    output_dim=784,
    hidden_dims=[512, 512],
    parameters={
        "num_coupling_layers": 8,
        "condition_dim": 10,  # e.g., one-hot class labels
    }
)
```

**Example:**

```python
from artifex.generative_models.models.flow.conditional import ConditionalRealNVP
import jax.numpy as jnp

# Create conditional model
rngs = nnx.Rngs(params=0, sample=1)
model = ConditionalRealNVP(config, rngs=rngs)

# Prepare conditioning (e.g., class labels)
batch_size = 16
class_labels = jax.random.randint(rngs.sample(), (batch_size,), 0, 10)
condition = jax.nn.one_hot(class_labels, 10)

# Conditional density estimation
x = jax.random.normal(rngs.sample(), (batch_size, 784))
log_prob = model.log_prob(x, condition=condition, rngs=rngs)

# Conditional generation
samples = model.generate(
    n_samples=16,
    condition=condition,
    rngs=rngs
)
```

---

## Configuration Reference

### Model Configuration

All flow models use `ModelConfig` for configuration:

```python
from artifex.generative_models.core.configuration import ModelConfig

config = ModelConfig(
    name: str,                    # Model name
    model_class: str,             # Full class path
    input_dim: int | tuple,       # Input dimensions
    output_dim: int | tuple,      # Output dimensions (often same as input)
    hidden_dims: list[int],       # Hidden layer dimensions
    parameters: dict,             # Model-specific parameters
    metadata: dict = {},          # Additional metadata
)
```

### RealNVP Configuration

```python
config = ModelConfig(
    name="realnvp",
    model_class="artifex.generative_models.models.flow.RealNVP",
    input_dim=784,
    output_dim=784,
    hidden_dims=[512, 512],
    parameters={
        "num_coupling_layers": 8,
        "mask_type": "checkerboard",  # or "channel-wise"
        "base_distribution": "normal",
        "base_distribution_params": {"loc": 0.0, "scale": 1.0},
    }
)
```

### Glow Configuration

```python
config = ModelConfig(
    name="glow",
    model_class="artifex.generative_models.models.flow.Glow",
    input_dim=(32, 32, 3),
    hidden_dims=[512, 512],
    parameters={
        "image_shape": (32, 32, 3),
        "num_scales": 3,
        "blocks_per_scale": 6,
    }
)
```

### MAF Configuration

```python
config = ModelConfig(
    name="maf",
    model_class="artifex.generative_models.models.flow.MAF",
    input_dim=64,
    output_dim=64,
    hidden_dims=[512],
    parameters={
        "num_layers": 5,
        "reverse_ordering": True,
    }
)
```

### IAF Configuration

```python
config = ModelConfig(
    name="iaf",
    model_class="artifex.generative_models.models.flow.IAF",
    input_dim=64,
    output_dim=64,
    hidden_dims=[512],
    parameters={
        "num_layers": 5,
        "reverse_ordering": True,
    }
)
```

### Neural Spline Flow Configuration

```python
config = ModelConfig(
    name="spline_flow",
    model_class="artifex.generative_models.models.flow.NeuralSplineFlow",
    input_dim=64,
    hidden_dims=[128, 128],
    metadata={
        "flow_params": {
            "num_layers": 8,
            "num_bins": 8,
            "tail_bound": 3.0,
            "base_distribution": "normal",
        }
    }
)
```

---

## Common Patterns

### Training Pattern

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Create model and optimizer (wrt=nnx.Param required in NNX 0.11.0+)
model = RealNVP(config, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Training step (JIT-compiled for speed)
@nnx.jit
def train_step(model, optimizer, batch, rngs):
    def loss_fn(model):
        outputs = model(batch, rngs=rngs, training=True)
        return -jnp.mean(outputs["log_prob"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API
    return {"loss": loss}

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        metrics = train_step(model, optimizer, batch, rngs)
        print(f"Loss: {metrics['loss']:.3f}")
```

### Density Estimation Pattern

```python
# Compute log-likelihood
log_probs = model.log_prob(data, rngs=rngs)

# Average log-likelihood
avg_ll = jnp.mean(log_probs)

# Bits per dimension
input_dim = jnp.prod(jnp.array(config.input_dim))
bpd = -avg_ll / (input_dim * jnp.log(2))

print(f"Average log-likelihood: {avg_ll:.3f}")
print(f"Bits per dimension: {bpd:.3f}")
```

### Generation Pattern

```python
# Generate samples
samples = model.generate(n_samples=16, rngs=rngs)

# With temperature
z = jax.random.normal(rngs.sample(), (16, latent_dim))
z = z * temperature  # temperature > 1: more diverse
samples, _ = model.inverse(z, rngs=rngs)
```

### Anomaly Detection Pattern

```python
# Compute log-likelihood for training data
train_log_probs = model.log_prob(train_data, rngs=rngs)

# Set threshold (e.g., 5th percentile)
threshold = jnp.percentile(train_log_probs, 5)

# Detect anomalies in test data
test_log_probs = model.log_prob(test_data, rngs=rngs)
is_anomaly = test_log_probs < threshold

print(f"Detected {jnp.sum(is_anomaly)} anomalies")
```

---

## Quick Reference

### Architecture Comparison

| Model | Forward | Inverse | Use Case |
|-------|---------|---------|----------|
| **RealNVP** | Fast | Fast | Balanced, general purpose |
| **Glow** | Fast | Fast | High-quality images |
| **MAF** | Fast | Slow | Density estimation |
| **IAF** | Slow | Fast | Fast sampling, VI |
| **Spline** | Fast | Fast | High expressiveness |

### Common Workflows

**Density Estimation:**

```python
model = MAF(config, rngs=rngs)
log_prob = model.log_prob(data, rngs=rngs)
```

**Fast Sampling:**

```python
model = IAF(config, rngs=rngs)
samples = model.generate(n_samples=100, rngs=rngs)
```

**Image Generation:**

```python
model = Glow(config, rngs=rngs)
samples = model.generate(n_samples=16, rngs=rngs)
```

**High Expressiveness:**

```python
model = NeuralSplineFlow(config, rngs=rngs)
log_prob = model.log_prob(data, rngs=rngs)
```

---

## See Also

- **User Guide**: [Flow Models Guide](../../user-guide/models/flow-guide.md) for practical examples
- **Concepts**: [Flow Explained](../../user-guide/concepts/flow-explained.md) for theory
- **Tutorial**: [Flow MNIST Example](../../examples/basic/flow-mnist.md) for hands-on learning

---

## References

- Dinh et al. (2016): "Density estimation using Real NVP"
- Kingma & Dhariwal (2018): "Glow: Generative Flow with Invertible 1x1 Convolutions"
- Papamakarios et al. (2017): "Masked Autoregressive Flow for Density Estimation"
- Kingma et al. (2016): "Improved Variational Inference with Inverse Autoregressive Flow"
- Durkan et al. (2019): "Neural Spline Flows"
