# Flow Models: Practical User Guide

This guide provides practical instructions for working with normalizing flow models in Artifex. We cover creating, training, and using various flow architectures for density estimation and generation.

## Quick Start

Here's a minimal example to get started with RealNVP:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.flow_config import (
    RealNVPConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import RealNVP

# Create RNG streams
rngs = nnx.Rngs(params=0, noise=1, sample=2, dropout=3)

# Configure the coupling network
coupling_config = CouplingNetworkConfig(
    name="coupling_mlp",
    hidden_dims=(256, 256),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Configure RealNVP
config = RealNVPConfig(
    name="realnvp_mnist",
    coupling_network=coupling_config,
    input_dim=784,              # MNIST flattened
    base_distribution="normal",
    num_coupling_layers=6,
    mask_type="checkerboard",
)

# Create model
model = RealNVP(config, rngs=rngs)

# Forward pass (data → latent)
batch = jax.random.normal(jax.random.key(0), (32, 784))
z, log_det = model.forward(batch)

# Compute log probability
log_prob = model.log_prob(batch)
loss = -jnp.mean(log_prob)

# Generate samples (latent → data)
model.eval()
samples = model.generate(n_samples=16)
```

## Creating Flow Models

Artifex provides multiple flow architectures. Choose based on your needs (see [Flow Concepts](../concepts/flow-explained.md) for detailed comparison).

### RealNVP (Recommended for Most Tasks)

**RealNVP** offers a good balance between performance and computational cost.

```python
from artifex.generative_models.core.configuration.flow_config import (
    RealNVPConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import RealNVP
from flax import nnx
import jax
import jax.numpy as jnp

# Create RNGs
rngs = nnx.Rngs(params=0, noise=1, sample=2, dropout=3)

# Configure coupling network
coupling_config = CouplingNetworkConfig(
    name="coupling_mlp",
    hidden_dims=(256, 256),      # Coupling network hidden layers
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Configure RealNVP
config = RealNVPConfig(
    name="realnvp_flow",
    coupling_network=coupling_config,
    input_dim=64,                # Feature dimension
    base_distribution="normal",
    num_coupling_layers=8,       # Number of coupling transformations
    mask_type="checkerboard",    # or "channel-wise" for images
)

# Create model
model = RealNVP(config, rngs=rngs)

# Forward pass (data to latent)
x = jax.random.normal(jax.random.key(0), (32, 64))
z, log_det = model.forward(x)

# Inverse pass (latent to data)
samples, _ = model.inverse(z)

# Compute log probability
log_prob = model.log_prob(x)
print(f"Log probability: {jnp.mean(log_prob):.3f}")
```

**Mask Types:**

- `"checkerboard"`: Alternates dimensions (good for tabular data)
- `"channel-wise"`: Splits along channels (better for images)

```python
# For image data (flattened), use larger coupling network
coupling_config_image = CouplingNetworkConfig(
    name="coupling_image",
    hidden_dims=(512, 512),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

config_image = RealNVPConfig(
    name="realnvp_image",
    coupling_network=coupling_config_image,
    input_dim=784,                  # MNIST flattened (28*28*1)
    base_distribution="normal",
    num_coupling_layers=12,
    mask_type="checkerboard",
)
```

### Glow (High-Quality Image Generation)

**Glow** uses a multi-scale architecture with ActNorm, invertible 1×1 convolutions, and coupling layers.

```python
from artifex.generative_models.core.configuration.flow_config import (
    GlowConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import Glow

# Configure coupling network for Glow
coupling_config = CouplingNetworkConfig(
    name="glow_coupling",
    hidden_dims=(512, 512),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Configure Glow
config = GlowConfig(
    name="glow_model",
    coupling_network=coupling_config,
    input_dim=3072,                  # 32*32*3 flattened
    base_distribution="normal",
    image_shape=(32, 32, 3),         # Image shape for multi-scale
    num_scales=3,                    # Multi-scale architecture
    blocks_per_scale=6,              # Flow steps per scale
)

# Create Glow model
rngs = nnx.Rngs(params=0, sample=1)
model = Glow(config, rngs=rngs)

# Training
images = jax.random.normal(rngs.sample(), (16, 32, 32, 3))
outputs = model(images, rngs=rngs)
loss = -jnp.mean(outputs["log_prob"])

# Generate high-quality samples
samples = model.generate(n_samples=16, rngs=rngs)
```

**Glow Architecture Parameters:**

- `num_scales`: Number of multi-scale levels (typically 2-4)
- `blocks_per_scale`: Flow steps at each scale (typically 4-8)
- Higher values = more expressive but slower

### MAF (Fast Density Estimation)

**MAF** (Masked Autoregressive Flow) excels at density estimation but has slow sampling.

```python
from artifex.generative_models.core.configuration.flow_config import (
    MAFConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import MAF

# Configure coupling network for MAF
coupling_config = CouplingNetworkConfig(
    name="maf_coupling",
    hidden_dims=(512,),              # MADE hidden dimensions
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Configure MAF
config = MAFConfig(
    name="maf_model",
    coupling_network=coupling_config,
    input_dim=64,
    base_distribution="normal",
    num_layers=5,                    # Number of MAF layers
    reverse_ordering=True,           # Alternate variable ordering
)

# Create MAF model
rngs = nnx.Rngs(params=0, sample=1)
model = MAF(config, rngs=rngs)

# Fast forward pass (density estimation)
x = jax.random.normal(rngs.sample(), (100, 64))
log_prob = model.log_prob(x, rngs=rngs)
print(f"Mean log-likelihood: {jnp.mean(log_prob):.3f}")

# Slow inverse pass (sampling)
samples = model.sample(n_samples=10, rngs=rngs)  # Sequential, slower
```

**When to Use MAF:**

- Primary goal is density estimation or anomaly detection
- Sampling speed is not critical
- Working with tabular or low-dimensional data
- Need high-quality likelihood estimates

### IAF (Fast Sampling)

**IAF** (Inverse Autoregressive Flow) provides fast sampling at the cost of slow density estimation.

```python
from artifex.generative_models.core.configuration.flow_config import (
    IAFConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import IAF

# Configure coupling network for IAF
coupling_config = CouplingNetworkConfig(
    name="iaf_coupling",
    hidden_dims=(512,),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Configure IAF
config = IAFConfig(
    name="iaf_model",
    coupling_network=coupling_config,
    input_dim=64,
    base_distribution="normal",
    num_layers=5,
    reverse_ordering=True,
)

# Create IAF model
rngs = nnx.Rngs(params=0, sample=1)
model = IAF(config, rngs=rngs)

# Fast sampling (parallel computation)
samples = model.sample(n_samples=100, rngs=rngs)  # Fast!

# Slow density estimation (sequential)
log_prob = model.log_prob(samples, rngs=rngs)  # Slower
```

**When to Use IAF:**

- Fast sampling is critical (real-time generation)
- Often used as variational posterior in VAEs
- Density estimation is secondary
- Generation frequency >> inference frequency

### Neural Spline Flows (Most Expressive)

**Neural Spline Flows** use rational quadratic splines for highly expressive transformations.

```python
from artifex.generative_models.core.configuration.flow_config import (
    NeuralSplineConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import NeuralSplineFlow

# Configure coupling network for spline flow
coupling_config = CouplingNetworkConfig(
    name="spline_coupling",
    hidden_dims=(128, 128),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Configure Neural Spline Flow
config = NeuralSplineConfig(
    name="spline_flow",
    coupling_network=coupling_config,
    input_dim=64,
    base_distribution="normal",
    num_layers=8,
    num_bins=8,                      # Number of spline segments
    tail_bound=3.0,                  # Spline domain bounds
)

# Create Neural Spline Flow
rngs = nnx.Rngs(params=0, sample=1)
model = NeuralSplineFlow(config, rngs=rngs)

# More expressive transformations
x = jax.random.normal(rngs.sample(), (32, 64))
log_prob = model.log_prob(x, rngs=rngs)

# Generate samples
samples = model.generate(n_samples=16, rngs=rngs)
```

**Spline Parameters:**

- `num_bins`: Number of spline segments (8-16 typical)
- `tail_bound`: Domain bounds for spline (3.0-5.0 typical)
- More bins = more expressive but higher memory cost

## Training Flow Models

### Basic Training Loop

Flow models are trained using maximum likelihood estimation.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.models.flow import RealNVP

# Initialize model and optimizer
rngs = nnx.Rngs(params=0, dropout=1, sample=2)
model = RealNVP(config, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-4), wrt=nnx.Param)

# Training step function
@nnx.jit
def train_step(model, optimizer, batch, rngs):
    """Single training step."""
    def loss_fn(model):
        # Forward pass through flow
        outputs = model(batch, rngs=rngs, training=True)

        # Negative log-likelihood loss
        log_prob = outputs["log_prob"]
        loss = -jnp.mean(log_prob)

        return loss, {"nll": loss, "mean_log_prob": jnp.mean(log_prob)}

    # Compute loss and gradients
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update parameters (NNX 0.11.0+ API)
    optimizer.update(model, grads)

    return metrics

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    epoch_metrics = []

    for batch in train_dataloader:
        # Preprocess: add uniform noise for dequantization
        batch = batch + jax.random.uniform(rngs.sample(), batch.shape) / 256.0

        # Scale to appropriate range
        batch = (batch - 0.5) / 0.5  # Scale to [-1, 1]

        # Training step
        metrics = train_step(model, optimizer, batch, rngs)
        epoch_metrics.append(metrics)

    # Log epoch statistics
    avg_nll = jnp.mean(jnp.array([m["nll"] for m in epoch_metrics]))
    print(f"Epoch {epoch+1}/{num_epochs}, NLL: {avg_nll:.3f}")
```

### Training with Gradient Clipping

Gradient clipping helps stabilize flow training:

```python
import optax

# Create optimizer with gradient clipping
optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients
    optax.adam(learning_rate=1e-4),
)

optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

# Training step with clipping
@nnx.jit
def train_step_clipped(model, optimizer, batch, rngs):
    def loss_fn(model):
        outputs = model(batch, rngs=rngs, training=True)
        loss = -jnp.mean(outputs["log_prob"])
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Optimizer applies gradient clipping automatically (NNX 0.11.0+ API)
    optimizer.update(model, grads)

    return {"loss": loss}
```

### Learning Rate Scheduling

Use learning rate warmup and decay for better convergence:

```python
import optax

# Learning rate schedule
warmup_steps = 1000
total_steps = 50000

schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-7,
    peak_value=1e-4,
    warmup_steps=warmup_steps,
    decay_steps=total_steps - warmup_steps,
    end_value=1e-6,
)

# Create optimizer with schedule (wrt=nnx.Param required in NNX 0.11.0+)
optimizer = nnx.Optimizer(
    model,
    optax.adam(learning_rate=schedule),
    wrt=nnx.Param
)

# Track global step
global_step = 0

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        metrics = train_step(model, optimizer, batch, rngs)
        global_step += 1

        # Learning rate automatically updated by optax
```

### Monitoring Training

Track important metrics during training:

```python
# Training with metrics tracking
@nnx.jit
def train_step_with_metrics(model, optimizer, batch, rngs):
    def loss_fn(model):
        # Forward pass
        z, log_det = model.forward(batch, rngs=rngs)

        # Base distribution log prob
        log_p_z = -0.5 * jnp.sum(z**2, axis=-1) - 0.5 * z.shape[-1] * jnp.log(2 * jnp.pi)

        # Total log probability
        log_prob = log_p_z + log_det

        # Loss
        loss = -jnp.mean(log_prob)

        # Additional metrics
        metrics = {
            "loss": loss,
            "log_p_z": jnp.mean(log_p_z),
            "log_det": jnp.mean(log_det),
            "log_prob": jnp.mean(log_prob),
            "z_norm": jnp.mean(jnp.linalg.norm(z, axis=-1)),
        }

        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    return metrics

# Training loop with logging
for epoch in range(num_epochs):
    metrics_list = []

    for batch in train_dataloader:
        batch = preprocess(batch)
        metrics = train_step_with_metrics(model, optimizer, batch, rngs)
        metrics_list.append(metrics)

    # Aggregate epoch metrics
    epoch_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
        for k in metrics_list[0].keys()
    }

    print(f"Epoch {epoch+1}: {epoch_metrics}")
```

**Important Metrics to Monitor:**

- `loss`: Negative log-likelihood (should decrease)
- `log_p_z`: Base distribution log-prob (should be near 0 for standard Gaussian)
- `log_det`: Jacobian log-determinant (tracks transformation magnitude)
- `z_norm`: Latent norm (should be near √d for d-dimensional Gaussian)

### Data Preprocessing for Flows

Proper preprocessing is crucial for flow models:

#### Dequantization

Images are discrete (0-255), but flows need continuous values:

```python
def dequantize(images, rngs):
    """Add uniform noise to dequantize discrete images."""
    # images should be in [0, 1] range
    noise = jax.random.uniform(rngs.sample(), images.shape)
    return images + noise / 256.0

# Apply during training
batch = dequantize(batch, rngs)
```

#### Logit Transform

Map bounded data to unbounded space:

```python
def logit_transform(x, alpha=0.05):
    """Apply logit transform with boundary handling."""
    # Squeeze to (alpha, 1-alpha) to avoid infinities
    x = alpha + (1 - 2*alpha) * x

    # Apply logit
    return jnp.log(x) - jnp.log(1 - x)

def inverse_logit_transform(y, alpha=0.05):
    """Inverse of logit transform."""
    x = jax.nn.sigmoid(y)
    return (x - alpha) / (1 - 2*alpha)

# Apply during training
batch = logit_transform(batch)
```

#### Normalization

Standardize data to zero mean and unit variance:

```python
# Compute statistics on training data
train_mean = jnp.mean(train_data, axis=0, keepdims=True)
train_std = jnp.std(train_data, axis=0, keepdims=True)

# Normalize
batch = (batch - train_mean) / (train_std + 1e-6)

# Remember to denormalize samples
samples = model.generate(n_samples=16, rngs=rngs)
samples = samples * train_std + train_mean
```

## Sampling and Generation

### Basic Sampling

Generate samples from a trained flow model:

```python
# Generate samples
n_samples = 16
samples = model.generate(n_samples=n_samples, rngs=rngs)

# For image data, reshape from flattened to spatial dimensions
# input_dim is always an int (e.g., 784 for MNIST)
H, W, C = 28, 28, 1
images = samples.reshape(n_samples, H, W, C)

# Denormalize for visualization
images = (images * 0.5) + 0.5  # From [-1, 1] to [0, 1]
images = jnp.clip(images, 0, 1)
```

### Temperature Sampling

Control sample diversity with temperature:

```python
def sample_with_temperature(model, n_samples, temperature, rngs):
    """Sample with temperature scaling.

    temperature > 1: More diverse samples
    temperature < 1: More conservative samples
    temperature = 1: Standard sampling
    """
    # Sample from base distribution
    z = jax.random.normal(rngs.sample(), (n_samples, model.latent_dim))

    # Scale by temperature
    z = z * temperature

    # Transform to data space
    samples, _ = model.inverse(z, rngs=rngs)

    return samples

# Conservative samples (sharper, less diverse)
conservative = sample_with_temperature(model, 16, temperature=0.7, rngs=rngs)

# Diverse samples (more variety, less sharp)
diverse = sample_with_temperature(model, 16, temperature=1.3, rngs=rngs)
```

### Conditional Sampling

Some flow architectures support conditional generation:

```python
# For conditional flows
from artifex.generative_models.core.configuration.flow_config import (
    RealNVPConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow import ConditionalRealNVP

# Configure coupling network
coupling_config = CouplingNetworkConfig(
    name="cond_coupling",
    hidden_dims=(512, 512),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# Create conditional model using RealNVPConfig
# ConditionalRealNVP extracts condition_dim via getattr with default=10
config = RealNVPConfig(
    name="conditional_realnvp",
    coupling_network=coupling_config,
    input_dim=784,
    base_distribution="normal",
    num_coupling_layers=8,
    mask_type="checkerboard",
)

model = ConditionalRealNVP(config, rngs=rngs)

# Sample conditioned on class labels
class_labels = jax.nn.one_hot(jnp.array([0, 1, 2]), 10)  # 3 classes
conditional_samples = model.generate(
    n_samples=3,
    condition=class_labels,
    rngs=rngs
)
```

### Interpolation in Latent Space

Interpolate between two data points:

```python
def interpolate(model, x1, x2, num_steps, rngs):
    """Interpolate between two data points in latent space."""
    # Encode to latent space
    z1, _ = model.forward(x1[None, ...], rngs=rngs)
    z2, _ = model.forward(x2[None, ...], rngs=rngs)

    # Linear interpolation in latent space
    alphas = jnp.linspace(0, 1, num_steps)
    z_interp = jnp.array([
        (1 - alpha) * z1 + alpha * z2
        for alpha in alphas
    ]).squeeze(1)

    # Decode to data space
    x_interp, _ = model.inverse(z_interp, rngs=rngs)

    return x_interp

# Interpolate between two images
x1 = train_data[0]  # First image
x2 = train_data[1]  # Second image
interpolations = interpolate(model, x1, x2, num_steps=10, rngs=rngs)
```

## Density Estimation and Evaluation

### Computing Log-Likelihood

Flow models provide exact log-likelihood:

```python
# Compute log-likelihood for test data
test_data = ...  # Your test dataset

log_likelihoods = []
for batch in test_dataloader:
    # Preprocess same as training
    batch = dequantize(batch, rngs)
    batch = (batch - 0.5) / 0.5

    # Compute log probability
    log_prob = model.log_prob(batch, rngs=rngs)
    log_likelihoods.append(log_prob)

# Average log-likelihood
all_log_probs = jnp.concatenate(log_likelihoods)
avg_log_likelihood = jnp.mean(all_log_probs)
print(f"Test log-likelihood: {avg_log_likelihood:.3f}")

# Bits per dimension (common metric)
input_dim = config.input_dim  # int, e.g., 784 for MNIST
bits_per_dim = -avg_log_likelihood / (input_dim * jnp.log(2))
print(f"Bits per dimension: {bits_per_dim:.3f}")
```

### Anomaly Detection

Use log-likelihood for anomaly detection:

```python
def detect_anomalies(model, data, threshold, rngs):
    """Detect anomalies using log-likelihood threshold."""
    # Compute log probabilities
    log_probs = model.log_prob(data, rngs=rngs)

    # Flag samples below threshold as anomalies
    is_anomaly = log_probs < threshold

    return is_anomaly, log_probs

# Set threshold (e.g., 5th percentile of training data)
train_log_probs = model.log_prob(train_data, rngs=rngs)
threshold = jnp.percentile(train_log_probs, 5)

# Detect anomalies in test data
anomalies, test_log_probs = detect_anomalies(
    model, test_data, threshold, rngs
)

print(f"Detected {jnp.sum(anomalies)} anomalies out of {len(test_data)} samples")
```

### Model Comparison

Compare different flow architectures using likelihood:

```python
# Train multiple models
models = {
    "RealNVP": realnvp_model,
    "Glow": glow_model,
    "MAF": maf_model,
    "Spline": spline_model,
}

# Evaluate on test set
results = {}
for name, model in models.items():
    log_probs = []
    for batch in test_dataloader:
        batch = preprocess(batch)
        log_prob = model.log_prob(batch, rngs=rngs)
        log_probs.append(log_prob)

    avg_log_prob = jnp.mean(jnp.concatenate(log_probs))
    results[name] = avg_log_prob

    print(f"{name}: {avg_log_prob:.3f} (higher is better)")

# Best model
best_model = max(results, key=results.get)
print(f"Best model: {best_model}")
```

## Common Patterns

### Multi-Modal Data Distribution

For data with multiple modes, increase model capacity:

```python
from artifex.generative_models.core.configuration.flow_config import (
    RealNVPConfig,
    NeuralSplineConfig,
    CouplingNetworkConfig,
)

# Increase number of layers
coupling_deep = CouplingNetworkConfig(
    name="deep_coupling",
    hidden_dims=(1024, 1024, 1024),  # Deeper networks
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

config = RealNVPConfig(
    name="multimodal_flow",
    coupling_network=coupling_deep,
    input_dim=64,
    base_distribution="normal",
    num_coupling_layers=16,          # More layers
    mask_type="checkerboard",
)

# Or use Neural Spline Flows for higher expressiveness
coupling_spline = CouplingNetworkConfig(
    name="spline_coupling",
    hidden_dims=(256, 256),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

config_spline = NeuralSplineConfig(
    name="multimodal_spline",
    coupling_network=coupling_spline,
    input_dim=64,
    base_distribution="normal",
    num_layers=12,
    num_bins=16,                     # More bins for expressiveness
    tail_bound=3.0,
)
```

### High-Dimensional Data

For very high-dimensional data (e.g., high-resolution images):

```python
from artifex.generative_models.core.configuration.flow_config import (
    GlowConfig,
    CouplingNetworkConfig,
)

# Use Glow with multi-scale architecture
coupling_config = CouplingNetworkConfig(
    name="glow_highres_coupling",
    hidden_dims=(512, 512),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

config = GlowConfig(
    name="glow_highres",
    coupling_network=coupling_config,
    input_dim=12288,                 # 64*64*3 flattened
    base_distribution="normal",
    image_shape=(64, 64, 3),
    num_scales=4,                    # More scales for higher resolution
    blocks_per_scale=8,
)

# Reduce memory by processing in patches
def train_on_patches(model, image, patch_size=32):
    """Train on image patches to reduce memory."""
    H, W, C = image.shape
    patches = []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)

    patches = jnp.array(patches)
    return model(patches, rngs=rngs)
```

### Tabular Data with Mixed Types

For tabular data with continuous and categorical features:

```python
# Preprocess mixed types
def preprocess_tabular(data, categorical_indices):
    """Preprocess tabular data with mixed types."""
    continuous = data.copy()

    # One-hot encode categorical features
    for idx in categorical_indices:
        # One-hot encode
        n_categories = int(jnp.max(data[:, idx])) + 1
        one_hot = jax.nn.one_hot(data[:, idx].astype(int), n_categories)

        # Replace categorical column with one-hot
        continuous = jnp.concatenate([
            continuous[:, :idx],
            one_hot,
            continuous[:, idx+1:],
        ], axis=1)

    return continuous

# Use MAF for tabular data (good density estimation)
from artifex.generative_models.core.configuration.flow_config import (
    MAFConfig,
    CouplingNetworkConfig,
)

coupling_config = CouplingNetworkConfig(
    name="tabular_coupling",
    hidden_dims=(512, 512),
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

config = MAFConfig(
    name="tabular_maf",
    coupling_network=coupling_config,
    input_dim=processed_dim,
    base_distribution="normal",
    num_layers=10,                   # More layers for complex dependencies
)
```

### Exact Reconstruction

Verify model invertibility:

```python
def test_invertibility(model, x, rngs, tolerance=1e-4):
    """Test that forward and inverse are true inverses."""
    # Forward then inverse
    z, _ = model.forward(x, rngs=rngs)
    x_recon, _ = model.inverse(z, rngs=rngs)

    # Compute reconstruction error
    error = jnp.max(jnp.abs(x - x_recon))

    print(f"Max reconstruction error: {error:.6f}")
    assert error < tolerance, f"Reconstruction error {error} exceeds tolerance {tolerance}"

# Test on random data
x = jax.random.normal(rngs.sample(), (10, 64))
test_invertibility(model, x, rngs)
```

## Troubleshooting

### Issue: NaN Loss During Training

**Symptoms**: Loss becomes NaN after a few iterations.

**Solutions**:

1. **Add gradient clipping**:

```python
optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    ),
    wrt=nnx.Param
)
```

2. **Check data preprocessing**:

```python
# Ensure data is properly scaled
assert jnp.all(jnp.isfinite(batch)), "Data contains NaN or Inf"
assert jnp.abs(jnp.mean(batch)) < 10, "Data not properly normalized"
```

3. **Reduce learning rate**:

```python
optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)  # Lower LR
```

4. **Check Jacobian stability**:

```python
# Monitor log-determinant magnitude
z, log_det = model.forward(batch, rngs=rngs)
print(f"Log-det range: [{jnp.min(log_det):.2f}, {jnp.max(log_det):.2f}]")

# If log_det has extreme values, reduce model capacity or layers
```

### Issue: Poor Sample Quality

**Symptoms**: Generated samples look noisy or unrealistic.

**Solutions**:

1. **Increase model capacity**:

```python
coupling_config = CouplingNetworkConfig(
    name="larger_coupling",
    hidden_dims=(1024, 1024),        # Larger networks
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

config = RealNVPConfig(
    name="larger_model",
    coupling_network=coupling_config,
    input_dim=784,
    base_distribution="normal",
    num_coupling_layers=16,          # More layers
    mask_type="checkerboard",
)
```

2. **Use more expressive architecture**:

```python
# Switch from RealNVP to Neural Spline Flows
model = NeuralSplineFlow(spline_config, rngs=rngs)
```

3. **Improve data preprocessing**:

```python
# Apply logit transform for bounded data
batch = logit_transform(batch)
```

4. **Train longer**:

```python
num_epochs = 200  # More epochs
# Monitor validation likelihood to check for convergence
```

### Issue: Slow Training

**Symptoms**: Training takes too long per iteration.

**Solutions**:

1. **Use JIT compilation**:

```python
@nnx.jit
def train_step(model, optimizer, batch, rngs):
    # ... training step code
    pass
```

2. **Reduce model complexity**:

```python
# Fewer coupling layers in RealNVPConfig
num_coupling_layers=6  # Instead of 16

# Smaller hidden dimensions in CouplingNetworkConfig
hidden_dims=(256, 256)  # Instead of (1024, 1024)
```

3. **Use IAF for fast sampling** (if sampling is the bottleneck):

```python
# IAF has fast sampling
model = IAF(config, rngs=rngs)
```

4. **Batch processing**:

```python
# Increase batch size (if memory allows)
batch_size = 128  # Instead of 32
```

### Issue: Mode Collapse

**Symptoms**: Model generates similar samples repeatedly.

**Solutions**:

1. **Check latent space coverage**:

```python
# Generate many samples and check latent codes
samples = model.generate(n_samples=1000, rngs=rngs)
z_samples, _ = model.forward(samples, rngs=rngs)

# Check if latents cover the expected distribution
z_mean = jnp.mean(z_samples, axis=0)
z_std = jnp.std(z_samples, axis=0)

print(f"Latent mean: {jnp.mean(z_mean):.3f} (should be ~0)")
print(f"Latent std: {jnp.mean(z_std):.3f} (should be ~1)")
```

2. **Increase model expressiveness**:

```python
# Use Neural Spline Flows
# Or increase number of flow layers
```

3. **Check for numerical issues**:

```python
# Ensure stable training
# Use gradient clipping and proper LR
```

### Issue: Memory Errors

**Symptoms**: Out of memory errors during training.

**Solutions**:

1. **Reduce batch size**:

```python
batch_size = 16  # Smaller batches
```

2. **Use gradient checkpointing** (if available):

```python
# Recompute intermediate activations during backward pass
# (implementation-specific)
```

3. **Reduce model size**:

```python
# Fewer layers or smaller hidden dimensions
# In CouplingNetworkConfig: hidden_dims=(256, 256)
# In RealNVPConfig: num_coupling_layers=6
```

4. **Use mixed precision training**:

```python
# Use float16 for some computations (implementation-specific)
```

## Best Practices

### DO

✅ **Preprocess data properly**:

```python
# Always dequantize discrete data
batch = dequantize(batch, rngs)

# Normalize to appropriate range
batch = (batch - 0.5) / 0.5
```

✅ **Monitor multiple metrics**:

```python
# Track loss, log_det, base log prob
metrics = {
    "loss": loss,
    "log_det": jnp.mean(log_det),
    "log_p_z": jnp.mean(log_p_z),
}
```

✅ **Use gradient clipping**:

```python
optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    ),
    wrt=nnx.Param
)
```

✅ **Validate invertibility**:

```python
# Periodically test reconstruction
test_invertibility(model, validation_batch, rngs)
```

✅ **Choose architecture for your task**:

```python
# MAF for density estimation
# IAF for fast sampling
# RealNVP for balance
# Glow for high-quality images
# Spline Flows for expressiveness
```

### DON'T

❌ **Don't skip data preprocessing**:

```python
# BAD: Using raw discrete images
model(raw_images, rngs=rngs)  # Will perform poorly!

# GOOD: Dequantize and normalize
processed = dequantize(raw_images, rngs)
processed = (processed - 0.5) / 0.5
model(processed, rngs=rngs)
```

❌ **Don't ignore numerical stability**:

```python
# BAD: No gradient clipping
# Can lead to NaN losses

# GOOD: Use gradient clipping (wrt=nnx.Param required in NNX 0.11.0+)
optimizer = nnx.Optimizer(
    model,
    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4)),
    wrt=nnx.Param
)
```

❌ **Don't use wrong architecture for task**:

```python
# BAD: Using IAF when density estimation is primary goal
# IAF has slow forward pass

# GOOD: Use MAF for density estimation
model = MAF(config, rngs=rngs)
```

❌ **Don't overtrain on small datasets**:

```python
# Monitor validation likelihood
# Use early stopping
```

## Hyperparameter Guidelines

### Number of Flow Layers

| Data Type | Recommended Layers | Notes |
|-----------|-------------------|-------|
| Tabular (low-dim) | 6-10 | Simpler distributions |
| Tabular (high-dim) | 10-16 | More complex dependencies |
| Images (low-res) | 8-12 | MNIST, CIFAR-10 |
| Images (high-res) | 12-24 | Higher resolution |

### Hidden Dimensions

| Model | Recommended | Notes |
|-------|------------|-------|
| RealNVP | [512, 512] | 2-3 layers sufficient |
| Glow | [512, 512] | Larger for high-res |
| MAF/IAF | [512] - [1024] | Single deep network |
| Spline | [128, 128] | Splines are expressive |

### Learning Rates

| Stage | Learning Rate | Notes |
|-------|--------------|-------|
| Warmup | 1e-7 → 1e-4 | First 1000 steps |
| Training | 1e-4 | Standard rate |
| Fine-tuning | 1e-5 | Near convergence |

### Batch Sizes

| Data Type | Batch Size | Notes |
|-----------|-----------|-------|
| Tabular | 256-1024 | Can use large batches |
| Images (32×32) | 64-128 | Memory dependent |
| Images (64×64) | 32-64 | Reduce for Glow |
| Images (128×128) | 16-32 | Limited by memory |

## Summary

### Quick Reference

**Model Selection**:

- **RealNVP**: Balanced performance, good for most tasks
- **Glow**: Best for high-quality image generation
- **MAF**: Optimal for density estimation
- **IAF**: Optimal for fast sampling
- **Spline Flows**: Most expressive transformations

**Training Checklist**:

1. ✅ Preprocess data (dequantize, normalize)
2. ✅ Use gradient clipping
3. ✅ Monitor multiple metrics
4. ✅ Validate invertibility
5. ✅ Apply learning rate warmup
6. ✅ Check for NaN/Inf values

**Common Workflows**:

```python
# Density estimation workflow
model = MAF(config, rngs=rngs)
log_probs = model.log_prob(data, rngs=rngs)

# Generation workflow
model = RealNVP(config, rngs=rngs)
samples = model.generate(n_samples=16, rngs=rngs)

# Anomaly detection workflow
model = MAF(config, rngs=rngs)
threshold = jnp.percentile(train_log_probs, 5)
anomalies = test_log_probs < threshold
```

## Next Steps

- **Theory**: See [Flow Concepts](../concepts/flow-explained.md) for mathematical foundations
- **API Reference**: Check [Flow API](../../api/models/flow.md) for complete documentation
- **Tutorial**: Follow [Flow MNIST Example](../../examples/basic/flow-mnist.md) for hands-on practice

## References

- Dinh et al. (2016): "Density estimation using Real NVP"
- Kingma & Dhariwal (2018): "Glow: Generative Flow with Invertible 1x1 Convolutions"
- Papamakarios et al. (2017): "Masked Autoregressive Flow for Density Estimation"
- Durkan et al. (2019): "Neural Spline Flows"
