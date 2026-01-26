# VAE User Guide

Complete guide to building, training, and using Variational Autoencoders with Artifex.

## Overview

This guide covers practical usage of VAEs in Artifex, from basic setup to advanced techniques. You'll learn how to:

<div class="grid cards" markdown>

- :material-cog: **Configure VAEs**

    ---

    Set up encoder/decoder architectures and configure hyperparameters

- :material-play: **Train Models**

    ---

    Train VAEs with proper loss functions and monitoring

- :material-creation: **Generate Samples**

    ---

    Sample from the prior and manipulate latent representations

- :material-tune: **Tune & Debug**

    ---

    Optimize hyperparameters and troubleshoot common issues

</div>

---

## Quick Start

### Basic VAE Example

```python
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.core.configuration.network_configs import (
    EncoderConfig,
    DecoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import VAE

# Initialize RNGs
rngs = nnx.Rngs(params=0, dropout=1, sample=2)

# Configuration
latent_dim = 20

# Create encoder config
encoder_config = EncoderConfig(
    name="mlp_encoder",
    hidden_dims=(256, 128),
    latent_dim=latent_dim,
    activation="relu",
    input_shape=(28, 28, 1),  # Image shape
)

# Create decoder config
decoder_config = DecoderConfig(
    name="mlp_decoder",
    hidden_dims=(128, 256),
    output_shape=(28, 28, 1),
    latent_dim=latent_dim,
    activation="relu",
)

# Create VAE config
vae_config = VAEConfig(
    name="basic_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",  # Use MLP encoder/decoder
    kl_weight=1.0,
)

# Create model
vae = VAE(config=vae_config, rngs=rngs)

# Forward pass
x = jnp.ones((32, 28, 28, 1))
outputs = vae(x)  # Model uses internal RNGs

# Get outputs
reconstructed = outputs["reconstructed"]
mean = outputs["mean"]
log_var = outputs["log_var"]
latent = outputs["z"]

print(f"Reconstruction shape: {reconstructed.shape}")
print(f"Latent shape: {latent.shape}")
```

---

## Creating VAE Models

Artifex uses a **config-based API** where you define configurations first, then create models from them. This provides type safety, validation, and easy serialization.

### 1. Encoder Configurations

#### MLP Encoder (Fully-Connected)

Best for tabular data and flattened images:

```python
from artifex.generative_models.core.configuration.network_configs import EncoderConfig
from artifex.generative_models.models.vae.encoders import MLPEncoder

# Define encoder configuration
encoder_config = EncoderConfig(
    name="mlp_encoder",
    hidden_dims=(512, 256, 128),  # Network depth (use tuples)
    latent_dim=32,                 # Latent space dimension
    activation="relu",             # Activation function
    input_shape=(784,),            # Flattened input size
)

# Create encoder from config
encoder = MLPEncoder(config=encoder_config, rngs=rngs)

# Forward pass returns (mean, log_var)
mean, log_var = encoder(x)
```

#### CNN Encoder (Convolutional)

Best for image data with spatial structure:

```python
from artifex.generative_models.core.configuration.network_configs import EncoderConfig
from artifex.generative_models.models.vae.encoders import CNNEncoder

encoder_config = EncoderConfig(
    name="cnn_encoder",
    hidden_dims=(32, 64, 128, 256),  # Channel progression
    latent_dim=64,
    activation="relu",
    input_shape=(28, 28, 1),          # (H, W, C)
)

encoder = CNNEncoder(config=encoder_config, rngs=rngs)

# Preserves spatial information through convolutions
mean, log_var = encoder(x)
```

#### Conditional Encoder

Add class conditioning using `ConditionalVAEConfig`:

```python
from artifex.generative_models.core.configuration.network_configs import EncoderConfig
from artifex.generative_models.core.configuration.vae_config import ConditionalVAEConfig
from artifex.generative_models.models.vae import ConditionalVAE

# ConditionalVAE handles label embedding and conditioning automatically.
# The encoder and decoder are created internally with conditional=True,
# so you only need to provide the base encoder/decoder configs:
cvae_config = ConditionalVAEConfig(
    name="conditional_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    num_classes=10,
    kl_weight=1.0,
)
cvae = ConditionalVAE(config=cvae_config, rngs=rngs)

# Integer labels are automatically one-hot encoded:
labels = jnp.array([0, 1, 2, 3])  # No need for manual one_hot
outputs = cvae(x, y=labels)
```

### 2. Decoder Configurations

#### MLP Decoder

```python
from artifex.generative_models.core.configuration.network_configs import DecoderConfig
from artifex.generative_models.models.vae.decoders import MLPDecoder

decoder_config = DecoderConfig(
    name="mlp_decoder",
    hidden_dims=(128, 256, 512),  # Reversed from encoder
    output_shape=(784,),          # Reconstruction size
    latent_dim=32,
    activation="relu",
)

decoder = MLPDecoder(config=decoder_config, rngs=rngs)

reconstructed = decoder(z)  # Returns JAX array
```

#### CNN Decoder (Transposed Convolutions)

```python
from artifex.generative_models.core.configuration.network_configs import DecoderConfig
from artifex.generative_models.models.vae.decoders import CNNDecoder

decoder_config = DecoderConfig(
    name="cnn_decoder",
    hidden_dims=(256, 128, 64, 32),  # Reversed channel progression
    output_shape=(28, 28, 1),         # Output image shape
    latent_dim=64,
    activation="relu",
)

decoder = CNNDecoder(config=decoder_config, rngs=rngs)

reconstructed = decoder(z)  # Returns (batch, 28, 28, 1)
```

#### Conditional Decoder

For conditional generation, use the full `ConditionalVAE` model which handles
conditioning in both encoder and decoder:

```python
# See "Conditional VAE" section below for the full config-based approach
```

### 3. Complete VAE Models

#### Standard VAE

```python
from artifex.generative_models.core.configuration.network_configs import (
    EncoderConfig,
    DecoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import VAE

# Define configurations
encoder_config = EncoderConfig(
    name="encoder",
    hidden_dims=(256, 128),
    latent_dim=32,
    activation="relu",
    input_shape=(28, 28, 1),
)

decoder_config = DecoderConfig(
    name="decoder",
    hidden_dims=(128, 256),
    output_shape=(28, 28, 1),
    latent_dim=32,
    activation="relu",
)

vae_config = VAEConfig(
    name="standard_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",     # "dense" for MLP, "cnn" for convolutional
    kl_weight=1.0,            # Beta parameter (1.0 = standard VAE)
)

# Create model
vae = VAE(config=vae_config, rngs=rngs)
```

#### β-VAE (Disentangled Representations)

```python
from artifex.generative_models.core.configuration.vae_config import BetaVAEConfig
from artifex.generative_models.models.vae import BetaVAE

beta_config = BetaVAEConfig(
    name="beta_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=4.0,                    # Higher beta = more disentanglement
    beta_warmup_steps=10000,             # Gradual beta annealing
    reconstruction_loss_type="mse",      # "mse" or "bce"
)

beta_vae = BetaVAE(config=beta_config, rngs=rngs)
```

#### Conditional VAE

```python
from artifex.generative_models.core.configuration.vae_config import ConditionalVAEConfig
from artifex.generative_models.models.vae import ConditionalVAE

cvae_config = ConditionalVAEConfig(
    name="conditional_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    num_classes=10,                      # Number of classes for conditioning
    kl_weight=1.0,
)

cvae = ConditionalVAE(config=cvae_config, rngs=rngs)

# Forward pass with condition (one-hot encoded labels)
labels = jax.nn.one_hot(jnp.array([0, 1, 2]), num_classes=10)
outputs = cvae(x, y=labels)
```

#### VQ-VAE (Discrete Latents)

```python
from artifex.generative_models.core.configuration.vae_config import VQVAEConfig
from artifex.generative_models.models.vae import VQVAE

vqvae_config = VQVAEConfig(
    name="vqvae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    num_embeddings=512,                  # Codebook size
    embedding_dim=64,                    # Embedding dimension
    commitment_cost=0.25,                # Commitment loss weight
)

vqvae = VQVAE(config=vqvae_config, rngs=rngs)
```

---

## Training VAEs

### Basic Training Loop

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    EncoderConfig,
    DecoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import VAE

# 1. Create synthetic data (replace with real data loading)
key = jax.random.key(42)
train_data = jax.random.uniform(key, (1000, 28, 28, 1))

# 2. Create model configuration
rngs = nnx.Rngs(params=0, dropout=1, sample=2)

encoder_config = EncoderConfig(
    name="encoder",
    hidden_dims=(256, 128),
    latent_dim=32,
    activation="relu",
    input_shape=(28, 28, 1),
)

decoder_config = DecoderConfig(
    name="decoder",
    hidden_dims=(128, 256),
    output_shape=(28, 28, 1),
    latent_dim=32,
    activation="relu",
)

vae_config = VAEConfig(
    name="mnist_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    kl_weight=1.0,
)

# 3. Initialize model and optimizer
vae = VAE(config=vae_config, rngs=rngs)
optimizer = nnx.Optimizer(vae, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

# 4. Training step (JIT-compiled for speed)
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        outputs = model(batch)  # Model uses internal RNGs
        losses = model.loss_fn(x=batch, outputs=outputs)
        return losses["loss"], losses

    (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    return losses

# 5. Training loop
batch_size = 32
num_epochs = 5

for epoch in range(num_epochs):
    # Simple batching
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i : i + batch_size]
        losses = train_step(vae, optimizer, batch)

    print(f"Epoch {epoch + 1} | Loss: {losses['loss']:.4f}")
```

### Training β-VAE with Annealing

```python
from artifex.generative_models.core.configuration.vae_config import BetaVAEConfig
from artifex.generative_models.models.vae import BetaVAE

beta_config = BetaVAEConfig(
    name="beta_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=4.0,
    beta_warmup_steps=10000,
    reconstruction_loss_type="mse",
)

beta_vae = BetaVAE(config=beta_config, rngs=rngs)
optimizer = nnx.Optimizer(beta_vae, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

step = 0
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i : i + batch_size]

        def loss_fn(model):
            outputs = model(batch)
            # Pass current step for beta annealing
            losses = model.loss_fn(x=batch, outputs=outputs, step=step)
            return losses["loss"], losses

        (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(beta_vae)
        optimizer.update(beta_vae, grads)
        step += 1

    print(f"Epoch {epoch + 1}, Beta: {losses.get('beta', 1.0):.4f}")
```

### Training Conditional VAE

```python
from artifex.generative_models.core.configuration.vae_config import ConditionalVAEConfig
from artifex.generative_models.models.vae import ConditionalVAE

cvae_config = ConditionalVAEConfig(
    name="conditional_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    num_classes=10,
    kl_weight=1.0,
)

cvae = ConditionalVAE(config=cvae_config, rngs=rngs)
optimizer = nnx.Optimizer(cvae, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

# Create synthetic labels (replace with real labels)
train_labels = jax.random.randint(jax.random.key(0), (1000,), 0, 10)
train_labels_onehot = jax.nn.one_hot(train_labels, num_classes=10)

for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch_x = train_data[i : i + batch_size]
        batch_y = train_labels_onehot[i : i + batch_size]

        def loss_fn(model):
            outputs = model(batch_x, y=batch_y)  # Condition on labels
            losses = model.loss_fn(x=batch_x, outputs=outputs)
            return losses["loss"], losses

        (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(cvae)
        optimizer.update(cvae, grads)

    print(f"Epoch {epoch + 1} | Loss: {losses['loss']:.4f}")
```

### Training VQ-VAE

```python
from artifex.generative_models.core.configuration.vae_config import VQVAEConfig
from artifex.generative_models.models.vae import VQVAE

vqvae_config = VQVAEConfig(
    name="vqvae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    num_embeddings=512,
    embedding_dim=64,
    commitment_cost=0.25,
)

vqvae = VQVAE(config=vqvae_config, rngs=rngs)
optimizer = nnx.Optimizer(vqvae, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i : i + batch_size]

        def loss_fn(model):
            outputs = model(batch)
            losses = model.loss_fn(x=batch, outputs=outputs)
            return losses["loss"], losses

        (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(vqvae)
        optimizer.update(vqvae, grads)

    # VQ-VAE specific metrics
    print(f"Epoch {epoch + 1} | Recon: {losses.get('reconstruction_loss', 0.0):.4f}")
```

---

## Generating and Sampling

### Generate New Samples

```python
# Sample from prior distribution
n_samples = 16
samples = vae.sample(n_samples, temperature=1.0)

# Temperature controls diversity
hot_samples = vae.sample(n_samples, temperature=2.0)   # More diverse
cold_samples = vae.sample(n_samples, temperature=0.5)  # More focused

# Using generate() method (alias for sample)
samples = vae.generate(n_samples, temperature=1.0)
```

### Conditional Generation

```python
# Generate samples for specific classes
target_classes = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # One of each digit
labels = jax.nn.one_hot(target_classes, num_classes=10)

samples = cvae.sample(n_samples=10, y=labels, temperature=1.0)
```

### Reconstruction

```python
# Stochastic reconstruction (uses internal RNGs)
reconstructed = vae.reconstruct(x, deterministic=False)

# Deterministic reconstruction (use mean of latent distribution)
deterministic_recon = vae.reconstruct(x, deterministic=True)
```

---

## Latent Space Manipulation

### Interpolation Between Images

```python
# Linear interpolation in latent space
x1 = test_images[0:1]  # First image (keep batch dim)
x2 = test_images[1:2]  # Second image

interpolated = vae.interpolate(
    x1=x1,
    x2=x2,
    steps=10,  # Number of interpolation steps
)

# interpolated.shape = (10, *input_shape)
```

### Latent Traversal (Disentanglement Analysis)

```python
# Traverse a single latent dimension
x = test_images[0:1]
dim_to_traverse = 3  # Which latent dimension to vary

traversal = vae.latent_traversal(
    x=x,
    dim=dim_to_traverse,
    range_vals=(-3.0, 3.0),  # Range of values
    steps=10,                 # Number of steps
)

# traversal.shape = (10, *input_shape)
```

### Manual Latent Manipulation

```python
# Encode image to latent space
mean, log_var = vae.encode(x)

# Manipulate specific dimensions
modified_mean = mean.at[:, 5].set(2.0)    # Increase dimension 5
modified_mean = modified_mean.at[:, 10].set(-1.5)  # Decrease dimension 10

# Decode modified latent
modified_image = vae.decode(modified_mean)
```

---

## Evaluation and Analysis

### Reconstruction Quality

```python
# Calculate reconstruction error
test_batch = test_images[:100]
reconstructed = vae.reconstruct(test_batch, deterministic=True)

mse = jnp.mean((test_batch - reconstructed) ** 2)
print(f"Reconstruction MSE: {mse:.4f}")
```

### ELBO (Evidence Lower Bound)

```python
# Full ELBO calculation
outputs = vae(test_batch)  # Model uses internal RNGs
losses = vae.loss_fn(x=test_batch, outputs=outputs)

elbo = -(losses['reconstruction_loss'] + losses['kl_loss'])
print(f"ELBO: {elbo:.4f}")
```

### Latent Space Statistics

```python
# Encode test set
all_means = []
all_logvars = []

batch_size = 32
for i in range(0, len(test_images), batch_size):
    batch = test_images[i : i + batch_size]
    mean, log_var = vae.encode(batch)  # Uses internal RNGs
    all_means.append(mean)
    all_logvars.append(log_var)

all_means = jnp.concatenate(all_means, axis=0)
all_logvars = jnp.concatenate(all_logvars, axis=0)

# Statistics per dimension
mean_per_dim = jnp.mean(all_means, axis=0)
std_per_dim = jnp.std(all_means, axis=0)
variance_per_dim = jnp.exp(jnp.mean(all_logvars, axis=0))

print(f"Latent mean: {mean_per_dim}")
print(f"Latent std: {std_per_dim}")
print(f"Average variance: {variance_per_dim}")
```

### Disentanglement Metrics

```python
# Per-dimension KL divergence (detect posterior collapse)
def per_dim_kl(mean, log_var):
    """Calculate KL divergence per dimension."""
    kl_per_dim = -0.5 * (1 + log_var - mean**2 - jnp.exp(log_var))
    return jnp.mean(kl_per_dim, axis=0)

kl_per_dimension = per_dim_kl(all_means, all_logvars)

# Dimensions with very low KL likely collapsed
inactive_dims = jnp.sum(kl_per_dimension < 0.01)
print(f"Inactive dimensions: {inactive_dims}/{vae.latent_dim}")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

```python
# Architecture
config = {
    # Network architecture
    "latent_dim": 64,              # 10-100 for images, 2-20 for simple data
    "hidden_dims": [512, 256, 128], # Deeper for complex data
    "activation": "relu",          # or "gelu", "swish"

    # Training
    "learning_rate": 1e-3,         # 1e-4 to 1e-3 typical
    "batch_size": 128,             # Larger is more stable
    "num_epochs": 100,

    # VAE-specific
    "kl_weight": 1.0,              # Beta parameter
    "reconstruction_loss": "mse",  # "mse" or "bce"
}
```

### Beta Tuning for β-VAE

```python
from artifex.generative_models.core.configuration.vae_config import BetaVAEConfig
from artifex.generative_models.models.vae import BetaVAE

# Grid search over beta values
beta_values = [0.5, 1.0, 2.0, 4.0, 8.0]
results = {}

for beta in beta_values:
    # Create config with different beta
    beta_config = BetaVAEConfig(
        name=f"beta_vae_{beta}",
        encoder=encoder_config,
        decoder=decoder_config,
        encoder_type="dense",
        beta_default=beta,
    )

    rngs = nnx.Rngs(params=0, dropout=1, sample=2)
    model = BetaVAE(config=beta_config, rngs=rngs)

    # Train and evaluate (implement your train/evaluate functions)
    # trained_model = train(model, train_data, num_epochs=50)
    # recon_error = evaluate_reconstruction(trained_model, test_data)

    results[beta] = {"beta": beta}

# Find best trade-off
print(results)
```

### Learning Rate Scheduling

```python
import optax

# Cosine decay schedule
schedule = optax.cosine_decay_schedule(
    init_value=1e-3,
    decay_steps=num_train_steps,
    alpha=0.1,  # Final learning rate = 0.1 * init_value
)

optimizer = nnx.Optimizer(vae, optax.adam(learning_rate=schedule), wrt=nnx.Param)
```

---

## Common Issues and Solutions

### Problem 1: Posterior Collapse

**Symptoms**: KL divergence near zero, poor generation quality

**Solutions**:

```python
from artifex.generative_models.core.configuration.vae_config import BetaVAEConfig
from artifex.generative_models.models.vae import BetaVAE

# Solution 1: Beta annealing - start with β=0, gradually increase
beta_config = BetaVAEConfig(
    name="beta_annealing_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=1.0,
    beta_warmup_steps=10000,  # Gradual warmup
)
beta_vae = BetaVAE(config=beta_config, rngs=rngs)

# Solution 2: Weaker decoder (make it harder to ignore latent)
# Use smaller hidden_dims in decoder than encoder
weak_decoder_config = DecoderConfig(
    name="weak_decoder",
    hidden_dims=(64, 128),  # Smaller than encoder
    output_shape=(28, 28, 1),
    latent_dim=32,
    activation="relu",
)
```

### Problem 2: Blurry Reconstructions

**Symptoms**: Overly smooth outputs, lack of detail

**Solutions**:

```python
# Solution 1: Lower kl_weight (emphasize reconstruction)
vae_config = VAEConfig(
    name="sharp_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    kl_weight=0.5,  # Lower than default 1.0
)
vae = VAE(config=vae_config, rngs=rngs)

# Solution 2: Use VQ-VAE (discrete latents often produce sharper outputs)
from artifex.generative_models.core.configuration.vae_config import VQVAEConfig
from artifex.generative_models.models.vae import VQVAE

vqvae_config = VQVAEConfig(
    name="vqvae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    num_embeddings=512,
    embedding_dim=64,
)
vqvae = VQVAE(config=vqvae_config, rngs=rngs)
```

### Problem 3: Unstable Training

**Symptoms**: Loss oscillations, NaN values

**Solutions**:

```python
# Solution 1: Gradient clipping
import optax

optimizer = nnx.Optimizer(
    vae,
    optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients
        optax.adam(learning_rate=1e-3),
    ),
    wrt=nnx.Param
)

# Solution 2: Lower learning rate
optimizer = nnx.Optimizer(vae, optax.adam(learning_rate=1e-4), wrt=nnx.Param)

# Solution 3: Batch normalization in encoder/decoder
# (implement custom encoder/decoder with normalization)
```

### Problem 4: Poor Disentanglement

**Symptoms**: Latent dimensions don't correspond to interpretable factors

**Solutions**:

```python
# Solution 1: Increase beta for more disentanglement
from artifex.generative_models.core.configuration.vae_config import BetaVAEConfig
from artifex.generative_models.models.vae import BetaVAE

beta_config = BetaVAEConfig(
    name="high_beta_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=4.0,  # Higher beta encourages disentanglement
)
beta_vae = BetaVAE(config=beta_config, rngs=rngs)

# Solution 2: More latent dimensions - give model more capacity
# Update encoder_config with larger latent_dim
encoder_config_large = EncoderConfig(
    name="encoder_large_latent",
    hidden_dims=(256, 128),
    latent_dim=128,  # Increased from 32
    activation="relu",
    input_shape=(28, 28, 1),
)
```

---

## Advanced Techniques

### Custom Loss Functions

```python
def custom_loss_fn(predictions, targets):
    """Custom reconstruction loss combining multiple terms.

    Note: follows JAX/Optax convention — (predictions, targets) order.
    """
    # L1 loss for sparsity
    l1_loss = jnp.mean(jnp.abs(predictions - targets))

    # L2 loss for overall quality
    l2_loss = jnp.mean((predictions - targets) ** 2)

    # Combine
    return 0.5 * l1_loss + 0.5 * l2_loss

# Use in training
losses = vae.loss_fn(
    x=batch,
    outputs=outputs,
    reconstruction_loss_fn=custom_loss_fn,
)
```

### Multi-GPU Training

```python
import jax
from jax import devices

# Check available devices
print(f"Available devices: {jax.devices()}")

# For multi-GPU training, use JAX's sharding API
# with Artifex's distributed training utilities
from artifex.generative_models.training.distributed import (
    DataParallel,
    DeviceMeshManager,
)
```

### Checkpointing

```python
import orbax.checkpoint as ocp

# Create checkpointer
checkpointer = ocp.StandardCheckpointer()

# Save model state
graphdef, state = nnx.split(vae)
checkpointer.save("/tmp/vae_checkpoint", state)

# Load model state
restored_state = checkpointer.restore("/tmp/vae_checkpoint")

# Create new model and merge state
new_vae = VAE(config=vae_config, rngs=nnx.Rngs(0))
_, new_state = nnx.split(new_vae)
# Merge restored state into new model
nnx.update(new_vae, restored_state)
```

---

## Best Practices

### DO ✅

- **Start simple**: Begin with standard VAE before trying variants
- **Monitor both losses**: Track reconstruction AND KL divergence
- **Use appropriate loss**: MSE for continuous, BCE for binary data
- **Visualize latent space**: Plot 2D projections to check structure
- **Test interpolation**: Smooth interpolation indicates good latent space
- **Check per-dim KL**: Detect posterior collapse early
- **Use beta annealing**: Helps avoid posterior collapse
- **Larger batch size**: More stable training (128+ recommended)

### DON'T ❌

- **Don't ignore KL**: Zero KL means model ignores latent code
- **Don't use too small latent**: Leads to underfitting
- **Don't overtrain**: Can lead to posterior collapse
- **Don't skip validation**: Regular evaluation prevents surprises
- **Don't forget temperature**: Use temperature for diverse sampling
- **Don't compare different betas directly**: Higher beta trades reconstruction for disentanglement

---

## Performance Tips

### Memory Optimization

```python
# Use gradient checkpointing for large models
from jax import checkpoint

@checkpoint
def encoder_forward(encoder, x):
    return encoder(x)

# Use lower precision for faster training
# Set precision in config or at JAX level
import jax
jax.config.update("jax_default_matmul_precision", "float32")  # or "bfloat16"
```

### Speed Optimization

```python
# JIT compile training step
@nnx.jit
def fast_train_step(model, optimizer, batch):
    def loss_fn(model):
        outputs = model(batch)  # Model uses internal RNGs
        losses = model.loss_fn(x=batch, outputs=outputs)
        return losses["loss"], losses

    (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return losses

# Vectorize sampling
vmapped_decode = jax.vmap(lambda z: vae.decode(z))
samples = vmapped_decode(latent_vectors)
```

---

## Summary

This guide covered:

- ✅ Creating encoders, decoders, and VAE models
- ✅ Training standard VAE, β-VAE, CVAE, and VQ-VAE
- ✅ Generating samples and manipulating latent space
- ✅ Evaluation metrics and diagnostics
- ✅ Hyperparameter tuning strategies
- ✅ Troubleshooting common issues
- ✅ Advanced techniques and optimizations

---

## Next Steps

- **[VAE Concepts](../concepts/vae-explained.md)** — Deep dive into theory
- **[VAE API Reference](../../api/models/vae.md)** — Complete API documentation
- **[VAE MNIST Example](../../examples/basic/vae-mnist.md)** — Hands-on tutorial
- **[Training Guide](../../training/trainer.md)** — Advanced training techniques
- **[Benchmarking](../../benchmarks/index.md)** — Evaluate your models
