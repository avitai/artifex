# VAE API Reference

Complete API documentation for Variational Autoencoder models in Artifex.

## Module Overview

```python
from artifex.generative_models.models.vae import (
    VAE,                    # Base VAE class
    BetaVAE,               # β-VAE with disentanglement
    BetaVAEWithCapacity,   # β-VAE with capacity control
    ConditionalVAE,        # Conditional VAE
    VQVAE,                 # Vector Quantized VAE
)

from artifex.generative_models.models.vae.encoders import (
    MLPEncoder,            # Fully-connected encoder
    CNNEncoder,            # Convolutional encoder
    ResNetEncoder,         # ResNet-based encoder
    ConditionalEncoder,    # Conditional wrapper
)

from artifex.generative_models.models.vae.decoders import (
    MLPDecoder,            # Fully-connected decoder
    CNNDecoder,            # Transposed convolutional decoder
    ResNetDecoder,         # ResNet-based decoder
    ConditionalDecoder,    # Conditional wrapper
)
```

---

## Base Classes

### VAE

::: artifex.generative_models.models.vae.base.VAE

The base VAE class implementing standard Variational Autoencoder functionality.

#### Class Definition

```python
class VAE(GenerativeModel):
    """Base class for Variational Autoencoders."""

    def __init__(
        self,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
        kl_weight: float = 1.0,
        precision: jax.lax.Precision | None = None,
    ) -> None:
        """Initialize a VAE."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder` | `nnx.Module` | Required | Encoder network mapping inputs to latent distributions |
| `decoder` | `nnx.Module` | Required | Decoder network mapping latent codes to reconstructions |
| `latent_dim` | `int` | Required | Dimensionality of the latent space (must be positive) |
| `rngs` | `nnx.Rngs` | Required | Random number generators for initialization and sampling |
| `kl_weight` | `float` | `1.0` | Weight for KL divergence term (β parameter) |
| `precision` | `jax.lax.Precision \| None` | `None` | Numerical precision for computations |

#### Methods

##### `encode`

```python
def encode(
    self,
    x: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, jax.Array]:
    """Encode input to latent distribution parameters."""
```

**Parameters:**

- `x` (Array): Input data with shape `(batch_size, *input_shape)`
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `tuple[Array, Array]`: Mean and log-variance of latent distribution
  - `mean`: Shape `(batch_size, latent_dim)`
  - `log_var`: Shape `(batch_size, latent_dim)`

**Raises:**

- `ValueError`: If encoder output format is invalid

**Example:**

```python
mean, log_var = vae.encode(x, rngs=rngs)
print(f"Mean shape: {mean.shape}")        # (32, 20)
print(f"Log-var shape: {log_var.shape}")  # (32, 20)
```

##### `decode`

```python
def decode(
    self,
    z: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> jax.Array:
    """Decode latent vectors to reconstructions."""
```

**Parameters:**

- `z` (Array): Latent vectors with shape `(batch_size, latent_dim)`
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `Array`: Reconstructed outputs with shape `(batch_size, *output_shape)`

**Example:**

```python
z = jax.random.normal(key, (32, 20))
reconstructed = vae.decode(z, rngs=rngs)
print(f"Reconstruction shape: {reconstructed.shape}")  # (32, 784)
```

##### `reparameterize`

```python
@nnx.jit
def reparameterize(
    self,
    mean: jax.Array,
    log_var: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> jax.Array:
    """Apply the reparameterization trick."""
```

**Parameters:**

- `mean` (Array): Mean vectors with shape `(batch_size, latent_dim)`
- `log_var` (Array): Log-variance vectors with shape `(batch_size, latent_dim)`
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `Array`: Sampled latent vectors with shape `(batch_size, latent_dim)`

**Details:**

Implements the reparameterization trick: $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$

Includes numerical stability via log-variance clipping to `[-20, 20]`.

**Example:**

```python
mean, log_var = vae.encode(x, rngs=rngs)
z = vae.reparameterize(mean, log_var, rngs=rngs)
```

##### `__call__`

```python
def __call__(
    self,
    x: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> dict[str, Any]:
    """Forward pass through the VAE."""
```

**Parameters:**

- `x` (Array): Input data with shape `(batch_size, *input_shape)`
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `dict`: Dictionary containing:
  - `reconstructed` (Array): Reconstructed outputs
  - `reconstruction` (Array): Alias for compatibility
  - `mean` (Array): Latent mean vectors
  - `log_var` (Array): Latent log-variance vectors
  - `logvar` (Array): Alias for compatibility
  - `z` (Array): Sampled latent vectors

**Example:**

```python
outputs = vae(x, rngs=rngs)
print(outputs.keys())
# dict_keys(['reconstructed', 'reconstruction', 'mean', 'log_var', 'logvar', 'z'])
```

##### `loss_fn`

```python
def loss_fn(
    self,
    params: dict | None = None,
    batch: dict | None = None,
    rng: jax.Array | None = None,
    x: jax.Array | None = None,
    outputs: dict[str, jax.Array] | None = None,
    beta: float | None = None,
    reconstruction_loss_fn: Callable | None = None,
    **kwargs,
) -> dict[str, jax.Array]:
    """Calculate VAE loss (ELBO)."""
```

**Parameters:**

- `params` (dict, optional): Model parameters (for Trainer compatibility)
- `batch` (dict, optional): Input batch (for Trainer compatibility)
- `rng` (Array, optional): Random number generator
- `x` (Array, optional): Input data if not in batch
- `outputs` (dict, optional): Pre-computed model outputs
- `beta` (float, optional): KL divergence weight override
- `reconstruction_loss_fn` (Callable, optional): Custom reconstruction loss
- `**kwargs`: Additional arguments

**Returns:**

- `dict`: Dictionary containing:
  - `reconstruction_loss` (Array): Reconstruction error
  - `recon_loss` (Array): Alias for compatibility
  - `kl_loss` (Array): KL divergence
  - `total_loss` (Array): Combined loss
  - `loss` (Array): Alias for compatibility

**Loss Formula:**

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] + \beta \cdot D_{KL}(q(z|x) \| p(z))
$$

**Example:**

```python
outputs = vae(x, rngs=rngs)
losses = vae.loss_fn(x=x, outputs=outputs)

print(f"Reconstruction: {losses['reconstruction_loss']:.4f}")
print(f"KL Divergence: {losses['kl_loss']:.4f}")
print(f"Total Loss: {losses['total_loss']:.4f}")
```

##### `sample`

```python
def sample(
    self,
    n_samples: int = 1,
    *,
    temperature: float = 1.0,
    rngs: nnx.Rngs | None = None
) -> jax.Array:
    """Sample from the prior distribution."""
```

**Parameters:**

- `n_samples` (int): Number of samples to generate
- `temperature` (float): Scaling factor for sampling diversity (higher = more diverse)
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `Array`: Generated samples with shape `(n_samples, *output_shape)`

**Example:**

```python
# Generate 16 samples
samples = vae.sample(n_samples=16, temperature=1.0, rngs=rngs)

# More diverse samples
hot_samples = vae.sample(n_samples=16, temperature=2.0, rngs=rngs)
```

##### `generate`

```python
def generate(
    self,
    n_samples: int = 1,
    *,
    temperature: float = 1.0,
    rngs: nnx.Rngs | None = None,
    **kwargs,
) -> jax.Array:
    """Generate samples (alias for sample)."""
```

Alias for `sample()` to maintain consistency with `GenerativeModel` interface.

##### `reconstruct`

```python
def reconstruct(
    self,
    x: jax.Array,
    *,
    deterministic: bool = False,
    rngs: nnx.Rngs | None = None
) -> jax.Array:
    """Reconstruct inputs."""
```

**Parameters:**

- `x` (Array): Input data
- `deterministic` (bool): If True, use mean instead of sampling
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `Array`: Reconstructed outputs

**Example:**

```python
# Stochastic reconstruction
recon = vae.reconstruct(x, deterministic=False, rngs=rngs)

# Deterministic reconstruction (use latent mean)
det_recon = vae.reconstruct(x, deterministic=True, rngs=rngs)
```

##### `interpolate`

```python
def interpolate(
    self,
    x1: jax.Array,
    x2: jax.Array,
    steps: int = 10,
    *,
    rngs: nnx.Rngs | None = None,
) -> jax.Array:
    """Interpolate between two inputs in latent space."""
```

**Parameters:**

- `x1` (Array): First input
- `x2` (Array): Second input
- `steps` (int): Number of interpolation steps (including endpoints)
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `Array`: Interpolated outputs with shape `(steps, *output_shape)`

**Example:**

```python
x1 = test_images[0]
x2 = test_images[1]
interpolation = vae.interpolate(x1, x2, steps=10, rngs=rngs)
```

##### `latent_traversal`

```python
def latent_traversal(
    self,
    x: jax.Array,
    dim: int,
    range_vals: tuple[float, float] = (-3.0, 3.0),
    steps: int = 10,
    *,
    rngs: nnx.Rngs | None = None,
) -> jax.Array:
    """Traverse a single latent dimension."""
```

**Parameters:**

- `x` (Array): Input data
- `dim` (int): Dimension to traverse (0 to latent_dim-1)
- `range_vals` (tuple): Range of values for traversal
- `steps` (int): Number of traversal steps
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `Array`: Decoded outputs from traversal with shape `(steps, *output_shape)`

**Raises:**

- `ValueError`: If dimension is out of range

**Example:**

```python
# Traverse dimension 5
traversal = vae.latent_traversal(
    x=test_image,
    dim=5,
    range_vals=(-3.0, 3.0),
    steps=15,
    rngs=rngs,
)
```

---

## VAE Variants

### BetaVAE

::: artifex.generative_models.models.vae.beta_vae.BetaVAE

β-VAE for learning disentangled representations.

#### Class Definition

```python
class BetaVAE(VAE):
    """Beta Variational Autoencoder for disentanglement."""

    def __init__(
        self,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        beta_default: float = 1.0,
        beta_warmup_steps: int = 0,
        reconstruction_loss_type: str = "mse",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a BetaVAE."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder` | `nnx.Module` | Required | Encoder network |
| `decoder` | `nnx.Module` | Required | Decoder network |
| `latent_dim` | `int` | Required | Latent space dimension |
| `beta_default` | `float` | `1.0` | Default β value for KL weighting |
| `beta_warmup_steps` | `int` | `0` | Steps for β annealing (0 = no annealing) |
| `reconstruction_loss_type` | `str` | `"mse"` | Loss type: `"mse"` or `"bce"` |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Key Differences from VAE

**Modified Loss Function:**

$$
\mathcal{L}_\beta = \mathbb{E}_{q(z|x)}[\log p(x|z)] + \beta \cdot D_{KL}(q(z|x) \| p(z))
$$

**Beta Annealing:**

When `beta_warmup_steps > 0`, β increases linearly from 0 to `beta_default`:

$$
\beta(t) = \min\left(\beta_{\text{default}}, \beta_{\text{default}} \cdot \frac{t}{T_{\text{warmup}}}\right)
$$

#### Example

```python
beta_vae = BetaVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=32,
    beta_default=4.0,              # Higher β encourages disentanglement
    beta_warmup_steps=10000,       # Gradually increase β
    reconstruction_loss_type="mse",
    rngs=rngs,
)

# Training step with beta annealing
for step in range(num_steps):
    outputs = beta_vae(batch, rngs=rngs)
    losses = beta_vae.loss_fn(x=batch, outputs=outputs, step=step)
    # losses['beta'] contains current β value
```

---

### BetaVAEWithCapacity

::: artifex.generative_models.models.vae.beta_vae.BetaVAEWithCapacity

β-VAE with Burgess et al. capacity control mechanism.

#### Class Definition

```python
class BetaVAEWithCapacity(BetaVAE):
    """β-VAE with capacity control."""

    def __init__(
        self,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        beta_default: float = 1.0,
        beta_warmup_steps: int = 0,
        reconstruction_loss_type: str = "mse",
        use_capacity_control: bool = False,
        capacity_max: float = 25.0,
        capacity_num_iter: int = 25000,
        gamma: float = 1000.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize BetaVAE with capacity control."""
```

#### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_capacity_control` | `bool` | `False` | Enable capacity control |
| `capacity_max` | `float` | `25.0` | Maximum capacity in nats |
| `capacity_num_iter` | `int` | `25000` | Steps to reach max capacity |
| `gamma` | `float` | `1000.0` | Weight for capacity loss |

#### Capacity Loss

$$
\mathcal{L}_C = \mathbb{E}_{q(z|x)}[\log p(x|z)] + \gamma \cdot |D_{KL}(q(z|x) \| p(z)) - C(t)|
$$

Where capacity $C(t)$ increases linearly:

$$
C(t) = \min\left(C_{\max}, C_{\max} \cdot \frac{t}{T_{\text{capacity}}}\right)
$$

#### Example

```python
capacity_vae = BetaVAEWithCapacity(
    encoder=encoder,
    decoder=decoder,
    latent_dim=32,
    beta_default=4.0,
    use_capacity_control=True,
    capacity_max=25.0,
    capacity_num_iter=25000,
    gamma=1000.0,
    rngs=rngs,
)

# Loss includes capacity terms
losses = capacity_vae.loss_fn(x=batch, outputs=outputs, step=step)
print(losses['current_capacity'])    # Current C value
print(losses['capacity_loss'])       # γ * |KL - C|
print(losses['kl_capacity_diff'])    # KL - C
```

---

### ConditionalVAE

::: artifex.generative_models.models.vae.conditional.ConditionalVAE

Conditional VAE for class-conditional generation.

#### Class Definition

```python
class ConditionalVAE(VAE):
    """Conditional Variational Autoencoder."""

    def __init__(
        self,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        condition_dim: int = 10,
        condition_type: str = "concat",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize Conditional VAE."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder` | `nnx.Module` | Required | Conditional encoder |
| `decoder` | `nnx.Module` | Required | Conditional decoder |
| `latent_dim` | `int` | Required | Latent dimension |
| `condition_dim` | `int` | `10` | Conditioning dimension |
| `condition_type` | `str` | `"concat"` | Conditioning strategy |
| `rngs` | `nnx.Rngs` | `None` | Random number generators |

#### Modified Methods

##### `__call__`

```python
def __call__(
    self,
    x: jax.Array,
    y: jax.Array | None = None,
    *,
    rngs: nnx.Rngs | None = None,
) -> dict[str, Any]:
    """Forward pass with conditioning."""
```

##### `encode`

```python
def encode(
    self,
    x: jax.Array,
    y: jax.Array | None = None,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, jax.Array]:
    """Encode with conditioning."""
```

##### `decode`

```python
def decode(
    self,
    z: jax.Array,
    y: jax.Array | None = None,
    *,
    rngs: nnx.Rngs | None = None
) -> jax.Array:
    """Decode with conditioning."""
```

##### `sample`

```python
def sample(
    self,
    n_samples: int = 1,
    *,
    temperature: float = 1.0,
    y: jax.Array | None = None,
    rngs: nnx.Rngs | None = None,
) -> jax.Array:
    """Sample with conditioning."""
```

#### Example

```python
# Create conditional encoder/decoder
conditional_encoder = ConditionalEncoder(
    encoder=base_encoder,
    num_classes=10,
    embed_dim=128,
    rngs=rngs,
)

conditional_decoder = ConditionalDecoder(
    decoder=base_decoder,
    num_classes=10,
    embed_dim=128,
    rngs=rngs,
)

cvae = ConditionalVAE(
    encoder=conditional_encoder,
    decoder=conditional_decoder,
    latent_dim=32,
    condition_dim=10,
    rngs=rngs,
)

# Forward pass with class labels
labels = jnp.array([0, 1, 2, 3, 4])
outputs = cvae(x, y=labels, rngs=rngs)

# Generate specific classes
target_labels = jnp.array([5, 5, 5, 5])
samples = cvae.sample(n_samples=4, y=target_labels, rngs=rngs)
```

---

### VQVAE

::: artifex.generative_models.models.vae.vq_vae.VQVAE

Vector Quantized VAE with discrete latent representations.

#### Class Definition

```python
class VQVAE(VAE):
    """Vector Quantized Variational Autoencoder."""

    def __init__(
        self,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize VQ-VAE."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder` | `nnx.Module` | Required | Encoder network |
| `decoder` | `nnx.Module` | Required | Decoder network |
| `latent_dim` | `int` | Required | Latent dimension |
| `num_embeddings` | `int` | `512` | Codebook size (number of embeddings) |
| `embedding_dim` | `int` | `64` | Dimension of each embedding |
| `commitment_cost` | `float` | `0.25` | Weight for commitment loss |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Key Method: `quantize`

```python
def quantize(
    self,
    encoding: jax.Array,
    *,
    rngs: nnx.Rngs | None = None
) -> tuple[jax.Array, dict[str, Any]]:
    """Quantize continuous encoding using codebook."""
```

**Parameters:**

- `encoding` (Array): Continuous encoding from encoder
- `rngs` (Rngs, optional): Random number generators

**Returns:**

- `tuple`: (quantized encoding, auxiliary info)
  - `quantized` (Array): Discrete quantized vectors
  - `aux` (dict): Contains `commitment_loss`, `codebook_loss`, `encoding_indices`

**Quantization Process:**

1. Find nearest codebook embedding for each encoding vector
2. Replace encoding with codebook embedding
3. Use straight-through estimator for gradients

#### Loss Function

VQ-VAE uses a specialized loss:

$$
\mathcal{L}_{VQ} = \|x - \hat{x}\|^2 + \|sg[z_e] - e\|^2 + \beta \|z_e - sg[e]\|^2
$$

Where:

- First term: Reconstruction loss
- Second term: Codebook loss (update embeddings)
- Third term: Commitment loss (encourage encoder to commit)

#### Example

```python
vqvae = VQVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=64,
    num_embeddings=512,    # 512 discrete codes
    embedding_dim=64,
    commitment_cost=0.25,
    rngs=rngs,
)

# Forward pass includes quantization
outputs = vqvae(x, rngs=rngs)
print(outputs['z_e'])              # Pre-quantization encoding
print(outputs['quantized'])        # Quantized (discrete) encoding
print(outputs['commitment_loss'])  # Commitment loss component
print(outputs['codebook_loss'])    # Codebook loss component

# Loss includes VQ-specific terms
losses = vqvae.loss_fn(x=batch, outputs=outputs)
print(losses['reconstruction_loss'])
print(losses['commitment_loss'])
print(losses['codebook_loss'])
```

---

## Encoders

### MLPEncoder

::: artifex.generative_models.models.vae.encoders.MLPEncoder

Fully-connected encoder for flattened inputs.

#### Class Definition

```python
class MLPEncoder(nnx.Module):
    """MLP encoder for VAE."""

    def __init__(
        self,
        hidden_dims: list,
        latent_dim: int,
        activation: str = "relu",
        input_dim: tuple | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize MLP encoder."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | Required | Hidden layer dimensions |
| `latent_dim` | `int` | Required | Latent space dimension |
| `activation` | `str` | `"relu"` | Activation function |
| `input_dim` | `tuple \| None` | `None` | Input dimensions for shape inference |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Example

```python
encoder = MLPEncoder(
    hidden_dims=[512, 256, 128],
    latent_dim=32,
    activation="relu",
    input_dim=(784,),
    rngs=rngs,
)

mean, log_var = encoder(x, rngs=rngs)
```

---

### CNNEncoder

::: artifex.generative_models.models.vae.encoders.CNNEncoder

Convolutional encoder for image inputs.

#### Class Definition

```python
class CNNEncoder(nnx.Module):
    """CNN encoder for VAE."""

    def __init__(
        self,
        hidden_dims: list,
        latent_dim: int,
        activation: str = "relu",
        input_dim: tuple | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CNN encoder."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | Required | Channel dimensions for conv layers |
| `latent_dim` | `int` | Required | Latent space dimension |
| `activation` | `str` | `"relu"` | Activation function |
| `input_dim` | `tuple \| None` | `None` | Input shape (H, W, C) |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Architecture

- Series of convolutional layers with stride 2
- Each layer reduces spatial dimensions by half
- Global pooling before projecting to latent space

#### Example

```python
encoder = CNNEncoder(
    hidden_dims=[32, 64, 128, 256],
    latent_dim=64,
    activation="relu",
    input_dim=(28, 28, 1),
    rngs=rngs,
)

# Input shape: (batch, 28, 28, 1)
mean, log_var = encoder(images, rngs=rngs)
# Output shapes: (batch, 64), (batch, 64)
```

---

### ConditionalEncoder

::: artifex.generative_models.models.vae.encoders.ConditionalEncoder

Wrapper that adds conditioning to any encoder.

#### Class Definition

```python
class ConditionalEncoder(nnx.Module):
    """Conditional encoder wrapper."""

    def __init__(
        self,
        encoder: nnx.Module,
        num_classes: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize conditional encoder."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder` | `nnx.Module` | Required | Base encoder to wrap |
| `num_classes` | `int` | Required | Number of conditioning classes |
| `embed_dim` | `int` | Required | Embedding dimension for labels |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Example

```python
base_encoder = MLPEncoder(
    hidden_dims=[512, 256],
    latent_dim=32,
    rngs=rngs,
)

conditional_encoder = ConditionalEncoder(
    encoder=base_encoder,
    num_classes=10,
    embed_dim=128,
    rngs=rngs,
)

# Pass class labels as integers or one-hot
labels = jnp.array([0, 1, 2, 3])
mean, log_var = conditional_encoder(x, condition=labels, rngs=rngs)
```

---

## Decoders

### MLPDecoder

::: artifex.generative_models.models.vae.decoders.MLPDecoder

Fully-connected decoder.

#### Class Definition

```python
class MLPDecoder(nnx.Module):
    """MLP decoder for VAE."""

    def __init__(
        self,
        hidden_dims: list[int],
        output_dim: tuple[int, ...],
        latent_dim: int,
        activation: str = "relu",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize MLP decoder."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | Required | Hidden layer dimensions (reversed from encoder) |
| `output_dim` | `tuple` | Required | Output reconstruction shape |
| `latent_dim` | `int` | Required | Latent space dimension |
| `activation` | `str` | `"relu"` | Activation function |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Example

```python
decoder = MLPDecoder(
    hidden_dims=[128, 256, 512],  # Reversed from encoder
    output_dim=(784,),
    latent_dim=32,
    activation="relu",
    rngs=rngs,
)

reconstructed = decoder(z)  # Shape: (batch, 784)
```

---

### CNNDecoder

::: artifex.generative_models.models.vae.decoders.CNNDecoder

Transposed convolutional decoder for images.

#### Class Definition

```python
class CNNDecoder(nnx.Module):
    """CNN decoder with transposed convolutions."""

    def __init__(
        self,
        hidden_dims: list[int],
        output_dim: tuple[int, ...],
        latent_dim: int,
        activation: str = "relu",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize CNN decoder."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | Required | Channel dimensions (reversed from encoder) |
| `output_dim` | `tuple` | Required | Output shape (H, W, C) |
| `latent_dim` | `int` | Required | Latent dimension |
| `activation` | `str` | `"relu"` | Activation function |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Example

```python
decoder = CNNDecoder(
    hidden_dims=[256, 128, 64, 32],  # Reversed channels
    output_dim=(28, 28, 1),
    latent_dim=64,
    activation="relu",
    rngs=rngs,
)

# Input: (batch, 64)
reconstructed = decoder(z)  # Output: (batch, 28, 28, 1)
```

---

### ConditionalDecoder

::: artifex.generative_models.models.vae.decoders.ConditionalDecoder

Wrapper that adds conditioning to any decoder.

#### Class Definition

```python
class ConditionalDecoder(nnx.Module):
    """Conditional decoder wrapper."""

    def __init__(
        self,
        decoder: nnx.Module,
        num_classes: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize conditional decoder."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decoder` | `nnx.Module` | Required | Base decoder to wrap |
| `num_classes` | `int` | Required | Number of conditioning classes |
| `embed_dim` | `int` | Required | Embedding dimension |
| `rngs` | `nnx.Rngs` | Required | Random number generators |

#### Example

```python
base_decoder = MLPDecoder(
    hidden_dims=[128, 256, 512],
    output_dim=(784,),
    latent_dim=32,
    rngs=rngs,
)

conditional_decoder = ConditionalDecoder(
    decoder=base_decoder,
    num_classes=10,
    embed_dim=128,
    rngs=rngs,
)

labels = jnp.array([0, 1, 2, 3])
reconstructed = conditional_decoder(z, condition=labels, rngs=rngs)
```

---

## Utility Functions

### `create_encoder_unified`

```python
def create_encoder_unified(
    *,
    config: ModelConfig,
    encoder_type: str,
    conditional: bool = False,
    num_classes: int | None = None,
    embed_dim: int | None = None,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create encoder from unified configuration."""
```

**Parameters:**

- `config` (ModelConfig): Model configuration
- `encoder_type` (str): Type of encoder: `"dense"`, `"cnn"`, `"resnet"`
- `conditional` (bool): Whether to wrap in conditional encoder
- `num_classes` (int, optional): Number of classes for conditioning
- `embed_dim` (int, optional): Embedding dimension
- `rngs` (Rngs): Random number generators

**Returns:**

- `nnx.Module`: Configured encoder

### `create_decoder_unified`

```python
def create_decoder_unified(
    *,
    config: ModelConfig,
    decoder_type: str,
    conditional: bool = False,
    num_classes: int | None = None,
    embed_dim: int | None = None,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create decoder from unified configuration."""
```

**Parameters:**

- `config` (ModelConfig): Model configuration
- `decoder_type` (str): Type of decoder: `"dense"`, `"cnn"`, `"resnet"`
- `conditional` (bool): Whether to wrap in conditional decoder
- `num_classes` (int, optional): Number of classes for conditioning
- `embed_dim` (int, optional): Embedding dimension
- `rngs` (Rngs): Random number generators

**Returns:**

- `nnx.Module`: Configured decoder

---

## Complete Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.models.vae import BetaVAE
from artifex.generative_models.models.vae.encoders import CNNEncoder
from artifex.generative_models.models.vae.decoders import CNNDecoder

# Initialize
rngs = nnx.Rngs(params=0, dropout=1, sample=2)

# Create encoder
encoder = CNNEncoder(
    hidden_dims=[32, 64, 128, 256],
    latent_dim=64,
    activation="relu",
    input_dim=(28, 28, 1),
    rngs=rngs,
)

# Create decoder
decoder = CNNDecoder(
    hidden_dims=[256, 128, 64, 32],
    output_dim=(28, 28, 1),
    latent_dim=64,
    activation="relu",
    rngs=rngs,
)

# Create β-VAE
model = BetaVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=64,
    beta_default=4.0,
    beta_warmup_steps=10000,
    reconstruction_loss_type="mse",
    rngs=rngs,
)

# Forward pass
x = jnp.ones((32, 28, 28, 1))
outputs = model(x, rngs=rngs)

# Calculate loss
losses = model.loss_fn(x=x, outputs=outputs, step=5000)
print(f"Total Loss: {losses['total_loss']:.4f}")
print(f"Beta: {losses['beta']:.4f}")

# Generate samples
samples = model.sample(n_samples=16, temperature=1.0, rngs=rngs)

# Latent traversal
traversal = model.latent_traversal(x[0], dim=10, steps=15, rngs=rngs)
```

---

## See Also

- **[VAE Concepts](../../user-guide/concepts/vae-explained.md)** — Theory and mathematical foundations
- **[VAE User Guide](../../user-guide/models/vae-guide.md)** — Practical usage and examples
- **[Training Guide](../../training/trainer.md)** — Training VAE models
- **[Loss Functions](../core/losses.md)** — Available loss functions
- **[Configuration](../core/configuration.md)** — Configuration system
