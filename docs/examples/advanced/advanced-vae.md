# Advanced VAE Examples

**Level:** Advanced | **Runtime:** Varies by variant (30-60 min per model) | **Format:** Python + Jupyter

Advanced Variational Autoencoder variants and techniques, including β-VAE, β-VAE with Capacity Control, Conditional VAE, and VQ-VAE.

## Prerequisites

**Required Knowledge:**

- Strong understanding of standard VAEs and ELBO
- Familiarity with the [Basic VAE Tutorial](../basic/vae-mnist.md)
- Experience with JAX and Flax NNX
- Understanding of latent space representations
- Knowledge of training dynamics and loss functions

**Skill Level:** Advanced - requires solid foundation in variational inference and generative modeling

**Estimated Time:** 2-3 hours to work through all variants

!!! info "Multiple Implementations"
    This guide contains **four complete VAE variant implementations**:

    - **β-VAE**: Disentangled representations with β-weighting and annealing
    - **β-VAE with Capacity Control**: Burgess et al. capacity-based training for stable disentanglement
    - **Conditional VAE**: Label-conditioned generation for controlled sampling
    - **VQ-VAE**: Discrete latent codes using vector quantization

    Each variant includes complete working code that you can run independently or integrate into your projects.

<div class="grid cards" markdown>

- :material-beta:{ .lg .middle } **β-VAE**

    ---

    Disentangled representations with β-weighting

    [:octicons-arrow-right-24: Learn more](#beta-vae)

- :material-grid:{ .lg .middle } **VQ-VAE**

    ---

    Vector-Quantized VAE for discrete latent spaces

    [:octicons-arrow-right-24: Learn more](#vq-vae)

- :material-filter:{ .lg .middle } **Conditional VAE**

    ---

    Condition generation on labels or attributes

    [:octicons-arrow-right-24: Learn more](#conditional-vae)

- :material-gauge:{ .lg .middle } **β-VAE with Capacity Control**

    ---

    Burgess et al. capacity-based training for stable disentanglement

    [:octicons-arrow-right-24: Learn more](#beta-vae-with-capacity-control)

</div>

## Beta-VAE

β-VAE adds a weight β to the KL divergence term, encouraging disentangled representations.

### Basic β-VAE

```python
from artifex.generative_models.core.configuration import (
    BetaVAEConfig,
    EncoderConfig,
    DecoderConfig,
)
from artifex.generative_models.models.vae import BetaVAE
from flax import nnx
import jax
import jax.numpy as jnp

# Create β-VAE configuration using frozen dataclass configs
encoder_config = EncoderConfig(
    name="beta_vae_encoder",
    input_shape=(64, 64, 3),  # RGB images
    latent_dim=10,  # Smaller latent dim encourages disentanglement
    hidden_dims=(512, 256, 128),  # Tuple for frozen dataclass
    activation="relu",
)

decoder_config = DecoderConfig(
    name="beta_vae_decoder",
    output_shape=(64, 64, 3),
    latent_dim=10,
    hidden_dims=(128, 256, 512),  # Tuple for frozen dataclass
    activation="relu",
)

config = BetaVAEConfig(
    name="beta_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=4.0,  # β > 1 for disentanglement
)

# Create model
rngs = nnx.Rngs(params=0, dropout=1, sample=2)
model = BetaVAE(config, rngs=rngs)

# Custom β-VAE loss
def beta_vae_loss(model, batch, beta=4.0):
    """β-VAE loss with weighted KL divergence."""
    output = model(batch["data"])

    # Reconstruction loss
    recon_loss = jnp.mean((batch["data"] - output["reconstruction"]) ** 2)

    # KL divergence
    kl_loss = -0.5 * jnp.mean(
        1 + output["logvar"] - output["mean"] ** 2 - jnp.exp(output["logvar"])
    )

    # β-weighted total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, {
        "loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
    }

# Training step
@jax.jit
def train_step(model_state, batch, optimizer_state, beta=4.0):
    """Training step with β-VAE loss."""
    model = nnx.merge(model_graphdef, model_state)

    (loss, metrics), grads = nnx.value_and_grad(
        lambda m: beta_vae_loss(m, batch, beta=beta),
        has_aux=True
    )(model)

    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    model_state = optax.apply_updates(model_state, updates)

    return model_state, optimizer_state, metrics

# Split model
model_graphdef, model_state = nnx.split(model)

# Training loop with different β values
for epoch in range(num_epochs):
    # Anneal β from 1.0 to 4.0
    beta = 1.0 + (4.0 - 1.0) * min(epoch / 10, 1.0)

    for batch in dataloader:
        model_state, optimizer_state, metrics = train_step(
            model_state, batch, optimizer_state, beta=beta
        )

    print(f"Epoch {epoch}, β={beta:.2f}, Loss={metrics['loss']:.4f}")
```

!!! tip "Choosing β Values"
    **β value guidelines:**

    - **β = 1.0**: Standard VAE (no disentanglement bias)
    - **β = 2.0-4.0**: Good balance for disentanglement on simple datasets
    - **β = 6.0-10.0**: Strong disentanglement, may sacrifice reconstruction quality
    - **β annealing**: Start at 1.0, gradually increase to target β over 10-20 epochs

    Higher β encourages independence between latent dimensions but can lead to posterior collapse if too large.

### Disentanglement Evaluation

```python
def evaluate_disentanglement(model, dataset, num_samples=1000):
    """Evaluate disentanglement of learned representations."""
    import numpy as np

    # Collect latent representations
    latents = []
    labels = []

    for batch in dataset.take(num_samples // 32):
        output = model(batch["data"])
        latents.append(np.array(output["mean"]))
        if "labels" in batch:
            labels.append(np.array(batch["labels"]))

    latents = np.concatenate(latents, axis=0)
    if labels:
        labels = np.concatenate(labels, axis=0)

    # Compute variance per latent dimension
    latent_variances = np.var(latents, axis=0)

    # Active dimensions (high variance)
    active_dims = latent_variances > 0.01

    print(f"Active dimensions: {np.sum(active_dims)} / {latents.shape[1]}")
    print(f"Latent variances: {latent_variances}")

    # If labels available, compute mutual information
    if labels:
        from sklearn.metrics import mutual_info_score

        mi_scores = []
        for dim in range(latents.shape[1]):
            # Discretize latent dimension
            latent_discrete = np.digitize(latents[:, dim], bins=10)

            # Compute MI with each label dimension
            for label_dim in range(labels.shape[1]):
                mi = mutual_info_score(label_dim, latent_discrete)
                mi_scores.append(mi)

        print(f"Mean mutual information: {np.mean(mi_scores):.4f}")

    return {
        "active_dimensions": int(np.sum(active_dims)),
        "latent_variances": latent_variances.tolist(),
    }

# Evaluate
results = evaluate_disentanglement(model, val_dataset)
```

### Latent Traversal Visualization

```python
def visualize_latent_traversals(model, z_base, dim, values=None):
    """Visualize effect of traversing a single latent dimension."""
    import matplotlib.pyplot as plt

    if values is None:
        values = jnp.linspace(-3, 3, 11)

    samples = []
    for value in values:
        z = z_base.copy()
        z[dim] = value
        sample = model.decode(z[None, :])[0]
        samples.append(sample)

    # Plot traversal
    fig, axes = plt.subplots(1, len(values), figsize=(15, 2))
    for i, (ax, sample) in enumerate(zip(axes, samples)):
        ax.imshow(sample, cmap="gray")
        ax.set_title(f"z[{dim}]={values[i]:.1f}")
        ax.axis("off")

    plt.suptitle(f"Latent Dimension {dim} Traversal")
    plt.tight_layout()
    return fig

# Get base latent vector
sample = next(iter(val_dataset))
output = model(sample["data"][:1])
z_base = jnp.array(output["mean"][0])

# Visualize each dimension
for dim in range(model.latent_dim):
    fig = visualize_latent_traversals(model, z_base, dim)
    # fig.savefig(f"traversal_dim_{dim}.png")
```

## VQ-VAE

Vector-Quantized VAE uses discrete latent codes from a learnable codebook.

### VQ-VAE Implementation

```python
from flax import nnx
import jax
import jax.numpy as jnp

class VectorQuantizer(nnx.Module):
    """Vector quantization layer."""

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        commitment_cost: float = 0.25,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Codebook
        self.embeddings = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (num_embeddings, embedding_dim),
                minval=-1.0,
                maxval=1.0,
            )
        )

    def __call__(self, z: jax.Array) -> tuple[jax.Array, dict]:
        """Quantize continuous latents.

        Args:
            z: Continuous latents (batch, ..., embedding_dim)

        Returns:
            (quantized, info_dict)
        """
        # Flatten spatial dimensions
        flat_z = z.reshape(-1, self.embedding_dim)

        # Compute distances to codebook vectors
        distances = (
            jnp.sum(flat_z ** 2, axis=1, keepdims=True)
            + jnp.sum(self.embeddings.value ** 2, axis=1)
            - 2 * flat_z @ self.embeddings.value.T
        )

        # Get nearest codebook indices
        indices = jnp.argmin(distances, axis=1)

        # Quantize
        quantized_flat = self.embeddings.value[indices]

        # Reshape to original shape
        quantized = quantized_flat.reshape(z.shape)

        # Compute losses
        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - z) ** 2)
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(z)) ** 2)

        # VQ loss
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + jax.lax.stop_gradient(quantized - z)

        return quantized, {
            "vq_loss": vq_loss,
            "perplexity": self._compute_perplexity(indices),
            "indices": indices,
        }

    def _compute_perplexity(self, indices: jax.Array) -> jax.Array:
        """Compute codebook perplexity (measure of usage)."""
        # Count frequency of each code
        counts = jnp.bincount(indices, length=self.num_embeddings)
        probs = counts / jnp.sum(counts)

        # Perplexity
        perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-10)))

        return perplexity


class VQVAE(nnx.Module):
    """VQ-VAE model."""

    def __init__(
        self,
        input_shape: tuple,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.input_shape = input_shape

        # Encoder (CNN for images)
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.Conv(64, 128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.Conv(128, embedding_dim, kernel_size=(3, 3), padding="SAME", rngs=rngs),
        )

        # Vector quantizer
        self.vq = VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            rngs=rngs,
        )

        # Decoder
        self.decoder = nnx.Sequential(
            nnx.Conv(embedding_dim, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.ConvTranspose(128, 64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.ConvTranspose(64, 3, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        """Forward pass.

        Args:
            x: Input images (batch, height, width, channels)

        Returns:
            Dictionary with reconstruction and losses
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_quantized, vq_info = self.vq(z)

        # Decode
        reconstruction = self.decoder(z_quantized)
        reconstruction = nnx.sigmoid(reconstruction)

        # Reconstruction loss
        recon_loss = jnp.mean((x - reconstruction) ** 2)

        # Total loss
        total_loss = recon_loss + vq_info["vq_loss"]

        return {
            "reconstruction": reconstruction,
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "vq_loss": vq_info["vq_loss"],
            "perplexity": vq_info["perplexity"],
        }


# Create VQ-VAE
vqvae = VQVAE(
    input_shape=(64, 64, 3),
    embedding_dim=64,
    num_embeddings=512,
    rngs=nnx.Rngs(0),
)

# Training
x = jnp.ones((32, 64, 64, 3))
output = vqvae(x)

print(f"Reconstruction loss: {output['reconstruction_loss']:.4f}")
print(f"VQ loss: {output['vq_loss']:.4f}")
print(f"Perplexity: {output['perplexity']:.2f}")
```

!!! warning "Monitor Codebook Usage"
    **Perplexity** measures how many codebook vectors are actively used:

    - **Perplexity = num_embeddings**: Perfect usage, all codes used equally
    - **Perplexity < 10%  of codebook**: Codebook collapse - many codes unused
    - **Healthy range**: 30-70% of codebook size

    **If perplexity is low:**

    - Increase commitment cost (e.g., 0.25 → 0.5)
    - Use exponential moving average (EMA) updates for codebook
    - Add codebook reset mechanism for unused codes
    - Reduce learning rate for decoder

## Conditional VAE

Conditional VAE generates samples conditioned on labels or attributes.

### Label-Conditional VAE

```python
class ConditionalVAE(nnx.Module):
    """VAE conditioned on labels."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Label embedding
        self.label_embedding = nnx.Embed(
            num_embeddings=num_classes,
            features=hidden_dims[0],
            rngs=rngs,
        )

        # Encoder (input + label embedding)
        encoder_layers = []
        encoder_layers.append(
            nnx.Linear(input_dim + hidden_dims[0], hidden_dims[0], rngs=rngs)
        )

        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(
                nnx.Linear(hidden_dims[i], hidden_dims[i + 1], rngs=rngs)
            )

        self.encoder = encoder_layers

        # Latent layers
        self.mean_layer = nnx.Linear(hidden_dims[-1], latent_dim, rngs=rngs)
        self.logvar_layer = nnx.Linear(hidden_dims[-1], latent_dim, rngs=rngs)

        # Decoder (latent + label embedding)
        decoder_layers = []
        decoder_layers.append(
            nnx.Linear(latent_dim + hidden_dims[0], hidden_dims[-1], rngs=rngs)
        )

        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(
                nnx.Linear(hidden_dims[i], hidden_dims[i - 1], rngs=rngs)
            )

        decoder_layers.append(
            nnx.Linear(hidden_dims[0], input_dim, rngs=rngs)
        )

        self.decoder = decoder_layers

    def encode(self, x: jax.Array, labels: jax.Array) -> dict:
        """Encode with label conditioning."""
        # Embed labels
        label_emb = self.label_embedding(labels)

        # Concatenate input and label
        h = jnp.concatenate([x, label_emb], axis=-1)

        # Forward through encoder
        for layer in self.encoder:
            h = nnx.relu(layer(h))

        # Latent parameters
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)

        return {"mean": mean, "logvar": logvar}

    def decode(self, z: jax.Array, labels: jax.Array) -> jax.Array:
        """Decode with label conditioning."""
        # Embed labels
        label_emb = self.label_embedding(labels)

        # Concatenate latent and label
        h = jnp.concatenate([z, label_emb], axis=-1)

        # Forward through decoder
        for layer in self.decoder:
            h = nnx.relu(layer(h))

        # Sigmoid output
        reconstruction = nnx.sigmoid(h)

        return reconstruction

    def __call__(
        self,
        x: jax.Array,
        labels: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> dict:
        """Forward pass with conditioning."""
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # Encode
        latent_params = self.encode(x_flat, labels)

        # Reparameterize
        if rngs is not None and "sample" in rngs:
            key = rngs.sample()
        else:
            key = jax.random.key(0)

        std = jnp.exp(0.5 * latent_params["logvar"])
        eps = jax.random.normal(key, latent_params["mean"].shape)
        z = latent_params["mean"] + eps * std

        # Decode
        reconstruction = self.decode(z, labels)

        # Reshape
        reconstruction = reconstruction.reshape(x.shape)

        # Loss
        recon_loss = jnp.mean((x_flat - reconstruction.reshape(batch_size, -1)) ** 2)
        kl_loss = -0.5 * jnp.mean(
            1 + latent_params["logvar"]
            - latent_params["mean"] ** 2
            - jnp.exp(latent_params["logvar"])
        )

        return {
            "reconstruction": reconstruction,
            "loss": recon_loss + kl_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }


# Create conditional VAE
cvae = ConditionalVAE(
    input_dim=784,  # 28x28
    latent_dim=20,
    num_classes=10,  # MNIST digits
    hidden_dims=[512, 256],
    rngs=nnx.Rngs(0),
)

# Training with labels
x = jnp.ones((32, 28, 28, 1))
labels = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 3 + [0, 1])

output = cvae(x, labels, rngs=nnx.Rngs(1))
print(f"Loss: {output['loss']:.4f}")

# Generate specific digit
z = jax.random.normal(jax.random.key(0), (10, 20))
target_labels = jnp.arange(10)  # One of each digit
samples = cvae.decode(z, target_labels)
samples = samples.reshape(10, 28, 28, 1)
```

!!! tip "Conditional Generation Trade-offs"
    **Benefits:**
    - **Controlled generation**: Produce specific classes or attributes on demand
    - **Better sample quality**: Conditioning provides additional guidance
    - **Interpretability**: Clear relationship between labels and outputs

    **Considerations:**

    - **Requires labeled data**: Training needs paired (data, label) samples
    - **Reduced diversity**: Model may ignore parts of latent space
    - **Label dependency**: Cannot generate without knowing target labels

    **Best for**: Classification tasks, attribute manipulation, targeted generation

## Beta-VAE with Capacity Control

β-VAE with Capacity Control (Burgess et al.) addresses the training instability of standard β-VAE by gradually increasing the KL capacity instead of using a fixed β weight.

### Key Concept

Instead of minimizing `L = reconstruction_loss + β * KL_loss`, capacity control minimizes:

```
L = reconstruction_loss + γ * |KL_loss - C|
```

Where:

- `C` is the current capacity (gradually increased from 0 to C_max)
- `γ` is a large weight (e.g., 1000) to enforce the capacity constraint
- The model learns to match the KL divergence to the target capacity

### Implementation

```python
from artifex.generative_models.core.configuration import (
    BetaVAEConfig,
    EncoderConfig,
    DecoderConfig,
    CapacityControlConfig,
)
from artifex.generative_models.models.vae import BetaVAEWithCapacity
from flax import nnx

# Create encoder config
encoder_config = EncoderConfig(
    name="capacity_encoder",
    input_shape=(28, 28, 1),  # MNIST
    latent_dim=10,
    hidden_dims=(512, 256),  # Tuple for frozen dataclass
    activation="relu",
)

# Create decoder config
decoder_config = DecoderConfig(
    name="capacity_decoder",
    output_shape=(28, 28, 1),
    latent_dim=10,
    hidden_dims=(256, 512),  # Tuple for frozen dataclass
    activation="relu",
)

# Capacity control config
capacity_config = CapacityControlConfig(
    capacity_max=25.0,  # Maximum KL capacity in nats
    capacity_num_iter=5000,  # Steps to reach max capacity
    gamma=1000.0,  # Weight for capacity constraint
)

# Create β-VAE with capacity control config
config = BetaVAEConfig(
    name="beta_vae_capacity",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=1.0,  # β fixed at 1.0 when using capacity control
    reconstruction_loss_type="mse",
)

# Create model
rngs = nnx.Rngs(params=10, dropout=11, sample=12)
model = BetaVAEWithCapacity(config, rngs=rngs)

# Training produces stable disentanglement
# Monitor: reconstruction_loss, kl_loss, capacity_loss, current_capacity
```

### Training Dynamics

```python
# Forward pass
outputs = model(x)

# Compute losses with step parameter for capacity annealing
losses = model.loss_fn(x=x, outputs=outputs, step=current_step)

# losses contains:
# - "loss": Total loss to optimize
# - "reconstruction_loss": Reconstruction term
# - "kl_loss": KL divergence
# - "capacity_loss": γ * |KL - C|
# - "current_capacity": Current capacity value C
```

!!! tip "Capacity Control Benefits"
    **Why use capacity control over fixed β:**

    - **More stable training**: Gradual capacity increase prevents KL collapse
    - **Better reconstructions**: Model isn't forced to compress early in training
    - **Easier to tune**: Set C_max based on desired disentanglement level
    - **Automatic scheduling**: No need to manually tune β annealing

    **Recommended settings for MNIST:**
    - `capacity_max=25.0`: Good balance of quality and disentanglement
    - `capacity_num_iter=5000-10000`: ~2-4 epochs on MNIST
    - `gamma=1000.0`: Strong enough to enforce constraint

### Monitoring Training

Track these metrics during training:

```python
history = {
    "loss": [],
    "reconstruction_loss": [],
    "kl_loss": [],
    "capacity_loss": [],
    "current_capacity": [],
}

# During training
for step in range(num_steps):
    losses = train_step(model, optimizer, batch, step)

    # Watch current_capacity increase from 0 to capacity_max
    # KL should track current_capacity closely
    print(f"Step {step}: KL={losses['kl_loss']:.2f}, C={losses['current_capacity']:.2f}")
```

## Best Practices

### DO

- ✅ **Tune β carefully** - start with β=1, increase gradually
- ✅ **Monitor KL divergence** - should not collapse to zero
- ✅ **Use β annealing** - gradually increase β during training
- ✅ **Evaluate disentanglement** - use traversals and metrics
- ✅ **Check codebook usage** in VQ-VAE - perplexity should be high
- ✅ **Condition on relevant attributes** - match task requirements
- ✅ **Monitor capacity in capacity-controlled β-VAE** - KL should track current capacity
- ✅ **Visualize latent space** - understand what's learned
- ✅ **Use adequate latent dimensions** - not too small
- ✅ **Save best models** - based on validation metrics

### DON'T

- ❌ **Don't use β=1** if you want disentanglement
- ❌ **Don't ignore posterior collapse** - KL should not be zero
- ❌ **Don't skip codebook monitoring** in VQ-VAE
- ❌ **Don't over-condition** - limits generation diversity
- ❌ **Don't use same architecture for all variants** - customize per model type
- ❌ **Don't skip capacity monitoring** in capacity-controlled β-VAE
- ❌ **Don't forget to normalize inputs** - affects reconstruction
- ❌ **Don't compare losses across variants** - different objectives
- ❌ **Don't skip visualization** - hard to debug otherwise
- ❌ **Don't use too small codebook** in VQ-VAE

## Summary

Advanced VAE variants covered:

1. **β-VAE**: Disentangled representations with β-weighting (β > 1)
2. **β-VAE with Capacity Control**: Stable disentanglement learning using gradual capacity increase
3. **Conditional VAE**: Generation conditioned on labels or attributes
4. **VQ-VAE**: Discrete latent space with vector quantization

Each variant offers different trade-offs:

- **β-VAE**: Better disentanglement through KL weighting, trade-off with reconstruction quality
- **β-VAE with Capacity Control**: More stable training than standard β-VAE, automatic capacity scheduling
- **Conditional VAE**: Controlled generation for specific classes, requires labeled data
- **VQ-VAE**: Discrete latent codes, excellent for compression and hierarchical generation

## Next Steps

<div class="grid cards" markdown>

- :material-creation:{ .lg .middle } **Advanced GANs**

    ---

    Explore StyleGAN and Progressive GAN techniques

    [:octicons-arrow-right-24: Advanced GANs](advanced-gan.md)

- :material-blur:{ .lg .middle } **Advanced Diffusion**

    ---

    Learn classifier guidance and advanced sampling

    [:octicons-arrow-right-24: Advanced Diffusion](advanced-diffusion.md)

- :material-vector-polyline:{ .lg .middle } **Advanced Flows**

    ---

    Implement continuous normalizing flows

    [:octicons-arrow-right-24: Advanced Flows](advanced-flow.md)

- :material-book-open-variant:{ .lg .middle } **VAE Guide**

    ---

    Return to the comprehensive VAE documentation

    [:octicons-arrow-right-24: VAE guide](../../user-guide/models/vae-guide.md)

</div>
