# VAE MNIST Example - Variational Autoencoder Demonstration

**Level:** Beginner | **Runtime:** ~2-3 minutes (CPU/GPU) | **Format:** Python + Jupyter

This example demonstrates how to build a Variational Autoencoder (VAE) on MNIST using Artifex's modular encoder/decoder components. It showcases explicit component creation, proper RNG handling, and VAE inference (no training - this is an architecture demonstration).

## Files

- **Python Script**: [`examples/generative_models/image/vae/vae_mnist.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/image/vae/vae_mnist.py)
- **Jupyter Notebook**: [`examples/generative_models/image/vae/vae_mnist.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/image/vae/vae_mnist.ipynb)

!!! info "Dual-Format Implementation"
    This example is available in two synchronized formats:

    - **Python Script** (.py) - For version control, IDE development, and CI/CD integration
    - **Jupyter Notebook** (.ipynb) - For interactive learning, experimentation, and exploration

    Both formats contain identical content and can be used interchangeably. Choose the format that best suits your workflow.

## Quick Start

```bash
# Activate Artifex environment
source activate.sh

# Run the Python script
python examples/generative_models/image/vae/vae_mnist.py

# Or launch Jupyter notebook
jupyter lab examples/generative_models/image/vae/vae_mnist.ipynb
```

## Overview

**Learning Objectives:**

- [ ] Understand VAE architecture: encoder → latent space → decoder
- [ ] Use Artifex's MLPEncoder and MLPDecoder components
- [ ] Handle RNGs properly in Flax NNX with sample streams
- [ ] Understand the reparameterization trick
- [ ] Generate samples from learned latent space
- [ ] Visualize reconstructions and generations

**Prerequisites:**

- Basic understanding of autoencoders and latent representations
- Familiarity with JAX and Flax NNX basics
- Understanding of variational inference concepts (ELBO, KL divergence)
- Artifex installed

**Estimated Time:** 5 minutes

### What's Covered

<div class="grid cards" markdown>

- :material-cube-outline: **Modular Components**

    ---

    MLPEncoder and MLPDecoder for building VAEs from reusable parts

- :material-shuffle-variant: **RNG Handling**

    ---

    Proper random number generation with separate streams for sampling

- :material-code-braces: **VAE Architecture**

    ---

    Encoder (x → μ, σ), reparameterization (z), decoder (z → x̂)

- :material-eye: **Visualization**

    ---

    Original, reconstructed, and generated samples side-by-side

</div>

**Expected Results:**

- Quick demonstration (~2-3 minutes on CPU, ~30 seconds on GPU)
- Synthetic MNIST-like data (for fast execution without downloads)
- Visualization showing three rows: original, reconstructed, generated images
- Understanding of how to assemble VAE from Artifex components

---

## Theory Background

### Variational Autoencoder (VAE)

A VAE is a generative model that learns a probabilistic latent representation:

**Mathematical Framework:**

- **Encoder**: $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ - Approximate posterior
- **Decoder**: $p_\theta(x|z)$ - Likelihood of data given latent code
- **Prior**: $p(z) = \mathcal{N}(0, I)$ - Standard normal prior

**VAE Loss (ELBO - Evidence Lower Bound):**

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

Where:

- **Reconstruction term**: $\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\|x - \hat{x}\|^2$ (MSE)
- **KL term**: $\text{KL}(q(z|x) \| p(z))$ has closed form for Gaussians

### Reparameterization Trick

To enable backpropagation through stochastic sampling:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This separates the stochastic component (ε) from learnable parameters (μ, σ), allowing gradients to flow through the sampling operation.

!!! tip "Why Reparameterization?"
    Without this trick, we couldn't backpropagate through random sampling because sampling is not differentiable. By expressing z as a deterministic function of ε, μ, and σ, we can compute gradients with respect to μ and σ.

---

## Imports

Import Artifex's modular VAE components:

- **MLPEncoder**: Maps inputs to latent distribution parameters (μ, log σ²)
- **MLPDecoder**: Maps latent codes to reconstructions
- **VAE**: Base VAE class that combines encoder + decoder with ELBO loss

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import VAE
```

!!! note "Config-Based API"
    Artifex uses frozen dataclass configurations for all model components. This provides type-safe, validated configuration with clear parameter structure. See the "Configuration Pattern" section below for details.

---

## Data Loading

For this example, we create synthetic MNIST-like data. In production, you would use real MNIST from `tensorflow_datasets` or `torchvision`.

**Data Format:**

- Images: 28×28×1 (grayscale)
- Values: [0, 1] range (normalized)
- Shape: (batch_size, height, width, channels)

```python
def load_mnist_data():
    """Load MNIST dataset.

    In this example, we use synthetic data for quick demonstration.
    Replace this with real MNIST loading for production use.

    Returns:
        Tuple of (train_images, test_images)

    Note:
        Real MNIST loading would look like:
        ```python
        import tensorflow_datasets as tfds
        ds = tfds.load('mnist', split='train', as_supervised=True)
        images = ds.map(lambda x, y: x / 255.0)  # Normalize to [0, 1]
        ```
    """
    # Create synthetic MNIST-like data with proper dimensions
    key = jax.random.key(42)
    train_key, test_key = jax.random.split(key)

    # Create synthetic data: 28×28×1 images in [0, 1] range
    train_images = jax.random.uniform(train_key, (1000, 28, 28, 1))
    test_images = jax.random.uniform(test_key, (100, 28, 28, 1))

    return train_images, test_images
```

---

## Visualization Function

The visualization function shows three rows to assess VAE quality:

1. **Original**: Input images from the dataset
2. **Reconstructed**: VAE reconstructions (tests encoder + decoder quality)
3. **Generated**: Samples from random latent codes (tests learned prior)

```python
def visualize_vae_results(original, reconstructed, generated, num_samples=5):
    """Visualize VAE results side-by-side.

    Args:
        original: Original images [batch, height, width, channels]
        reconstructed: Reconstructed images (same shape as original)
        generated: Generated images from random latent codes
        num_samples: Number of samples to display (default: 5)

    Returns:
        matplotlib.figure.Figure: The created figure

    Note:
        All images should be in [0, 1] range for proper visualization.
        Images are clipped to [0, 1] before display to handle any overshooting.
    """
    fig, axes = plt.subplots(3, num_samples, figsize=(12, 7))

    for i in range(num_samples):
        # Row 1: Original images
        axes[0, i].imshow(jnp.clip(original[i, :, :, 0], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12, fontweight="bold")

        # Row 2: Reconstructed images
        axes[1, i].imshow(jnp.clip(reconstructed[i, :, :, 0], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=12, fontweight="bold")

        # Row 3: Generated images (from random latent codes)
        axes[2, i].imshow(jnp.clip(generated[i, :, :, 0], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Generated", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig
```

---

## Main Pipeline

The main function demonstrates the VAE workflow using Artifex's modular components:

1. **Setup**: Initialize RNG for reproducibility
2. **Data**: Load MNIST (synthetic in this demo)
3. **Encoder**: Create MLPEncoder to map x → (μ, log σ²)
4. **Decoder**: Create MLPDecoder to map z → x̂
5. **VAE**: Combine encoder + decoder into full VAE
6. **Forward Pass**: Run reconstruction
7. **Generation**: Sample from prior
8. **Visualization**: Display results

!!! tip "Why Explicit Component Creation?"
    We explicitly create encoder and decoder to demonstrate:

    - How to configure Artifex's components
    - The modular design pattern (reusable, swappable parts)
    - How components connect in the VAE
    - Easy customization (swap MLP → CNN, adjust layers, etc.)

    This approach gives you full control and understanding. For production workflows where you want consistency across model types, the factory pattern (see below) is also available.

```python
def main():
    """Run the VAE MNIST example.

    This function demonstrates the complete VAE pipeline using Artifex's
    modular encoder/decoder components.
    """
    print("=" * 80)
    print("VAE MNIST Example - Using Artifex's MLPEncoder & MLPDecoder")
    print("=" * 80)
```

---

### Step 1: Setup RNG

In Flax NNX, we use `nnx.Rngs` to manage random number generation. We need separate streams for:

- **params**: Parameter initialization
- **dropout**: Dropout layers (if used)
- **sample**: Stochastic sampling in VAE (reparameterization trick)

```python
    # Step 1: Set random seed for reproducibility
    seed = 42
    key = jax.random.key(seed)
    params_key, dropout_key, sample_key = jax.random.split(key, 3)

    # Create RNG streams for different purposes
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key, sample=sample_key)
```

!!! tip "RNG Best Practices"
    Always split your RNG into separate streams for different purposes. This ensures:

    - Reproducibility across runs
    - Proper handling of stochastic operations
    - Thread-safe random state management
    - No interference between parameter init and sampling

---

### Step 2: Load Data

MNIST consists of 28×28 grayscale images of handwritten digits (0-9).

- Training set: 60,000 images (we use 1,000 synthetic for demo)
- Test set: 10,000 images (we use 100 synthetic for demo)

Images are normalized to [0, 1] range for stable training.

```python
    # Step 2: Load data
    print("\n📊 Loading MNIST data...")
    train_images, test_images = load_mnist_data()
    print(f"  Train data shape: {train_images.shape}")  # (1000, 28, 28, 1)
    print(f"  Test data shape: {test_images.shape}")  # (100, 28, 28, 1)
```

**Output:**

```
📊 Loading MNIST data...
  Train data shape: (1000, 28, 28, 1)
  Test data shape: (100, 28, 28, 1)
```

---

### Step 3: Create Encoder

Artifex's `MLPEncoder` maps inputs to latent distribution parameters:

- **Input**: x (28×28×1 = 784 features after flattening)
- **Output**: (mean, log_var) for latent distribution q(z|x)

**Parameters:**

- `hidden_dims=[256, 128]`: Two hidden layers with decreasing dimensions
- `latent_dim=32`: Dimension of latent space z
- `activation="relu"`: ReLU activation between layers
- `input_dim=(28, 28, 1)`: Shape of input images (auto-flattened to 784)

```python
    # Step 3: Create encoder config using Artifex's EncoderConfig
    print("\n🔧 Creating VAE components using Artifex APIs...")

    latent_dim = 32

    # Create encoder configuration (frozen dataclass)
    encoder_config = EncoderConfig(
        name="vae_encoder",
        hidden_dims=(256, 128),  # Encoder architecture (tuple for frozen dataclass)
        latent_dim=latent_dim,  # Latent space dimension
        activation="relu",  # Activation function
        input_shape=(28, 28, 1),  # Input image shape
    )
    print(f"  ✅ Encoder config: hidden_dims={encoder_config.hidden_dims}, latent_dim={latent_dim}")
```

**Output:**

```
🔧 Creating VAE components using Artifex APIs...
  ✅ Encoder config: hidden_dims=(256, 128), latent_dim=32
```

---

### Step 4: Create Decoder

Artifex's `MLPDecoder` maps latent codes to reconstructions:

- **Input**: z (32-dimensional latent vector)
- **Output**: x̂ (28×28×1 reconstructed image)

**Parameters:**

- `hidden_dims=[128, 256]`: Reversed encoder dims (symmetric architecture)
- `output_dim=(28, 28, 1)`: Shape of reconstructed images
- `latent_dim=32`: Dimension of latent space (must match encoder)
- `activation="relu"`: ReLU activation (except final layer uses sigmoid)

!!! note "Automatic Output Activation"
    The decoder automatically applies sigmoid activation to the output to ensure pixel values are in [0, 1] range. This matches the input image range.

```python
    # Step 4: Create decoder config using Artifex's DecoderConfig
    decoder_config = DecoderConfig(
        name="vae_decoder",
        hidden_dims=(128, 256),  # Decoder architecture (reversed, tuple)
        output_shape=(28, 28, 1),  # Output image shape
        latent_dim=latent_dim,  # Latent space dimension
        activation="relu",  # Activation function
    )
    print(f"  ✅ Decoder config: hidden_dims={decoder_config.hidden_dims}, output_shape={decoder_config.output_shape}")
```

**Output:**

```
  ✅ Decoder config: hidden_dims=(128, 256), output_shape=(28, 28, 1)
```

---

### Step 5: Create VAE Model

Artifex's `VAE` class combines encoder + decoder with:

- **Forward pass**: x → encoder → (μ, log σ²) → sample z → decoder → x̂
- **ELBO loss**: Reconstruction loss + KL divergence
- **Sampling methods**: Generate from prior p(z) = N(0, I)

**Parameters:**

- `encoder`: The MLPEncoder we created above
- `decoder`: The MLPDecoder we created above
- `latent_dim=32`: Must match encoder/decoder latent dimensions
- `kl_weight=1.0`: Weight for KL term (β-VAE uses β≠1 for disentanglement)

```python
    # Step 5: Create VAE model with config
    vae_config = VAEConfig(
        name="vae_mnist",
        encoder=encoder_config,
        decoder=decoder_config,
        encoder_type="dense",  # MLP encoder
        kl_weight=1.0,  # Standard VAE (β=1), increase for β-VAE
        reconstruction_loss_type="mse",
    )

    model = VAE(config=vae_config, rngs=rngs)
    print(f"  ✅ VAE model created: latent_dim={vae_config.encoder.latent_dim}, kl_weight={vae_config.kl_weight}")
```

**Output:**

```
  ✅ VAE model created: latent_dim=32, kl_weight=1.0
```

---

### Step 6: Forward Pass (Reconstruction)

The forward pass demonstrates the full VAE pipeline:

1. **Encoding**: x → encoder → (μ, log σ²)
2. **Reparameterization**: z = μ + σ ⊙ ε, where ε ~ N(0, I)
3. **Decoding**: z → decoder → x̂ (reconstruction)

**Output Dictionary:**

- `reconstructed` or `reconstruction`: Reconstructed images x̂
- `mean`: Latent distribution mean μ
- `log_var` or `logvar`: Latent distribution log variance log σ²
- `z`: Sampled latent codes (used for reconstruction)

!!! warning "RNG for Sampling"
    We pass `rngs` with a `sample` stream for the reparameterization trick's random sampling. Without this, the VAE would use deterministic (mean) latent codes.

```python
    # Step 6: Test the model with a batch
    print("\n🧪 Testing model forward pass...")
    test_batch = train_images[:8]  # Use 8 images for testing

    # Forward pass - the VAE uses its internal rngs for reparameterization
    # The 'sample' RNG stream is used internally for the reparameterization trick
    outputs = model(test_batch)

    # Extract reconstructions (check both possible keys)
    reconstructed = outputs.get("reconstructed")
    if reconstructed is None:
        reconstructed = outputs["reconstruction"]
    print(f"  ✅ Reconstruction shape: {reconstructed.shape}")

    # Extract latent codes
    latent = outputs.get("z")
    if latent is None:
        latent = outputs["latent"]
    print(f"  ✅ Latent shape: {latent.shape}")

    # Show latent statistics to verify reasonable values
    print("  📊 Latent statistics:")
    print(f"     Mean: {jnp.mean(latent):.4f} (should be near 0)")
    print(f"     Std: {jnp.std(latent):.4f} (should be near 1 for standard normal)")
```

**Output:**

```
🧪 Testing model forward pass...
  ✅ Reconstruction shape: (8, 28, 28, 1)
  ✅ Latent shape: (8, 32)
  📊 Latent statistics:
     Mean: 0.0315 (should be near 0)
     Std: 1.0909 (should be near 1 for standard normal)
```

!!! tip "Interpreting Latent Statistics"
    The latent codes should have:

    - **Mean near 0**: The KL term pushes the posterior toward the prior N(0, I)
    - **Std near 1**: Standard deviation close to 1 indicates good regularization

    If mean or std deviate significantly, consider adjusting `kl_weight` or using KL annealing during training.

---

### Step 7: Generation from Prior

To generate new samples:

1. Sample z ~ N(0, I) from the standard normal prior
2. Decode: x_new = decoder(z)

This tests whether the VAE has learned a meaningful latent space.

**Quality Indicators:**

- **Diversity**: Generated samples should vary (not all identical)
- **Realism**: Samples should resemble training data distribution
- **Smoothness**: Similar z should produce similar x (interpolation works)

!!! note "Synthetic Data Limitation"
    With synthetic random data, generations won't be realistic digits, but the shapes should match the training distribution. With real MNIST, you'd see clear digit reconstructions and realistic generated digits.

```python
    # Step 7: Generate new samples from the prior
    print("\n🎨 Generating new samples from prior...")
    n_samples = 5
    generated = model.generate(n_samples=n_samples)  # VAE uses internal rngs
    print(f"  ✅ Generated shape: {generated.shape}")
    print(f"  📊 Generated pixels range: [{jnp.min(generated):.3f}, {jnp.max(generated):.3f}]")
```

**Output:**

```
🎨 Generating new samples from prior...
  ✅ Generated shape: (5, 28, 28, 1)
  📊 Generated pixels range: [0.059, 0.943]
```

---

### Step 8: Visualization

The visualization shows:

- **Top row**: Original input images
- **Middle row**: Reconstructions (tests encoder + decoder quality)
- **Bottom row**: Generated samples (tests learned prior)

**What to look for:**

- Reconstructions should closely match originals (good reconstruction loss)
- Generated samples should look plausible (good latent space)
- Diversity in generated samples indicates good latent space coverage

```python
    # Step 8: Visualize results
    print("\n📊 Visualizing results...")
    fig = visualize_vae_results(
        original=test_batch[:n_samples],
        reconstructed=reconstructed[:n_samples],
        generated=generated[:n_samples],
    )

    # Step 9: Save figure
    import os

    output_dir = "examples_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vae_mnist_results.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Results saved to {output_path}")
```

**Output:**

```
📊 Visualizing results...
  ✅ Results saved to examples_output/vae_mnist_results.png
```

---

## Summary

In this example, you learned:

- ✅ **VAE architecture**: encoder → latent space → decoder
- ✅ **Reparameterization trick**: enables backpropagation through sampling
- ✅ **Artifex's MLPEncoder and MLPDecoder**: modular, reusable components
- ✅ **Proper RNG handling**: use rngs with 'sample' stream for stochastic operations
- ✅ **VAE base class**: handles ELBO loss computation automatically

**Key Insights:**

- VAEs trade reconstruction quality for smooth, structured latent spaces
- The latent dimension (32) controls representation capacity
- KL weight controls reconstruction vs. regularization tradeoff
- Modular design allows easy swapping (MLP → CNN, different layers, etc.)

**Artifex APIs Used:**

- `MLPEncoder`: Maps inputs → (μ, log σ²)
- `MLPDecoder`: Maps latent codes → reconstructions
- `VAE`: Combines encoder/decoder with ELBO loss

---

## Configuration Pattern

Artifex uses frozen dataclass configurations for type-safe, validated model setup. The nested config pattern provides clear structure:

```python
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import VAE

# Create encoder configuration
encoder_config = EncoderConfig(
    name="vae_encoder",
    hidden_dims=(256, 128),  # Must be tuple (frozen dataclass)
    latent_dim=32,
    activation="relu",
    input_shape=(28, 28, 1),
)

# Create decoder configuration
decoder_config = DecoderConfig(
    name="vae_decoder",
    hidden_dims=(128, 256),
    output_shape=(28, 28, 1),
    latent_dim=32,
    activation="relu",
)

# Create VAE configuration with nested configs
vae_config = VAEConfig(
    name="vae_mnist",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    kl_weight=1.0,
    reconstruction_loss_type="mse",
)

# Create model with config
model = VAE(config=vae_config, rngs=rngs)
```

**Benefits of config-based API:**

| **Feature** | **Benefit** |
|-------------|-------------|
| ✅ Type validation | Catches errors at config creation time |
| ✅ Frozen dataclasses | Immutable, hashable configurations |
| ✅ Nested structure | Clear component relationships |
| ✅ Serializable | Easy to save/load configurations |

This is the recommended approach for all Artifex models.

---

## Experiments to Try

### 1. CNN Architecture

CNNs often work better for image data than MLPs:

```python
# Use CNN encoder type in config
encoder_config = EncoderConfig(
    name="cnn_encoder",
    hidden_dims=(32, 64, 128),  # Channel progression
    latent_dim=32,
    activation="relu",
    input_shape=(28, 28, 1),
)

decoder_config = DecoderConfig(
    name="cnn_decoder",
    hidden_dims=(128, 64, 32),
    output_shape=(28, 28, 1),
    latent_dim=32,
    activation="relu",
)

vae_config = VAEConfig(
    name="cnn_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="cnn",  # Use CNN encoder
    kl_weight=1.0,
)

model = VAE(config=vae_config, rngs=rngs)
```

### 2. Latent Dimension Experiments

- Try `latent_dim=16`: Smaller capacity, faster training, may lose details
- Try `latent_dim=64`: Larger capacity, better reconstructions
- Try `latent_dim=128`: Very high capacity, risk of overfitting

**Trade-off:** Larger latent dims → better reconstruction but less structured space

### 3. β-VAE for Disentanglement

```python
from artifex.generative_models.core.configuration.vae_config import BetaVAEConfig
from artifex.generative_models.models.vae import BetaVAE

beta_config = BetaVAEConfig(
    name="beta_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",
    beta_default=4.0,  # β=4 encourages disentangled representations
    beta_warmup_steps=1000,  # Gradual warmup
)

model = BetaVAE(config=beta_config, rngs=rngs)
```

- **Higher β** (4.0-10.0): More regularization, worse reconstruction, better disentanglement
- **Lower β** (0.1-0.5): Better reconstruction, less structured latent space
- **β=1**: Standard VAE

### 4. Architecture Variations

```python
# Deeper network with different activation
encoder_config = EncoderConfig(
    name="deep_encoder",
    hidden_dims=(512, 256, 128),  # More layers
    latent_dim=32,
    activation="gelu",  # GELU often works better than ReLU
    input_shape=(28, 28, 1),
)
```

- **More layers**: Higher capacity but slower training
- **Different activations**: GELU often works better than ReLU
- **Batch normalization**: Can help with deeper networks

### 5. Real MNIST Data

```python
import tensorflow_datasets as tfds

# Load real MNIST
ds = tfds.load('mnist', split='train', as_supervised=True)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

ds = ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
```

Real MNIST will give much better results and realistic digit generation.

---

## Troubleshooting

### Encoder/decoder creation fails

**Solution**: Check that `input_dim` matches your data shape

**Common mistake**: Forgetting to pass `rngs` parameter

**Fix**: All Artifex modules require `rngs` for initialization

### Reconstructions are very blurry

**Solution**: This is expected with MSE loss on images

**Explanation**: MSE averages over pixel space, causing blur

**Alternatives**:

- Use CNNEncoder/CNNDecoder instead of MLP
- Try perceptual loss or adversarial training
- Use VQVAE for sharper reconstructions

### Generated samples look like noise

**Solution**: VAE needs training; this example only demonstrates architecture

**Note**: With synthetic random data, generations won't be meaningful

**Fix**: Train on real MNIST with a proper training loop (see training examples)

### KL collapse (all latent codes become identical)

**Solution**: Reduce `kl_weight` to allow more latent variance

**Monitoring**: Check `jnp.std(latent)` - should be > 0.5

**Fix**: Use KL annealing schedule (start with kl_weight=0.1, increase gradually)

### Model output shape mismatch

**Solution**: Ensure encoder `latent_dim` matches decoder `latent_dim`

**Check**: Verify `output_dim` matches input shape for reconstruction

---

## Next Steps

After understanding this basic VAE demonstration, explore:

### Related Examples

<div class="grid cards" markdown>

- :material-school: **Training VAE**

    ---

    See training examples for full training loops with optimizers and loss monitoring

- :material-star-four-points: **Advanced VAEs**

    ---

    β-VAE, Conditional VAE, VQ-VAE, Hierarchical VAE with disentanglement

- :material-chart-line: **Disentanglement**

    ---

    Multi-β-VAE benchmark with MIG score evaluation and metrics

- :material-compare: **VAE vs GAN**

    ---

    Compare `simple_gan.py` to understand trade-offs between approaches

</div>

### Documentation Resources

- **[VAE Concepts](../../user-guide/concepts/vae-explained.md)**: Deep dive into VAE theory
- **[VAE User Guide](../../user-guide/models/vae-guide.md)**: Advanced usage patterns
- **[VAE API Reference](../../api/models/vae.md)**: Complete API documentation
- **[Training Guide](../../user-guide/training/training-guide.md)**: How to train VAEs from scratch

### Research Papers

1. **Auto-Encoding Variational Bayes** (Kingma & Welling, 2014)
   Original VAE paper: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

2. **β-VAE: Learning Basic Visual Concepts** (Higgins et al., 2017)
   Disentanglement via β parameter: [https://openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)

3. **Understanding disentangling in β-VAE** (Burgess et al., 2018)
   Analysis of β-VAE disentanglement: [https://arxiv.org/abs/1804.03599](https://arxiv.org/abs/1804.03599)

---

**Congratulations! You've learned how to build VAEs with Artifex's modular components!** 🎉
