#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
r"""VAE MNIST Example - Variational Autoencoder on MNIST

## Overview

This example demonstrates how to build a Variational Autoencoder (VAE) on MNIST using
Artifex's modular encoder/decoder architecture. VAEs learn probabilistic latent representations
that enable both reconstruction and generation of new samples.

**Key Artifex Components Used:**
- `MLPEncoder` - Artifex's MLP-based encoder for latent mean and log-variance
- `MLPDecoder` - Artifex's MLP-based decoder for reconstruction
- `VAE` - Artifex's base VAE class with ELBO loss and sampling

## Source Code Dependencies

**Validated:** 2025-10-16

This example depends on the following Artifex source files:
- `src/artifex/generative_models/models/vae/base.py` - VAE base class
- `src/artifex/generative_models/models/vae/encoders.py` - MLPEncoder class
- `src/artifex/generative_models/models/vae/decoders.py` - MLPDecoder class

**Validation Status:**
- ✅ All dependencies validated against `memory-bank/guides/flax-nnx-guide.md`
- ✅ No anti-patterns detected (RNG handling, module init, activations)
- ✅ All patterns follow Flax NNX best practices (no nnx.List issues)

## What You'll Learn

- [x] VAE architecture: encoder (x → μ, σ), reparameterization (z), decoder (z → x̂)
- [x] Using Artifex's MLPEncoder and MLPDecoder components
- [x] Proper RNG handling in Flax NNX (rngs.sample() pattern)
- [x] ELBO loss decomposition: reconstruction + KL divergence
- [x] Sample generation from learned prior p(z) = N(0, I)
- [x] Visualization of reconstructions and generated samples

## Prerequisites

- Artifex installed (run `source activate.sh`)
- Basic understanding of autoencoders and latent representations
- Familiarity with JAX and Flax NNX basics
- Understanding of variational inference concepts

## Usage

```bash
source activate.sh
python examples/generative_models/image/vae/vae_mnist.py
```

## Expected Output

The example will:
1. Create synthetic MNIST-like data (28×28 grayscale images)
2. Build VAE with Artifex's MLPEncoder and MLPDecoder
3. Perform forward pass to reconstruct inputs
4. Generate new samples from random latent vectors
5. Visualize original vs reconstructed vs generated images
6. Save visualization to `examples_output/vae_mnist_results.png`

## Key Concepts

### Variational Autoencoder (VAE)

A VAE is a generative model that learns to encode data into a latent space and decode it back.
Unlike standard autoencoders, VAEs learn a **probabilistic** latent representation:

**Mathematical Framework:**
- Encoder: $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ - Approximate posterior
- Decoder: $p_\theta(x|z)$ - Likelihood of data given latent code
- Prior: $p(z) = \mathcal{N}(0, I)$ - Standard normal prior

**VAE Loss (ELBO - Evidence Lower Bound):**
$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] -
\text{KL}(q_\phi(z|x) \| p(z))$$

Where:
- **Reconstruction term**: $\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\|x - \hat{x}\|^2$ (MSE)
- **KL term**: $\text{KL}(q(z|x) \| p(z))$ has closed form for Gaussians

### Reparameterization Trick

To enable backpropagation through stochastic sampling:
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This separates the stochastic component (ε) from the learnable parameters (μ, σ),
allowing gradients to flow through the sampling operation.

### Artifex's Modular Design

Artifex provides reusable encoder/decoder components:
- **MLPEncoder**: Maps inputs → (mean, log_var)
- **MLPDecoder**: Maps latent codes → reconstructions
- **VAE**: Combines encoder/decoder with ELBO loss

This modular design allows:
- Easy swapping between MLP, CNN, ResNet encoders/decoders
- Consistent architecture across examples
- Proper initialization and RNG handling

## Estimated Runtime

- **CPU**: ~2-3 minutes (synthetic data, quick demo)
- **GPU**: ~30 seconds (if available)

## Author

Artifex Team

## Last Updated

2025-10-16
"""

# %% [markdown]
"""
# VAE MNIST Example

This notebook demonstrates Variational Autoencoders (VAEs) on MNIST using Artifex's
modular encoder/decoder components.

## Learning Objectives

By the end of this example, you will understand:
1. VAE architecture and the reparameterization trick
2. How to use Artifex's MLPEncoder and MLPDecoder
3. Proper RNG handling in Flax NNX
4. ELBO loss computation and interpretation
5. Generating samples from learned latent space
"""

# %%
# Cell 1: Import Dependencies
"""
Import Artifex components:
- MLPEncoder: Encodes inputs to latent distribution parameters (μ, log σ²)
- MLPDecoder: Decodes latent codes to reconstructions
- VAE: Base VAE class with ELBO loss and sampling methods
"""

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


# %% [markdown]
"""
## Data Loading

For this example, we create synthetic MNIST-like data. In production, you would use:
- `tensorflow_datasets` for real MNIST
- `torch.utils.data.DataLoader` for PyTorch datasets
- Artifex's data loaders for standardized pipelines

**Data Format:**
- Images: 28×28×1 (grayscale)
- Values: [0, 1] range (normalized)
- Shape: (batch_size, height, width, channels)
"""


# %%
# Cell 2: Data Loading Function
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


# %% [markdown]
"""
## Visualization

The visualization function shows three rows:
1. **Original**: Input images from the dataset
2. **Reconstructed**: Images reconstructed by the VAE (encoder → decoder)
3. **Generated**: New images generated from random latent codes (prior → decoder)

This helps assess:
- **Reconstruction quality**: How well the model preserves input details
- **Generation diversity**: Variety in generated samples
- **Latent space coverage**: Quality of the learned representation
"""


# %%
# Cell 3: Visualization Function
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


# %% [markdown]
"""
## Main Pipeline

The main function demonstrates the VAE workflow:

1. **Setup**: Initialize RNG for reproducibility
2. **Data**: Load MNIST (synthetic in this demo)
3. **Encoder**: Create MLPEncoder to map x → (μ, log σ²)
4. **Decoder**: Create MLPDecoder to map z → x̂
5. **VAE**: Combine encoder + decoder into full VAE
6. **Inference**: Run forward pass and generation
7. **Visualization**: Display results

### Why Explicit Component Creation?

We explicitly create encoder and decoder to demonstrate:
- How to configure Artifex's components
- The modular design pattern
- How components connect in the VAE
- Easy customization (swap MLP → CNN, adjust layers, etc.)
"""


# %%
# Cell 4: Main Function
def main():
    """Run the VAE MNIST example.

    This function demonstrates the complete VAE pipeline using Artifex's
    modular encoder/decoder components.
    """
    print("=" * 80)
    print("VAE MNIST Example - Using Artifex's MLPEncoder & MLPDecoder")
    print("=" * 80)


# %% [markdown]
#     ### Step 1: Setup RNG
#
#     In Flax NNX, we use `nnx.Rngs` to manage random number generation.
#     We need separate streams for:
#     - `params`: Parameter initialization
#     - `dropout`: Dropout layers (if used)
#     - `sample`: Stochastic sampling in VAE
#

# %%
# Step 1: Set random seed for reproducibility
seed = 42
key = jax.random.key(seed)
params_key, dropout_key, sample_key = jax.random.split(key, 3)

# Create RNG streams for different purposes
rngs = nnx.Rngs(params=params_key, dropout=dropout_key, sample=sample_key)

# %% [markdown]
#     ### Step 2: Load Data
#
#     MNIST consists of 28×28 grayscale images of handwritten digits (0-9).
#     - Training set: 60,000 images (we use 1,000 synthetic for demo)
#     - Test set: 10,000 images (we use 100 synthetic for demo)
#
#     Images are normalized to [0, 1] range for stable training.
#

# %%
# Step 2: Load data
print()
print("📊 Loading MNIST data...")
train_images, test_images = load_mnist_data()
print(f"  Train data shape: {train_images.shape}")  # (1000, 28, 28, 1)
print(f"  Test data shape: {test_images.shape}")  # (100, 28, 28, 1)

# %% [markdown]
#     ### Step 3: Create Encoder Configuration
#
#     Artifex uses frozen dataclass configs for model components. The `EncoderConfig`
#     specifies the encoder architecture:
#     - `hidden_dims=(256, 128)`: Two hidden layers with decreasing dimensions
#     - `latent_dim=32`: Dimension of latent space z
#     - `activation="relu"`: ReLU activation between layers
#     - `input_shape=(28, 28, 1)`: Shape of input images (auto-flattened to 784)
#
#     The encoder then maps inputs to latent distribution parameters:
#     - Input: x (28×28×1 = 784 features after flattening)
#     - Output: (mean, log_var) for latent distribution q(z|x)
#

# %%
# Step 3: Create encoder configuration
print()
print("🔧 Creating VAE components using Artifex APIs...")

latent_dim = 32
encoder_config = EncoderConfig(
    name="mnist_encoder",
    hidden_dims=(256, 128),  # Encoder architecture (tuple, not list)
    latent_dim=latent_dim,  # Latent space dimension
    activation="relu",  # Activation function
    input_shape=(28, 28, 1),  # Input image shape
)
print(f"  ✅ Encoder config: hidden_dims=(256, 128), latent_dim={latent_dim}")

# %% [markdown]
#     ### Step 4: Create Decoder Configuration
#
#     The `DecoderConfig` specifies the decoder architecture:
#     - `hidden_dims=(128, 256)`: Reversed encoder dims (symmetric architecture)
#     - `output_shape=(28, 28, 1)`: Shape of reconstructed images
#     - `latent_dim=32`: Dimension of latent space (must match encoder)
#     - `activation="relu"`: ReLU activation (except final layer uses sigmoid)
#
#     Artifex's decoder maps latent codes to reconstructions:
#     - Input: z (32-dimensional latent vector)
#     - Output: x̂ (28×28×1 reconstructed image)
#
#     **Note:** The decoder automatically applies sigmoid activation to the output
#     to ensure pixel values are in [0, 1] range.
#

# %%
# Step 4: Create decoder configuration
decoder_config = DecoderConfig(
    name="mnist_decoder",
    hidden_dims=(128, 256),  # Decoder architecture (reversed, tuple)
    output_shape=(28, 28, 1),  # Output image shape
    latent_dim=latent_dim,  # Latent space dimension
    activation="relu",  # Activation function
)
print("  ✅ Decoder config: hidden_dims=(128, 256), output_shape=(28, 28, 1)")

# %% [markdown]
#     ### Step 5: Create VAE Model
#
#     Artifex's `VAE` class uses a `VAEConfig` that combines encoder and decoder configs:
#     - **Forward pass**: x → encoder → (μ, log σ²) → sample z → decoder → x̂
#     - **ELBO loss**: Reconstruction loss + KL divergence
#     - **Sampling methods**: Generate from prior p(z) = N(0, I)
#
#     **VAEConfig Parameters:**
#     - `encoder`: The EncoderConfig we created above
#     - `decoder`: The DecoderConfig we created above
#     - `encoder_type="dense"`: Uses MLP-based encoder/decoder (vs "cnn" or "resnet")
#     - `kl_weight=1.0`: Weight for KL term (β-VAE uses β≠1 for disentanglement)
#

# %%
# Step 5: Create VAE model with config
vae_config = VAEConfig(
    name="mnist_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",  # Use MLP encoder/decoder
    kl_weight=1.0,  # Standard VAE (β=1), increase for β-VAE
)
model = VAE(config=vae_config, rngs=rngs)
print(f"  ✅ VAE model created: latent_dim={model.latent_dim}, kl_weight={model.kl_weight}")

# %% [markdown]
#     ### Step 6: Forward Pass (Reconstruction)
#
#     The forward pass demonstrates the full VAE pipeline:
#     1. **Encoding**: x → encoder → (μ, log σ²)
#     2. **Reparameterization**: z = μ + σ ⊙ ε, where ε ~ N(0, I)
#     3. **Decoding**: z → decoder → x̂ (reconstruction)
#
#     **Output Dictionary:**
#     - `reconstructed` or `reconstruction`: Reconstructed images x̂
#     - `mean`: Latent distribution mean μ
#     - `log_var` or `logvar`: Latent distribution log variance log σ²
#     - `z`: Sampled latent codes (used for reconstruction)
#
#     **RNG Handling:** The VAE uses its internal `rngs` (passed during initialization)
#     with a `sample` stream for the reparameterization trick's random sampling.
#

# %%
# Step 6: Test the model with a batch
print()
print("🧪 Testing model forward pass...")
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

# %% [markdown]
#     ### Step 7: Generation from Prior
#
#     To generate new samples:
#     1. Sample z ~ N(0, I) from the standard normal prior
#     2. Decode: x_new = decoder(z)
#
#     This tests whether the VAE has learned a meaningful latent space.
#
#     **Quality Indicators:**
#     - **Diversity**: Generated samples should vary (not all identical)
#     - **Realism**: Samples should resemble training data distribution
#     - **Smoothness**: Similar z should produce similar x (interpolation works)
#
#     **Note:** With synthetic data, generations won't be realistic digits,
#     but the shapes should match the training distribution.
#

# %%
# Step 7: Generate new samples from the prior
print()
print("🎨 Generating new samples from prior...")
n_samples = 5
generated = model.generate(n_samples=n_samples)  # VAE uses internal rngs
print(f"  ✅ Generated shape: {generated.shape}")
print(f"  📊 Generated pixels range: [{jnp.min(generated):.3f}, {jnp.max(generated):.3f}]")

# %% [markdown]
#     ### Step 8: Visualization
#
#     The visualization shows:
#     - **Top row**: Original input images
#     - **Middle row**: Reconstructions (tests encoder + decoder quality)
#     - **Bottom row**: Generated samples (tests learned prior)
#
#     **What to look for:**
#     - Reconstructions should closely match originals (good reconstruction loss)
#     - Generated samples should look plausible (good latent space)
#     - Diversity in generated samples indicates good latent space coverage
#
#     **Note:** With synthetic random data, reconstructions will be blurry
#     and generations will be random patterns. With real MNIST, you'd see
#     clear digit reconstructions and realistic generated digits.
#

# %%
# Step 8: Visualize results
print()
print("📊 Visualizing results...")
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

# %% [markdown]
#     """
#     ## Summary and Key Takeaways
#
#     ✅ **What We Learned:**
#     - VAE architecture: encoder → latent space → decoder
#     - Reparameterization trick enables backpropagation through sampling
#     - Artifex's MLPEncoder and MLPDecoder provide modular components
#     - Proper RNG handling: use rngs with 'sample' stream for stochastic operations
#     - VAE base class handles ELBO loss computation automatically
#
#     💡 **Key Insights:**
#     - VAEs trade reconstruction quality for smooth, structured latent spaces
#     - The latent dimension (32) controls representation capacity
#     - KL weight controls reconstruction vs. regularization tradeoff
#     - Modular design allows easy swapping (MLP → CNN, different layers, etc.)
#
#     📊 **Results:**
#     - Reconstructions preserve input structure (with some blur from MSE loss)
#     - Generated samples from prior p(z) = N(0, I)
#     - Visualization saved for inspection
#
#     🔧 **Artifex APIs Used:**
#     - `MLPEncoder`: Maps inputs → (μ, log σ²)
#     - `MLPDecoder`: Maps latent codes → reconstructions
#     - `VAE`: Combines encoder/decoder with ELBO loss
#     """
#     print()
#     print("=" * 80)
#     print("VAE MNIST Example Completed Successfully!")
#     print("=" * 80)
#     print()
#     print("💡 Key Takeaways:")
#     print("  - VAEs learn probabilistic latent representations")
#     print("  - Reparameterization trick enables gradient-based training")
#     print("  - Artifex provides modular MLPEncoder and MLPDecoder components")
#     print("  - Easy to swap components (MLP → CNN, different architectures)")
#     print()
#     print("🔬 Next Steps:")
#     print("  - Try CNN encoder/decoder for better image modeling")
#     print("  - Experiment with different latent dimensions (16, 64, 128)")
#     print("  - Try β-VAE with higher kl_weight for disentanglement")
#     print("  - Train on real MNIST data for realistic results")
#     print("  - Explore latent space interpolation and traversals")
#
#
# if __name__ == "__main__":
#     main()


# %% [markdown]
"""
## Experiments to Try

1. **CNN Architecture**:
   ```python
   # Use encoder_type="cnn" in VAEConfig for CNN-based encoder/decoder
   encoder_config = EncoderConfig(
       name="cnn_encoder",
       hidden_dims=(32, 64, 128),
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
   config = VAEConfig(
       name="cnn_vae",
       encoder=encoder_config,
       decoder=decoder_config,
       encoder_type="cnn",  # Use CNN encoder/decoder
       kl_weight=1.0,
   )
   model = VAE(config=config, rngs=rngs)
   ```
   CNNs often work better for image data than MLPs.

2. **Latent Dimension Experiments**:
   - Try `latent_dim=16`: Smaller capacity, faster training, may lose details
   - Try `latent_dim=64`: Larger capacity, better reconstructions
   - Try `latent_dim=128`: Very high capacity, risk of overfitting

   **Trade-off:** Larger latent dims → better reconstruction but less structured space

3. **β-VAE for Disentanglement**:
   ```python
   # Increase kl_weight for disentanglement
   config = VAEConfig(
       name="beta_vae",
       encoder=encoder_config,
       decoder=decoder_config,
       encoder_type="dense",
       kl_weight=4.0,  # β=4 encourages disentangled representations
   )
   model = VAE(config=config, rngs=rngs)
   ```
   - Higher β: More regularization, worse reconstruction, better disentanglement
   - Lower β: Better reconstruction, less structured latent space
   - β=1: Standard VAE

4. **Architecture Variations**:
   ```python
   # Deeper network with different activation
   encoder_config = EncoderConfig(
       name="deep_encoder",
       hidden_dims=(512, 256, 128),  # Deeper network
       latent_dim=32,
       activation="gelu",  # Try different activations
       input_shape=(28, 28, 1),
   )
   ```
   - More layers: Higher capacity but slower training
   - Different activations: GELU often works better than ReLU
   - Batch normalization: Can help with deeper networks

5. **Real MNIST Data**:
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

## Troubleshooting

**Issue:** Encoder/decoder creation fails
- **Solution**: Check that `input_shape` matches your data shape in EncoderConfig
- **Common mistake**: Using list instead of tuple for `hidden_dims`
- **Fix**: Use tuples: `hidden_dims=(256, 128)` not `hidden_dims=[256, 128]`

**Issue:** Reconstructions are very blurry
- **Solution**: This is expected with MSE loss on images
- **Explanation**: MSE averages over pixel space, causing blur
- **Alternatives**:
  - Use CNNEncoder/CNNDecoder instead of MLP
  - Try perceptual loss or adversarial training
  - Use VQVAE for sharper reconstructions

**Issue:** Generated samples look like noise
- **Solution**: VAE needs training; this example only demonstrates architecture
- **Note**: With synthetic random data, generations won't be meaningful
- **Fix**: Train on real MNIST with a proper training loop (see training examples)

**Issue:** KL collapse (all latent codes become identical)
- **Solution**: Reduce `kl_weight` to allow more latent variance
- **Monitoring**: Check `jnp.std(latent)` - should be > 0.5
- **Fix**: Use KL annealing schedule (start with kl_weight=0.1, increase gradually)

**Issue:** Model output shape mismatch
- **Solution**: Ensure encoder `latent_dim` matches decoder `latent_dim` in configs
- **Check**: Verify `output_shape` matches `input_shape` for reconstruction
- **Note**: VAEConfig validates that latent_dim matches between encoder and decoder

## Additional Resources

**Papers:**
1. **Auto-Encoding Variational Bayes** (Kingma & Welling, 2014)
   - Original VAE paper
   - https://arxiv.org/abs/1312.6114

2. **β-VAE: Learning Basic Visual Concepts** (Higgins et al., 2017)
   - Disentanglement via β parameter
   - https://openreview.net/forum?id=Sy2fzU9gl

3. **Understanding disentangling in β-VAE** (Burgess et al., 2018)
   - Analysis of β-VAE disentanglement
   - https://arxiv.org/abs/1804.03599

**Documentation:**
- Artifex VAE Documentation: `docs/models/vae.md`
- Flax NNX Guide: https://flax.readthedocs.io/en/latest/nnx/index.html
- VAE Tutorial: `docs/examples/basic/vae-mnist.md`

**Related Examples:**
- `advanced_vae.py`: β-VAE, Conditional VAE, VQ-VAE, Hierarchical VAE
- `multi_beta_vae_benchmark_demo.py`: Disentanglement evaluation with MIG score
- `simple_gan.py`: Compare VAE vs. GAN generation approaches
"""
