# Loss Functions for Generative Models

**Level:** Intermediate | **Runtime:** ~30 seconds (CPU) | **Format:** Python + Jupyter

**Prerequisites:** Basic understanding of loss functions and JAX | **Target Audience:** Users learning Artifex's loss API

## Overview

This example provides a comprehensive guide to loss functions in Artifex, covering everything from simple functional losses to advanced composable loss systems. Learn how to use built-in losses, create custom compositions, and apply specialized losses for VAEs, GANs, and geometric models.

## What You'll Learn

<div class="grid cards" markdown>

- :material-function: **Functional Losses**

    ---

    Simple loss functions (MSE, MAE) with flexible reduction modes

- :material-link-variant: **Composable System**

    ---

    Combine weighted losses with component tracking

- :material-robot-outline: **VAE Losses**

    ---

    Reconstruction + KL divergence for variational autoencoders

- :material-sword-cross: **GAN Losses**

    ---

    Generator and discriminator losses (Standard, LS-GAN, Wasserstein)

- :material-calendar-clock: **Scheduled Losses**

    ---

    Time-varying loss weights for curriculum learning

- :material-cube-outline: **Geometric Losses**

    ---

    Chamfer distance and mesh losses for 3D data

</div>

## Files

This example is available in two formats:

- **Python Script**: [`loss_examples.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/loss_examples.py)
- **Jupyter Notebook**: [`loss_examples.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/loss_examples.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/loss_examples.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/loss_examples.ipynb
```

## Key Concepts

### 1. Functional Losses

Simple, stateless loss functions for common use cases:

```python
from artifex.generative_models.core.losses import mse_loss, mae_loss

# Mean Squared Error
loss = mse_loss(predictions, targets, reduction="mean")

# Mean Absolute Error
loss = mae_loss(predictions, targets, reduction="sum")
```

**Available Reductions:**

- `"mean"`: Average over all elements (default)
- `"sum"`: Sum all elements
- `"none"`: Return per-element losses

### 2. Weighted Losses

Apply fixed weights to loss components:

```python
from artifex.generative_models.core.losses import WeightedLoss

# Create weighted loss
weighted_mse = WeightedLoss(
    loss_fn=mse_loss,
    weight=2.0,
    name="weighted_reconstruction"
)

# Compute weighted loss
loss_value = weighted_mse(predictions, targets)
```

### 3. Composite Losses

Combine multiple loss functions:

$$L_{\text{total}} = \sum_{i=1}^{n} w_i \cdot L_i(\theta)$$

```python
from artifex.generative_models.core.losses import CompositeLoss

composite = CompositeLoss([
    WeightedLoss(mse_loss, weight=1.0, name="reconstruction"),
    WeightedLoss(mae_loss, weight=0.5, name="l1_penalty"),
], return_components=True)

# Get total loss and components
total_loss, components = composite(predictions, targets)
# components = {"reconstruction": 0.15, "l1_penalty": 0.08}
```

### 4. VAE Losses

VAE loss combines reconstruction and KL divergence:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{\beta \cdot \text{KL}(q(z|x) \| p(z))}_{\text{Regularization}}$$

```python
def vae_loss(reconstruction, targets, mean, logvar, beta=1.0):
    # Reconstruction loss
    recon_loss = mse_loss(reconstruction, targets)

    # KL divergence (assuming standard normal prior)
    kl_loss = -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar))
    kl_loss = kl_loss / targets.shape[0]  # Normalize by batch size

    # Total VAE loss
    return recon_loss + beta * kl_loss
```

**β Parameter:**

- `β = 1.0`: Standard VAE
- `β > 1.0`: β-VAE (encourages disentanglement)
- `β < 1.0`: Less regularization, better reconstruction

### 5. GAN Losses

Artifex provides pre-configured GAN loss suites:

```python
from artifex.generative_models.core.losses import create_gan_loss_suite

# Create GAN losses
gen_loss, disc_loss = create_gan_loss_suite(
    generator_loss_type="lsgan",
    discriminator_loss_type="lsgan"
)

# Generator loss (want discriminator to output 1 for fake)
g_loss = gen_loss(fake_scores)

# Discriminator loss (real→1, fake→0)
d_loss = disc_loss(real_scores, fake_scores)
```

**Available GAN Loss Types:**

- `"standard"`: Binary cross-entropy (original GAN)
- `"lsgan"`: Least-squares GAN (more stable)
- `"wgan"`: Wasserstein GAN (requires gradient penalty)

### 6. Scheduled Losses

Time-varying loss weights for curriculum learning:

```python
from artifex.generative_models.core.losses import ScheduledLoss

# Define schedule function
def warmup_schedule(step):
    """Linear warmup from 0 to 1 over 1000 steps."""
    return jnp.minimum(1.0, step / 1000.0)

# Create scheduled loss
scheduled_loss = ScheduledLoss(
    loss_fn=perceptual_loss,
    schedule_fn=warmup_schedule,
    name="scheduled_perceptual"
)

# Loss weight increases with training steps
loss_value = scheduled_loss(..., step=500)  # weight = 0.5
```

### 7. Geometric Losses

Specialized losses for 3D data:

#### Chamfer Distance

Measures point cloud similarity:

$$L_{\text{Chamfer}}(X, Y) = \frac{1}{|X|}\sum_{x \in X} \min_{y \in Y} \|x - y\|^2 + \frac{1}{|Y|}\sum_{y \in Y} \min_{x \in X} \|x - y\|^2$$

```python
from artifex.generative_models.core.losses import chamfer_distance

# Point clouds: (batch, num_points, 3)
pred_points = jax.random.normal(key, (4, 1000, 3))
target_points = jax.random.normal(key, (4, 1000, 3))

loss = chamfer_distance(pred_points, target_points)
```

#### Mesh Loss

Multi-component loss for mesh quality:

```python
from artifex.generative_models.core.losses import MeshLoss

mesh_loss = MeshLoss(
    vertex_weight=1.0,      # Vertex position accuracy
    normal_weight=0.1,      # Surface normal consistency
    edge_weight=0.1,        # Edge length preservation
    laplacian_weight=0.01   # Smoothness regularization
)

# Mesh format: (vertices, faces, normals)
pred_mesh = (vertices_pred, faces, normals_pred)
target_mesh = (vertices_target, faces, normals_target)

loss = mesh_loss(pred_mesh, target_mesh)
```

### 8. Perceptual Loss

Feature-based loss using pre-trained networks:

```python
from artifex.generative_models.core.losses import PerceptualLoss

perceptual = PerceptualLoss(
    content_weight=1.0,
    style_weight=0.01
)

# Requires feature extraction from images
loss = perceptual(
    pred_images=generated_images,
    target_images=real_images,
    features_pred=extracted_features_pred,
    features_target=extracted_features_target
)
```

### 9. Total Variation Loss

Smoothness regularization for images:

```python
from artifex.generative_models.core.losses import total_variation_loss

# Encourages spatial smoothness
tv_loss = total_variation_loss(generated_images)

# Often combined with other losses
total_loss = reconstruction_loss + 0.001 * tv_loss
```

## Code Structure

The example demonstrates seven loss usage patterns:

1. **Functional Usage** - Simple MSE and MAE losses
2. **Composable Loss** - Weighted loss combination
3. **VAE Training** - Reconstruction + KL divergence
4. **GAN Training** - Generator and discriminator losses
5. **Scheduled Loss** - Progressive loss weight ramping
6. **Geometric Losses** - Chamfer distance and mesh losses
7. **Complete Training** - Full training loop with losses

## Features Demonstrated

- ✅ Functional losses with flexible reduction modes
- ✅ Weighted loss composition with component tracking
- ✅ VAE loss (reconstruction + KL divergence)
- ✅ GAN loss suites (standard, LS-GAN, Wasserstein)
- ✅ Scheduled losses for curriculum learning
- ✅ Geometric losses for 3D data (Chamfer, mesh)
- ✅ Perceptual loss with feature extraction
- ✅ Total variation loss for smoothness
- ✅ Integration with Flax NNX training loops

## Experiments to Try

1. **Adjust Loss Weights**

   ```python
   # Try different β values for VAE
   composite = CompositeLoss([
       WeightedLoss(recon_loss, weight=1.0, name="recon"),
       WeightedLoss(kl_loss, weight=4.0, name="kl"),  # β = 4.0
   ])
   ```

2. **Compare GAN Loss Types**

   ```python
   # Standard GAN
   gen_loss, disc_loss = create_gan_loss_suite("standard", "standard")

   # LS-GAN (often more stable)
   gen_loss, disc_loss = create_gan_loss_suite("lsgan", "lsgan")
   ```

3. **Custom Schedule Functions**

   ```python
   # Exponential warmup
   def exp_schedule(step):
       return 1.0 - jnp.exp(-step / 1000.0)

   # Cosine annealing
   def cosine_schedule(step):
       return 0.5 * (1 + jnp.cos(jnp.pi * step / total_steps))
   ```

4. **Geometric Loss Weights**

   ```python
   # Adjust mesh loss components
   mesh_loss = MeshLoss(
       vertex_weight=2.0,      # Emphasize position accuracy
       normal_weight=0.5,      # More weight on normals
       edge_weight=0.1,
       laplacian_weight=0.01
   )
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-arrow-right: **VAE Examples**

    ---

    Apply losses in VAE training

    [:octicons-arrow-right-24: VAE MNIST Tutorial](../basic/vae-mnist.md)

- :material-arrow-right: **GAN Examples**

    ---

    Use GAN losses in training

    [:octicons-arrow-right-24: GAN MNIST Tutorial](../basic/simple-gan.md)

- :material-arrow-right: **Geometric Models**

    ---

    Apply geometric losses

    [:octicons-arrow-right-24: Geometric Benchmark](../geometric/geometric-benchmark-demo.md)

- :material-arrow-right: **Framework Features**

    ---

    Understand composable design

    [:octicons-arrow-right-24: Framework Demo](../framework/framework-features-demo.md)

</div>

## Troubleshooting

### Shape Mismatch Errors

**Symptom:** `ValueError` about incompatible shapes

**Solution:** Ensure predictions and targets have the same shape

```python
print(f"Predictions: {predictions.shape}")
print(f"Targets: {targets.shape}")

# Reshape if needed
predictions = predictions.reshape(targets.shape)
```

### NaN in KL Divergence

**Symptom:** KL loss becomes NaN during VAE training

**Cause:** Numerical instability in `exp(logvar)` for large `logvar`

**Solution:** Clip logvar values

```python
logvar = jnp.clip(logvar, -10.0, 10.0)
kl_loss = -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar))
```

### GAN Loss Not Converging

**Symptom:** Generator or discriminator loss diverges

**Solution:** Try LS-GAN loss instead of standard GAN

```python
# LS-GAN is often more stable
gen_loss, disc_loss = create_gan_loss_suite("lsgan", "lsgan")
```

### Composite Loss Component Mismatch

**Symptom:** `KeyError` when accessing loss components

**Solution:** Set `return_components=True` in CompositeLoss

```python
composite = CompositeLoss([...], return_components=True)
total, components = composite(pred, target)  # Returns tuple
```

## Additional Resources

### Documentation

- [Loss Functions API Reference](../../api/core/losses.md) - Complete loss function documentation

### Related Examples

- [Framework Features Demo](../framework/framework-features-demo.md) - Composable loss system
- [VAE MNIST Tutorial](../basic/vae-mnist.md) - VAE loss in practice
- [GAN MNIST Tutorial](../basic/simple-gan.md) - GAN loss in practice
- [Geometric Benchmark](../geometric/geometric-benchmark-demo.md) - Geometric losses

### Papers

- **VAE**: [Auto-Encoding Variational Bayes (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
- **β-VAE**: [β-VAE: Learning Basic Visual Concepts (Higgins et al., 2017)](https://openreview.net/forum?id=Sy2fzU9gl)
- **LS-GAN**: [Least Squares GAN (Mao et al., 2017)](https://arxiv.org/abs/1611.04076)
- **Perceptual Loss**: [Perceptual Losses (Johnson et al., 2016)](https://arxiv.org/abs/1603.08155)
- **Chamfer Distance**: [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/abs/1707.02392)
