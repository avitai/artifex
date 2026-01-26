# Multi-β VAE Controllable Generation Benchmark

**Level:** Intermediate | **Runtime:** ~2-3 minutes (CPU) / ~1 minute (GPU) | **Format:** Python + Jupyter

**Prerequisites:** Understanding of VAEs and disentangled representations | **Target Audience:** Researchers in controllable generation and representation learning

## Overview

This example demonstrates how to benchmark β-VAE models for controllable generation using disentanglement and image quality metrics. Learn how the β parameter affects the trade-off between disentanglement and reconstruction quality, and how to systematically evaluate models using MIG, FID, LPIPS, and SSIM metrics.

## What You'll Learn

<div class="grid cards" markdown>

- :material-tune: **β-VAE Framework**

    ---

    Understand how β controls disentanglement vs reconstruction trade-off

- :material-chart-bell-curve: **MIG Score**

    ---

    Measure disentanglement using Mutual Information Gap

- :material-image-multiple: **Image Quality Metrics**

    ---

    Evaluate generation quality with FID, LPIPS, and SSIM

- :material-scale-balance: **Quality Trade-offs**

    ---

    Balance disentanglement, reconstruction, and training time

- :material-compare-horizontal: **Model Comparison**

    ---

    Systematically compare different model configurations

- :material-benchmark: **Benchmark Suite**

    ---

    Comprehensive evaluation across multiple metrics

</div>

## Files

This example is available in two formats:

- **Python Script**: [`multi_beta_vae_benchmark_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/vae/multi_beta_vae_benchmark_demo.py)
- **Jupyter Notebook**: [`multi_beta_vae_benchmark_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/vae/multi_beta_vae_benchmark_demo.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the benchmark demo
python examples/generative_models/vae/multi_beta_vae_benchmark_demo.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/vae/multi_beta_vae_benchmark_demo.ipynb
```

## Key Concepts

### 1. β-VAE Framework

β-VAE modifies the standard VAE loss by adding a weight β to the KL divergence term:

$$\mathcal{L}_{\beta\text{-VAE}} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{\beta \cdot \text{KL}(q(z|x) \| p(z))}_{\text{Regularization}}$$

**β Parameter Effects:**

- **β = 1**: Standard VAE
- **β > 1**: Encourages disentanglement, may reduce reconstruction quality
- **β < 1**: Prioritizes reconstruction, may reduce disentanglement

```python
# Different β values for different goals
beta_configs = {
    "reconstruction_focused": 0.5,   # Better reconstruction
    "balanced": 1.0,                 # Standard VAE
    "disentanglement_focused": 4.0,  # Better disentanglement
}
```

### 2. Disentanglement

Disentangled representations have independent latent dimensions that each capture a single factor of variation:

```
Disentangled Latent Space:
z[0] → Controls rotation
z[1] → Controls size
z[2] → Controls color
...

Entangled Latent Space:
z[0] → Affects rotation AND size
z[1] → Affects size AND color
z[2] → Affects rotation AND color
```

**Why Disentanglement Matters:**

- Better interpretability
- More controllable generation
- Improved generalization
- Easier downstream task learning

### 3. MIG Score (Mutual Information Gap)

MIG measures how much each latent dimension encodes a single ground-truth factor:

$$\text{MIG} = \frac{1}{K} \sum_{k=1}^{K} \frac{I(z_j^{(k)}; v_k) - I(z_j^{(k-1)}; v_k)}{H(v_k)}$$

where $I$ is mutual information, $v_k$ are ground-truth factors, and $j^{(k)}$ is the latent dimension with highest MI for factor $k$.

```python
from artifex.benchmarks.metrics.disentanglement import MIGMetric

# Compute MIG score
mig_metric = MIGMetric(rngs=rngs)
mig_score = mig_metric.compute(
    latent_codes=z,          # (batch, latent_dim)
    ground_truth_factors=factors  # (batch, num_factors)
)
# mig_score: 0-1 (higher is better)
# >0.3: Good disentanglement
# >0.5: Excellent disentanglement
```

### 4. FID Score (Fréchet Inception Distance)

Measures distribution distance between real and generated images in feature space:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

```python
from artifex.benchmarks.metrics.image import FIDMetric

fid_metric = FIDMetric(rngs=rngs)
fid_score = fid_metric.compute(
    real_images=real_imgs,
    generated_images=gen_imgs
)
# fid_score: 0-∞ (lower is better)
# <30: Excellent quality
# <50: Good quality
# <100: Acceptable quality
```

### 5. LPIPS (Learned Perceptual Image Patch Similarity)

Uses deep features to measure perceptual similarity:

```python
from artifex.benchmarks.metrics.image import LPIPSMetric

lpips_metric = LPIPSMetric(rngs=rngs)
lpips_score = lpips_metric.compute(
    images1=original_imgs,
    images2=reconstructed_imgs
)
# lpips_score: 0-1 (lower is better)
# <0.1: Excellent perceptual quality
# <0.2: Good perceptual quality
# <0.3: Acceptable perceptual quality
```

### 6. SSIM (Structural Similarity Index)

Measures structural similarity between images:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

```python
from artifex.benchmarks.metrics.image import SSIMMetric

ssim_metric = SSIMMetric(rngs=rngs)
ssim_score = ssim_metric.compute(
    images1=original_imgs,
    images2=reconstructed_imgs
)
# ssim_score: 0-1 (higher is better)
# >0.9: Excellent structural similarity
# >0.8: Good structural similarity
# >0.7: Acceptable structural similarity
```

### 7. Multi-β VAE Benchmark Suite

Comprehensive evaluation across all metrics:

```python
from artifex.benchmarks.suites.multi_beta_vae_suite import MultiBetaVAEBenchmarkSuite

suite = MultiBetaVAEBenchmarkSuite(
    dataset_config={
        "num_samples": 100,
        "image_size": 64,
        "include_attributes": True,  # For disentanglement metrics
    },
    benchmark_config={
        "num_samples": 50,
        "batch_size": 10,
    },
    rngs=rngs
)

# Run evaluation
results = suite.run_all(model)
# results = {
#     "multi_beta_vae_benchmark": {
#         "mig_score": 0.35,
#         "fid_score": 42.3,
#         "lpips_score": 0.18,
#         "ssim_score": 0.84,
#         "training_time_per_epoch": 7.5,
#     }
# }
```

## Code Structure

The example consists of three main components:

1. **MockMultiBetaVAE** - Simulates β-VAE with controllable quality levels
   - Three quality modes: low, medium, high
   - Demonstrates encode-decode-generate pipeline
   - Shows proper RNG handling patterns

2. **Benchmark Suite** - Comprehensive evaluation system
   - Disentanglement metrics (MIG)
   - Image quality metrics (FID, LPIPS, SSIM)
   - Training efficiency metrics

3. **Model Comparison** - Systematic evaluation
   - Compare across quality levels
   - Analyze trade-offs
   - Performance targets

## Features Demonstrated

- ✅ β-VAE framework for controllable generation
- ✅ Disentanglement evaluation (MIG score)
- ✅ Image quality assessment (FID, LPIPS, SSIM)
- ✅ Model comparison across quality levels
- ✅ Trade-off analysis (disentanglement vs quality vs training time)
- ✅ Comprehensive benchmark suite
- ✅ Performance target assessment

## Experiments to Try

1. **Adjust Latent Dimensionality**

   ```python
   model = MockMultiBetaVAE(
       latent_dim=256,  # Try different sizes: 16, 32, 64, 128, 256
       image_size=64,
       quality_level="high",
       rngs=rngs
   )
   ```

2. **Vary β Values**

   ```python
   # In real β-VAE training
   beta_values = [0.5, 1.0, 2.0, 4.0, 8.0]
   for beta in beta_values:
       model = BetaVAE(latent_dim=64, beta=beta, rngs=rngs)
       # Train and evaluate
   ```

3. **Change Dataset Size**

   ```python
   suite = MultiBetaVAEBenchmarkSuite(
       dataset_config={
           "num_samples": 500,  # More samples for stable metrics
           "image_size": 128,   # Higher resolution
       },
       # ...
   )
   ```

4. **Custom Quality Configurations**

   ```python
   model.quality_params["custom"] = {
       "mig_score": 0.40,
       "fid_score": 35.0,
       "lpips_score": 0.12,
       "ssim_score": 0.88,
   }
   model.quality_level = "custom"
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-arrow-right: **VAE Training**

    ---

    Train β-VAE on real datasets

    [:octicons-arrow-right-24: VAE MNIST Tutorial](../basic/vae-mnist.md)

- :material-arrow-right: **Advanced VAE**

    ---

    Explore FactorVAE and β-TCVAE

    [:octicons-arrow-right-24: Advanced VAE](../advanced/advanced-vae.md)

- :material-arrow-right: **Latent Space Analysis**

    ---

    Visualize and interpret disentangled representations

    [:octicons-arrow-right-24: Latent Space Tutorial](#)

- :material-arrow-right: **Loss Functions**

    ---

    Understand VAE loss components

    [:octicons-arrow-right-24: Loss Examples](../losses/loss-examples.md)

</div>

## Troubleshooting

### Benchmark Runs Slowly

**Symptom:** Evaluation takes too long

**Solution:** Reduce dataset or batch size

```python
dataset_config = {
    "num_samples": 50,  # Smaller dataset
}
benchmark_config = {
    "batch_size": 5,    # Smaller batches
}
```

### Models Don't Meet Targets

**Symptom:** All models fail to meet performance targets

**Cause:** Insufficient model capacity or training

**Solution:** Increase latent dimensionality or improve quality

```python
model = MockMultiBetaVAE(
    latent_dim=128,      # Larger capacity
    quality_level="high", # Better quality
    rngs=rngs
)
```

### High Memory Usage

**Symptom:** Out of memory errors during evaluation

**Solution:** Reduce image size or batch size

```python
dataset_config = {
    "image_size": 32,   # Smaller images
}
benchmark_config = {
    "batch_size": 4,    # Smaller batches
}
```

### Inconsistent MIG Scores

**Symptom:** MIG scores vary significantly between runs

**Cause:** Too few samples for stable metric computation

**Solution:** Increase number of evaluation samples

```python
benchmark_config = {
    "num_samples": 100,  # More samples for stability
}
```

## Additional Resources

### Documentation

- [VAE Guide](../../user-guide/models/vae-guide.md) - Complete VAE documentation
- [Benchmark Suite](../../benchmarks/multi_beta_vae_suite.md) - Benchmarking guide

### Related Examples

- [VAE MNIST Tutorial](../basic/vae-mnist.md) - Basic VAE training
- [Advanced VAE](../advanced/advanced-vae.md) - Advanced VAE variants
- [Loss Examples](../losses/loss-examples.md) - VAE loss functions
- [Framework Features Demo](../framework/framework-features-demo.md) - Configuration system

### Papers and Resources

- **β-VAE**: ["β-VAE: Learning Basic Visual Concepts with a Constrained VAE" (Higgins et al., 2017)](https://openreview.net/forum?id=Sy2fzU9gl)
- **FactorVAE**: ["Disentangling by Factorising" (Kim & Mnih, 2018)](https://arxiv.org/abs/1802.05983)
- **β-TCVAE**: ["Isolating Sources of Disentanglement in VAEs" (Chen et al., 2018)](https://arxiv.org/abs/1802.04942)
- **MIG**: ["A Framework for the Quantitative Evaluation of Disentangled Representations" (Chen et al., 2018)](https://openreview.net/forum?id=By-7dz-AZ)
- **FID**: ["GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (Heusel et al., 2017)](https://arxiv.org/abs/1706.08500)
- **LPIPS**: ["The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (Zhang et al., 2018)](https://arxiv.org/abs/1801.03924)

### External Libraries

- **disentanglement_lib**: [Google's disentanglement benchmark](https://github.com/google-research/disentanglement_lib)
- **pytorch-fid**: [FID score implementation](https://github.com/mseitzer/pytorch-fid)
- **LPIPS**: [Official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity)
