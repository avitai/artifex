# Diffusion Model API Demo (MNIST)

A lightweight demonstration of Artifex's DDPM (Denoising Diffusion Probabilistic Model) API using MNIST. This example shows how to use the DDPMModel without training, focusing on API usage and different sampling techniques.

**⏱️ Duration:** 5-10 minutes | **💻 Level:** Beginner | **🎓 Prerequisites:** Basic Python

## Overview

This demo covers:

1. Creating a DDPM model with Artifex's API
2. Understanding forward diffusion (noise addition)
3. Sampling with DDPM (1000 steps)
4. Fast sampling with DDIM (50 steps, 20x speedup)
5. Visualizing progressive denoising

**What This Demo Is NOT:**

- This is not a training tutorial (see [diffusion-mnist.md](diffusion-mnist.md) for full training)
- Uses a freshly initialized model (not trained)
- Generates abstract patterns, not realistic digits
- Focused on API demonstration, not production use

## Quick Start

```bash
# Activate environment
source activate.sh

# Run the demo
python examples/generative_models/image/diffusion/diffusion_mnist.py
```

**Expected Output:**

- 4 visualizations saved to `examples_output/`
- Runtime: ~2-3 minutes on GPU, ~5-10 minutes on CPU

## Code Walkthrough

### 1. Model Creation

The demo shows how to create a DDPM model using Artifex's unified configuration:

```python
from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.models.diffusion.ddpm import DDPMModel

# Configure DDPM
config = ModelConfig(
    name="ddpm_mnist",
    model_class="DDPMModel",
    input_dim=(28, 28, 1),
    parameters={
        "noise_steps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "beta_schedule": "linear",
    },
)

# Create model
model = DDPMModel(config, rngs=rngs)
```

**Key Points:**

- `ModelConfig` provides unified config across all Artifex models
- `input_dim=(28, 28, 1)` specifies MNIST dimensions (grayscale 28x28)
- `noise_steps=1000` sets the number of diffusion timesteps
- Beta schedule controls noise levels at each step

### 2. Forward Diffusion

The demo visualizes how diffusion models add noise to images:

```python
# Add noise at different timesteps
t_tensor = jnp.array([timestep])
noisy_x, added_noise = model.forward_diffusion(image, t_tensor, rngs=rngs)
```

**Output:** Visualization showing image → progressive noise levels → pure noise

### 3. Model Forward Pass

Shows how the model predicts noise:

```python
# Predict noise for a batch
outputs = model(noisy_images, timesteps, rngs=rngs)
predicted_noise = outputs["predicted_noise"]
```

**Key API:** `model(x, t, rngs)` returns dictionary with `"predicted_noise"` key

### 4. DDPM Sampling (Slow but High Quality)

Generate samples using the full 1000-step process:

```python
samples_ddpm = model.sample(
    n_samples_or_shape=8,
    scheduler="ddpm",  # Use DDPM scheduler
    rngs=rngs,
)
```

**Characteristics:**

- ✅ Original DDPM algorithm
- ✅ 1000 denoising steps
- ⏱️ Slow (~40 seconds for 8 samples on GPU)
- 🎨 High quality (after training)

### 5. DDIM Sampling (Fast with Comparable Quality)

Generate samples with only 50 steps:

```python
samples_ddim = model.sample(
    n_samples_or_shape=8,
    scheduler="ddim",  # Use DDIM scheduler
    steps=50,          # Only 50 steps!
    rngs=rngs,
)
```

**Characteristics:**

- ✅ DDIM algorithm (deterministic)
- ✅ Only 50 steps (configurable)
- ⚡ **20x faster** than DDPM
- 🎨 Comparable quality to DDPM

**Speedup Comparison:**

```
DDPM (1000 steps): ~40s
DDIM (50 steps):   ~2s
Speedup:           20x
```

### 6. Progressive Denoising

Visualize how the model transforms noise into structure:

```python
# Capture snapshots during denoising
snapshots = []
for t in tqdm(range(model.noise_steps - 1, -1, -1)):
    x_denoised = denoise_step(x, t)
    if t % snapshot_interval == 0:
        snapshots.append(x_denoised)
```

**Output:** Shows the gradual transformation from noise → structured patterns

## Generated Outputs

The demo generates 4 visualization files:

1. **`diffusion_mnist_forward.png`**
   - Shows forward diffusion (clean → noisy)
   - 5 timesteps: t=0, 250, 500, 750, 999

2. **`diffusion_mnist_ddpm_samples.png`**
   - 8 samples generated with DDPM
   - 1000-step sampling process

3. **`diffusion_mnist_ddim_samples.png`**
   - 8 samples generated with DDIM
   - 50-step sampling (20x faster)

4. **`diffusion_mnist_trajectory.png`**
   - Progressive denoising over 6 snapshots
   - Shows noise → pattern transformation

## Key Takeaways

### Artifex API Patterns

1. **Model Creation:**

   ```python
   model = DDPMModel(config, rngs=rngs)
   ```

2. **Forward Diffusion:**

   ```python
   noisy_x, noise = model.forward_diffusion(x, t, rngs=rngs)
   ```

3. **Noise Prediction:**

   ```python
   outputs = model(x, t, rngs=rngs)
   noise_pred = outputs["predicted_noise"]
   ```

4. **Sampling:**

   ```python
   # DDPM (slow)
   samples = model.sample(n, scheduler="ddpm", rngs=rngs)

   # DDIM (fast)
   samples = model.sample(n, scheduler="ddim", steps=50, rngs=rngs)
   ```

### DDPM vs DDIM

| Aspect | DDPM | DDIM |
|--------|------|------|
| Steps | 1000 (fixed) | Configurable (20-100) |
| Speed | Slow | 10-50x faster |
| Quality | High (baseline) | Comparable |
| Stochasticity | Stochastic | Deterministic |
| Use Case | Best quality | Production/fast iteration |

### When to Use Each

**Use DDPM when:**

- You want the original algorithm
- Quality is critical
- Speed is not a concern
- Following research papers exactly

**Use DDIM when:**

- You need fast sampling
- Deploying to production
- Iterating quickly during development
- GPU memory is limited

## Experiments to Try

### 1. Different Step Counts (DDIM)

```python
# Very fast (lower quality)
model.sample(8, scheduler="ddim", steps=20, rngs=rngs)

# Balanced (recommended)
model.sample(8, scheduler="ddim", steps=50, rngs=rngs)

# Slower but better
model.sample(8, scheduler="ddim", steps=100, rngs=rngs)
```

### 2. Different Beta Schedules

```python
# Try cosine schedule
config.parameters["beta_schedule"] = "cosine"
model = DDPMModel(config, rngs=rngs)
```

### 3. Different Image Sizes

```python
# Larger images (CIFAR-10 size)
config = ModelConfig(
    name="ddpm_cifar",
    model_class="DDPMModel",
    input_dim=(32, 32, 3),  # RGB images
    parameters={"noise_steps": 1000},
)
```

## Limitations of This Demo

⚠️ **Important Limitations:**

1. **Untrained Model:** The model is randomly initialized, not trained
   - Generates abstract patterns, not realistic digits
   - For training, see [diffusion-mnist.md](diffusion-mnist.md)

2. **Dummy Data:** Uses synthetic data (random noise)
   - Not real MNIST images
   - Just for API demonstration

3. **No Evaluation:** No metrics or quality assessment
   - See training tutorial for FID scores and evaluation

4. **Simplified:** Focuses on core API, not advanced techniques
   - No conditional generation
   - No inpainting or interpolation
   - No classifier guidance

## Next Steps

### For Learning

1. **[Training Tutorial](diffusion-mnist.md)**
   - Complete end-to-end training
   - Real MNIST data
   - Evaluation metrics
   - Model checkpointing

2. **[Diffusion Concepts](../../user-guide/concepts/diffusion-explained.md)**
   - Mathematical foundations
   - Forward and reverse processes
   - Noise schedules

3. **[Advanced Techniques](../../user-guide/models/diffusion-guide.md)**
   - Conditional generation
   - Inpainting and interpolation
   - Classifier-free guidance

### For Development

1. **Train Your Own Model:**

   ```bash
   python examples/generative_models/image/diffusion/diffusion_mnist_training.py
   ```

2. **Try Other Models:**
   - `vae_mnist.py` - Variational Autoencoders
   - `gan_mnist.py` - Generative Adversarial Networks
   - `flow_mnist.py` - Normalizing Flows

3. **Explore Advanced Examples:**
   - `dit_demo.py` - Diffusion Transformers
   - `latent_diffusion.py` - High-resolution generation

## Complete Code

The complete code is available at:

```
examples/generative_models/image/diffusion/diffusion_mnist.py
```

Or as a Jupyter notebook:

```
examples/generative_models/image/diffusion/diffusion_mnist.ipynb
```

## Troubleshooting

### Issue: Import Error

**Error:** `ModuleNotFoundError: No module named 'artifex'`

**Solution:**

```bash
# Make sure environment is activated
source activate.sh

# Verify installation
python -c "import artifex; print(artifex.__version__)"
```

### Issue: Slow Execution

**Problem:** Demo takes too long to run

**Solutions:**

1. Use GPU if available (20x faster)
2. Reduce number of samples: `n_samples_or_shape=4`
3. Use DDIM with fewer steps: `steps=20`
4. Reduce noise steps in config: `noise_steps=100`

### Issue: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**

```python
# Reduce batch size
n_samples_or_shape=4  # Instead of 8

# Use CPU instead
# JAX will automatically fallback to CPU

# Use DDIM with fewer steps
steps=20  # Instead of 50
```

## Additional Resources

<div class="grid cards" markdown>

- :material-school-outline:{ .lg .middle } **[Full Training Tutorial](diffusion-mnist.md)**

    ---

    Complete DDPM training pipeline with real data

- :material-book-open-variant:{ .lg .middle } **[Diffusion Guide](../../user-guide/models/diffusion-guide.md)**

    ---

    Comprehensive guide to diffusion models

- :material-api:{ .lg .middle } **[API Reference](../../api/models/diffusion.md)**

    ---

    Complete API documentation

- :material-file-document:{ .lg .middle } **[Paper: DDPM](https://arxiv.org/abs/2006.11239)**

    ---

    Original paper by Ho et al., 2020

- :material-file-document:{ .lg .middle } **[Paper: DDIM](https://arxiv.org/abs/2010.02502)**

    ---

    Fast sampling by Song et al., 2020

- :material-github:{ .lg .middle } **[More Examples](https://github.com/avitai/artifex)**

    ---

    Additional code examples and notebooks

</div>

## Summary

This demo introduced you to:

- ✅ Artifex's DDPMModel API
- ✅ Forward and reverse diffusion
- ✅ DDPM vs DDIM sampling
- ✅ Visualization techniques
- ✅ Speed vs quality tradeoffs

**Ready to train?** Check out the [complete training tutorial](diffusion-mnist.md)!
