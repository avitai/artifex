# Simple Diffusion Example

**Level:** Beginner | **Runtime:** ~30 seconds (CPU) / ~10 seconds (GPU) | **Format:** Python + Jupyter

## Overview

This example demonstrates the fundamentals of diffusion models using a simple implementation. It covers the core concepts of the forward diffusion process, reverse denoising, and sample generation.

## What You'll Learn

- How to create and configure a basic diffusion model
- Understanding noise schedules (beta schedules)
- Forward diffusion process (adding noise)
- Reverse process (denoising)
- Generating samples from random noise
- Visualizing diffusion model outputs

## Files

- **Python Script**: [`examples/generative_models/diffusion/simple_diffusion_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/diffusion/simple_diffusion_example.py)
- **Jupyter Notebook**: [`examples/generative_models/diffusion/simple_diffusion_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/diffusion/simple_diffusion_example.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/diffusion/simple_diffusion_example.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/diffusion/simple_diffusion_example.ipynb
```

## Key Concepts

### Forward Diffusion Process

The forward process gradually adds Gaussian noise to data according to a variance schedule:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

Where $\beta_t$ is the noise schedule at timestep $t$.

### Reverse Process

The model learns to reverse this process, removing noise step by step:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### Noise Schedules

The example demonstrates different beta schedules:

- **Linear schedule**: Simple linear increase from $\beta_{min}$ to $\beta_{max}$
- **Cosine schedule**: Smoother noise addition following a cosine curve

## Code Structure

The example is organized into clear sections:

1. **Model Definition**: Creating a `SimpleDiffusionModel` class
2. **Configuration**: Setting up model parameters and noise schedule
3. **Model Instantiation**: Creating the model with proper RNG handling
4. **Sample Generation**: Generating samples from random noise
5. **Visualization**: Displaying generated samples

## Example Code

```python
from flax import nnx
import jax.numpy as jnp
from artifex.generative_models.core.base import GenerativeModel

# Create RNG
rngs = nnx.Rngs(params=42, dropout=1, sample=2)

# Model configuration
config = {
    "input_dim": (32, 32, 3),  # Image shape (H, W, C)
    "noise_steps": 50,          # Number of denoising steps
    "beta_start": 1e-4,         # Starting noise level
    "beta_end": 0.02,           # Ending noise level
}

# The SimpleDiffusionModel in the example demonstrates:
# - Inheriting from GenerativeModel
# - Linear beta schedule for noise control
# - Simplified denoising process with three phases:
#   1. Early steps: Reduce noise magnitude
#   2. Middle steps: Introduce spatial structure
#   3. Late steps: Refine and sharpen output

# Run the example to see the full implementation:
# python examples/generative_models/diffusion/simple_diffusion_example.py

# Output will be saved to: examples_output/diffusion_samples.png
```

## Features Demonstrated

### SimpleDiffusionModel Creation

- Custom model class extending `GenerativeModel`
- Proper initialization with RNG handling
- Beta schedule setup for noise control

### Noise Schedule

- Linear schedule implementation
- Alpha and alpha_bar calculations
- Understanding variance schedules

### Sample Generation

- Starting from random noise
- Iterative denoising process
- Controlling generation quality

### Visualization

- Displaying generated samples
- Comparing different timesteps
- Analyzing generation quality

## Experiments to Try

1. **Modify the noise schedule**: Try different beta ranges or cosine schedules
2. **Change timesteps**: Experiment with different numbers of diffusion steps
3. **Vary sample size**: Generate different numbers of samples
4. **Add conditioning**: Extend to conditional generation
5. **Custom architecture**: Implement different denoising networks

## Next Steps

After understanding this basic example:

1. **[DiT Demo](dit-demo.md)**: Learn about Diffusion Transformers for more advanced architectures
2. **Training**: Implement a full training loop for your own dataset
3. **Advanced Schedules**: Explore more sophisticated noise schedules
4. **Conditional Generation**: Add class or text conditioning

## Troubleshooting

### Import Errors

Make sure you've activated the Artifex environment:

```bash
source activate.sh
```

### CUDA Issues

If you encounter GPU errors, try running on CPU:

```bash
export JAX_PLATFORMS=cpu
python examples/generative_models/diffusion/simple_diffusion_example.py
```

### Memory Issues

Reduce the batch size or number of timesteps if you run out of memory.

## Additional Resources

- **Paper**: [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- **Artifex Diffusion Guide**: [Diffusion Models Guide](../../user-guide/models/diffusion-guide.md)
- **API Reference**: [Diffusion API](../../api/models/diffusion.md)

## Related Examples

- [DiT Demo](dit-demo.md) - Advanced diffusion with transformers
- [Simple EBM](../energy/simple-ebm.md) - Energy-based models with MCMC sampling
