# Artifex Framework Examples

This directory contains comprehensive examples demonstrating how to use the Artifex framework for generative modeling with JAX/Flax.

## 🚀 Quick Start

### Prerequisites

```bash
# Install Artifex with all dependencies
uv sync --all-extras

# For CPU-only execution (recommended for testing)
export JAX_PLATFORMS=cpu
```

### Running Examples

```bash
# Basic examples
python examples/generative_models/diffusion/simple_diffusion_example.py
python examples/generative_models/vae/vae_mnist.py
python examples/generative_models/gan/simple_gan.py

# Framework features demonstration
python examples/generative_models/framework_features_demo.py
```

## 📁 Directory Structure

```
examples/
├── README.md                              # This file
├── generative_models/                     # All generative modeling examples
│   ├── audio/                            # Audio generation examples
│   ├── config/                           # Configuration examples
│   ├── deployments/                      # Model deployment examples
│   ├── diffusion/                        # Diffusion model examples
│   │   └── simple_diffusion_example.py   # Basic diffusion implementation
│   ├── distributed/                      # Distributed training examples
│   ├── energy/                            # Energy-Based Model examples
│   │   └── simple_ebm_example.py         # Basic EBM implementation
│   ├── geometric/                        # Geometric modeling examples
│   │   ├── geometric_benchmark_demo.py   # Benchmark suite
│   │   ├── geometric_losses_demo.py      # Specialized loss functions
│   │   ├── geometric_models_demo.py      # Model architectures
│   │   └── simple_point_cloud_example.py # Point cloud processing
│   ├── image/                            # Image generation examples
│   │   ├── diffusion/                    # Image diffusion models
│   │   ├── gan/                          # GAN models
│   │   └── vae/                          # VAE models
│   ├── multimodal/                       # Multi-modal examples
│   ├── optimization/                     # Optimization examples
│   ├── protein/                          # Protein modeling examples
│   │   ├── protein_diffusion_example.py  # Comprehensive protein diffusion
│   │   ├── protein_diffusion_tech_validation.py # Technical validation
│   │   ├── protein_extensions_example.py # Extension mechanisms
│   │   ├── protein_extensions_with_config.py # Config-driven modeling
│   │   ├── protein_ligand_benchmark_demo.py # Protein-ligand interactions
│   │   ├── protein_model_extension.py    # Model extensions
│   │   ├── protein_model_with_modality.py # Multi-modal protein modeling
│   │   └── protein_point_cloud_example.py # Point cloud representations
│   ├── sampling/                         # Advanced sampling techniques
│   │   └── blackjax_example.py          # MCMC sampling with BlackJAX
│   ├── text/                             # Text generation examples
│   ├── vae/                              # VAE examples
│   │   └── multi_beta_vae_benchmark_demo.py # β-VAE benchmarking
│   └── loss_examples.py                  # Comprehensive loss functions
├── utils/                                 # Utility modules
│   └── demo_utils.py                     # Shared helper functions
└── tests/                                 # Example-specific tests
```

## 🎯 Featured Examples

### Protein Modeling

#### `generative_models/protein/protein_diffusion_example.py`

**Comprehensive protein diffusion modeling example** - Our flagship example showcasing both high-level API and direct model approaches for protein structure generation.

- **High-level API with extensions**: Demonstrates protein-specific extensions and constraints
- **Direct model creation**: Shows how to create and manipulate models directly
- **Visualization & quality assessment**: Complete pipeline from generation to analysis
- **Size matching fixes**: Robust handling of different structure sizes

#### `generative_models/protein/protein_extensions_with_config.py`

Configuration-driven protein modeling with extension system.

#### `generative_models/protein/protein_ligand_benchmark_demo.py`

Benchmark demonstrations for protein-ligand interaction modeling.

#### `generative_models/protein/protein_point_cloud_example.py`

Point cloud representations for protein structure modeling.

### Geometric Modeling

#### `generative_models/geometric/geometric_benchmark_demo.py`

Comprehensive benchmark suite for geometric generative models.

#### `generative_models/geometric/geometric_losses_demo.py`

Specialized loss functions for geometric data.

#### `generative_models/geometric/geometric_models_demo.py`

Various geometric model architectures and applications.

#### `generative_models/geometric/simple_point_cloud_example.py`

Basic point cloud generation and processing.

### Advanced Sampling

#### `generative_models/sampling/blackjax_example.py`

**MCMC sampling integration** - Advanced sampling techniques using BlackJAX.

- HMC (Hamiltonian Monte Carlo)
- NUTS (No-U-Turn Sampler)
- MALA (Metropolis-Adjusted Langevin Algorithm)

### Energy-Based Models

#### `generative_models/energy/simple_ebm_example.py`

**Energy-Based Model implementation** - Comprehensive example of EBMs with MCMC sampling.

- Energy function computation
- Langevin dynamics sampling
- Contrastive divergence training
- Persistent contrastive divergence with sample buffers
- Deep EBM architectures

### Diffusion Models

#### `generative_models/diffusion/simple_diffusion_example.py`

**Basic diffusion model implementation** - Simple example demonstrating diffusion model fundamentals.

- SimpleDiffusionModel creation and configuration
- Noise schedule setup (beta schedules)
- Sample generation from noise
- Visualization of generated samples

#### `generative_models/diffusion/dit_demo.py`

**Diffusion Transformer (DiT) demonstration** - Advanced diffusion model using transformer architecture.

- DiffusionTransformer backbone architecture
- DiT model sizes (S, B, L, XL configurations)
- Conditional generation with classifier-free guidance (CFG)
- Patch-based image processing
- Performance benchmarking across model sizes
- Sample generation and visualization

### VAE Models

#### `generative_models/vae/multi_beta_vae_benchmark_demo.py`

Multi-scale β-VAE benchmarking and evaluation.

### Image Generation

The `generative_models/image/` directory contains comprehensive examples for:

- **Diffusion models**: DDPM, LDM for various datasets
- **GANs**: StyleGAN, CycleGAN implementations
- **VAEs**: VQ-VAE, standard VAE for image generation

### Text Generation

The `generative_models/text/` directory includes:

- **Transformers**: Language models and fine-tuning
- **Compression**: VQ-VAE for text

## 🚀 Quick Start

### Basic Usage

```python
# Run a simple diffusion example
python examples/generative_models/diffusion/simple_diffusion_example.py

# Run Energy-Based Model example
python examples/generative_models/energy/simple_ebm_example.py

# Run protein diffusion (comprehensive)
python examples/generative_models/protein/protein_diffusion_example.py

# Run BlackJAX sampling
python examples/generative_models/sampling/blackjax_example.py

# Run geometric benchmarks
python examples/generative_models/geometric/geometric_benchmark_demo.py
```

### Prerequisites

Make sure you have activated the Artifex environment:

```bash
source ./activate.sh
```

### GPU Support

Most examples are optimized for GPU usage and will automatically use CUDA when available:

```python
import jax
print(f"Backend: {jax.default_backend()}")  # Should show 'gpu' if CUDA is available
```

## 🔧 Example Categories

### 🧬 Protein Modeling

- Structure generation and prediction
- Physical constraints and validation
- Multi-modal protein representations
- Benchmarking and evaluation
- Located in: `generative_models/protein/`

### 📐 Geometric Modeling

- Point cloud generation
- 3D shape modeling
- Geometric loss functions
- Benchmark suites
- Located in: `generative_models/geometric/`

### 🎲 Advanced Sampling

- MCMC methods (HMC, NUTS, MALA)
- Custom sampling strategies
- Convergence analysis
- Located in: `generative_models/sampling/`

### ⚡ Energy-Based Models

- Energy function learning
- MCMC sampling techniques
- Contrastive divergence training
- Persistent sample buffers
- Located in: `generative_models/energy/`

### 🌊 Diffusion Models

- DDPM implementations
- Custom noise schedules
- Conditional generation
- Located in: `generative_models/diffusion/`

### 🖼️ Image Generation

- Diffusion models for images
- GAN architectures
- VAE variants
- Located in: `generative_models/image/`

### 📝 Text Generation

- Transformer models
- Text compression
- Language modeling
- Located in: `generative_models/text/`

### 📊 Benchmarking

- Performance evaluation
- Model comparison
- Metrics computation
- Distributed across relevant directories

## 💡 Usage Tips

### Running Examples

1. **Environment Setup**:

   ```bash
   source ./activate.sh  # Activate Artifex environment
   ```

2. **Navigate to Category**:

   ```bash
   cd examples/generative_models/protein/  # For protein examples
   cd examples/generative_models/geometric/  # For geometric examples
   # etc.
   ```

3. **Run Example**:

   ```bash
   python protein_diffusion_example.py
   ```

### GPU Optimization

Examples automatically detect and use GPU when available. Check with:

```python
import jax
print(jax.devices())  # Should show CUDA devices
```

### Modifying Examples

All examples are designed to be educational and modifiable:

- **Configuration**: Most examples have clear configuration sections at the top
- **Modularity**: Functions are well-separated for easy customization
- **Documentation**: Comprehensive docstrings explain each component

### Integration with Artifex

Examples demonstrate integration with Artifex's core components:

- **Models**: `artifex.generative_models.models.*`
- **Factory**: `artifex.generative_models.factory.*`
- **Config**: `artifex.generative_models.core.configuration.*`
- **Modalities**: `artifex.generative_models.modalities.*`
- **Extensions**: `artifex.generative_models.extensions.*`

## 🧪 Testing Examples

To verify examples work correctly:

```bash
# Test imports
python -c "
from examples.generative_models.diffusion import simple_diffusion_example
from examples.generative_models.protein import protein_diffusion_example
print('✅ Basic imports work')
"

# Run specific examples
python examples/generative_models/diffusion/simple_diffusion_example.py
python examples/generative_models/protein/protein_diffusion_example.py
```

## 📚 Learn More

- **Main Documentation**: See the main README.md in the repository root
- **API Reference**: Check `docs/` directory
- **Configuration**: See `src/artifex/generative_models/core/configuration/`
- **Factory System**: See `src/artifex/generative_models/factory/`

## 📝 Best Practices

### Configuration Management

Always use the unified configuration system:

```python
from artifex.generative_models.core.configuration import (
    ModelConfig,
    OptimizerConfig,
    TrainingConfig
)

# Never use raw dictionaries
# ❌ Wrong
config = {"latent_dim": 32, "beta": 1.0}

# ✅ Correct
config = ModelConfig(
    name="my_model",
    model_class="...",
    parameters={"latent_dim": 32, "beta": 1.0}
)
```

### Factory Pattern

Always use the factory system for model creation:

```python
from artifex.generative_models.factory import create_model

# ❌ Wrong: Direct instantiation
model = VAE(config)

# ✅ Correct: Factory pattern
model = create_model(config, rngs=rngs)
```

### RNG Management

Proper random number generator handling:

```python
# Create RNGs
key = jax.random.key(seed)
rngs = nnx.Rngs(params=key, dropout=key)

# In training loops, generate keys outside loss functions
train_key, step_key = jax.random.split(train_key)
z = jax.random.normal(step_key, shape)

def loss_fn(model):
    # Use pre-generated z, don't call RNG here
    output = model(z)
    ...
```

### Loss Composition

Use the composable loss system:

```python
from artifex.generative_models.core.losses import (
    CompositeLoss,
    WeightedLoss
)

loss = CompositeLoss([
    WeightedLoss(mse_loss, weight=1.0, name="reconstruction"),
    WeightedLoss(kl_divergence, weight=0.5, name="kl")
], return_components=True)

total_loss, components = loss(predictions, targets)
```

## 📄 Adding New Examples

When creating new examples, use this template:

```python
#!/usr/bin/env python
"""Brief description of what this example demonstrates.

This example shows:
- Feature 1
- Feature 2
"""

import jax
from flax import nnx

from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.factory import create_model

def main():
    """Run the example."""
    print("=" * 60)
    print("Example Name")
    print("=" * 60)

    # Configuration
    config = ModelConfig(...)

    # Create model
    rngs = nnx.Rngs(params=jax.random.key(42))
    model = create_model(config, rngs=rngs)

    # Demonstrate functionality
    ...

    print("Example completed successfully!")

if __name__ == "__main__":
    main()
```

**Checklist for new examples:**

1. Place in appropriate category directory
2. Use configuration objects (not raw dicts)
3. Include docstrings documenting purpose
4. Test both CPU/GPU compatibility
5. Update documentation as needed

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've activated the environment with `source ./activate.sh`
2. **CUDA Issues**: Run `python scripts/verify_gpu_setup.py` to diagnose GPU problems
3. **Memory Issues**: Try reducing batch sizes in example configurations
4. **Module Not Found**: Ensure you're running from the artifex root directory

### Getting Help

- Check example docstrings for parameter explanations
- Review the main Artifex documentation
- Look at similar examples in the same category for reference patterns
- Check the imports at the top of each example for required dependencies

---

**Note**: Examples are continuously updated to demonstrate the latest Artifex features. The organized structure makes it easier to find relevant examples for your use case.
