# Artifex Framework Examples Guide

## Overview

This guide provides comprehensive documentation for all Artifex framework examples, demonstrating best practices and proper usage of the framework's features.

## Example Categories

### 🎯 Core Framework Demonstrations

#### `framework_features_demo.py`

Comprehensive demonstration of Artifex's core features:

- **Unified Configuration System**: Using frozen dataclass configs (`VAEConfig`, `DDPMConfig`, `PointCloudConfig`, etc.)
- **Factory Pattern**: Creating models with `create_model()`
- **Composable Losses**: Building complex loss functions
- **Sampling Methods**: MCMC, SDE sampling
- **Modality System**: Understanding domain-specific adapters

```python
# Key pattern: Always use specific frozen dataclass configurations
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)

encoder_config = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),
)
decoder_config = DecoderConfig(
    name="decoder",
    output_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(128, 256),
)
config = VAEConfig(
    name="my_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    kl_weight=1.0,
)
model = create_model(config, rngs=rngs)
```

#### `advanced_training_example.py`

Complete training pipeline implementation:

- **Training Loop**: Proper epoch/batch iteration
- **Metrics Tracking**: Loss and accuracy monitoring
- **Validation**: Regular validation during training
- **Checkpointing**: Model saving and restoration
- **Visualization**: Training curves plotting

### 🖼️ Generative Models

#### VAE (Variational Autoencoders)

**`image/vae/vae_mnist.py`**

- MNIST digit generation using VAE
- Demonstrates unified configuration
- Shows reconstruction and generation
- Includes visualization utilities

```python
# VAE-specific configuration
parameters={
    "latent_dim": 32,
    "beta": 1.0,  # β-VAE parameter
    "kl_weight": 0.5,
    "reconstruction_loss": "mse"
}
```

#### GAN (Generative Adversarial Networks)

**`image/gan/simple_gan.py`**

- 2D data generation with GAN
- Proper RNG handling in training loops
- Generator/Discriminator architecture
- Training dynamics visualization

```python
# Key pattern: Handle RNGs outside loss functions
z = jax.random.normal(key, (batch_size, latent_dim))
fake_batch = generator(z)  # Generate outside loss_fn

def disc_loss_fn(discriminator):
    # Use pre-generated fake_batch
    fake_scores = discriminator(fake_batch)
    ...
```

#### Diffusion Models

**`diffusion/simple_diffusion_example.py`**

- Simplified diffusion process demonstration
- Noise scheduling
- Denoising process visualization

### 🔊 Domain-Specific Examples

#### Audio Generation

**`audio/simple_audio_generation.py`**

- Waveform synthesis using neural networks
- Spectrogram generation
- Audio variation generation

```python
# Audio-specific features
generator = SimpleAudioGenerator(
    sample_rate=16000,  # 16 kHz
    duration=0.5,        # 0.5 seconds
    latent_dim=32,
    rngs=rngs
)
```

#### Text Generation

**`text/simple_text_generation.py`**

- Character-level language modeling
- Temperature-controlled generation
- Batch text processing

#### Geometric Models

**`geometric/simple_point_cloud_example.py`**

- 3D point cloud generation
- Proper use of `PointCloudConfig`
- Visualization utilities

### 🔀 Multimodal Models

**`multimodal/simple_image_text.py`**

- Joint image-text embedding
- Cross-modal retrieval
- Embedding space visualization

## Best Practices

### 1. Configuration Management

Always use specific frozen dataclass configurations:

```python
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
)

# Never use raw dictionaries
# ❌ Wrong
config = {"latent_dim": 32, "beta": 1.0}

# ✅ Correct - Use specific config classes with nested configs
encoder_config = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),
)
config = VAEConfig(
    name="my_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    kl_weight=1.0,
)
```

### 2. Factory Pattern

Always use the factory system for model creation:

```python
from artifex.generative_models.factory import create_model

# ❌ Wrong: Direct instantiation
model = VAE(config)

# ✅ Correct: Factory pattern
model = create_model(config, rngs=rngs)
```

### 3. RNG Management

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

### 4. Loss Composition

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

### 5. Training Loops

Implement proper training structure:

```python
# Training step function
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        outputs = model(batch)
        loss = compute_loss(outputs, batch)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API
    return loss

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(model, optimizer, batch)

    # Validation
    val_loss = evaluate(model, val_loader)

    # Checkpoint
    if epoch % save_freq == 0:
        save_checkpoint(model, epoch)
```

## Common Issues and Solutions

### Issue: CUDA/GPU Segmentation Faults

**Solution**: Run with CPU-only mode

```bash
export JAX_PLATFORMS=cpu
python your_example.py
```

### Issue: Configuration Validation Errors

**Solution**: Check required fields and use correct config types

```python
# Use specific config classes, not generic ModelConfiguration
# OptimizerConfig requires specific fields
optimizer_config = OptimizerConfig(
    name="optimizer",  # Required
    optimizer_type="adam",  # Required
    learning_rate=1e-3,  # Depends on optimizer
    # ... other fields
)
```

### Issue: RNG Context Errors in Training

**Solution**: Generate random values outside loss functions

```python
# ❌ Wrong
def loss_fn(model):
    z = jax.random.normal(rngs.sample(), shape)  # Error!

# ✅ Correct
z = jax.random.normal(key, shape)
def loss_fn(model):
    output = model(z)  # Use pre-generated z
```

### Issue: Factory Creation Fails

**Solution**: Provide fallback implementation

```python
try:
    model = create_model(config, rngs=rngs)
except Exception as e:
    print(f"Factory failed: {e}, using fallback")
    model = SimpleImplementation(config, rngs=rngs)
```

## Running Examples

### Individual Examples

```bash
# Set environment
export JAX_PLATFORMS=cpu

# Run specific example
python examples/generative_models/vae/vae_mnist.py
```

### Run All Examples

```bash
# Make script executable
chmod +x examples/run_all_examples.sh

# Run all tests
./examples/run_all_examples.sh
```

### With GPU

```bash
# For GPU execution (if CUDA is properly configured)
unset JAX_PLATFORMS
python examples/generative_models/diffusion/simple_diffusion_example.py
```

## Adding New Examples

When creating new examples:

1. **Follow the structure**: Place in appropriate category directory
2. **Use configurations**: Always use specific config classes (`VAEConfig`, `DDPMConfig`, `PointCloudConfig`, etc.)
3. **Include docstrings**: Document purpose and usage
4. **Test both CPU/GPU**: Ensure compatibility
5. **Add to run script**: Update `run_all_examples.sh`
6. **Update documentation**: Add to this guide

### Example Template

```python
#!/usr/bin/env python
"""Brief description of what this example demonstrates.

This example shows:
- Feature 1
- Feature 2
"""

import jax
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.factory import create_model

def main():
    """Run the example."""
    print("=" * 60)
    print("Example Name")
    print("=" * 60)

    # Configuration - use specific config classes with nested configs
    network_config = PointCloudNetworkConfig(
        name="network",
        hidden_dims=(128, 128),  # Tuples for frozen dataclass
        activation="gelu",
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout_rate=0.1,
    )
    config = PointCloudConfig(
        name="model",
        network=network_config,
        num_points=512,
        dropout_rate=0.1,
    )

    # Create model
    rngs = nnx.Rngs(params=jax.random.key(42))
    model = create_model(config, rngs=rngs)

    # Demonstrate functionality
    ...

    print("✅ Example completed successfully!")

if __name__ == "__main__":
    main()
```

## Resources

- **Artifex Documentation**: `../docs/`
- **API Reference**: `../docs/api/`
- **Technical Guidelines**: See project documentation
- **README.md**: Project documentation

## Support

For issues or questions:

1. Check this guide first
2. Review error messages carefully
3. Consult the Artifex documentation
4. Check existing examples for patterns

---

Last updated: 2025
