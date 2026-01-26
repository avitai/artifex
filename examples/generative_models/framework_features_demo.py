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

# %% [markdown]
"""
# Artifex Framework Features Demonstration

This example demonstrates the core features of the Artifex framework for generative modeling:

- **Unified Configuration System**: Type-safe configuration using frozen dataclasses
- **Factory Pattern**: Consistent model creation across all model types
- **Composable Loss Functions**: Flexible loss composition with weighted components
- **Sampling Methods**: MCMC and SDE sampling for generation
- **Modality System**: Domain-specific adapters for images, text, audio, and more

**Target Audience**: Users who want to understand the framework's architecture and best practices

**Prerequisites**: Basic understanding of generative models and JAX

**Runtime**: ~1-2 minutes (CPU)
"""

# %% [markdown]
"""
## Setup and Imports

We'll import the core framework components:
- Configuration classes for models, training, data, and optimizers
- Loss functions and composition utilities
- Sampling methods for generation
- Factory functions for model creation
"""

# %%
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from artifex.generative_models.core.losses import (
    CompositeLoss,
    mae_loss,
    mse_loss,
    WeightedLoss,
)
from artifex.generative_models.core.sampling import mcmc_sampling, sde_sampling
from artifex.generative_models.factory import create_model


# %% [markdown]
"""
## 1. Unified Configuration System

Artifex uses a unified configuration system based on frozen dataclasses. This provides:

- **Type safety**: Automatic validation of configuration parameters
- **Immutability**: Frozen dataclasses prevent accidental mutation
- **Serialization**: Easy save/load of configurations via YAML/dict
- **Composition**: Configurations can be nested and composed
- **JAX-native**: No metaclasses, fully JIT-safe

### Why use configurations?

- Reproducibility: Save exact hyperparameters
- Experimentation: Easy parameter sweeps
- Validation: Catch errors before training
- Documentation: Self-describing experiments
"""


# %%
def demonstrate_configuration_system():
    """Demonstrate the unified configuration system."""
    print("=" * 60)
    print("1. UNIFIED CONFIGURATION SYSTEM")
    print("=" * 60)

    # Model configuration
    model_config = ModelConfig(
        name="demo_vae",
        model_class="artifex.generative_models.models.vae.VAE",
        input_dim=(28, 28, 1),
        hidden_dims=(256, 128),  # Tuple for frozen dataclass
        output_dim=32,  # Latent dimension
        activation="relu",
        dropout_rate=0.1,
        parameters={
            "latent_dim": 32,
            "beta": 1.0,
            "kl_weight": 0.5,
            "reconstruction_loss": "mse",
        },
    )

    print("\nModel Configuration:")
    print(f"  Name: {model_config.name}")
    print(f"  Model class: {model_config.model_class}")
    print(f"  Input dim: {model_config.input_dim}")
    print(f"  Hidden dims: {model_config.hidden_dims}")
    print(f"  Parameters: {model_config.parameters}")

    # Optimizer configuration
    optimizer_config = OptimizerConfig(
        name="demo_optimizer",
        optimizer_type="adam",
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
    )

    # Training configuration
    training_config = TrainingConfig(
        name="demo_training",
        batch_size=32,
        num_epochs=100,
        optimizer=optimizer_config,
        gradient_clip_norm=1.0,
    )

    print("\nTraining Configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Num epochs: {training_config.num_epochs}")
    print(f"  Optimizer: {training_config.optimizer.optimizer_type}")
    print(f"  Learning rate: {training_config.optimizer.learning_rate}")

    # Data configuration
    data_config = DataConfig(
        name="demo_data",
        dataset_name="mnist",
        data_dir="/tmp/data",
        augmentation=True,
        augmentation_params={"normalize": True, "random_flip": True},
    )

    print("\nData Configuration:")
    print(f"  Dataset: {data_config.dataset_name}")
    print(f"  Augmentation: {data_config.augmentation}")
    print(f"  Augmentation params: {data_config.augmentation_params}")

    return model_config


# Run configuration demonstration
model_config = demonstrate_configuration_system()

# %% [markdown]
"""
## 2. Factory Pattern

The factory pattern provides a unified interface for creating all model types. Benefits:

- **Consistency**: Same creation pattern for VAEs, GANs, diffusion models, etc.
- **Flexibility**: Easy to swap model types without changing code
- **Validation**: Factory validates compatibility before instantiation
- **Extensibility**: Easy to add new model types

### Key Concepts:

1. **Configuration-driven**: Models are created from configuration objects
2. **RNG management**: Proper random number generator handling
3. **Type checking**: Factory ensures model types match configurations
"""


# %%
def demonstrate_factory_pattern(model_config):
    """Demonstrate the factory pattern for model creation."""
    print("\n" + "=" * 60)
    print("2. FACTORY PATTERN")
    print("=" * 60)

    # Set up RNGs
    key = jax.random.key(42)
    rngs = nnx.Rngs(params=key, dropout=key)

    model = None  # Initialize model variable
    try:
        # Create model using factory
        print("\nCreating model using factory...")
        model = create_model(model_config, rngs=rngs)
        print(f"✓ Model created: {type(model).__name__}")

        # Test the model
        batch_size = 4
        test_input = jax.random.normal(key, (batch_size, *model_config.input_dim))

        # VAE forward pass
        outputs = model(test_input, rngs=rngs)
        print("✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")

        # Generate samples
        if hasattr(model, "generate"):
            samples = model.generate(num_samples=2, rngs=rngs)
            print("✓ Generation successful")
            print(f"  Generated shape: {samples.shape}")

    except Exception as e:
        print(f"Note: Factory creation encountered: {e}")
        print("This is expected for some model types in the example")

    return model


# Run factory demonstration
model = demonstrate_factory_pattern(model_config)

# %% [markdown]
r"""
## 3. Composable Loss System

Artifex provides a flexible loss composition system that allows:

- **Single losses**: Standard loss functions (MSE, MAE, cross-entropy)
- **Weighted losses**: Apply weights to individual loss components
- **Composite losses**: Combine multiple losses with different weights
- **Component tracking**: Monitor individual loss components during training

### Mathematical Formulation:

For a composite loss with components L_1, L_2, ..., L_n and weights w_1, w_2, ..., w_n:

$$L_{total} = \\sum_{i=1}^{n} w_i \\cdot L_i(predictions, targets)$$

This is essential for multi-objective training in generative models (e.g., VAE with
reconstruction + KL loss).
"""


# %%
def demonstrate_loss_system():
    """Demonstrate the composable loss system."""
    print("\n" + "=" * 60)
    print("3. COMPOSABLE LOSS SYSTEM")
    print("=" * 60)

    # Create dummy data
    key = jax.random.key(42)
    predictions = jax.random.normal(key, (8, 32))
    targets = jax.random.normal(key, (8, 32))

    # Single loss
    print("\nSingle loss function:")
    loss_value = mse_loss(predictions, targets)
    print(f"  MSE loss: {loss_value:.4f}")

    # Weighted loss
    print("\nWeighted loss:")
    weighted_mse = WeightedLoss(mse_loss, weight=2.0, name="weighted_mse")
    weighted_value = weighted_mse(predictions, targets)
    print(f"  Weighted MSE (2x): {weighted_value:.4f}")

    # Composite loss
    print("\nComposite loss:")
    composite = CompositeLoss(
        [
            WeightedLoss(mse_loss, weight=1.0, name="reconstruction"),
            WeightedLoss(mae_loss, weight=0.5, name="l1_penalty"),
        ],
        return_components=True,
    )

    total_loss, components = composite(predictions, targets)
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Components: {components}")


# Run loss system demonstration
demonstrate_loss_system()

# %% [markdown]
"""
## 4. Sampling Methods

Artifex provides two main sampling paradigms for generation:

### MCMC Sampling

Markov Chain Monte Carlo sampling for energy-based models:
- Uses Metropolis-Hastings or Langevin dynamics
- Samples from arbitrary probability distributions
- Requires only a log probability function

### SDE Sampling

Stochastic Differential Equation sampling for diffusion models:
- Solves reverse-time SDEs
- Flexible drift and diffusion functions
- Used in DDPM, score matching, etc.

Both methods are JIT-compiled for performance.
"""


# %%
def demonstrate_sampling_methods():
    """Demonstrate sampling methods."""
    print("\n" + "=" * 60)
    print("4. SAMPLING METHODS")
    print("=" * 60)

    # Define a simple log probability function
    def log_prob_fn(x):
        # Simple Gaussian
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(42)
    init_state = jnp.zeros(5)

    # MCMC sampling
    print("\nMCMC Sampling:")
    mcmc_samples = mcmc_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=100,
        n_burnin=50,
        step_size=0.1,
    )
    print(f"  Samples shape: {mcmc_samples.shape}")
    print(f"  Mean: {jnp.mean(mcmc_samples, axis=0)}")
    print(f"  Std: {jnp.std(mcmc_samples, axis=0)}")

    # SDE sampling (for diffusion models)
    print("\nSDE Sampling:")

    def drift_fn(x, _t):
        return -x  # Simple mean-reverting drift

    def diffusion_fn(x, _t):
        return jnp.ones_like(x) * 0.1  # Constant diffusion

    sde_samples = sde_sampling(
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        init_state=init_state,
        t_span=(0.0, 1.0),
        key=key,
        n_steps=100,
    )
    print(f"  Final sample: {sde_samples}")


# Run sampling demonstration
demonstrate_sampling_methods()

# %% [markdown]
"""
## 5. Modality System

Artifex's modality system provides domain-specific features for different data types:

### Available Modalities:

- **Image**: Conv layers, attention, data augmentation
- **Text**: Tokenization, embeddings, language-specific metrics
- **Audio**: Spectrograms, waveform processing, audio generation
- **Protein**: Structure prediction, sequence modeling
- **Geometric**: Point clouds, meshes, 3D transformations

### Key Benefits:

1. **Specialized datasets**: Modality-aware data loading
2. **Domain metrics**: FID for images, perplexity for text, etc.
3. **Architecture adapters**: Modality-specific network components
4. **Pre/post-processing**: Standardized transformations

Each modality provides:
- `create_dataset()`: Load and preprocess data
- `evaluate()`: Compute domain-specific metrics
- `get_adapter()`: Get modality-specific model components
"""


# %%
def demonstrate_modality_system():
    """Demonstrate the modality system."""
    print("\n" + "=" * 60)
    print("5. MODALITY SYSTEM")
    print("=" * 60)

    # Note: The modality system requires specific setup
    print("\nAvailable modalities in Artifex:")
    print("  - image: Image generation and processing")
    print("  - text: Text generation")
    print("  - audio: Audio synthesis")
    print("  - protein: Protein structure modeling")
    print("  - geometric: Point clouds and 3D data")

    # Example of how modalities work
    print("\nModality usage pattern:")
    print("""
    # Get a modality
    image_modality = get_modality('image', rngs=rngs)

    # Use modality-specific features
    dataset = image_modality.create_dataset(config)
    metrics = image_modality.evaluate(model, data)

    # Apply modality adapter to model
    adapter = image_modality.get_adapter('vae')
    adapted_model = adapter.adapt(model, config)
    """)


# Run modality demonstration
demonstrate_modality_system()

# %% [markdown]
"""
## Summary and Key Takeaways

### Framework Benefits:

1. **Type-safe configurations**: Catch errors before training starts
2. **Unified interfaces**: Same patterns across all model types
3. **Composable components**: Mix and match losses, samplers, modalities
4. **Extensible design**: Easy to add new models and features
5. **Production-ready**: Built on JAX for performance and scalability

### Best Practices:

- Always use `ModelConfig` instead of direct instantiation
- Leverage the factory pattern for consistency
- Compose losses for multi-objective training
- Use appropriate samplers for your model type
- Apply modality adapters for domain-specific features

### Next Steps:

- Explore specific model examples (VAE, GAN, diffusion)
- Implement custom loss functions
- Add new modality support
- Build custom model architectures using the framework
"""


# %%
def main():
    """Run all demonstrations."""
    print("\n" + "🚀 " * 20)
    print("ARTIFEX FRAMEWORK FEATURES DEMONSTRATION")
    print("🚀 " * 20 + "\n")

    print("This example demonstrates proper usage of Artifex framework features:")
    print("- Unified configuration system")
    print("- Factory pattern for model creation")
    print("- Composable loss functions")
    print("- Sampling methods")
    print("- Modality system")

    # Run demonstrations
    model_config = demonstrate_configuration_system()
    demonstrate_factory_pattern(model_config)
    demonstrate_loss_system()
    demonstrate_sampling_methods()
    demonstrate_modality_system()

    print("\n" + "=" * 60)
    print("✅ Framework features demonstration completed!")
    print("=" * 60)

    print("\nKey takeaways:")
    print("1. Use ModelConfig for all model definitions")
    print("2. Use the factory system (create_model) instead of direct instantiation")
    print("3. Leverage the composable loss system for complex objectives")
    print("4. Use provided sampling methods for generation")
    print("5. Apply modality adapters for domain-specific features")


if __name__ == "__main__":
    main()
