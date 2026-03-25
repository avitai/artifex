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
- **Explicit Loss Composition**: JAX-native multi-term objectives without wrapper APIs
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
- Loss functions and explicit composition patterns
- Sampling methods for generation
- Factory functions for model creation
"""

# %%
import logging

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    OptimizerConfig,
    TrainingConfig,
    VAEConfig,
)
from artifex.generative_models.core.losses import (
    mae_loss,
    mse_loss,
)
from artifex.generative_models.core.sampling import mcmc_sampling, sde_sampling
from artifex.generative_models.factory import create_model


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def echo(message: object = "") -> None:
    """Emit example output through the standard example logger."""
    LOGGER.info("%s", message)


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
    echo("=" * 60)
    echo("1. UNIFIED CONFIGURATION SYSTEM")
    echo("=" * 60)

    # Model configuration
    encoder_config = EncoderConfig(
        name="demo_encoder",
        input_shape=(28, 28, 1),
        latent_dim=32,
        hidden_dims=(256, 128),
        activation="relu",
    )
    decoder_config = DecoderConfig(
        name="demo_decoder",
        latent_dim=32,
        output_shape=(28, 28, 1),
        hidden_dims=(128, 256),
        activation="relu",
    )
    model_config = VAEConfig(
        name="demo_vae",
        encoder=encoder_config,
        decoder=decoder_config,
        kl_weight=0.5,
    )

    echo("\nModel Configuration:")
    echo(f"  Name: {model_config.name}")
    echo(f"  Encoder latent dim: {model_config.encoder.latent_dim}")
    echo(f"  Input shape: {model_config.encoder.input_shape}")
    echo(f"  Encoder hidden dims: {model_config.encoder.hidden_dims}")
    echo(f"  KL weight: {model_config.kl_weight}")

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

    echo("\nTraining Configuration:")
    echo(f"  Batch size: {training_config.batch_size}")
    echo(f"  Num epochs: {training_config.num_epochs}")
    echo(f"  Optimizer: {training_config.optimizer.optimizer_type}")
    echo(f"  Learning rate: {training_config.optimizer.learning_rate}")

    # Data configuration
    data_config = DataConfig(
        name="demo_data",
        dataset_name="mnist",
        data_dir="/tmp/data",
        augmentation=True,
        augmentation_params={"normalize": True, "random_flip": True},
    )

    echo("\nData Configuration:")
    echo(f"  Dataset: {data_config.dataset_name}")
    echo(f"  Augmentation: {data_config.augmentation}")
    echo(f"  Augmentation params: {data_config.augmentation_params}")

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
    echo("\n" + "=" * 60)
    echo("2. FACTORY PATTERN")
    echo("=" * 60)

    # Set up RNGs
    key = jax.random.key(42)
    rngs = nnx.Rngs(params=key, dropout=key)

    model = None  # Initialize model variable
    try:
        # Create model using factory
        echo("\nCreating model using factory...")
        model = create_model(model_config, rngs=rngs)
        echo(f"✓ Model created: {type(model).__name__}")

        # Test the model
        batch_size = 4
        test_input = jax.random.normal(key, (batch_size, *model_config.encoder.input_shape))

        # VAE forward pass
        outputs = model(test_input, rngs=rngs)
        echo("✓ Forward pass successful")
        echo(f"  Output keys: {list(outputs.keys())}")

        # Generate samples
        if hasattr(model, "generate"):
            samples = model.generate(num_samples=2, rngs=rngs)
            echo("✓ Generation successful")
            echo(f"  Generated shape: {samples.shape}")

    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
        echo(f"Note: Factory creation encountered: {e}")
        echo("This is expected for some model types in the example")

    return model


# Run factory demonstration
model = demonstrate_factory_pattern(model_config)

# %% [markdown]
r"""
## 3. Explicit Loss Composition

Artifex keeps multi-term objectives explicit:

- **Single losses**: Standard loss functions (MSE, MAE, cross-entropy)
- **Weighted terms**: Apply scalar weights directly in the objective
- **Named components**: Track individual terms in plain dictionaries
- **JAX-native math**: No extra wrapper API between the loss primitive and the model

### Mathematical Formulation:

For loss components L_1, L_2, ..., L_n and weights w_1, w_2, ..., w_n:

$$L_{total} = \\sum_{i=1}^{n} w_i \\cdot L_i(predictions, targets)$$

This keeps multi-objective training in generative models explicit and easy to audit.
"""


# %%
def demonstrate_loss_system():
    """Demonstrate explicit loss composition."""
    echo("\n" + "=" * 60)
    echo("3. EXPLICIT LOSS COMPOSITION")
    echo("=" * 60)

    # Create dummy data
    key = jax.random.key(42)
    predictions = jax.random.normal(key, (8, 32))
    targets = jax.random.normal(key, (8, 32))

    # Single loss
    echo("\nSingle loss function:")
    loss_value = mse_loss(predictions, targets)
    echo(f"  MSE loss: {loss_value:.4f}")

    # Weighted loss
    echo("\nWeighted loss:")
    weighted_value = 2.0 * mse_loss(predictions, targets)
    echo(f"  Weighted MSE (2x): {weighted_value:.4f}")

    # Explicit multi-term objective
    echo("\nExplicit multi-term objective:")
    reconstruction_loss = mse_loss(predictions, targets)
    l1_penalty = mae_loss(predictions, targets)
    total_loss = reconstruction_loss + 0.5 * l1_penalty
    components = {
        "reconstruction": reconstruction_loss,
        "l1_penalty": l1_penalty,
    }
    echo(f"  Total loss: {total_loss:.4f}")
    echo(f"  Components: {components}")


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
    echo("\n" + "=" * 60)
    echo("4. SAMPLING METHODS")
    echo("=" * 60)

    # Define a simple log probability function
    def log_prob_fn(x):
        # Simple Gaussian
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(42)
    init_state = jnp.zeros(5)

    # MCMC sampling
    echo("\nMCMC Sampling:")
    mcmc_samples = mcmc_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=100,
        n_burnin=50,
        step_size=0.1,
    )
    echo(f"  Samples shape: {mcmc_samples.shape}")
    echo(f"  Mean: {jnp.mean(mcmc_samples, axis=0)}")
    echo(f"  Std: {jnp.std(mcmc_samples, axis=0)}")

    # SDE sampling (for diffusion models)
    echo("\nSDE Sampling:")

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
    echo(f"  Final sample: {sde_samples}")


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
    echo("\n" + "=" * 60)
    echo("5. MODALITY SYSTEM")
    echo("=" * 60)

    # Note: The modality system requires specific setup
    echo("\nAvailable modalities in Artifex:")
    echo("  - image: Image generation and processing")
    echo("  - text: Text generation")
    echo("  - audio: Audio synthesis")
    echo("  - protein: Protein structure modeling")
    echo("  - geometric: Point clouds and 3D data")

    # Example of how modalities work
    echo("\nModality usage pattern:")
    echo("""
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
3. **Composable components**: Mix and match models, samplers, modalities
4. **Extensible design**: Easy to add new models and features
5. **Production-ready**: Built on JAX for performance and scalability

### Best Practices:

- Use the family-specific typed config that matches the model you are creating
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
    echo("\n" + "🚀 " * 20)
    echo("ARTIFEX FRAMEWORK FEATURES DEMONSTRATION")
    echo("🚀 " * 20 + "\n")

    echo("This example demonstrates proper usage of Artifex framework features:")
    echo("- Unified configuration system")
    echo("- Factory pattern for model creation")
    echo("- Explicit loss composition")
    echo("- Sampling methods")
    echo("- Modality system")

    # Run demonstrations
    model_config = demonstrate_configuration_system()
    demonstrate_factory_pattern(model_config)
    demonstrate_loss_system()
    demonstrate_sampling_methods()
    demonstrate_modality_system()

    echo("\n" + "=" * 60)
    echo("✅ Framework features demonstration completed!")
    echo("=" * 60)

    echo("\nKey takeaways:")
    echo("1. Use family-specific typed configs for model definitions")
    echo("2. Use the factory system (create_model) instead of direct instantiation")
    echo("3. Keep multi-term objectives explicit and JAX-native")
    echo("4. Use provided sampling methods for generation")
    echo("5. Apply modality adapters for domain-specific features")


if __name__ == "__main__":
    main()
