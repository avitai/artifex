# Design Philosophy

Artifex is designed with a clear set of guiding principles that inform every architectural and implementation decision.

## Core Principles

### Research-First Design

Artifex prioritizes research workflows and experimentation:

- **Modularity**: Components can be easily swapped and combined for experimentation
- **Clarity**: Implementations favor readability over clever optimizations
- **Extensibility**: Simple to add new models, losses, and domain-specific functionality
- **Reproducibility**: Deterministic execution with clear configuration management

### Modern Stack

Built on the latest JAX ecosystem:

- **JAX Native**: Leverages JAX's functional programming paradigm for composable transformations
- **Flax NNX**: Uses the modern object-oriented API (not Linen) for neural network definitions
- **Optax**: Standard optimization library for gradient-based learning
- **BlackJAX**: MCMC sampling for energy-based models

### Type Safety and Protocols

Strong typing throughout:

- **Protocol-Based Design**: All major components implement Python Protocols for type-safe interfaces
- **Frozen Dataclass Configs**: Immutable configuration objects with automatic validation
- **Full Type Annotations**: Comprehensive type hints for IDE support and static analysis

## Configuration Philosophy

### Why Frozen Dataclasses?

Artifex uses frozen dataclass configurations instead of dictionaries or mutable config objects:

```python
from artifex.generative_models.core.configuration import VAEConfig, EncoderConfig

# Immutable, validated configuration
encoder_config = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=64,
    hidden_dims=(256, 128),  # Tuple, not list
    activation="relu",
)
```

**Benefits:**

1. **Immutability**: Prevents accidental modification during training
2. **Validation**: Automatic type checking and value validation at construction
3. **Serialization**: Easy JSON/YAML export for reproducibility
4. **IDE Support**: Full autocomplete and type checking
5. **Nested Composition**: Complex models built from simple, composable configs

### Configuration Composition

Models use nested configurations for clear separation of concerns:

```python
config = VAEConfig(
    name="my_vae",
    encoder=encoder_config,  # Nested encoder config
    decoder=decoder_config,  # Nested decoder config
    kl_weight=1.0,
)
```

## Module Design

### Factory Pattern

All models are created through factories for consistent initialization:

```python
from artifex.generative_models.factory import create_model

model = create_model(config, rngs=rngs)
```

**Why factories?**

- Consistent validation before instantiation
- Proper RNG management
- Easy to swap models for experimentation
- Clear error messages for misconfiguration

### RNG Management

Flax NNX requires explicit RNG streams:

```python
from flax import nnx

rngs = nnx.Rngs(params=42, dropout=43, sample=44)
model = create_model(config, rngs=rngs)
```

Different streams serve different purposes:

- `params`: Parameter initialization
- `dropout`: Dropout randomness
- `sample`: Generative sampling

## Testing Philosophy

### Test-Driven Development

- Write tests first, then implement functionality
- Tests define expected behavior, not current implementation
- Never modify tests to accommodate flawed implementations
- Minimum 80% coverage for new code

### Test Organization

Tests mirror source structure:

- `tests/standalone/`: Isolated component tests
- `tests/artifex/`: Integrated system tests
- GPU tests marked with `@pytest.mark.gpu`

## What We Don't Do

### No Backward Compatibility Hacks

- Breaking changes are acceptable for better foundations
- No deprecated parameter forwarding
- No compatibility shims
- Clean removal of unused code

### No Over-Engineering

- Only make changes directly requested or clearly necessary
- Don't add features, refactor code, or make "improvements" beyond what was asked
- Don't add error handling for scenarios that can't happen
- Don't create abstractions for one-time operations

## Framework Constraints

### Flax NNX Only

Artifex exclusively uses Flax NNX:

- **NEVER** use Flax Linen
- **NEVER** use PyTorch or TensorFlow
- Always call `super().__init__()` in module constructors
- Use `nnx.gelu`, `nnx.relu` not `jax.nn.gelu`, `jax.nn.relu`

### JAX Compatibility

Inside NNX modules:

- Use `jax.numpy` not `numpy`
- Use `jax.scipy` not `scipy`
- No numpy-based packages (scipy, sklearn)

## See Also

- [Core Concepts](../getting-started/core-concepts.md) - Architecture overview
- [Quickstart Guide](../getting-started/quickstart.md) - Practical getting started
