# Extensions

Extensions in Artifex provide domain-specific functionality that enhances base generative models with specialized knowledge and constraints. This modular system allows you to add protein modeling, NLP preprocessing, audio processing, and vision augmentation capabilities to any model.

## Overview

<div class="grid cards" markdown>

- :material-molecule:{ .lg .middle } **Protein Extensions**

    ---

    Add protein-specific constraints, backbone geometry, and amino acid features to geometric models

    [:octicons-arrow-right-24: Protein Extensions](#protein-extensions)

- :material-text:{ .lg .middle } **NLP Extensions**

    ---

    Tokenization, embeddings, and text preprocessing for language models

    [:octicons-arrow-right-24: NLP Extensions](#nlp-extensions)

- :material-waveform:{ .lg .middle } **Audio Extensions**

    ---

    Spectral processing and temporal features for audio generation

    [:octicons-arrow-right-24: Audio Extensions](#audio-extensions)

- :material-image:{ .lg .middle } **Vision Extensions**

    ---

    Image augmentation and preprocessing for visual models

    [:octicons-arrow-right-24: Vision Extensions](#vision-extensions)

</div>

## Quick Start

Extensions integrate seamlessly with Artifex models through the extension system:

```python
import jax
from flax import nnx
from artifex.generative_models.core.configuration import (
    ProteinExtensionConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.extensions.protein import (
    BondLengthExtension,
    BondAngleExtension,
    ProteinMixinExtension,
)
from artifex.generative_models.models.geometric.point_cloud import PointCloudModel

# Configure extensions using frozen dataclass configs
mixin_config = ProteinMixinConfig(
    name="protein_mixin",
    weight=1.0,
    enabled=True,
    embedding_dim=64,
    num_aa_types=20,
)

# Create extension instance
key = jax.random.key(42)
protein_mixin = ProteinMixinExtension(
    config=mixin_config,
    rngs=nnx.Rngs(params=key),
)

# Wrap in nnx.Dict for NNX compatibility
extensions = nnx.Dict({"protein_mixin": protein_mixin})

# Create model with extensions
model = PointCloudModel(model_config, extensions=extensions, rngs=nnx.Rngs(params=key))
```

---

## Protein Extensions

Protein extensions add domain knowledge about molecular structure to geometric models, enabling physically realistic protein generation.

### Available Extensions

| Extension | Description | Key Features |
|-----------|-------------|--------------|
| **ProteinMixinExtension** | Amino acid integration | 20 AA type embeddings, residue features |
| **ProteinBackboneConstraint** | Backbone geometry | N, CA, C, O atom indices, geometric constraints |
| **BondLengthExtension** | Bond distance monitoring | Violation detection, loss contribution |
| **BondAngleExtension** | Bond angle monitoring | Peptide bond angles, backbone geometry |

### Usage Example

```python
import jax
from flax import nnx
from artifex.generative_models.core.configuration import (
    ProteinExtensionConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.extensions.protein import (
    BondLengthExtension,
    BondAngleExtension,
    ProteinMixinExtension,
)
from artifex.generative_models.extensions.protein.constraints import (
    ProteinBackboneConstraint,
)

key = jax.random.key(42)

# Create multiple protein extensions
extensions_dict = {}

# Amino acid type integration with frozen dataclass config
extensions_dict["protein_mixin"] = ProteinMixinExtension(
    config=ProteinMixinConfig(
        name="protein_mixin",
        weight=1.0,
        enabled=True,
        embedding_dim=64,
        num_aa_types=20,
    ),
    rngs=nnx.Rngs(params=key),
)

# Backbone constraints with explicit fields
extensions_dict["backbone"] = ProteinBackboneConstraint(
    config=ProteinExtensionConfig(
        name="backbone",
        weight=1.0,
        enabled=True,
        bond_length_weight=1.0,
        bond_angle_weight=0.5,
    ),
    rngs=nnx.Rngs(params=key),
)

# Bond length monitoring
extensions_dict["bond_length"] = BondLengthExtension(
    config=ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        enabled=True,
        bond_length_weight=1.0,
        ideal_bond_lengths={"N-CA": 1.45, "CA-C": 1.52, "C-N": 1.33},
    ),
    rngs=nnx.Rngs(params=key),
)

# Wrap for NNX
extensions = nnx.Dict(extensions_dict)
```

### Documentation

- [Backbone Extension](backbone.md) - Backbone atom handling
- [Constraints](constraints.md) - Geometric constraints
- [Mixin Extension](mixin.md) - Amino acid integration
- [Utilities](utils.md) - Protein utility functions

---

## NLP Extensions

NLP extensions provide text processing capabilities for language models and multimodal systems.

### Available Extensions

| Extension | Description | Key Features |
|-----------|-------------|--------------|
| **Tokenization** | Text tokenization | BPE, SentencePiece, character-level |
| **Embeddings** | Token embeddings | Positional encoding, learned embeddings |

### Documentation

- [Tokenization](tokenization.md) - Tokenization methods
- [Embeddings](embeddings.md) - Embedding systems

---

## Audio Extensions

Audio extensions add signal processing capabilities for audio generation models.

### Available Extensions

| Extension | Description | Key Features |
|-----------|-------------|--------------|
| **Spectral** | Frequency analysis | STFT, mel-spectrograms, spectrogram inversion |
| **Temporal** | Time-domain features | Envelope extraction, onset detection |

### Documentation

- [Spectral Processing](spectral.md) - Frequency domain operations
- [Temporal Features](temporal.md) - Time domain processing

---

## Vision Extensions

Vision extensions provide image preprocessing and augmentation for visual models.

### Available Extensions

| Extension | Description | Key Features |
|-----------|-------------|--------------|
| **Augmentation** | Data augmentation | Flips, rotations, color jitter, cutout |

### Documentation

- [Augmentation](augmentation.md) - Image augmentation methods

---

## Extension Architecture

### Configuration Classes

Extensions use frozen dataclass configurations from `core.configuration`:

```python
from artifex.generative_models.core.configuration import (
    ExtensionConfig,           # Base extension config
    ConstraintExtensionConfig, # For constraint extensions
    ProteinExtensionConfig,    # Protein-specific constraints
    ProteinMixinConfig,        # Protein amino acid features
    ChemicalConstraintConfig,  # Chemical/molecular constraints
    ImageAugmentationConfig,   # Vision augmentation
    AudioSpectralConfig,       # Audio spectral processing
    TextEmbeddingConfig,       # NLP embeddings
)

# Base ExtensionConfig for simple extensions
config = ExtensionConfig(
    name="my_extension",      # Unique identifier
    weight=1.0,               # Loss contribution weight
    enabled=True,             # Enable/disable toggle
)

# Domain-specific configs have explicit fields (no extensions dict)
protein_config = ProteinExtensionConfig(
    name="backbone",
    weight=1.0,
    enabled=True,
    bond_length_weight=1.0,   # Explicit field, not in extensions dict
    bond_angle_weight=0.5,
    ideal_bond_lengths={"N-CA": 1.45, "CA-C": 1.52},
)
```

### Extension Registry

Extensions can be registered and discovered through the registry:

```python
from artifex.generative_models.extensions.registry import (
    register_extension,
    get_extension,
    list_extensions,
)

# List available extensions
available = list_extensions()
print(f"Available extensions: {available}")

# Get extension by name
ExtensionClass = get_extension("protein_mixin")
```

### Documentation

- [Extensions Base](extensions.md) - Base extension classes
- [Registry](registry.md) - Extension registration system
- [Features](features.md) - Feature extraction utilities

---

## Creating Custom Extensions

You can create custom extensions by inheriting from the base extension class:

```python
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import ModelExtension


# Define a custom frozen dataclass config for your extension
@dataclasses.dataclass(frozen=True)
class MyExtensionConfig(ExtensionConfig):
    """Custom extension configuration."""
    my_param: float = 1.0
    another_param: int = 10


class MyCustomExtension(ModelExtension):
    """Custom extension for domain-specific processing."""

    def __init__(
        self,
        config: MyExtensionConfig | ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs=rngs)
        # Use explicit config fields (frozen dataclass pattern)
        if isinstance(config, MyExtensionConfig):
            self.my_param = config.my_param
            self.another_param = config.another_param
        else:
            # Defaults for base ExtensionConfig
            self.my_param = 1.0
            self.another_param = 10

    def __call__(self, inputs, model_outputs, **kwargs) -> dict:
        """Process inputs and model outputs.

        Args:
            inputs: Input batch dictionary
            model_outputs: Model predictions
            **kwargs: Additional arguments

        Returns:
            Extension outputs dictionary
        """
        if not self.enabled:
            return {"extension_type": "my_custom"}

        # Implement extension logic
        result = self._process(inputs, model_outputs)
        return {"my_output": result, "extension_type": "my_custom"}

    def loss_fn(self, batch: dict, model_outputs, **kwargs) -> jax.Array:
        """Compute extension-specific loss.

        Args:
            batch: Input batch
            model_outputs: Model outputs

        Returns:
            Loss value (scalar JAX array)
        """
        if not self.enabled:
            return jnp.array(0.0)

        # Implement loss computation using pure JAX operations
        return self._compute_loss(batch, model_outputs)
```

---

## Best Practices

!!! success "DO"
    - Use frozen dataclass configs from `core.configuration`
    - Use domain-specific configs (e.g., `ProteinExtensionConfig`) with explicit fields
    - Wrap extensions in `nnx.Dict` for NNX compatibility
    - Set appropriate weights for multi-extension setups
    - Disable unused extensions for efficiency
    - Use pure JAX operations in `loss_fn` for JIT compatibility

!!! danger "DON'T"
    - Don't use `extensions={}` dict pattern (old Pydantic style)
    - Don't use raw dictionaries instead of `nnx.Dict`
    - Don't forget to pass `rngs` to extension constructors
    - Don't use conflicting extension names
    - Don't enable extensions without proper configuration
    - Don't mutate RNGs inside traced functions (JIT/grad)

---

## Summary

Extensions provide a modular way to add domain-specific functionality:

- **Protein**: Physical constraints and amino acid features
- **NLP**: Tokenization and text embeddings
- **Audio**: Spectral and temporal processing
- **Vision**: Image augmentation

All extensions follow consistent patterns for configuration, registration, and integration with Artifex models.
