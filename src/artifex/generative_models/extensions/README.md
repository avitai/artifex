# Extension Mechanism for Generative Models

This package provides a flexible extension mechanism for adding domain-specific functionality to generative models without modifying core implementations.

## Directory Structure

```
extensions/
├── __init__.py         # Main exports
├── base/               # Base extension classes
│   ├── __init__.py
│   └── extensions.py   # ModelExtension, ConstraintExtension, etc.
├── protein/            # Protein-specific extensions
│   ├── __init__.py
│   ├── backbone.py     # Protein backbone constraints
│   ├── mixin.py        # Protein-specific mixins
│   └── utils.py        # Utility functions for creating extensions
└── README.md           # This file
```

## Key Features

- **Separation of Concerns**: Keep domain-agnostic model code separate from domain-specific extensions
- **Composability**: Mix and match different extension components
- **Type Safety**: Strong typing with JAX's typing system
- **Configuration Integration**: Use the artifex configuration system to create and customize extensions
- **Domain-Specific Extensions**: Ready-to-use protein-specific extensions

## Extension Types

- **ModelExtension**: Base class for all model extensions
- **ConstraintExtension**: Extensions that enforce domain-specific constraints
- **Protein Extensions**: Extensions specific to protein modeling (bond lengths, angles, etc.)

## Usage Examples

### Basic Usage

```python
import jax
from flax import nnx

from artifex.generative_models.extensions import (
    BondLengthExtension,
    BondAngleExtension,
    ProteinMixinExtension,
)
from artifex.generative_models.extensions.base import ExtensionConfig
from artifex.generative_models.models.geometric import PointCloudModel

# Initialize random number generator
rng_key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng_key)

# Create extension components individually
extensions = {
    "bond_length": BondLengthExtension(
        ExtensionConfig(
            name="bond_length",
            weight=1.0,
            extensions={"ideal_lengths": [1.45, 1.52, 1.33]}
        ),
        rngs=rngs,
    ),
    "bond_angle": BondAngleExtension(
        ExtensionConfig(
            name="bond_angle",
            weight=0.5,
            extensions={"ideal_angles": [1.94, 2.10]}
        ),
        rngs=rngs,
    ),
}

# Create model with extensions using factory
from artifex.generative_models.core.configuration import ModelConfiguration
from artifex.generative_models.factory import create_model

model_config = ModelConfiguration(
    name="point_cloud_with_extensions",
    model_class="artifex.generative_models.models.geometric.PointCloudModel",
    input_dim=(128, 3),
    hidden_dims=[64],
    output_dim=(128, 3),
    parameters={
        "num_points": 128,
        "embed_dim": 64,
        "num_layers": 3,
        "num_heads": 8,
        "dropout": 0.1
    }
)

model = create_model(model_config, extensions=extensions, rngs=rngs)
```

### Using the Helper Function

```python
from artifex.generative_models.extensions.protein import create_protein_extensions

# Create protein extensions with a simple configuration
protein_config = {
    "use_backbone_constraints": True,
    "bond_length_weight": 1.0,
    "bond_angle_weight": 0.5,
    "use_protein_mixin": True,
}

extensions = create_protein_extensions(protein_config, rngs=rngs)
```

## How Extensions Work

Extensions are attached to models during initialization and can:

1. **Process Inputs/Outputs**: Extensions can process model inputs and outputs
2. **Add Loss Terms**: Extensions can contribute additional loss terms
3. **Project Outputs**: Constraint extensions can project outputs to satisfy constraints
4. **Validate Results**: Extensions can validate model outputs against constraints

## Creating Custom Extensions

To create a custom extension:

1. Subclass `ModelExtension` or `ConstraintExtension`
2. Implement the required methods (`__call__`, `loss_fn`, etc.)
3. Register your extension in the appropriate location

## Migration Notes

This extension mechanism was restructured from the previous location under `models/geometric/extensions` to improve separation of concerns. The restructuring:

1. Moved base extension classes to `extensions/base/`
2. Moved protein-specific extensions to `extensions/protein/`
3. Ensured backward compatibility with existing code
4. Added proper configuration integration

If you're updating existing code, you'll need to change imports:

```python
# Old imports
from artifex.generative_models.models.geometric.extensions import ...

# New imports
from artifex.generative_models.extensions import ...
from artifex.generative_models.extensions.protein import ...
```
