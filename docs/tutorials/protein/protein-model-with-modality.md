# Protein Models with Modality Architecture

**Level:** Intermediate | **Runtime:** ~10 seconds (CPU/GPU) | **Format:** Python + Jupyter

## Overview

This example demonstrates Artifex's modality architecture for protein-oriented geometric workflows. The modality system provides a unified interface for working with different data types while keeping the shared factory on the generic model family selected by the typed config. You will also see where the typed protein extension bundle fits when you need retained protein-specific runtime behavior.

## What You'll Learn

- Understand Artifex's modality architecture and its benefits
- Create protein-oriented models using the factory system with modalities
- Work with different model types (PointCloudModel, GeometricModel) for protein data
- Choose the correct typed config family for the model you want to build
- Identify where the typed protein extension bundle fits in the runtime story

## Files

- **Python Script**: [`examples/generative_models/protein/protein_model_with_modality.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_model_with_modality.py)
- **Jupyter Notebook**: [`examples/generative_models/protein/protein_model_with_modality.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_model_with_modality.ipynb)

## Quick Start

### Run the Python Script

```bash
python examples/generative_models/protein/protein_model_with_modality.py
```

### Run the Jupyter Notebook

```bash
jupyter lab examples/generative_models/protein/protein_model_with_modality.ipynb
```

## Key Concepts

### Modality Architecture

Artifex uses a modality-based design where each data type has its own modality class that handles:

- **Adapter lookup**: Choose the retained protein adapter for compatible model families
- **Typed extension bundles**: Build protein-specific runtime extensions with `ProteinExtensionsConfig`
- **Input conventions**: Keep protein-shaped batches aligned with generic geometric models

**Benefits**:

- Consistent interface across different data types
- Explicit protein boundaries without changing the selected model family
- Easy switching between different model architectures through config choice
- Protein-specific runtime behavior stays on the typed protein extension bundle

### Factory System

The `create_model()` factory provides a unified way to instantiate models:

```python
from artifex.generative_models.factory import create_model

model = create_model(
    config=model_config,
    modality="protein",  # Keeps the generic model family selected by the typed config
    rngs=rngs
)
```

**Important**: `modality="protein"` is an explicit adapter and typed extension boundary.
It does not swap in `ProteinPointCloudModel` or `ProteinGraphModel` automatically.
Use `PointCloudConfig` when you want a point-cloud model and `GeometricConfig`
when you want the generic geometric base path.

### Protein Data Structure

Protein models in Artifex expect input data with the following structure:

```python
protein_data = {
    "aatype": jnp.array,        # [batch, num_residues] - amino acid types
    "atom_positions": jnp.array,  # [batch, num_residues, num_atoms, 3] - 3D coords
    "atom_mask": jnp.array,      # [batch, num_residues, num_atoms] - presence mask
}
```

## Code Structure

The example demonstrates four major sections:

1. **Available Modalities**: List all registered modalities in the system
2. **Model Configuration**: Create typed configuration objects
3. **Factory with Modality**: Create protein models using the factory
4. **Model Usage**: Perform inference with protein data

## Example Code

### Listing Available Modalities

```python
from artifex.generative_models.modalities import list_modalities

# See what modalities are available
modalities = list_modalities()
for name, cls in modalities.items():
    print(f"  - {name}: {cls.__name__}")
# Output:
#   - image: ImageModality
#   - molecular: MolecularModality
#   - protein: ProteinModality
```

### Creating Model Configuration

```python
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)

# Create network config for point cloud
network_config = PointCloudNetworkConfig(
    name="protein_point_cloud_network",
    hidden_dims=(64, 64),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.1,
)

# Create PointCloudConfig with nested network config
model_config = PointCloudConfig(
    name="protein_point_cloud",
    network=network_config,
    num_points=128,  # Total points (residues × atoms per residue)
    dropout_rate=0.1,
)
```

### Factory with Modality Parameter

```python
from artifex.generative_models.factory import create_model
from flax import nnx

# Initialize RNG
rngs = nnx.Rngs(params=42)

# Create model with protein modality
model = create_model(
    config=model_config,
    modality="protein",  # Keeps the generic model family selected by the typed config
    rngs=rngs,
)

print(f"Created model: {model.__class__.__name__}")
# Output: Created model: PointCloudModel
```

### Using Different Model Types

```python
# Create configuration for GeometricModel instead
geometric_config = GeometricConfig(
    name="protein_geometric",
    dropout_rate=0.1,
)

# Create with same modality
geometric_model = create_model(
    config=geometric_config,
    modality="protein",
    rngs=rngs,
)
```

### Model Inference

```python
import jax.numpy as jnp

# Create dummy protein data
num_residues = 10
num_atoms = 4
batch_size = 2

protein_input = {
    "aatype": jnp.full((batch_size, num_residues), 7),  # All glycine
    "atom_positions": jnp.ones((batch_size, num_residues, num_atoms, 3)),
    "atom_mask": jnp.ones((batch_size, num_residues, num_atoms)),
}

# Generate output
outputs = model(protein_input, deterministic=True)

# Inspect results
for key, value in outputs.items():
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
# Output might include:
#   coordinates: (2, 40, 3)
#   features: (2, 40, ...)
```

## Features Demonstrated

- **Modality Registration**: Discovery of the retained registry-backed modalities
- **Factory Pattern**: Unified model creation interface
- **Typed Config Families**: Config type chooses the generic model family
- **Typed Protein Extensions**: Protein-specific runtime behavior lives on `ProteinExtensionsConfig`
- **Model Flexibility**: Same modality boundary works with different model architectures
- **Type Safety**: Configuration validation through frozen dataclass configs

## Experiments to Try

1. **Explore Other Modalities**: Try changing `modality="protein"` to `modality="image"` and see how the interface remains consistent

   ```python
   image_model = create_model(
       config=image_config,
       modality="image",
       rngs=rngs
   )
   ```

2. **Modify Extension Weights**: Build a typed protein extension bundle to change constraint weights explicitly

   ```python
   from artifex.generative_models.core.configuration import (
       ProteinExtensionConfig,
       ProteinExtensionsConfig,
   )

   extensions = ProteinExtensionsConfig(
       name="protein_extensions",
       bond_length=ProteinExtensionConfig(
           name="bond_length",
           bond_length_weight=2.0,
       ),
       bond_angle=ProteinExtensionConfig(
           name="bond_angle",
           bond_angle_weight=1.0,
       ),
   )
   ```

3. **Scale to Larger Proteins**: Increase `num_residues` and `num_atoms` to work with larger protein structures

   ```python
   num_residues = 100  # Larger protein
   num_atoms = 14      # All heavy atoms
   ```

4. **Add Training Loop**: Extend this example to include a simple training loop using the model's loss function

   ```python
   loss_fn = model.get_loss_fn()
   for batch in data_loader:
       outputs = model(batch)
       loss = loss_fn(batch, outputs)
       # Update parameters
   ```

## Troubleshooting

### Common Issues

#### ValueError: network is required and cannot be None

**Symptom:**

```
ValueError: network is required and cannot be None
```

**Cause:**
`PointCloudConfig` requires a `PointCloudNetworkConfig`. Use `GeometricConfig`
only when you intentionally want the generic geometric base path.

**Solution:**

```python
# ❌ WRONG - Generic geometric config when you need a point-cloud model
config = GeometricConfig(name="protein_geometric")

# ✅ CORRECT - Point-cloud model uses PointCloudConfig
config = PointCloudConfig(
    name="protein_point_cloud",
    network=network_config,
    num_points=128,
)
```

#### KeyError: 'sample'

**Symptom:**

```
KeyError: 'sample'
```

**Cause:**
Missing RNG keys for stochastic operations.

**Solution:**

```python
# Ensure all required RNG keys are initialized
rngs = nnx.Rngs(params=42, dropout=43, sample=44)
```

#### Shape mismatch errors

**Symptom:**

```
ValueError: Expected shape (batch, num_points, 3), got (batch, num_residues, num_atoms, 3)
```

**Cause:**
`num_points` parameter doesn't match the actual number of points in input data.

**Solution:**

```python
# Ensure num_points = num_residues × num_atoms
parameters = {
    "num_points": num_residues * num_atoms,  # e.g., 10 × 4 = 40
}
```

## Summary

In this example, you learned:

- ✅ Artifex's modality architecture provides a unified interface for different data types
- ✅ The factory system with `create_model()` keeps config choice and modality choice explicit
- ✅ Typed config families choose the generic model family
- ✅ The same modality boundary can work with different model architectures
- ✅ Protein-specific runtime behavior lives on the typed protein extension bundle

**Key Takeaways:**

1. **Modality System**: Separates domain logic from model architecture
2. **Factory Pattern**: Keeps model-family choice and modality choice explicit
3. **Typed Config Families**: `PointCloudConfig` and `GeometricConfig` choose different generic model families
4. **Protein Extensions**: The typed protein extension bundle owns retained protein-specific runtime behavior

## Next Steps

<div class="grid cards" markdown>

- :material-molecule: **Protein Point Cloud Models**

    ---

    Learn more about protein-specific geometric models

    [:octicons-arrow-right-24: protein-point-cloud-example.py](protein-point-cloud-example.md)

- :material-link-variant: **Protein Extensions**

    ---

    Explore protein constraint extensions in detail

    [:octicons-arrow-right-24: protein-extensions-example.py](protein-extensions-example.md)

- :material-cog: **Model Configuration**

    ---

    Deep dive into configuration system and parameters

    [:octicons-arrow-right-24: Configuration Guide](../../guides/configuration.md)

- :material-factory: **Factory System**

    ---

    Complete guide to the model factory

    [:octicons-arrow-right-24: Factory Guide](../../guides/factory.md)

</div>

## Additional Resources

- **Artifex Documentation**: [Modality System](../../guides/modalities.md)
- **Artifex Documentation**: [Factory System](../../guides/factory.md)
- **Artifex Documentation**: [Protein Extensions](../../guides/protein-extensions.md)
- **API Reference**: [Configuration API](../../api/configuration.md)
- **API Reference**: [create_model](../../api/factory.md)

## Related Examples

- [Protein Point Cloud Example](protein-point-cloud-example.md) - Detailed protein geometric modeling
- [Protein Extensions Example](protein-extensions-example.md) - Using protein-specific constraints
- [Geometric Benchmark Demo](../geometric/geometric-benchmark-demo.md) - Evaluating geometric models
