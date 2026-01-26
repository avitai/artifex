# Protein Models with Modality Architecture

**Level:** Intermediate | **Runtime:** ~10 seconds (CPU/GPU) | **Format:** Python + Jupyter

## Overview

This example demonstrates Artifex's modality architecture for creating protein-specific generative models. The modality system provides a unified interface for working with different data types (image, text, protein, etc.) while maintaining domain-specific capabilities. You'll learn how to use the factory system to create protein models with automatic modality-specific enhancements.

## What You'll Learn

- Understand Artifex's modality architecture and its benefits
- Create protein models using the factory system with modalities
- Work with different model types (PointCloudModel, GeometricModel) for proteins
- Use full module paths when working with the factory system
- Apply domain-specific extensions through modality configuration

## Files

- **Python Script**: [`examples/generative_models/protein/protein_model_with_modality.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_model_with_modality.py)
- **Jupyter Notebook**: [`examples/generative_models/protein/protein_model_with_modality.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_model_with_modality.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/protein/protein_model_with_modality.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/protein/protein_model_with_modality.ipynb
```

## Key Concepts

### Modality Architecture

Artifex uses a modality-based design where each data type has its own modality class that handles:

- **Domain-specific preprocessing**: Convert protein data to appropriate representations
- **Evaluation metrics**: Compute relevant metrics for the domain (RMSD, TM-score, etc.)
- **Model adaptations**: Apply domain-specific enhancements and extensions

**Benefits**:

- Consistent interface across different data types
- Automatic application of domain expertise
- Easy switching between different model architectures
- Built-in best practices for each domain

### Factory System

The `create_model()` factory provides a unified way to instantiate models:

```python
from artifex.generative_models.factory import create_model

model = create_model(
    config=model_config,
    modality="protein",  # Automatically applies protein-specific enhancements
    rngs=rngs
)
```

**Important**: When using the factory system, model classes **must be specified with full module paths**:

- ✅ `"artifex.generative_models.models.geometric.point_cloud.PointCloudModel"`
- ❌ `"PointCloudModel"` (will raise ValueError)

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
2. **Model Configuration**: Create configuration with full module paths
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
#   - protein: MolecularModality
#   - image: ImageModality
#   - text: TextModality
#   ...
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
    modality="protein",  # Applies protein-specific enhancements
    rngs=rngs,
)

print(f"Created model: {model.__class__.__name__}")
# Output: Created model: PointCloudModel
```

### Using Different Model Types

```python
# Create configuration for GeometricModel instead
geometric_config = ModelConfig(
    name="protein_geometric",
    # Different model class, same modality system
    model_class="artifex.generative_models.models.geometric.base.GeometricModel",
    input_dim=128,
    output_dim=128,
    hidden_dims=[64, 64],
    parameters=model_config.parameters,
    metadata=model_config.metadata,
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
#   bond_length: {...}
#   bond_angle: {...}
```

## Features Demonstrated

- **Modality Registration**: Automatic discovery of available modalities
- **Factory Pattern**: Unified model creation interface
- **Full Module Paths**: Required syntax for model class specification
- **Domain Extensions**: Protein-specific constraints and mixins
- **Model Flexibility**: Same modality works with different model architectures
- **Type Safety**: Configuration validation through Pydantic models

## Experiments to Try

1. **Explore Other Modalities**: Try changing `modality="protein"` to `modality="image"` and see how the interface remains consistent

   ```python
   image_model = create_model(
       config=image_config,
       modality="image",
       rngs=rngs
   )
   ```

2. **Modify Extension Weights**: Change `bond_length_weight` and `bond_angle_weight` to see their effect on the model's constraint enforcement

   ```python
   metadata = {
       "extensions": {
           "bond_length_weight": 2.0,  # Increased from 1.0
           "bond_angle_weight": 1.0,   # Increased from 0.5
       }
   }
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

#### ValueError: Invalid model class

**Symptom:**

```
ValueError: model_class must be a fully qualified module path
```

**Cause:**
Using short names like `"PointCloudModel"` instead of full module paths.

**Solution:**

```python
# ❌ WRONG - Short name
model_class="PointCloudModel"

# ✅ CORRECT - Full module path
model_class="artifex.generative_models.models.geometric.point_cloud.PointCloudModel"
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
- ✅ The factory system with `create_model()` simplifies model creation
- ✅ Full module paths are required when specifying model classes
- ✅ The same modality can work with different model architectures
- ✅ Domain-specific extensions are automatically applied through modality configuration

**Key Takeaways:**

1. **Modality System**: Separates domain logic from model architecture
2. **Factory Pattern**: Consistent API across all model types
3. **Full Paths Required**: Always use complete module paths for `model_class`
4. **Flexibility**: Easy to switch between different model types while maintaining domain capabilities

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

    Comprehensive guide to the model factory

    [:octicons-arrow-right-24: Factory Guide](../../guides/factory.md)

</div>

## Additional Resources

- **Artifex Documentation**: [Modality System](../../guides/modalities.md)
- **Artifex Documentation**: [Factory System](../../guides/factory.md)
- **Artifex Documentation**: [Protein Extensions](../../guides/protein-extensions.md)
- **API Reference**: [ModelConfig](../../api/configuration.md)
- **API Reference**: [create_model](../../api/factory.md)

## Related Examples

- [Protein Point Cloud Example](protein-point-cloud-example.md) - Detailed protein geometric modeling
- [Protein Extensions Example](protein-extensions-example.md) - Using protein-specific constraints
- [Geometric Benchmark Demo](../geometric/geometric-benchmark-demo.md) - Evaluating geometric models
