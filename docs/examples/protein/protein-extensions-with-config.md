# Protein Extensions with Configuration System

**Level:** Intermediate | **Runtime:** ~10 seconds (CPU/GPU) | **Format:** Python + Jupyter

## Overview

This example demonstrates how to use protein extensions with Artifex's Pydantic-based configuration system. Protein extensions add domain-specific capabilities (backbone constraints, amino acid embeddings) to geometric models through a modular, composable architecture. You'll learn how to load configurations from YAML files, create extensions programmatically, and integrate them with geometric models.

## What You'll Learn

- Understand protein extensions and their modular architecture
- Load extension configurations from YAML files
- Create protein extensions programmatically with Pydantic models
- Integrate extensions with geometric models (PointCloudModel)
- Use configuration validation and serialization features
- Calculate extension-specific losses (bond length, bond angle)

## Files

- **Python Script**: [`examples/generative_models/protein/protein_extensions_with_config.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_with_config.py)
- **Jupyter Notebook**: [`examples/generative_models/protein/protein_extensions_with_config.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_with_config.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/protein/protein_extensions_with_config.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/protein/protein_extensions_with_config.ipynb
```

## Key Concepts

### Protein Extensions

Modular components that add protein-specific functionality to generic geometric models:

**Backbone Constraints Extension**:

- Enforces realistic bond lengths between atoms
- Enforces realistic bond angles
- Penalizes violations during training

**Protein Mixin Extension**:

- Embeds amino acid types (20 standard amino acids)
- Processes sequence information
- Integrates with geometric features

**Extensibility**:

- Easy to add new protein-specific features
- Composable: mix and match extensions
- Minimal coupling with base models

### Configuration System

Artifex uses Pydantic models for type-safe, validated configurations:

**Type Safety**:

```python
class ProteinExtensionConfig(BaseModel):
    name: str                              # Required field
    use_backbone_constraints: bool = True  # With default
    bond_length_weight: float = 1.0        # Validated type
```

**YAML Integration**:

```yaml
# protein.yaml
name: "protein_extensions"
use_backbone_constraints: true
bond_length_weight: 1.0
```

**Benefits**:

- Automatic validation at load time
- Self-documenting through schemas
- Easy to serialize/deserialize
- Version control friendly

## Code Structure

The example demonstrates nine major sections:

1. **Setup**: Import libraries and initialize RNGs
2. **Load Configuration**: From YAML or create programmatically
3. **Convert Config**: Map to extension parameters
4. **Create Extensions**: Using factory function
5. **Configure Model**: Set up PointCloudModel
6. **Create Model**: With extensions attached
7. **Prepare Data**: Synthetic protein structures
8. **Forward Pass**: Test model and extensions
9. **Calculate Losses**: Reconstruction + extension losses

## Example Code

### Loading Configuration from YAML

```python
from artifex.configs.schema.extensions import ProteinExtensionConfig
from artifex.configs.utils import create_config_from_yaml

# Load from YAML file
config_path = "configs/protein.yaml"
extension_config = create_config_from_yaml(config_path, ProteinExtensionConfig)

# Automatic validation
print(f"Loaded config: {extension_config.name}")
print(f"Backbone constraints: {extension_config.use_backbone_constraints}")
```

### Creating Configuration Programmatically

```python
# Fallback: create config in code
extension_config = ProteinExtensionConfig(
    name="my_protein_extensions",
    description="Custom protein extension config",
    use_backbone_constraints=True,
    use_protein_mixin=True,
)

# Pydantic validates all fields automatically
```

### Creating Protein Extensions

```python
from artifex.generative_models.extensions.protein import create_protein_extensions

# Convert config to extension parameters
protein_config = {
    "use_backbone_constraints": True,
    "bond_length_weight": 1.0,
    "bond_angle_weight": 0.5,
    "use_protein_mixin": True,
    "aa_embedding_dim": 16,
    "num_aa_types": 20,
}

# Create extensions
extensions = create_protein_extensions(protein_config, rngs=rngs)
print(f"Created extensions: {', '.join(extensions.keys())}")
# Output: Created extensions: backbone_constraints, protein_mixin
```

### Integrating with Models

```python
from artifex.generative_models.models.geometric import PointCloudModel
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)

# Create network config for point cloud processing
network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(64, 64),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout_rate=0.1,
)

# Create model config with nested network config
model_config = PointCloudConfig(
    name="protein_point_cloud",
    network=network_config,
    num_points=40,
    dropout_rate=0.1,
)

# Create model with extensions
model = PointCloudModel(model_config, extensions=extensions, rngs=rngs)
```

### Extension Loss Calculation

```python
# Forward pass
outputs = model(coords, deterministic=True)

# Calculate extension losses
total_loss = 0.0
for ext_name, extension in model.extensions.items():
    if hasattr(extension, "loss_fn"):
        ext_loss = extension.loss_fn(batch, outputs)
        total_loss += ext_loss
        print(f"{ext_name}: {ext_loss:.6f}")

# Output:
#   backbone_constraints: 0.452000
#   protein_mixin: 0.123000
```

## Features Demonstrated

- **Modular Extensions**: Composable protein-specific capabilities
- **Configuration System**: Type-safe Pydantic models with validation
- **YAML Support**: Load/save configurations from files
- **Integration**: Extensions seamlessly integrate with geometric models
- **Loss Composition**: Extension losses combine with reconstruction loss
- **Type Safety**: Automatic validation prevents configuration errors

## Experiments to Try

1. **Adjust Constraint Weights**: Control strength of geometric constraints

   ```python
   protein_config["bond_length_weight"] = 2.0  # Stricter
   protein_config["bond_angle_weight"] = 1.0
   ```

2. **Disable Extensions**: Compare with/without specific extensions

   ```python
   protein_config["use_backbone_constraints"] = False
   # Observe effect on loss and generated structures
   ```

3. **Modify Embedding Dimension**: Change amino acid representation capacity

   ```python
   protein_config["aa_embedding_dim"] = 32  # Larger embeddings
   ```

4. **Save Configuration to YAML**: Version control your settings

   ```python
   import yaml

   with open("my_config.yaml", "w") as f:
       yaml.dump(extension_config.model_dump(), f)
   ```

5. **Create Custom Extension**: Add your own protein-specific functionality

   ```python
   # Implement custom extension class
   # Add to create_protein_extensions factory
   ```

## Troubleshooting

### Common Issues

#### Configuration validation error

**Symptom:**

```
pydantic.ValidationError: 1 validation error for ProteinExtensionConfig
```

**Cause:**
Invalid value for a configuration field (wrong type, out of range, etc.)

**Solution:**

```python
# Check field requirements
print(ProteinExtensionConfig.model_json_schema())

# Fix the config
extension_config = ProteinExtensionConfig(
    name="valid_name",  # Must be string
    bond_length_weight=1.0,  # Must be float
)
```

#### YAML file not found

**Symptom:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'protein.yaml'
```

**Cause:**
Config file doesn't exist at specified path.

**Solution:**

```python
# Use absolute path
import os
config_path = os.path.join(os.getcwd(), "configs/protein.yaml")

# Or create programmatically as fallback
try:
    config = create_config_from_yaml(config_path, ProteinExtensionConfig)
except FileNotFoundError:
    config = ProteinExtensionConfig(...)  # Fallback
```

#### Extension has no loss_fn

**Symptom:**

```
AttributeError: 'BackboneConstraints' object has no attribute 'loss_fn'
```

**Cause:**
Not all extensions implement loss functions.

**Solution:**

```python
# Check before calling
if hasattr(extension, "loss_fn"):
    loss = extension.loss_fn(batch, outputs)
else:
    print(f"{ext_name} has no loss function")
```

## Summary

In this example, you learned:

- ✅ How protein extensions add modular, domain-specific capabilities to models
- ✅ How to use Pydantic-based configurations for type safety and validation
- ✅ How to load configurations from YAML files for version control
- ✅ How extensions integrate with geometric models and contribute to losses
- ✅ How the configuration system provides serialization and documentation

**Key Takeaways:**

1. **Modularity**: Extensions are composable and loosely coupled
2. **Type Safety**: Pydantic validates configs automatically
3. **YAML Integration**: Version control friendly configuration files
4. **Loss Composition**: Extensions contribute domain-specific losses

## Next Steps

<div class="grid cards" markdown>

- :material-molecule: **Protein Point Cloud**

    ---

    Deep dive into protein point cloud modeling

    [:octicons-arrow-right-24: protein-point-cloud-example.md](protein-point-cloud-example.md)

- :material-factory: **Protein with Modality**

    ---

    Learn about the modality architecture

    [:octicons-arrow-right-24: protein-model-with-modality.md](protein-model-with-modality.md)

- :material-cog: **Configuration Guide**

    ---

    Complete guide to Artifex's config system

    [:octicons-arrow-right-24: configuration-guide.md](../../user-guide/training/configuration.md)

- :material-puzzle: **Custom Extensions**

    ---

    Create your own domain-specific extensions

    [:octicons-arrow-right-24: custom-extensions.md](../../guides/extensions.md)

</div>

## Additional Resources

- **Artifex Documentation**: [Configuration System](../../guides/configuration.md)
- **Artifex Documentation**: [Protein Extensions](../../guides/protein-extensions.md)
- **Pydantic Documentation**: [Models and Validation](https://docs.pydantic.dev/)
- **API Reference**: [ProteinExtensionConfig](../../api/configs/protein-extension.md)
- **API Reference**: [create_protein_extensions](../../api/extensions/protein.md)

## Related Examples

- [Protein Point Cloud Example](protein-point-cloud-example.md) - Detailed protein geometric modeling
- [Protein Model with Modality](protein-model-with-modality.md) - Modality architecture integration
- [Geometric Benchmark Demo](../geometric/geometric-benchmark-demo.md) - Evaluating geometric models
