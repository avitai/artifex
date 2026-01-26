# Protein Model Extensions Example

![Level](https://img.shields.io/badge/Level-Intermediate-orange)
![Runtime](https://img.shields.io/badge/Runtime-~10s-green)
![Format](https://img.shields.io/badge/Format-Script%20%2B%20Notebook-blue)

Demonstrate how to use protein-specific extensions with Artifex's geometric model framework, combining domain knowledge with general-purpose geometric models.

## Files

- **Python Script**: [`protein_model_extension.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_model_extension.py)
- **Jupyter Notebook**: [`protein_model_extension.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_model_extension.ipynb)

## Quick Start

```bash
# Run the Python script
source activate.sh
python examples/generative_models/protein/protein_model_extension.py

# Or use Jupyter notebook
jupyter notebook examples/generative_models/protein/protein_model_extension.ipynb
```

## Overview

This example shows how to enhance a point cloud model with protein-specific extensions. Extensions add domain knowledge about protein structure, chemistry, and geometry to improve model predictions and learning.

### Learning Objectives

- [ ] Understand the extension system in Artifex
- [ ] Learn how to create and configure protein-specific extensions
- [ ] See how extensions enhance model outputs with domain knowledge
- [ ] Understand extension contribution to loss calculations
- [ ] Learn to combine multiple extensions for complex domain knowledge

### Prerequisites

- Basic understanding of protein structure (residues, backbone atoms)
- Familiarity with point cloud models
- Knowledge of Flax NNX modules
- Understanding of JAX random number generation

## Background: Protein Structure and Extensions

### Protein Basics

Proteins are polymers composed of amino acid residues connected by peptide bonds. The backbone consists of repeating units with four key atoms per residue:

- **N** (Nitrogen): Backbone nitrogen
- **Cα** (Alpha Carbon): Central carbon with side chain
- **C** (Carbonyl Carbon): Carbon of carbonyl group
- **O** (Oxygen): Carbonyl oxygen

### Geometric Constraints

Protein structures follow specific geometric constraints due to chemical bonding:

**Bond Lengths**: Relatively fixed distances between bonded atoms

$$d_{\text{C-N}} \approx 1.33\text{Å}, \quad d_{\text{N-Cα}} \approx 1.46\text{Å}, \quad d_{\text{Cα-C}} \approx 1.52\text{Å}$$

**Bond Angles**: Preferred angles between consecutive bonds

$$\theta_{\text{N-Cα-C}} \approx 110°, \quad \theta_{\text{Cα-C-N}} \approx 120°$$

**Torsion Angles**: Backbone flexibility through φ (phi) and ψ (psi) angles

These constraints make protein structure prediction a constrained optimization problem, which extensions help enforce.

### Extension System

Artifex's extension system allows adding domain-specific knowledge to base models. For proteins, we have:

1. **Protein Mixin Extension**: Integrates amino acid type information
2. **Protein Constraints Extension**: Enforces backbone geometry
3. **Bond Length Extension**: Monitors and penalizes bond violations
4. **Bond Angle Extension**: Monitors and penalizes angle violations

## Code Walkthrough

### 1. Import Required Modules

```python
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.extensions.base.extensions import ExtensionConfig
from artifex.generative_models.extensions.protein import (
    BondAngleExtension,
    BondLengthExtension,
    ProteinMixinExtension,
)
from artifex.generative_models.extensions.protein.constraints import (
    ProteinBackboneConstraint,
)
from artifex.generative_models.models.geometric.point_cloud import (
    PointCloudModel,
)
```

We import:

- JAX for array operations and random number generation
- Flax NNX for neural network modules
- Artifex configuration and extension classes
- Protein-specific extensions
- Point cloud model

### 2. Configure Protein Structure

```python
num_residues = 10
atoms_per_residue = 4  # N, CA, C, O
num_points = num_residues * atoms_per_residue
embedding_dim = 64

# Create network config for point cloud processing
network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(embedding_dim,) * 2,  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=embedding_dim,
    num_heads=4,
    num_layers=2,
    dropout_rate=0.1,
)

# Create point cloud config with nested network config
model_config = PointCloudConfig(
    name="protein_point_cloud",
    network=network_config,
    num_points=num_points,
    dropout_rate=0.1,
)
```

Key configuration points:

- **num_points**: Must be specified to override default (1024)
- **hidden_dims**: List of hidden dimensions for each layer
- **parameters**: Additional model-specific parameters
- **dropout_rate**: Regularization during training

### 3. Create Protein-Specific Extensions

#### 3.1 Protein Mixin Extension

```python
mixin_config = ExtensionConfig(
    name="protein_mixin",
    weight=1.0,
    enabled=True,
    extensions={
        "embedding_dim": embedding_dim,
        "num_aa_types": 20,
    },
)
extensions_dict["protein_mixin"] = ProteinMixinExtension(
    config=mixin_config,
    rngs=nnx.Rngs(params=mixin_key),
)
```

The mixin extension learns embeddings for all 20 standard amino acid types, allowing the model to incorporate sequence information.

#### 3.2 Protein Constraints Extension

```python
constraint_config = ExtensionConfig(
    name="protein_constraints",
    weight=1.0,
    enabled=True,
    extensions={
        "num_residues": num_residues,
        "backbone_indices": [0, 1, 2, 3],
    },
)
extensions_dict["protein_constraints"] = ProteinBackboneConstraint(
    config=constraint_config,
    rngs=nnx.Rngs(params=constraint_key),
)
```

This extension enforces geometric constraints on backbone atoms during generation.

#### 3.3 Bond Length Extension

```python
bond_length_config = ExtensionConfig(
    name="bond_length",
    weight=1.0,
    enabled=True,
    extensions={
        "num_residues": num_residues,
        "backbone_indices": [0, 1, 2, 3],
    },
)
extensions_dict["bond_length"] = BondLengthExtension(
    config=bond_length_config,
    rngs=nnx.Rngs(params=constraint_key),
)
```

Monitors bond lengths and calculates violations for use in the loss function.

#### 3.4 Bond Angle Extension

```python
bond_angle_config = ExtensionConfig(
    name="bond_angle",
    weight=0.5,  # Lower weight - angles are more flexible
    enabled=True,
    extensions={
        "num_residues": num_residues,
        "backbone_indices": [0, 1, 2, 3],
    },
)
extensions_dict["bond_angle"] = BondAngleExtension(
    config=bond_angle_config,
    rngs=nnx.Rngs(params=backbone_key),
)
```

Note the lower weight (0.5) compared to bond lengths, reflecting the fact that bond angles are more flexible than bond lengths in real proteins.

### 4. Wrap Extensions in nnx.Dict

```python
extensions = nnx.Dict(extensions_dict)
```

Flax NNX 0.12.0+ requires extensions to be wrapped in `nnx.Dict` for proper parameter tracking and serialization.

### 5. Create Model with Extensions

```python
model = PointCloudModel(
    model_config,
    extensions=extensions,
    rngs=nnx.Rngs(params=model_key)
)
```

The point cloud model now has access to all four extensions and will use them during forward passes and loss calculation.

### 6. Create Test Batch

```python
batch = {
    "aatype": aatype,        # Shape: (batch_size, num_residues)
    "positions": coords,      # Shape: (batch_size, num_points, 3)
    "mask": mask,            # Shape: (batch_size, num_points)
}
```

The batch contains:

- **aatype**: Amino acid types (integers 0-19 for 20 amino acids)
- **positions**: 3D coordinates of all atoms
- **mask**: Binary mask indicating valid atoms

### 7. Forward Pass with Extensions

```python
outputs = model(batch)
```

During the forward pass:

1. Model processes input through transformer layers
2. Each enabled extension runs on the intermediate representations
3. Extension outputs are collected and returned alongside main output

Extension outputs might include:

- Amino acid embeddings (from mixin)
- Constraint violation metrics
- Bond statistics
- Angle statistics

### 8. Calculate Loss with Extensions

```python
loss_fn = model.get_loss_fn()
loss_outputs = loss_fn(batch, outputs)
```

The loss function combines:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \sum_{i} w_i \mathcal{L}_{\text{ext}_i}$$

Where:

- $\mathcal{L}_{\text{MSE}}$: Main reconstruction loss
- $w_i$: Extension weight
- $\mathcal{L}_{\text{ext}_i}$: Extension-specific loss

This multi-objective loss encourages the model to:

1. Reconstruct input positions accurately
2. Respect bond length constraints
3. Maintain realistic bond angles
4. Utilize amino acid type information

## Expected Output

```
Created extensions: protein_mixin, protein_constraints, bond_length, bond_angle
Created model: PointCloudModel

Model outputs:
- Main output shape: (2, 40, 3)
- Extension outputs:
  - protein_mixin
  - protein_constraints
  - bond_length
  - bond_angle

Loss calculation:
- Available loss keys: ['total_loss', 'mse_loss', 'protein_mixin', 'protein_constraints', 'bond_length', 'bond_angle']
- Total loss: 89.77
- total_loss: 89.77
- mse_loss: 87.01

Protein model extension demo completed successfully!
```

## Key Concepts

### Extension Configuration

Extensions use `ExtensionConfig` with:

- **name**: Identifier for the extension
- **weight**: Contribution to total loss (0-1 or higher)
- **enabled**: Whether extension is active
- **extensions**: Extension-specific parameters dict

### Extension Weights

Weights control the relative importance of different constraints:

- **Bond lengths**: Weight 1.0 (strict constraint)
- **Bond angles**: Weight 0.5 (more flexible)
- **Constraints**: Weight 1.0 (enforce geometry)
- **Mixin**: Weight 1.0 (sequence information)

Adjust weights based on your application's priorities.

### Flax NNX 0.12.0+ Compatibility

Always wrap extension dictionaries in `nnx.Dict`:

```python
# CORRECT
extensions = nnx.Dict(extensions_dict)

# WRONG (will fail in NNX 0.12.0+)
extensions = extensions_dict
```

### Random Number Generation

Each extension receives its own RNG for parameter initialization:

```python
key, mixin_key, constraint_key, backbone_key = jax.random.split(key, 4)

extensions_dict["protein_mixin"] = ProteinMixinExtension(
    config=mixin_config,
    rngs=nnx.Rngs(params=mixin_key),  # Separate key
)
```

This ensures independent randomness across extensions.

## Experiments to Try

1. **Adjust Extension Weights**

   ```python
   # Emphasize bond lengths more
   bond_length_config.weight = 2.0
   bond_angle_config.weight = 0.1
   ```

2. **Disable Specific Extensions**

   ```python
   # See impact of removing angle constraints
   bond_angle_config.enabled = False
   ```

3. **Increase Protein Size**

   ```python
   num_residues = 50  # Larger protein
   atoms_per_residue = 4
   ```

4. **Add Custom Extensions**

   Create your own extension for other properties (e.g., secondary structure, hydrophobicity).

5. **Visualize Bond Statistics**

   Extract and plot bond length/angle distributions from extension outputs.

6. **Compare With/Without Extensions**

   Train two models (with and without extensions) and compare structure quality.

## Next Steps

Explore related examples to deepen your understanding:

<div class="grid cards" markdown>

- :material-molecule:{ .lg .middle } **Protein Extensions Deep Dive**

    ---

    Learn more about individual protein extensions and their implementation.

    [:octicons-arrow-right-24: protein_extensions_example.py](protein-extensions-example.md)

- :material-chart-scatter-plot:{ .lg .middle } **Protein Point Cloud Model**

    ---

    Explore the ProteinPointCloudModel that combines point clouds with protein constraints.

    [:octicons-arrow-right-24: protein_point_cloud_example.py](protein-point-cloud-example.md)

- :material-wave:{ .lg .middle } **Protein Diffusion**

    ---

    Use diffusion models for protein structure generation with extensions.

    [:octicons-arrow-right-24: protein_diffusion_example.py](protein-diffusion-example.md)

- :material-flask:{ .lg .middle } **Protein Benchmarks**

    ---

    Evaluate protein models with domain-specific metrics.

    [:octicons-arrow-right-24: protein_ligand_benchmark_demo.py](protein-ligand-benchmark-demo.md)

</div>

## Troubleshooting

### Extension Not Contributing to Loss

**Problem**: Extension appears in outputs but not in loss.

**Solution**: Check that:

1. Extension weight is non-zero
2. Extension is enabled (`enabled=True`)
3. Extension implements `compute_loss()` method

### nnx.Dict Error

**Problem**: `TypeError: extensions must be nnx.Dict`

**Solution**: Wrap your extensions dictionary:

```python
extensions = nnx.Dict(extensions_dict)
```

### Bond Violations Too High

**Problem**: Bond length/angle violations are unreasonably large.

**Solution**:

1. Check input coordinates are in correct units (Angstroms)
2. Verify backbone indices match your atom ordering
3. Increase extension weights to penalize violations more

### Out of Memory

**Problem**: GPU runs out of memory with extensions.

**Solution**:

1. Reduce batch size
2. Reduce number of residues
3. Reduce embedding dimension
4. Disable less critical extensions

## Additional Resources

- [Extension System Documentation](../../guides/extensions.md)
- [Point Cloud Models Guide](../../guides/geometric-models.md)
- [Protein Modeling Tutorial](../../tutorials/protein-modeling.md)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/)
- [JAX Documentation](https://jax.readthedocs.io/)

## Citation

If you use this example in your research, please cite:

```bibtex
@software{artifex2025,
  title={Artifex: Modular Generative Modeling Library},
  author={Artifex Contributors},
  year={2025},
  url={https://github.com/avitai/artifex}
}
```
