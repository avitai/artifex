# Protein Point Cloud Model Example

**Level:** Intermediate | **Runtime:** ~15 seconds (CPU) / ~5 seconds (GPU) | **Format:** Python + Jupyter

## Overview

This example demonstrates the ProteinPointCloudModel, a specialized geometric model designed for protein structure generation and refinement. It combines point cloud processing with protein-specific constraints (bond lengths, bond angles) to generate physically plausible protein structures. You'll learn how to work with backbone-only representations and apply geometric constraints to ensure chemical validity.

## What You'll Learn

- Represent proteins as 3D point clouds (atoms as points in space)
- Configure and create ProteinPointCloudModel with attention mechanisms
- Generate synthetic protein data with alpha-helix geometry
- Apply geometric constraints (bond lengths, angles) during generation
- Evaluate model outputs using reconstruction and constraint losses
- Work with backbone-only protein representations (N, CA, C, O atoms)

## Files

- **Python Script**: [`examples/generative_models/protein/protein_point_cloud_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_point_cloud_example.py)
- **Jupyter Notebook**: [`examples/generative_models/protein/protein_point_cloud_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_point_cloud_example.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/protein/protein_point_cloud_example.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/protein/protein_point_cloud_example.ipynb
```

## Key Concepts

### Point Cloud Representation

Proteins can be represented as sets of 3D points, where each point corresponds to an atom's position in space:

```python
atom_positions: [num_residues, num_atoms, 3]  # 3D coordinates
atom_mask: [num_residues, num_atoms]           # Presence indicator
```

**Benefits**:

- **Invariance**: Naturally invariant to rotation and translation
- **Flexibility**: Can handle variable-length structures
- **Geometric**: Directly represents 3D spatial structure

### Backbone Atoms

The protein backbone consists of four atoms present in every amino acid:

- **N** (Nitrogen): Index 0, forms peptide bond
- **CA** (Alpha Carbon): Index 1, central carbon atom
- **C** (Carbon): Index 2, carbonyl carbon
- **O** (Oxygen): Index 4, carbonyl oxygen

These atoms determine the overall protein structure (folds and secondary structures).

### Geometric Constraints

Physical constraints ensure generated structures are chemically valid:

**Bond Length Constraints**:
$$\mathcal{L}_{\text{bond}} = \sum_{i} \left| \| p_i - p_{i+1} \| - d_{\text{ideal}} \right|$$

Where $p_i$ are atomic positions and $d_{\text{ideal}}$ is the ideal bond length.

**Bond Angle Constraints**:
$$\mathcal{L}_{\text{angle}} = \sum_{i} \left| \angle(p_{i-1}, p_i, p_{i+1}) - \theta_{\text{ideal}} \right|$$

These constraints penalize deviations from ideal bond geometry.

### Attention Mechanisms

The model uses multi-head attention to capture:

- **Long-range interactions**: Between distant amino acids
- **Structural context**: How each residue affects neighbors
- **Folding patterns**: Secondary and tertiary structure

## Code Structure

The example demonstrates eight major sections:

1. **Setup and Initialization**: Import libraries and create RNG keys
2. **Model Configuration**: Define architecture and constraint parameters
3. **Model Creation**: Instantiate ProteinPointCloudModel
4. **Synthetic Data Generation**: Create alpha-helix structures for testing
5. **Dataset Loading**: Load protein data using ProteinDataset
6. **Forward Pass**: Generate protein structures
7. **Loss Calculation**: Compute reconstruction and constraint losses
8. **Summary**: Key takeaways and experiments

## Example Code

### Model Configuration

```python
from artifex.generative_models.core.configuration import (
    PointCloudNetworkConfig,
    ProteinConstraintConfig,
    ProteinPointCloudConfig,
)

# Create network config for point cloud processing
network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(128, 128, 128, 128),  # 4 layers for hierarchical processing
    activation="gelu",
    embed_dim=128,      # Embedding dimension for each atom
    num_heads=4,        # Number of attention heads
    num_layers=4,       # Number of transformer layers
    dropout_rate=0.1,   # Dropout rate for regularization
)

# Create constraint config for structural constraints
constraint_config = ProteinConstraintConfig(
    bond_weight=1.0,    # Weight for bond length constraints
    angle_weight=0.5,   # Weight for bond angle constraints
)

# Create protein point cloud config with nested configs
config = ProteinPointCloudConfig(
    name="protein_example",
    network=network_config,
    num_points=128 * 4,  # num_residues × num_atoms (flattened)
    dropout_rate=0.1,
    num_residues=128,    # Maximum number of residues
    num_atoms_per_residue=4,  # Only backbone atoms (N, CA, C, O)
    backbone_indices=(0, 1, 2, 3),  # Sequential indices for backbone-only view
    use_constraints=True,
    constraint_config=constraint_config,
)
```

### Creating the Model

```python
from artifex.generative_models.models.geometric.protein_point_cloud import (
    ProteinPointCloudModel,
)
from flax import nnx
import jax

# Initialize RNGs
key = jax.random.key(42)
key, params_key, dropout_key = jax.random.split(key, 3)
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

# Create model
model = ProteinPointCloudModel(config, rngs=rngs)
```

### Generating Synthetic Data

```python
import numpy as np

# Create alpha-helix geometry
seq_length = 64
atom_positions = np.zeros((seq_length, num_atoms, 3))

for i in range(seq_length):
    t = i * 0.5  # Helix parameter

    # CA (alpha carbon) along helix
    atom_positions[i, 1, 0] = 3.0 * np.sin(t)
    atom_positions[i, 1, 1] = 3.0 * np.cos(t)
    atom_positions[i, 1, 2] = 1.5 * t  # Rise along z-axis

    # N (nitrogen) relative to CA
    atom_positions[i, 0, :] = atom_positions[i, 1, :] + np.array([-1.45, 0, 0])

    # C (carbon) relative to CA
    atom_positions[i, 2, :] = atom_positions[i, 1, :] + np.array([1.52, 0, 0])

    # O (oxygen) relative to C
    atom_positions[i, 4, :] = atom_positions[i, 2, :] + np.array([0, 1.23, 0])
```

### Model Forward Pass

```python
from artifex.data.protein.dataset import ProteinDataset

# Load dataset
dataset = ProteinDataset("data/synthetic_proteins.pkl", backbone_only=True)
batch = dataset.get_batch([0, 1, 2, 3])

# Forward pass
outputs = model(batch)

print("Outputs:")
for key, value in outputs.items():
    if hasattr(value, "shape"):
        print(f"  {key}: {value.shape}")
# Output:
#   coordinates: (4, 512, 3)  # Predicted atom positions
#   constraints: {...}         # Constraint violations
```

### Loss Calculation

```python
# Get loss function
loss_fn = model.get_loss_fn()

# Calculate losses
loss_dict = loss_fn(batch, outputs)

print("Losses:")
for key, value in loss_dict.items():
    print(f"  {key}: {value:.4f}")
# Output:
#   total_loss: 2.5634
#   reconstruction_loss: 2.1234
#   bond_loss: 0.3200
#   angle_loss: 0.1200
```

## Features Demonstrated

- **Point Cloud Processing**: Representing proteins as unordered sets of 3D points
- **Attention Mechanisms**: Multi-head attention for capturing structural context
- **Geometric Constraints**: Enforcing physical validity through bond/angle constraints
- **Backbone Representation**: Working with minimal backbone atoms (N, CA, C, O)
- **Synthetic Data**: Generating alpha-helix structures for testing
- **Loss Computation**: Combining reconstruction and constraint losses

## Experiments to Try

1. **Modify Constraint Weights**: Adjust the balance between bond and angle constraints

   ```python
   config.parameters["constraint_config"] = {
       "bond_weight": 2.0,  # Stricter bond constraints
       "angle_weight": 1.0,  # Stricter angle constraints
   }
   ```

   **Expected Effect**: Stronger enforcement of ideal geometry, potentially slower convergence

2. **Change Architecture Size**: Increase model capacity for larger proteins

   ```python
   config.parameters.update({
       "embed_dim": 256,     # Increased from 128
       "num_layers": 8,      # Increased from 4
       "num_heads": 8,       # Increased from 4
   })
   ```

   **Expected Effect**: Better capacity for complex structures, more parameters to train

3. **Generate Beta-Sheets**: Modify synthetic data to create beta-sheet geometry

   ```python
   # Extended strand geometry instead of helix
   for i in range(seq_length):
       # CA positions along extended strand
       atom_positions[i, 1, 0] = i * 3.8  # Extended along x
       atom_positions[i, 1, 1] = 0.0
       atom_positions[i, 1, 2] = 0.0
   ```

   **Expected Effect**: Test model's ability to handle different secondary structures

4. **Increase Protein Length**: Scale to longer sequences

   ```python
   config.parameters["num_residues"] = 256  # Increased from 128
   max_seq_length = 256
   ```

   **Expected Effect**: Test scaling behavior and memory requirements

## Troubleshooting

### Common Issues

#### Shape mismatch error

**Symptom:**

```
ValueError: Expected shape (batch, 512, 3), got (batch, 256, 3)
```

**Cause:**
Mismatch between `num_residues × num_atoms` in config and actual data shape.

**Solution:**

```python
# Ensure consistency
num_residues = 128
num_atoms = 4
config.parameters["num_residues"] = num_residues
config.parameters["num_atoms"] = num_atoms
config.input_dim = num_residues * num_atoms
config.output_dim = num_residues * num_atoms
```

#### OOM (Out of Memory) error

**Symptom:**

```
jax.errors.OutOfMemoryError: RESOURCE_EXHAUSTED
```

**Cause:**
Model or sequence length too large for available GPU memory.

**Solutions:**

1. **Reduce sequence length**:

   ```python
   config.parameters["num_residues"] = 64  # Reduced from 128
   ```

2. **Reduce model size**:

   ```python
   config.parameters.update({
       "embed_dim": 64,      # Reduced from 128
       "num_layers": 2,      # Reduced from 4
   })
   ```

3. **Use CPU instead of GPU**:

   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
   ```

#### Constraint loss is NaN

**Symptom:**

```
bond_loss: nan
angle_loss: nan
```

**Cause:**
Invalid atom positions (NaN or Inf values) in data.

**Solution:**

```python
# Validate data before training
import numpy as np

def validate_batch(batch):
    for key, value in batch.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains NaN or Inf values")
    return batch

batch = validate_batch(batch)
```

## Summary

In this example, you learned:

- ✅ How to represent proteins as 3D point clouds for geometric modeling
- ✅ How to configure ProteinPointCloudModel with attention mechanisms
- ✅ How to generate synthetic alpha-helix structures for testing
- ✅ How to apply geometric constraints to ensure chemical validity
- ✅ How to evaluate models using reconstruction and constraint losses
- ✅ How to work with backbone-only protein representations

**Key Takeaways:**

1. **Point Clouds**: Effective representation for geometric protein modeling
2. **Attention**: Captures long-range interactions in protein structures
3. **Constraints**: Essential for generating physically plausible structures
4. **Backbone**: Minimal representation sufficient for overall structure

## Next Steps

<div class="grid cards" markdown>

- :material-factory: **Protein Model with Modality**

    ---

    Learn about the modality architecture for protein modeling

    [:octicons-arrow-right-24: protein-model-with-modality.md](protein-model-with-modality.md)

- :material-link-variant: **Protein Extensions**

    ---

    Deep dive into protein-specific constraint extensions

    [:octicons-arrow-right-24: protein-extensions-example.py](protein-extensions-example.md)

- :material-benchmark: **Geometric Benchmark**

    ---

    Evaluate geometric models on protein datasets

    [:octicons-arrow-right-24: geometric-benchmark-demo.md](../geometric/geometric-benchmark-demo.md)

- :material-flask: **Protein Ligand Benchmark**

    ---

    Advanced protein-ligand interaction modeling

    [:octicons-arrow-right-24: protein-ligand-benchmark-demo.md](protein-ligand-benchmark-demo.md)

</div>

## Additional Resources

- **Artifex Documentation**: [Protein Modeling Guide](../../guides/protein-modeling.md)
- **Artifex Documentation**: [Geometric Models](../../guides/geometric-models.md)
- **Artifex Documentation**: [Point Cloud Processing](../../guides/point-clouds.md)
- **API Reference**: [ProteinPointCloudModel](../../api/models/protein-point-cloud.md)
- **API Reference**: [ProteinDataset](../../api/data/protein-dataset.md)
- **Paper**: [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) - Protein structure prediction with message passing

## Related Examples

- [Protein Model with Modality](protein-model-with-modality.md) - Using the modality architecture
- [Protein Extensions Example](protein-extensions-example.md) - Protein-specific constraints
- [Geometric Benchmark Demo](../geometric/geometric-benchmark-demo.md) - Evaluating geometric models
