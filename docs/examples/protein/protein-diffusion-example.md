# Protein Diffusion Example

**Level:** Advanced
**Runtime:** ~5 minutes
**Format:** Dual (.py script | .ipynb notebook)

Comprehensive protein diffusion modeling with two approaches: high-level API with extensions and direct model creation, including quality assessment and visualization.

## Files

- **Python Script:** [`examples/generative_models/protein/protein_diffusion_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_example.py)
- **Jupyter Notebook:** [`examples/generative_models/protein/protein_diffusion_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_example.ipynb)

## Quick Start

```bash
# Run the Python script
python examples/generative_models/protein/protein_diffusion_example.py

# Or open the Jupyter notebook
jupyter notebook examples/generative_models/protein/protein_diffusion_example.ipynb
```

## Overview

This comprehensive example demonstrates how to build and use protein diffusion models for generating 3D protein structures. You'll learn two distinct approaches to protein modeling, understand protein-specific geometric constraints, and explore quality assessment techniques.

### Learning Objectives

After completing this example, you will understand:

- [x] How to create protein diffusion models with Artifex's high-level API
- [x] Direct model creation and manipulation for protein structures
- [x] Protein-specific loss functions and geometric constraints
- [x] Quality assessment metrics for generated proteins
- [x] Visualization techniques for 3D protein structures

### Prerequisites

- Understanding of diffusion models and denoising processes
- Familiarity with protein structure representations
- Knowledge of geometric constraints in biomolecules
- Experience with JAX and Flax NNX

## Theory and Key Concepts

### Protein Structure Representation

Proteins are complex biomolecules composed of amino acid residues. Each residue contains multiple atoms with specific 3D coordinates:

**Backbone Atoms**: The main chain of every protein contains four atoms per residue:

- **N** (Nitrogen): Backbone nitrogen
- **CA** (Alpha Carbon): Central carbon atom
- **C** (Carbonyl Carbon): Carbonyl carbon
- **O** (Oxygen): Carbonyl oxygen

**Representation Approaches**:

1. **Point Cloud**: Unordered set of 3D points representing atom positions
   - Advantages: Simple, flexible, good for local geometry
   - Use case: Backbone modeling, local structure refinement

2. **Graph**: Nodes (residues/atoms) connected by edges (bonds)
   - Advantages: Captures connectivity, enforces topology
   - Use case: Full protein modeling, contact prediction

### Geometric Constraints

Valid protein structures must satisfy strict geometric constraints:

**Bond Lengths**: Distance between bonded atoms must fall within specific ranges:

- C-C bonds: ~1.5 Å
- C-N bonds: ~1.3 Å
- C=O bonds: ~1.2 Å

**Bond Angles**: Angles between consecutive bonds follow specific distributions:

- Tetrahedral angles: ~109.5°
- Planar peptide bonds: ~120°

**Dihedral Angles**: Rotation around bonds defines protein conformation:

- **Phi (φ)**: Rotation around N-CA bond
- **Psi (ψ)**: Rotation around CA-C bond
- Ramachandran plot: Shows allowed (φ, ψ) combinations

### Protein-Specific Loss Functions

**RMSD (Root Mean Square Deviation)**: Measures structural similarity between predicted and target structures:

$$
\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^N ||x_i - y_i||^2}
$$

where $N$ is the number of atoms, $x_i$ is the predicted position, and $y_i$ is the target position.

**Backbone Loss**: Enforces correct backbone geometry:

$$
\mathcal{L}_{\text{backbone}} = \sum_{i=1}^{N-1} ||d(x_i, x_{i+1}) - d_{\text{ideal}}||^2
$$

where $d(x_i, x_{i+1})$ is the distance between consecutive residues and $d_{\text{ideal}}$ is the ideal distance.

**Composite Loss**: Combines multiple geometric constraints:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{rmsd}} \mathcal{L}_{\text{rmsd}} + \lambda_{\text{backbone}} \mathcal{L}_{\text{backbone}} + \lambda_{\text{angle}} \mathcal{L}_{\text{angle}}
$$

## Code Walkthrough

### Part 1: High-Level API with Extensions

The example demonstrates using Artifex's extension system for protein modeling:

```python
# Create model with extensions
extension_config = {
    "name": "protein_diffusion_extensions",
    "description": "Extensions for protein diffusion model",
    "enabled": True,
    "use_backbone_constraints": True,
    "use_protein_mixin": True,
}

extensions = create_protein_extensions(extension_config, rngs=rngs)
model = nnx.Module()
model.extensions = extensions
```

The extension system provides:

- **Backbone Constraints**: Automatic enforcement of backbone geometry
- **Protein Mixin**: Domain-specific operations for proteins
- **Quality Assessment**: Built-in metrics for structure validation

### Part 2: Direct Model Creation

For full control, create models directly:

```python
from artifex.generative_models.core.configuration import (
    PointCloudNetworkConfig,
    ProteinConstraintConfig,
    ProteinPointCloudConfig,
)

# Create network config for point cloud processing
network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(128, 128, 128, 128),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    dropout_rate=0.1,
)

# Create constraint config for structural constraints
constraint_config = ProteinConstraintConfig(
    bond_weight=1.0,
    angle_weight=0.5,
)

# Create protein point cloud config with nested configs
config = ProteinPointCloudConfig(
    name="protein_point_cloud_model",
    network=network_config,
    num_points=64 * 4,  # num_residues × num_atoms (flattened)
    dropout_rate=0.1,
    num_residues=64,
    num_atoms_per_residue=4,
    backbone_indices=(0, 1, 2, 3),  # N, CA, C, O
    use_constraints=True,
    constraint_config=constraint_config,
)

model = ProteinPointCloudModel(config, rngs=rngs)
```

### Dataset Preparation

Load synthetic or real protein datasets:

```python
# Create synthetic dataset for demonstration
dataset = create_synthetic_protein_dataset(
    num_proteins=50,
    min_seq_length=32,
    max_seq_length=64,
    random_seed=42,
)

# Prepare batch
batch = prepare_batch(dataset, batch_size=8, random_seed=42)

# Add noise for diffusion training
noisy_batch = add_noise_to_batch(batch, noise_level=0.1, random_seed=42)
```

### Loss Function Configuration

Combine multiple protein-specific losses:

```python
from artifex.generative_models.modalities.protein.losses import (
    CompositeLoss,
    create_backbone_loss,
    create_rmsd_loss,
)

loss_fn = CompositeLoss({
    "rmsd": (create_rmsd_loss(), 1.0),      # Weight: 1.0
    "backbone": (create_backbone_loss(), 0.5),  # Weight: 0.5
})

# Calculate losses
outputs = model(noisy_batch)
losses = loss_fn(batch, outputs)
```

### Visualization and Quality Assessment

Visualize generated structures and assess quality:

```python
from artifex.visualization.protein_viz import ProteinVisualizer

# Extract positions
target_pos = batch["atom_positions"][0]
pred_pos = outputs["positions"][0]
mask = batch["atom_mask"][0]

# Calculate dihedral angles
target_phi, target_psi = ProteinVisualizer.calculate_dihedral_angles(target_pos, mask)
pred_phi, pred_psi = ProteinVisualizer.calculate_dihedral_angles(pred_pos, mask)

# Plot Ramachandran plots
ProteinVisualizer.plot_ramachandran(target_phi, target_psi, title="Target")
ProteinVisualizer.plot_ramachandran(pred_phi, pred_psi, title="Predicted")

# 3D visualization (requires py3Dmol)
viewer = ProteinVisualizer.visualize_structure(
    pred_pos,
    mask,
    show_sidechains=False,
    color_by="chain"
)
viewer.show()
```

## Expected Output

The example runs both approaches and displays results:

```
=== Protein Diffusion Examples ===
This example demonstrates two approaches to protein diffusion:
1. High-level API with extension components
2. Direct model creation and manipulation

=== Running Extensions Example ===

Model structure:
- Type: Module
- Extensions: ['bond_length', 'bond_angle', 'protein_mixin']
Generated 2 protein samples
- Sample shape: (2, 64, 4, 3)
- Atom mask shape: (2, 64, 4)

Quality metrics:
- rmsd: 1.2345
- bond_violations: 0.0234
- angle_violations: 0.0156

=== Running Direct Model Example ===

Creating model...
Loading dataset...
Preparing batch...
Adding noise to batch...
Creating loss function...
Running model...
Calculating losses...
Losses:
  rmsd: 0.1234
  backbone: 0.0567
  total: 0.1801

Displaying results...
```

The example also generates:

- 2D plots of protein structures
- Ramachandran plots showing dihedral angle distributions
- 3D interactive visualizations (if py3Dmol is installed)

## Experiments to Try

1. **Compare Model Types**: Test point cloud vs graph representations

   ```python
   point_cloud_model = create_protein_diffusion_model(model_type="point_cloud")
   graph_model = create_protein_diffusion_model(model_type="graph")
   ```

2. **Adjust Constraint Weights**: Balance different geometric constraints

   ```python
   constraint_config = {
       "backbone_weight": 2.0,  # Emphasize backbone connectivity
       "bond_weight": 1.5,      # Strong bond length enforcement
       "angle_weight": 1.0,     # Moderate angle constraints
       "dihedral_weight": 0.5,  # Soft dihedral constraints
   }
   ```

3. **Larger Proteins**: Scale to longer sequences

   ```python
   model = create_protein_diffusion_model(
       num_residues=128,  # Double the default size
       hidden_dim=256,    # Increase capacity
   )
   ```

4. **Custom Loss Functions**: Create domain-specific losses

   ```python
   def create_contact_loss():
       """Enforce protein contact map constraints."""
       def loss_fn(batch, outputs):
           # Calculate contact map loss
           return contact_loss
       return loss_fn

   loss_fn = CompositeLoss({
       "rmsd": (create_rmsd_loss(), 1.0),
       "backbone": (create_backbone_loss(), 0.5),
       "contact": (create_contact_loss(), 0.3),
   })
   ```

5. **Real Datasets**: Load actual protein structures

   ```python
   dataset = ProteinDataset(
       data_dir="path/to/pdb/files",
       max_seq_length=128,
       random_seed=42,
   )
   ```

## Troubleshooting

### Size Mismatch Warnings

If you see "Target size doesn't match prediction size":

- Check that `num_residues` matches between model and data
- Ensure batch collation handles variable-length sequences
- Use masking to handle different protein lengths

### Geometric Constraint Violations

If structures have high constraint violations:

- Increase constraint weights in `constraint_config`
- Add more training epochs for constraint satisfaction
- Use smaller noise levels during training

### Visualization Issues

If 3D visualization fails:

- Install py3Dmol: `pip install py3Dmol`
- For Jupyter notebooks, ensure proper widget support
- Fall back to 2D plots if 3D is unavailable

### Memory Issues

For large proteins:

- Reduce batch size
- Use gradient checkpointing
- Process proteins in chunks

## Next Steps

<div class="grid cards" markdown>

- :material-protein: **Advanced Protein Models**

    ---

    Explore AlphaFold-style architectures and multi-scale modeling

    [:octicons-arrow-right-24: Protein Extensions](protein-extensions-example.md)

- :material-chart-scatter-plot: **Point Cloud Models**

    ---

    Learn specialized techniques for point cloud protein representations

    [:octicons-arrow-right-24: Point Cloud Example](protein-point-cloud-example.md)

- :material-chart-timeline-variant: **Diffusion Training**

    ---

    Master advanced diffusion techniques for proteins

    [:octicons-arrow-right-24: Diffusion Guide](../diffusion/advanced-diffusion.md)

- :material-test-tube: **Protein Benchmarks**

    ---

    Evaluate protein models with standard benchmarks

    [:octicons-arrow-right-24: Benchmarking](../../benchmarks/protein_benchmarks.md)

</div>

## Additional Resources

- [Protein Data Bank (PDB)](https://www.rcsb.org/) - Repository of 3D protein structures
- [AlphaFold Documentation](https://alphafold.ebi.ac.uk/) - State-of-the-art protein structure prediction
- [Diffusion Models for Proteins](https://arxiv.org/abs/2205.15019) - Research paper on protein diffusion
- [Artifex Protein Modeling Guide](../../guides/protein-modeling.md) - Comprehensive guide
- [Ramachandran Plot](https://en.wikipedia.org/wiki/Ramachandran_plot) - Understanding dihedral angles
