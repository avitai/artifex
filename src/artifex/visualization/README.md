# Visualization Utilities

This directory contains visualization utilities for generative models and their outputs.

## Overview

The visualization utilities provide tools for rendering and analyzing generated samples from various model types, with special focus on protein structures and geometric models.

## Components

### Protein Visualization (`protein_viz.py`)

The `ProteinVisualizer` class provides comprehensive visualization tools for protein structures:

- **3D Visualization**: Interactive 3D visualization of protein structures using py3Dmol
- **Ramachandran Plots**: 2D visualization of dihedral angles in protein backbones
- **Structure Analysis**: Tools for analyzing protein structure geometry
- **PDB Export**: Utilities for converting to PDB format for interoperability
- **2D Projections**: Multi-view 2D projections of protein structures

### Recent Improvements

- **Scatter Plot Color Fix**: Added proper color array parameters to scatter plots to avoid matplotlib warnings
- **Layout Management**: Replaced `tight_layout()` with explicit `subplots_adjust()` for more reliable figure layouts
- **Color Mapping**: Improved color mapping for better visualization of residue indices
- **Figure Handling**: Enhanced figure creation and management for better memory usage
- **Error Handling**: Improved backend availability detection and error messaging
- **NaN Value Handling**: Added explicit handling of NaN values in Ramachandran plots
- **Figure Saving**: Added `save_path` parameter to visualization methods for direct file output
- **Graceful Degradation**: Improved error handling when optional dependencies like py3Dmol are not available

## Usage Example

### Protein Visualization

```python
import jax.numpy as jnp
import numpy as np
from artifex.visualization.protein_viz import ProteinVisualizer

# Create sample protein data
num_residues = 50
num_atoms_per_residue = 4  # N, CA, C, O
atom_positions = np.random.normal(size=(num_residues, num_atoms_per_residue, 3))
atom_mask = np.ones((num_residues, num_atoms_per_residue))

# Calculate dihedral angles
phi, psi = ProteinVisualizer.calculate_dihedral_angles(atom_positions, atom_mask)

# Create Ramachandran plot
fig = ProteinVisualizer.plot_ramachandran(
    phi_angles=phi,
    psi_angles=psi,
    residue_indices=np.arange(num_residues),
    title="Sample Protein Ramachandran Plot",
    highlight_outliers=True,
    show_regions=True,
    figsize=(10, 8),
    save_path="ramachandran_plot.png"  # Optional: save to file
)

# To display the figure in a non-notebook environment
# plt.show()

# Create 2D structure visualization (showing projections of CA atoms)
fig = ProteinVisualizer.visualize_protein_structure(
    atom_positions=atom_positions,
    figsize=(12, 10),
    save_path="protein_structure_2d.png"  # Optional: save to file
)

# To display the figure in a non-notebook environment
# plt.show()

# Create PDB string (for export or 3D visualization)
pdb_string = ProteinVisualizer.to_pdb_string(
    atom_positions=atom_positions,
    atom_mask=atom_mask,
    chain_id="A"
)

# 3D visualization (in Jupyter notebook)
try:
    view = ProteinVisualizer.visualize_structure(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        show_sidechains=False,
        show_mainchains=True,
        show_surface=True,
        color_by="chain",
        surface_opacity=0.5,
        width=800,
        height=600
    )
    # In notebook: display the interactive viewer
    if view is not None:
        # view.show()  # Uncomment in Jupyter notebook
        print("3D visualization created successfully")
    else:
        print("py3Dmol not available - 3D visualization skipped")
except Exception as e:
    print(f"Error creating 3D visualization: {e}")
```

## Integration

The visualization utilities integrate seamlessly with model outputs from the `models` module, particularly the geometric and protein models.
