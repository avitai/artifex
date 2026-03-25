# Protein Visualization

This package currently ships one protein-specific helper module:
`artifex.visualization.protein_viz`.

## Retained Surface

`ProteinVisualizer` is the canonical public owner for protein visualization in
Artifex. It provides:

- PDB export through `to_pdb_string(...)`, `coords_to_pdb(...)`, and
  `export_to_pdb(...)`
- backbone analysis through `calculate_dihedral_angles(...)` and
  `plot_ramachandran(...)`
- structure views through `visualize_structure(...)` and
  `visualize_protein_structure(...)`

Coordinate methods accept either one structure with shape
`[num_res, num_atoms, 3]` or a batched tensor with shape
`[batch, num_res, num_atoms, 3]`. When a batch dimension is present, the first
structure is exported or visualized.

## Scope

- This directory is not a generic visualization subsystem for every model
  family.
- General plotting helpers should live with their owning package.
- Benchmark-specific plots remain under `artifex.benchmarks.visualization`.

## Usage Example

```python
import numpy as np

from artifex.visualization.protein_viz import ProteinVisualizer

atom_positions = np.random.normal(size=(32, 4, 3))
atom_mask = np.ones((32, 4))

pdb_string = ProteinVisualizer.to_pdb_string(atom_positions, atom_mask)
ProteinVisualizer.export_to_pdb(
    {"atom_positions": atom_positions, "atom_mask": atom_mask},
    "protein_structure.pdb",
)

phi, psi = ProteinVisualizer.calculate_dihedral_angles(atom_positions, atom_mask)
fig = ProteinVisualizer.plot_ramachandran(phi, psi)
viewer = ProteinVisualizer.visualize_structure(atom_positions, atom_mask)
```
