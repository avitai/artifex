# Protein Viz

`artifex.visualization.protein_viz` is the canonical public owner for protein
visualization in Artifex.

## Import

```python
from artifex.visualization.protein_viz import ProteinVisualizer
```

## Retained Methods

- `to_pdb_string(...)`: export a protein structure to a PDB string
- `coords_to_pdb(...)`: compatibility alias for the same PDB export path
- `export_to_pdb(...)`: write a PDB file from either a structure dict or raw
  coordinates
- `calculate_dihedral_angles(...)`: compute phi and psi backbone angles
- `plot_ramachandran(...)`: visualize backbone angles in 2D
- `visualize_structure(...)`: create a `py3Dmol` 3D view when the optional
  dependency is installed
- `visualize_protein_structure(...)`: create Matplotlib 2D backbone projections

## Shape Contract

Coordinate-bearing methods accept either an unbatched structure with shape
`[num_res, num_atoms, 3]` or a batched tensor with shape
`[batch, num_res, num_atoms, 3]`. Batched calls normalize to the first
structure before export or visualization.

## Example

```python
import numpy as np

from artifex.visualization.protein_viz import ProteinVisualizer

atom_positions = np.random.normal(size=(48, 4, 3))
atom_mask = np.ones((48, 4))

pdb_string = ProteinVisualizer.to_pdb_string(atom_positions, atom_mask)
ProteinVisualizer.export_to_pdb(
    {"atom_positions": atom_positions, "atom_mask": atom_mask},
    "protein_structure.pdb",
)

phi, psi = ProteinVisualizer.calculate_dihedral_angles(atom_positions, atom_mask)
fig = ProteinVisualizer.plot_ramachandran(phi, psi)
viewer = ProteinVisualizer.visualize_structure(atom_positions, atom_mask)
```
