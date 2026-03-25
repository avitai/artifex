# Visualization

The top-level `artifex.visualization` namespace currently ships one retained
public helper: `ProteinVisualizer` from `artifex.visualization.protein_viz`.

## Canonical Import

```python
from artifex.visualization.protein_viz import ProteinVisualizer
```

## Retained Surface

- `ProteinVisualizer.to_pdb_string(...)` and `coords_to_pdb(...)` export one
  structure to PDB text
- `ProteinVisualizer.export_to_pdb(...)` writes a PDB file
- `ProteinVisualizer.calculate_dihedral_angles(...)` and
  `ProteinVisualizer.plot_ramachandran(...)` support backbone inspection
- `ProteinVisualizer.visualize_structure(...)` provides optional 3D rendering
  through `py3Dmol`
- `ProteinVisualizer.visualize_protein_structure(...)` produces 2D backbone
  projections with Matplotlib

## Shape Contract

Methods that accept atom coordinates support either:

- `[num_res, num_atoms, 3]`
- `[batch, num_res, num_atoms, 3]`

When a batch dimension is present, the first structure is exported or
visualized.

## Example

```python
import numpy as np

from artifex.visualization.protein_viz import ProteinVisualizer

atom_positions = np.random.normal(size=(32, 4, 3))
atom_mask = np.ones((32, 4))
aatype = np.zeros(32, dtype=np.int32)

pdb_string = ProteinVisualizer.to_pdb_string(atom_positions, atom_mask, aatype)
ProteinVisualizer.export_to_pdb(
    {"atom_positions": atom_positions, "atom_mask": atom_mask, "aatype": aatype},
    "protein_structure.pdb",
)

phi, psi = ProteinVisualizer.calculate_dihedral_angles(atom_positions, atom_mask)
fig = ProteinVisualizer.plot_ramachandran(phi, psi)
viewer = ProteinVisualizer.visualize_structure(atom_positions, atom_mask)
```

## Scope Notes

- `artifex.visualization` is not a generic sample-grid, latent-space, or
  training-plot subsystem.
- Benchmark plotting remains under `artifex.benchmarks.visualization`.
- Package-local plotting helpers should be documented with their owning package
  instead of through a generic visualization facade.

## Related Pages

- [Protein Viz](protein_viz.md)
- [Protein Diffusion Example](../examples/protein/protein-diffusion-example.md)
