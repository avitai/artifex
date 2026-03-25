# Protein Visualization Compatibility

**Status:** `Supported runtime utility`
**Module:** `artifex.visualization.protein_viz`
**Source:** `src/artifex/visualization/protein_viz.py`

The canonical protein visualization owner is
`artifex.visualization.protein_viz`.

`artifex.generative_models.utils.visualization.protein.ProteinVisualizer`
remains only as a thin compatibility alias to that same class so older imports
do not break immediately.

## Preferred Import

```python
from artifex.visualization.protein_viz import ProteinVisualizer
```

## Compatibility Notes

- the compatibility alias resolves to the same `ProteinVisualizer` class
- no separate generative-models visualization implementation is maintained
- new docs and examples should not treat `generative_models.utils.visualization`
  as an independent protein visualization owner
