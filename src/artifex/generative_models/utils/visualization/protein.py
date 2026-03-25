"""Compatibility alias for the canonical protein visualization helpers.

`artifex.generative_models.utils.visualization.protein` is retained only as a
thin import-compatibility shim. The canonical public owner lives at
`artifex.visualization.protein_viz`.
"""

from artifex.visualization.protein_viz import ProteinVisualizer


__all__ = ["ProteinVisualizer"]
