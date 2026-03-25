"""Compatibility exports for protein visualization helpers.

The canonical public owner now lives at :mod:`artifex.visualization.protein_viz`.
This package only re-exports the protein visualizer for older import paths.
"""

from artifex.generative_models.utils.visualization.protein import (
    ProteinVisualizer,
)


__all__ = [
    "ProteinVisualizer",
]
