"""Compatibility re-export for the retained protein benchmark adapter.

The canonical implementation lives in
`artifex.benchmarks.model_adapters.protein_adapters`.
"""

from artifex.benchmarks.model_adapters.protein_adapters import ProteinPointCloudAdapter


__all__ = ["ProteinPointCloudAdapter"]
