"""Protein-specific data handling utilities.

This package contains data loading, processing, and transformation utilities
for protein structure data.
"""

from artifex.data.protein.dataset import (
    ATOM_TYPES,
    BACKBONE_ATOM_INDICES,
    pdb_to_protein_example,
    ProteinDataset,
)


__all__ = [
    "ProteinDataset",
    "pdb_to_protein_example",
    "ATOM_TYPES",
    "BACKBONE_ATOM_INDICES",
]
