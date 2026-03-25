"""Protein-specific data handling utilities.

This package contains data loading, processing, and transformation utilities
for protein structure data, backed by datarax DataSourceModule.
"""

from artifex.data.protein.dataset import (
    AA_TYPE_TO_IDX,
    AA_TYPES,
    ATOM_TYPES,
    BACKBONE_ATOM_INDICES,
    create_synthetic_protein_dataset,
    MAX_SEQ_LENGTH,
    pdb_to_protein_example,
    protein_collate_fn,
    ProteinDataset,
    ProteinDatasetConfig,
    ProteinStructure,
)


__all__ = [
    "AA_TYPE_TO_IDX",
    "AA_TYPES",
    "ATOM_TYPES",
    "BACKBONE_ATOM_INDICES",
    "MAX_SEQ_LENGTH",
    "ProteinDataset",
    "ProteinDatasetConfig",
    "ProteinStructure",
    "create_synthetic_protein_dataset",
    "pdb_to_protein_example",
    "protein_collate_fn",
]
