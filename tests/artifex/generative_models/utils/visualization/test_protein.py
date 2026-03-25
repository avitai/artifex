"""Compatibility tests for protein visualization imports."""

import numpy as np

from artifex.generative_models.utils.visualization.protein import (
    ProteinVisualizer as CompatProteinVisualizer,
)
from artifex.visualization.protein_viz import ProteinVisualizer


def _batched_protein_payload() -> dict[str, np.ndarray]:
    """Create a small batched protein payload for compatibility checks."""
    positions = np.zeros((1, 3, 4, 3), dtype=np.float32)
    mask = np.ones((1, 3, 4), dtype=np.float32)
    aatype = np.array([[0, 1, 2]], dtype=np.int32)
    return {
        "coords": positions,
        "mask": mask,
        "aatype": aatype,
    }


def test_compatibility_alias_points_to_canonical_owner() -> None:
    """The old generative-models path should resolve to the canonical class."""
    assert CompatProteinVisualizer is ProteinVisualizer


def test_compatibility_alias_preserves_pdb_helpers(tmp_path) -> None:
    """The compatibility alias should retain the normalized PDB export helpers."""
    protein_data = _batched_protein_payload()

    pdb_string = CompatProteinVisualizer.coords_to_pdb(
        protein_data["coords"],
        protein_data["mask"],
        protein_data["aatype"],
    )

    assert pdb_string.startswith("MODEL     1")
    assert "ATOM  " in pdb_string

    output_path = tmp_path / "compat_protein.pdb"
    CompatProteinVisualizer.export_to_pdb(protein_data, str(output_path))

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").endswith("END")
