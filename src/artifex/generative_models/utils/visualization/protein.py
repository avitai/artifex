"""Protein structure visualization utilities.

This module provides visualization utilities for protein structures, with support
for both interactive visualization in notebooks and exporting to standard formats.
"""

import tempfile
from typing import Any

import jax
import numpy as np


class ProteinVisualizer:
    """Visualization tools for protein structures.

    This class provides methods for visualizing protein structures using different
    backends (py3Dmol or nglview) and for exporting protein structures to standard
    formats like PDB.
    """

    def __init__(self, backend: str = "py3dmol"):
        """Initialize the visualizer with the specified backend.

        Args:
            backend: Visualization backend to use. Options: "py3dmol", "nglview".
        """
        self.backend = backend.lower()

        # Import the appropriate backend
        if self.backend == "py3dmol":
            try:
                import py3Dmol

                self._py3Dmol = py3Dmol
            except ImportError as err:
                raise ImportError(
                    "The py3Dmol package is required for protein visualization. "
                    "Install it with 'pip install py3dmol'."
                ) from err
        elif self.backend == "nglview":
            try:
                import nglview

                self._nglview = nglview
            except ImportError as err:
                raise ImportError(
                    "The nglview package is required for protein visualization. "
                    "Install it with 'pip install nglview'."
                ) from err
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Supported backends: 'py3dmol', 'nglview'."
            )

    def visualize(
        self,
        protein_data: dict[str, Any],
        style: str | None = "cartoon",
        color_scheme: str | None = "chainname",
        width: int = 500,
        height: int = 500,
        **kwargs,
    ) -> Any:
        """Visualize protein structure.

        Args:
            protein_data: Dictionary with protein data (coords, mask, aatype).
            style: Visualization style (cartoon, surface, ball_and_stick).
            color_scheme: Color scheme for visualization.
            width: Width of the visualization in pixels.
            height: Height of the visualization in pixels.
            **kwargs: Additional visualization parameters.

        Returns:
            Visualization object from the backend.
        """
        # Call the appropriate backend visualization method
        if self.backend == "py3dmol":
            return self._visualize_py3dmol(
                protein_data, style, color_scheme, width, height, **kwargs
            )
        elif self.backend == "nglview":
            return self._visualize_nglview(
                protein_data, style, color_scheme, width, height, **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def export_to_pdb(self, protein_data: dict[str, Any], output_path: str) -> None:
        """Export protein structure to PDB file.

        Args:
            protein_data: Dictionary with protein data (coords, mask, aatype).
            output_path: Path to output PDB file.
        """
        # Extract data
        coords = protein_data.get("coords", protein_data.get("atom_positions"))
        mask = protein_data.get("mask", protein_data.get("atom_mask"))
        aatype = protein_data.get("aatype", None)

        # Generate PDB string
        pdb_str = self.coords_to_pdb(coords, mask, aatype)

        # Write to file
        with open(output_path, "w") as f:
            f.write(pdb_str)

    def coords_to_pdb(
        self,
        coords: np.ndarray | jax.Array,
        mask: np.ndarray | jax.Array | None = None,
        aatype: np.ndarray | jax.Array | None = None,
        chain_ids: list[str] | None = None,
    ) -> str:
        """Convert coordinates to PDB format string.

        Args:
            coords: Protein coordinates with shape [batch, seq_len, num_atoms, 3].
            mask: Optional atom mask with shape [batch, seq_len, num_atoms].
            aatype: Optional amino acid types with shape [batch, seq_len].
            chain_ids: Optional list of chain IDs to use.

        Returns:
            PDB format string.
        """
        # Convert JAX arrays to numpy if needed
        if hasattr(coords, "device_buffer"):
            coords = np.array(coords)
        if mask is not None and hasattr(mask, "device_buffer"):
            mask = np.array(mask)
        if aatype is not None and hasattr(aatype, "device_buffer"):
            aatype = np.array(aatype)

        # Always use the first batch element if multiple batches are provided
        if coords.shape[0] > 1:
            coords = coords[0]
            if mask is not None:
                mask = mask[0]
            if aatype is not None:
                aatype = aatype[0]
        elif coords.shape[0] == 1:
            # Extract single batch dimension for consistency
            coords = coords[0]
            if mask is not None:
                mask = mask[0]
            if aatype is not None:
                aatype = aatype[0]

        # Create all-ones mask if none provided
        if mask is None:
            mask = np.ones(coords.shape[:-1])

        # Create default chain ID (A) if none provided
        if chain_ids is None:
            chain_ids = ["A"]

        # Define atom names (assuming backbone atoms: N, CA, C, O)
        atom_names = ["N", "CA", "C", "O"]
        if coords.shape[-2] > 4:
            # Add placeholder names for additional atoms
            atom_names.extend([f"X{i}" for i in range(5, coords.shape[-2] + 1)])

        # Amino acid mapping (1-letter to 3-letter codes)
        aa_map = {
            0: "GLY",
            1: "ALA",
            2: "VAL",
            3: "LEU",
            4: "ILE",
            5: "MET",
            6: "PHE",
            7: "TRP",
            8: "PRO",
            9: "SER",
            10: "THR",
            11: "CYS",
            12: "TYR",
            13: "ASN",
            14: "GLN",
            15: "ASP",
            16: "GLU",
            17: "LYS",
            18: "ARG",
            19: "HIS",
        }

        # Initialize PDB string
        pdb_lines = []
        pdb_lines.append("MODEL     1")

        # Current atom index
        atom_index = 1

        # Get sequence length and number of atoms
        seq_len, num_atoms = coords.shape[-3], coords.shape[-2]

        # Generate PDB lines
        for i in range(seq_len):
            # Determine residue name
            if aatype is None:
                res_name = "GLY"  # Default to glycine if no aatype provided
            else:
                # Handle array types safely by getting a scalar value
                if isinstance(aatype[i], (np.ndarray, list)) and len(aatype[i]) == 1:
                    aa_idx = int(aatype[i][0])
                else:
                    aa_idx = int(aatype[i])
                res_name = aa_map.get(aa_idx, "GLY")

            # Determine chain ID
            chain_id = chain_ids[0]  # Use first chain ID for now

            # Residue index (1-indexed in PDB)
            res_idx = i + 1

            # Add atom lines for this residue
            for j in range(num_atoms):
                # Skip if atom is masked out - convert comparison to scalar
                if float(mask[i, j]) < 0.5:
                    continue

                # Get atom coordinates
                x, y, z = coords[i, j]

                # Get atom name
                atom_name = atom_names[j] if j < len(atom_names) else f"X{j + 1}"

                # Format atom line according to PDB standard
                atom_line = (
                    f"ATOM  {atom_index:5d} {atom_name:<4s} {res_name:3s} "
                    f"{chain_id:1s}{res_idx:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           "
                    f"{atom_name[0]:1s}  "
                )
                pdb_lines.append(atom_line)
                atom_index += 1

        # Add end-of-model and end-of-file markers
        pdb_lines.append("ENDMDL")
        pdb_lines.append("END")

        # Join lines and return
        return "\n".join(pdb_lines)

    def _visualize_py3dmol(
        self,
        protein_data: dict[str, Any],
        style: str | None = "cartoon",
        color_scheme: str | None = "chainname",
        width: int = 500,
        height: int = 500,
        surface: bool = False,
        **kwargs,
    ) -> Any:
        """Visualize protein structure using py3Dmol backend.

        Args:
            protein_data: Dictionary with protein data.
            style: Visualization style.
            color_scheme: Color scheme for visualization.
            width: Width of the visualization in pixels.
            height: Height of the visualization in pixels.
            surface: Whether to add surface representation.
            **kwargs: Additional visualization parameters.

        Returns:
            py3Dmol viewer object.
        """
        # Extract data
        coords = protein_data.get("coords", protein_data.get("atom_positions"))
        mask = protein_data.get("mask", protein_data.get("atom_mask"))
        aatype = protein_data.get("aatype", None)

        # Convert to PDB string
        pdb_str = self.coords_to_pdb(coords, mask, aatype)

        # Create viewer
        viewer = self._py3Dmol.view(width=width, height=height)

        # Add model from PDB string
        viewer.addModel(pdb_str, "pdb")

        # Apply style
        style_dict = {}
        if style == "cartoon":
            style_dict["cartoon"] = {"color": color_scheme}
        elif style == "surface":
            style_dict["cartoon"] = {"color": color_scheme}
        elif style == "ball_and_stick":
            style_dict["stick"] = {}
            style_dict["sphere"] = {"scale": 0.3}
        else:
            style_dict["cartoon"] = {"color": color_scheme}

        # Set style
        viewer.setStyle(style_dict)

        # Add surface if requested
        if surface or style == "surface":
            viewer.addSurface(
                self._py3Dmol.VDW,
                {"opacity": kwargs.get("opacity", 0.7), "color": "white"},
            )

        # Set background color and zoom
        viewer.setBackgroundColor(kwargs.get("background_color", "white"))
        viewer.zoomTo()

        return viewer

    def _visualize_nglview(
        self,
        protein_data: dict[str, Any],
        style: str | None = "cartoon",
        color_scheme: str | None = "chainname",
        width: int = 500,
        height: int = 500,
        **kwargs,
    ) -> Any:
        """Visualize protein structure using nglview backend.

        Args:
            protein_data: Dictionary with protein data.
            style: Visualization style.
            color_scheme: Color scheme for visualization.
            width: Width of the visualization in pixels.
            height: Height of the visualization in pixels.
            **kwargs: Additional visualization parameters.

        Returns:
            nglview viewer object.
        """
        # Extract data
        coords = protein_data.get("coords", protein_data.get("atom_positions"))
        mask = protein_data.get("mask", protein_data.get("atom_mask"))
        aatype = protein_data.get("aatype", None)

        # Convert to PDB string
        pdb_str = self.coords_to_pdb(coords, mask, aatype)

        # Need to declare view outside the tempfile context
        view = None

        # Create temporary file to save PDB (nglview requires a file)
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
            # Write PDB string to file
            tmp.write(pdb_str)
            tmp.flush()

            # Create viewer
            view = self._nglview.show_file(tmp.name)

        # Now all operations on view are outside the tempfile context
        # Set size
        view._remote_call("setSize", target="Widget", args=[width, height])

        # Clear default representations
        view.clear_representations()

        # Add appropriate representation
        if style == "cartoon":
            view.add_representation("cartoon", color_scheme=color_scheme)
        elif style == "surface":
            view.add_representation("cartoon", color_scheme=color_scheme)
            view.add_representation("surface", opacity=kwargs.get("opacity", 0.7))
        elif style == "ball_and_stick":
            view.add_representation("ball+stick")
        else:
            view.add_representation("cartoon", color_scheme=color_scheme)

        return view
