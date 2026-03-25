"""Protein-specific visualization helpers.

This top-level module is the canonical public owner for protein visualization in
Artifex. It accepts either one structure with shape ``[num_res, num_atoms, 3]``
or a batched payload with shape ``[batch, num_res, num_atoms, 3]`` and
normalizes batched inputs to the first structure for export or visualization.
"""

from typing import Any, cast, TypeAlias

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from artifex.utils.file_utils import ensure_valid_output_path


ArrayLike: TypeAlias = np.ndarray | jax.Array


class ProteinVisualizer:
    """Protein-specific visualization utilities."""

    @staticmethod
    def _to_numpy(value: ArrayLike | None) -> np.ndarray | None:
        """Convert optional JAX arrays to NumPy arrays."""
        if value is None:
            return None
        if isinstance(value, jax.Array):
            return np.array(value)
        return value

    @staticmethod
    def _normalize_optional_batch_dimension(
        value: ArrayLike | None,
        *,
        name: str,
        expected_ndim: int,
    ) -> np.ndarray | None:
        """Normalize optional arrays to one structure by taking the first batch item."""
        array = ProteinVisualizer._to_numpy(value)
        if array is None:
            return None
        if array.ndim == expected_ndim:
            return array
        if array.ndim == expected_ndim + 1:
            if array.shape[0] == 0:
                raise ValueError(f"{name} must contain at least one batch element")
            return array[0]
        raise ValueError(f"{name} must have {expected_ndim} or {expected_ndim + 1} dims")

    @staticmethod
    def _normalize_structure_inputs(
        atom_positions: ArrayLike,
        atom_mask: ArrayLike | None = None,
        aatype: ArrayLike | None = None,
        b_factors: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Normalize batched or unbatched protein tensors to one structure."""
        positions = ProteinVisualizer._normalize_optional_batch_dimension(
            atom_positions,
            name="atom_positions",
            expected_ndim=3,
        )
        if positions is None or positions.shape[-1] != 3:
            raise ValueError("atom_positions must have shape [num_res, num_atoms, 3]")

        mask = ProteinVisualizer._normalize_optional_batch_dimension(
            atom_mask,
            name="atom_mask",
            expected_ndim=2,
        )
        residue_types = ProteinVisualizer._normalize_optional_batch_dimension(
            aatype,
            name="aatype",
            expected_ndim=1,
        )
        factors = ProteinVisualizer._normalize_optional_batch_dimension(
            b_factors,
            name="b_factors",
            expected_ndim=2,
        )

        return positions, mask, residue_types, factors

    @staticmethod
    def coords_to_pdb(
        coords: ArrayLike,
        mask: ArrayLike | None = None,
        aatype: ArrayLike | None = None,
        chain_ids: list[str] | None = None,
    ) -> str:
        """Compatibility alias for :meth:`to_pdb_string`."""
        chain_id = chain_ids[0] if chain_ids else "A"
        return ProteinVisualizer.to_pdb_string(
            atom_positions=coords,
            atom_mask=mask,
            aatype=aatype,
            chain_id=chain_id,
        )

    @staticmethod
    def export_to_pdb(
        protein_data: dict[str, Any] | ArrayLike,
        output_path: str,
        atom_mask: ArrayLike | None = None,
        aatype: ArrayLike | None = None,
        chain_id: str = "A",
    ) -> None:
        """Export protein coordinates to a PDB file."""
        mask = atom_mask
        residue_types = aatype

        if isinstance(protein_data, dict):
            positions = protein_data.get("coords", protein_data.get("atom_positions"))
            mask = protein_data.get("mask", protein_data.get("atom_mask", atom_mask))
            residue_types = protein_data.get("aatype", aatype)

            if positions is None:
                raise ValueError("protein_data must provide `coords` or `atom_positions`")
            atom_positions = cast(ArrayLike, positions)
        else:
            atom_positions = protein_data

        pdb_string = ProteinVisualizer.to_pdb_string(
            atom_positions=atom_positions,
            atom_mask=mask,
            aatype=residue_types,
            chain_id=chain_id,
        )
        valid_output_path = ensure_valid_output_path(output_path)
        with open(valid_output_path, "w", encoding="utf-8") as f:
            f.write(pdb_string)

    @staticmethod
    def to_pdb_string(
        atom_positions: ArrayLike,
        atom_mask: ArrayLike | None = None,
        aatype: ArrayLike | None = None,
        atom_types: list[str] | None = None,
        b_factors: ArrayLike | None = None,
        chain_id: str = "A",
    ) -> str:
        """Convert atom positions to a PDB string.

        Args:
            atom_positions: Atom positions with shape ``[num_res, num_atoms, 3]``
                or ``[batch, num_res, num_atoms, 3]``.
            atom_mask: Optional mask with shape ``[num_res, num_atoms]`` or
                ``[batch, num_res, num_atoms]``.
            aatype: Optional amino acid types with shape ``[num_res]`` or
                ``[batch, num_res]``.
            atom_types: Optional atom types, default is ``N``, ``CA``, ``C``, ``O``.
            b_factors: Optional B-factors with shape ``[num_res, num_atoms]`` or
                ``[batch, num_res, num_atoms]``.
            chain_id: Chain ID for the PDB file.

        Returns:
            PDB format string.
        """
        positions, mask, residue_types, factors = ProteinVisualizer._normalize_structure_inputs(
            atom_positions,
            atom_mask,
            aatype,
            b_factors,
        )

        if atom_types is None:
            atom_types = ["N", "CA", "C", "O"]

        if mask is None:
            mask = np.ones(positions.shape[:-1])
        if residue_types is None:
            residue_types = np.zeros(positions.shape[0], dtype=np.int32)
        if factors is None:
            factors = np.zeros(positions.shape[:-1])

        restype_3to1 = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }
        restype_1to3 = {v: k for k, v in restype_3to1.items()}
        restypes = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]

        if (
            len(positions.shape) != 3
            or positions.shape[2] != 3
            or positions.shape[1] != len(atom_types)
        ):
            msg = f"atom_positions should have shape [num_res, {len(atom_types)}, 3]"
            raise ValueError(msg)

        pdb_lines = ["MODEL     1"]
        atom_index = 1

        for res_index in range(positions.shape[0]):
            res_type_idx = int(residue_types[res_index])
            if 0 <= res_type_idx < len(restypes):
                res_type_1letter = restypes[res_type_idx]
                res_type_3letter = restype_1to3.get(res_type_1letter, "UNK")
            else:
                res_type_3letter = "UNK"

            for atom_index_in_res, atom_type in enumerate(atom_types):
                if mask[res_index, atom_index_in_res] < 0.5:
                    continue

                x, y, z = positions[res_index, atom_index_in_res]
                b_factor = factors[res_index, atom_index_in_res]
                element = atom_type[0]
                pdb_line = (
                    f"ATOM  {atom_index:5d} {atom_type:^4s} "
                    f"{res_type_3letter:3s} {chain_id:1s}{res_index + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  "
                    f"1.00{b_factor:6.2f}          {element:>2s}  "
                )
                pdb_lines.append(pdb_line)
                atom_index += 1

        pdb_lines.append("ENDMDL")
        pdb_lines.append("END")
        return "\n".join(pdb_lines)

    @staticmethod
    def visualize_structure(
        atom_positions: ArrayLike,
        atom_mask: ArrayLike | None = None,
        aatype: ArrayLike | None = None,
        atom_types: list[str] | None = None,
        width: int = 800,
        height: int = 600,
        show_sidechains: bool = False,
        show_mainchains: bool = True,
        show_surface: bool = False,
        color_by: str = "chain",
        surface_opacity: float = 0.5,
    ):
        """Visualize one protein structure in 3D."""
        try:
            import py3Dmol
        except ImportError:
            import warnings

            warnings.warn(
                "py3Dmol not installed, 3D visualization not available. "
                "Install with `pip install py3Dmol`"
            )
            return None

        atom_positions, atom_mask, aatype, _ = ProteinVisualizer._normalize_structure_inputs(
            atom_positions,
            atom_mask,
            aatype,
        )
        pdb_string = ProteinVisualizer.to_pdb_string(atom_positions, atom_mask, aatype, atom_types)

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(pdb_string, "pdb")

        if color_by == "chain":
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
        elif color_by == "residue":
            viewer.setStyle({"cartoon": {"colorscheme": "amino"}})
        elif color_by == "atom":
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
        elif color_by == "b_factor":
            viewer.setStyle({"cartoon": {"color": "spectrum", "colorscheme": "rwb"}})

        if not show_mainchains:
            viewer.setStyle({"cartoon": {"display": False}})

        if show_sidechains:
            viewer.addStyle(
                {"hetflag": False},
                {"stick": {"colorscheme": "amino", "radius": 0.15}},
            )

        if show_surface:
            viewer.addSurface(
                py3Dmol.VDW,
                {"opacity": surface_opacity, "colorscheme": "WhiteCarbon"},
            )

        viewer.zoomTo()
        return viewer

    @staticmethod
    def plot_ramachandran(
        phi_angles: np.ndarray | jax.Array,
        psi_angles: np.ndarray | jax.Array,
        residue_indices: np.ndarray | jax.Array | None = None,
        title: str = "Ramachandran Plot",
        highlight_outliers: bool = True,
        show_regions: bool = True,
        figsize: tuple[int, int] = (10, 8),
        save_path: str | None = None,
    ) -> Figure:
        """Create a Ramachandran plot for the given protein structure."""
        if isinstance(phi_angles, jax.Array):
            phi_angles = np.array(phi_angles)
        if isinstance(psi_angles, jax.Array):
            psi_angles = np.array(psi_angles)
        if residue_indices is not None and isinstance(residue_indices, jax.Array):
            residue_indices = np.array(residue_indices)

        valid_mask = ~np.isnan(phi_angles) & ~np.isnan(psi_angles)
        if not np.any(valid_mask):
            raise ValueError("No valid angles found for Ramachandran plot")

        phi_valid = phi_angles[valid_mask]
        psi_valid = psi_angles[valid_mask]
        residue_indices_valid = (
            residue_indices[valid_mask] if residue_indices is not None else np.ones_like(phi_valid)
        )

        alpha_helix_region = [(-90, -30), (-60, 0)]
        beta_sheet_region = [(-180, -45), (90, 180)]
        left_helix_region = [(30, 90), (0, 60)]

        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            phi_valid * 180 / np.pi,
            psi_valid * 180 / np.pi,
            c=residue_indices_valid,
            cmap="viridis",
            alpha=0.7,
            s=50,
        )

        if residue_indices is not None:
            plt.colorbar(scatter, label="Residue Index")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("Phi (φ) Angle (degrees)")
        ax.set_ylabel("Psi (ψ) Angle (degrees)")
        ax.set_title(title)
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)

        if show_regions:
            ax.add_patch(
                Rectangle(
                    (alpha_helix_region[0][0], alpha_helix_region[1][0]),
                    alpha_helix_region[0][1] - alpha_helix_region[0][0],
                    alpha_helix_region[1][1] - alpha_helix_region[1][0],
                    fill=True,
                    color="red",
                    alpha=0.1,
                    label="Alpha Helix",
                )
            )
            ax.add_patch(
                Rectangle(
                    (beta_sheet_region[0][0], beta_sheet_region[1][0]),
                    beta_sheet_region[0][1] - beta_sheet_region[0][0],
                    beta_sheet_region[1][1] - beta_sheet_region[1][0],
                    fill=True,
                    color="blue",
                    alpha=0.1,
                    label="Beta Sheet",
                )
            )
            ax.add_patch(
                Rectangle(
                    (left_helix_region[0][0], left_helix_region[1][0]),
                    left_helix_region[0][1] - left_helix_region[0][0],
                    left_helix_region[1][1] - left_helix_region[1][0],
                    fill=True,
                    color="green",
                    alpha=0.1,
                    label="Left-handed Helix",
                )
            )
            ax.legend(loc="lower right")

        if highlight_outliers:

            def in_region(phi: float, psi: float, region: list[tuple[int, int]]) -> bool:
                x_in = region[0][0] <= phi * 180 / np.pi <= region[0][1]
                y_in = region[1][0] <= psi * 180 / np.pi <= region[1][1]
                return x_in and y_in

            outliers = []
            for i, (phi, psi) in enumerate(zip(phi_valid, psi_valid)):
                if not (
                    in_region(phi, psi, alpha_helix_region)
                    or in_region(phi, psi, beta_sheet_region)
                    or in_region(phi, psi, left_helix_region)
                ):
                    outliers.append(i)

            if outliers:
                ax.scatter(
                    phi_valid[outliers] * 180 / np.pi,
                    psi_valid[outliers] * 180 / np.pi,
                    color="red",
                    s=100,
                    facecolors="none",
                    linewidth=2,
                    label="Outliers",
                )

        plt.tight_layout()

        if save_path:
            valid_save_path = ensure_valid_output_path(save_path)
            plt.savefig(valid_save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def calculate_dihedral_angles(
        atom_positions: ArrayLike,
        atom_mask: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate phi and psi angles from atom positions."""
        atom_positions, atom_mask, _, _ = ProteinVisualizer._normalize_structure_inputs(
            atom_positions,
            atom_mask,
        )

        n_pos = atom_positions[:, 0]
        ca_pos = atom_positions[:, 1]
        c_pos = atom_positions[:, 2]

        num_residues = atom_positions.shape[0]
        phi_angles = np.zeros(num_residues)
        psi_angles = np.zeros(num_residues)
        phi_angles[0] = np.nan
        psi_angles[-1] = np.nan

        def calculate_dihedral(
            p1: np.ndarray,
            p2: np.ndarray,
            p3: np.ndarray,
            p4: np.ndarray,
        ) -> float:
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3
            b2_norm = b2 / np.linalg.norm(b2)
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            return np.arctan2(np.dot(np.cross(n1, n2), b2_norm), np.dot(n1, n2))

        for i in range(1, num_residues):
            phi_angles[i] = calculate_dihedral(c_pos[i - 1], n_pos[i], ca_pos[i], c_pos[i])

        for i in range(num_residues - 1):
            psi_angles[i] = calculate_dihedral(n_pos[i], ca_pos[i], c_pos[i], n_pos[i + 1])

        if atom_mask is not None:
            valid_phi = np.ones(num_residues, dtype=bool)
            valid_phi[0] = False
            for i in range(1, num_residues):
                if (
                    atom_mask[i - 1, 2] < 0.5
                    or atom_mask[i, 0] < 0.5
                    or atom_mask[i, 1] < 0.5
                    or atom_mask[i, 2] < 0.5
                ):
                    valid_phi[i] = False

            valid_psi = np.ones(num_residues, dtype=bool)
            valid_psi[-1] = False
            for i in range(num_residues - 1):
                if (
                    atom_mask[i, 0] < 0.5
                    or atom_mask[i, 1] < 0.5
                    or atom_mask[i, 2] < 0.5
                    or atom_mask[i + 1, 0] < 0.5
                ):
                    valid_psi[i] = False

            phi_angles = np.where(valid_phi, phi_angles, np.nan)
            psi_angles = np.where(valid_psi, psi_angles, np.nan)

        return phi_angles, psi_angles

    @staticmethod
    def visualize_protein_structure(
        atom_positions: ArrayLike,
        figsize: tuple[int, int] = (12, 10),
        save_path: str | None = None,
    ) -> Figure:
        """Create a 2D visualization of a protein backbone structure."""
        atom_positions, _, _, _ = ProteinVisualizer._normalize_structure_inputs(atom_positions)
        ca_positions = atom_positions[:, 1]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].plot(ca_positions[:, 0], ca_positions[:, 1], "b-", linewidth=2)
        axes[0].scatter(
            ca_positions[:, 0],
            ca_positions[:, 1],
            c=range(len(ca_positions)),
            cmap="viridis",
        )
        axes[0].set_title("X-Y Projection")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        axes[1].plot(ca_positions[:, 0], ca_positions[:, 2], "g-", linewidth=2)
        axes[1].scatter(
            ca_positions[:, 0],
            ca_positions[:, 2],
            c=range(len(ca_positions)),
            cmap="viridis",
        )
        axes[1].set_title("X-Z Projection")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")

        axes[2].plot(ca_positions[:, 1], ca_positions[:, 2], "r-", linewidth=2)
        axes[2].scatter(
            ca_positions[:, 1],
            ca_positions[:, 2],
            c=range(len(ca_positions)),
            cmap="viridis",
        )
        axes[2].set_title("Y-Z Projection")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(0, len(ca_positions) - 1))
        cbar = fig.colorbar(sm, ax=axes)
        cbar.set_label("Residue Index")
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2)

        if save_path:
            valid_save_path = ensure_valid_output_path(save_path)
            plt.savefig(valid_save_path, dpi=300, bbox_inches="tight")

        return fig
