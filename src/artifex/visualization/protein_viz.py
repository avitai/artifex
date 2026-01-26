"""Protein visualization utilities.

This module provides visualization utilities for protein structures, including
3D visualization with py3Dmol and 2D plots for Ramachandran analysis.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from artifex.utils.file_utils import ensure_valid_output_path


class ProteinVisualizer:
    """Visualization utilities for protein structures."""

    @staticmethod
    def to_pdb_string(
        atom_positions: np.ndarray,
        atom_mask: np.ndarray | None = None,
        aatype: np.ndarray | None = None,
        atom_types: list[str] | None = None,
        b_factors: np.ndarray | None = None,
        chain_id: str = "A",
    ) -> str:
        """Convert atom positions to PDB string.

        Args:
            atom_positions: Atom positions with shape [num_res, num_atoms, 3]
            atom_mask: Optional mask with shape [num_res, num_atoms]
            aatype: Optional amino acid types with shape [num_res]
            atom_types: Optional atom types, default is N, CA, C, O
            b_factors: Optional B-factors with shape [num_res, num_atoms]
            chain_id: Chain ID for the PDB file

        Returns:
            PDB format string
        """
        if atom_types is None:
            atom_types = ["N", "CA", "C", "O"]

        # Default mask (all atoms are present)
        if atom_mask is None:
            atom_mask = np.ones(atom_positions.shape[:-1])

        # Default amino acid types (all alanine)
        if aatype is None:
            aatype = np.zeros(atom_positions.shape[0], dtype=np.int32)

        # Default B-factors (all zeros)
        if b_factors is None:
            b_factors = np.zeros(atom_positions.shape[:-1])

        # Standard amino acid 3-letter codes
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

        # Map numeric amino acid types to 3-letter codes
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

        # Check inputs
        if (
            len(atom_positions.shape) != 3
            or atom_positions.shape[2] != 3
            or atom_positions.shape[1] != len(atom_types)
        ):
            msg = f"atom_positions should have shape [num_res, {len(atom_types)}, 3]"
            raise ValueError(msg)

        pdb_lines = []
        pdb_lines.append("MODEL     1")
        atom_index = 1

        # Iterate over residues
        for res_index in range(atom_positions.shape[0]):
            # Get amino acid type
            res_type_idx = aatype[res_index]
            if 0 <= res_type_idx < len(restypes):
                res_type_1letter = restypes[res_type_idx]
                res_type_3letter = restype_1to3.get(res_type_1letter, "UNK")
            else:
                res_type_3letter = "UNK"

            # Iterate over atoms in residue
            for atom_index_in_res, atom_type in enumerate(atom_types):
                if atom_mask[res_index, atom_index_in_res] < 0.5:
                    # Skip atoms that are masked out
                    continue

                # Get atom position
                pos = atom_positions[res_index, atom_index_in_res]
                x, y, z = pos

                # Get B-factor
                b_factor = b_factors[res_index, atom_index_in_res]

                # Element is first character of atom type
                element = atom_type[0]

                # Format PDB line according to spec
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
        atom_positions: np.ndarray | jax.Array,
        atom_mask: np.ndarray | jax.Array | None = None,
        aatype: np.ndarray | jax.Array | None = None,
        atom_types: list[str] | None = None,
        width: int = 800,
        height: int = 600,
        show_sidechains: bool = False,
        show_mainchains: bool = True,
        show_surface: bool = False,
        color_by: str = "chain",
        surface_opacity: float = 0.5,
    ):
        """Visualize protein structure in 3D.

        Args:
            atom_positions: Atom positions with shape [num_res, num_atoms, 3]
            atom_mask: Optional mask with shape [num_res, num_atoms]
            aatype: Optional amino acid types with shape [num_res]
            atom_types: Optional atom types, default is N, CA, C, O
            width: Viewer width
            height: Viewer height
            show_sidechains: Whether to show sidechains
            show_mainchains: Whether to show mainchains
            show_surface: Whether to show surface
            color_by: How to color the structure ("chain", "residue", etc.)
            surface_opacity: Opacity of the surface

        Returns:
            py3Dmol viewer object or None if py3Dmol is not installed
        """
        try:
            import py3Dmol
        except ImportError:
            import warnings

            warnings.warn(
                "py3Dmol not installed, 3D visualization not available. "
                "Install with `pip install py3Dmol`"
            )
            return None

        # Convert JAX arrays to numpy if needed
        if isinstance(atom_positions, jax.Array):
            atom_positions = np.array(atom_positions)
        if atom_mask is not None and isinstance(atom_mask, jax.Array):
            atom_mask = np.array(atom_mask)
        if aatype is not None and isinstance(aatype, jax.Array):
            aatype = np.array(aatype)

        # Generate PDB string
        pdb_string = ProteinVisualizer.to_pdb_string(atom_positions, atom_mask, aatype, atom_types)

        # Create viewer
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(pdb_string, "pdb")

        # Apply visualization options
        if color_by == "chain":
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
        elif color_by == "residue":
            viewer.setStyle({"cartoon": {"colorscheme": "amino"}})
        elif color_by == "atom":
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
        elif color_by == "b_factor":
            viewer.setStyle({"cartoon": {"color": "spectrum", "colorscheme": "rwb"}})

        # Show/hide elements
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

        # Center and zoom
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
    ) -> plt.Figure:
        """Create a Ramachandran plot for the given protein structure.

        Args:
            phi_angles: Phi angles in radians
            psi_angles: Psi angles in radians
            residue_indices: Optional residue indices
            title: Plot title
            highlight_outliers: Whether to highlight outliers
            show_regions: Whether to show allowed regions
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        # Convert to numpy if needed
        if isinstance(phi_angles, jax.Array):
            phi_angles = np.array(phi_angles)
        if isinstance(psi_angles, jax.Array):
            psi_angles = np.array(psi_angles)
        if residue_indices is not None and isinstance(residue_indices, jax.Array):
            residue_indices = np.array(residue_indices)

        # Filter out NaN values
        valid_mask = ~np.isnan(phi_angles) & ~np.isnan(psi_angles)
        if not np.any(valid_mask):
            msg = "No valid angles found for Ramachandran plot"
            raise ValueError(msg)

        phi_valid = phi_angles[valid_mask]
        psi_valid = psi_angles[valid_mask]

        if residue_indices is not None:
            residue_indices_valid = residue_indices[valid_mask]
        else:
            residue_indices_valid = np.ones_like(phi_valid)

        # Allowed regions for different secondary structures
        # Values are approximations based on common Ramachandran plots
        alpha_helix_region = [(-90, -30), (-60, 0)]
        beta_sheet_region = [(-180, -45), (90, 180)]
        left_helix_region = [(30, 90), (0, 60)]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot points
        scatter = ax.scatter(
            phi_valid * 180 / np.pi,  # Convert to degrees
            psi_valid * 180 / np.pi,  # Convert to degrees
            c=residue_indices_valid,
            cmap="viridis",
            alpha=0.7,
            s=50,
        )

        # Add colorbar if using residue indices
        if residue_indices is not None:
            plt.colorbar(scatter, label="Residue Index")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Set axis limits and labels
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("Phi (φ) Angle (degrees)")
        ax.set_ylabel("Psi (ψ) Angle (degrees)")
        ax.set_title(title)

        # Add lines at 0 degrees
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)

        # Show allowed regions
        if show_regions:
            # Alpha helix region
            ax.add_patch(
                plt.Rectangle(
                    (alpha_helix_region[0][0], alpha_helix_region[1][0]),
                    alpha_helix_region[0][1] - alpha_helix_region[0][0],
                    alpha_helix_region[1][1] - alpha_helix_region[1][0],
                    fill=True,
                    color="red",
                    alpha=0.1,
                    label="Alpha Helix",
                )
            )

            # Beta sheet region
            ax.add_patch(
                plt.Rectangle(
                    (beta_sheet_region[0][0], beta_sheet_region[1][0]),
                    beta_sheet_region[0][1] - beta_sheet_region[0][0],
                    beta_sheet_region[1][1] - beta_sheet_region[1][0],
                    fill=True,
                    color="blue",
                    alpha=0.1,
                    label="Beta Sheet",
                )
            )

            # Left-handed helix region
            ax.add_patch(
                plt.Rectangle(
                    (left_helix_region[0][0], left_helix_region[1][0]),
                    left_helix_region[0][1] - left_helix_region[0][0],
                    left_helix_region[1][1] - left_helix_region[1][0],
                    fill=True,
                    color="green",
                    alpha=0.1,
                    label="Left-handed Helix",
                )
            )

            # Add legend
            ax.legend(loc="lower right")

        # Highlight outliers
        if highlight_outliers:
            # Define a mask for points not in any of the allowed regions
            def in_region(phi, psi, region):
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

        # Adjust layout
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            # Ensure save_path is within appropriate directory
            valid_save_path = ensure_valid_output_path(save_path)
            plt.savefig(valid_save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def calculate_dihedral_angles(
        atom_positions: np.ndarray | jax.Array,
        atom_mask: np.ndarray | jax.Array | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate phi and psi angles from atom positions.

        Args:
            atom_positions: Atom positions with shape [num_res, num_atoms, 3]
            atom_mask: Optional mask with shape [num_res, num_atoms]

        Returns:
            Tuple of phi and psi angles in radians
        """
        # Convert to numpy if needed
        if isinstance(atom_positions, jax.Array):
            atom_positions = np.array(atom_positions)
        if atom_mask is not None and isinstance(atom_mask, jax.Array):
            atom_mask = np.array(atom_mask)

        # Atom order is assumed to be N, CA, C, O
        n_pos = atom_positions[:, 0]  # N atoms
        ca_pos = atom_positions[:, 1]  # CA atoms
        c_pos = atom_positions[:, 2]  # C atoms

        # Phi: C(i-1), N(i), CA(i), C(i)
        # Psi: N(i), CA(i), C(i), N(i+1)

        num_residues = atom_positions.shape[0]
        phi_angles = np.zeros(num_residues)
        psi_angles = np.zeros(num_residues)

        # First residue phi and last residue psi should be NaN
        phi_angles[0] = np.nan
        psi_angles[-1] = np.nan

        # Function to calculate dihedral angle between 4 points
        def calculate_dihedral(p1, p2, p3, p4):
            # Calculate vectors between points
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3

            # Normalize b2
            b2_norm = b2 / np.linalg.norm(b2)

            # Calculate the normal vectors to the planes
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)

            # Calculate the angle
            angle = np.arctan2(np.dot(np.cross(n1, n2), b2_norm), np.dot(n1, n2))
            return angle

        # Calculate phi angles (starting from second residue)
        for i in range(1, num_residues):
            phi_angles[i] = calculate_dihedral(c_pos[i - 1], n_pos[i], ca_pos[i], c_pos[i])

        # Calculate psi angles (ending at penultimate residue)
        for i in range(num_residues - 1):
            psi_angles[i] = calculate_dihedral(n_pos[i], ca_pos[i], c_pos[i], n_pos[i + 1])

        # Apply mask if provided
        if atom_mask is not None:
            # Ensure atom_mask matches the size of atom_positions
            mask_num_residues = atom_mask.shape[0]
            if mask_num_residues != num_residues:
                # Use only the smaller size to avoid index errors
                effective_size = min(num_residues, mask_num_residues)
                atom_positions = atom_positions[:effective_size]
                atom_mask = atom_mask[:effective_size]
                num_residues = effective_size
                # Adjust phi and psi arrays
                phi_angles = phi_angles[:num_residues]
                psi_angles = psi_angles[:num_residues]
                if num_residues > 0:
                    psi_angles[-1] = np.nan  # Last residue doesn't have a psi angle

            # For phi angles, we need atoms from two consecutive residues
            valid_phi = np.ones(num_residues, dtype=bool)
            valid_phi[0] = False  # First residue doesn't have a phi angle

            for i in range(1, num_residues):
                # Check if all necessary atoms exist
                if (
                    atom_mask[i - 1, 2] < 0.5  # C(i-1)
                    or atom_mask[i, 0] < 0.5  # N(i)
                    or atom_mask[i, 1] < 0.5  # CA(i)
                    or atom_mask[i, 2] < 0.5  # C(i)
                ):
                    valid_phi[i] = False

            # For psi angles, we also need atoms from two consecutive residues
            valid_psi = np.ones(num_residues, dtype=bool)
            valid_psi[-1] = False  # Last residue doesn't have a psi angle

            for i in range(num_residues - 1):
                # Check if all necessary atoms exist
                if (
                    atom_mask[i, 0] < 0.5  # N(i)
                    or atom_mask[i, 1] < 0.5  # CA(i)
                    or atom_mask[i, 2] < 0.5  # C(i)
                    or atom_mask[i + 1, 0] < 0.5  # N(i+1)
                ):
                    valid_psi[i] = False

            # Mask invalid angles with NaN
            phi_angles = np.where(valid_phi, phi_angles, np.nan)
            psi_angles = np.where(valid_psi, psi_angles, np.nan)

        return phi_angles, psi_angles

    @staticmethod
    def visualize_protein_structure(
        atom_positions: np.ndarray | jax.Array,
        figsize: tuple[int, int] = (12, 10),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create a 2D visualization of a protein backbone structure.

        Args:
            atom_positions: Atom positions with shape [num_res, num_atoms, 3]
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure with 3 subplots (projections)
        """
        # Convert to numpy if needed
        if isinstance(atom_positions, jax.Array):
            atom_positions = np.array(atom_positions)

        # Extract CA atoms (usually at index 1)
        ca_positions = atom_positions[:, 1]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot X-Y projection
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

        # Plot X-Z projection
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

        # Plot Y-Z projection
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

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, len(ca_positions) - 1))
        cbar = fig.colorbar(sm, ax=axes)
        cbar.set_label("Residue Index")

        # Instead of tight_layout, manually adjust figure
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2)

        # Save the figure if a path is provided
        if save_path:
            # Ensure save_path is within appropriate directory
            valid_save_path = ensure_valid_output_path(save_path)
            plt.savefig(valid_save_path, dpi=300, bbox_inches="tight")

        return fig
