"""Protein dataset implementation for diffusion models.

This module provides dataset classes for loading, processing, and batching
protein structure data for diffusion models.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class ProteinStructure:
    """Protein structure data representation.

    Attributes:
        atom_positions: Atom positions with shape [num_res, num_atoms, 3]
        atom_mask: Binary mask for atom positions [num_res, num_atoms]
        aatype: Amino acid types [num_res]
        residue_index: Residue indices [num_res]
        chain_index: Chain indices [num_res]
        b_factors: B-factors for each atom [num_res, num_atoms]
        resolution: Structure resolution
        seq_length: Sequence length (number of residues)
        sequence: Amino acid sequence as string
        pdb_id: PDB identifier
    """

    atom_positions: jnp.ndarray
    atom_mask: jnp.ndarray
    aatype: jnp.ndarray | None = None
    residue_index: jnp.ndarray | None = None
    chain_index: jnp.ndarray | None = None
    b_factors: jnp.ndarray | None = None
    resolution: float | None = None
    seq_length: int | None = None
    sequence: str | None = None
    pdb_id: str | None = None

    @classmethod
    def from_numpy(cls, data: dict[str, np.ndarray]) -> "ProteinStructure":
        """Create a ProteinStructure from numpy arrays.

        Args:
            data: Dictionary with numpy arrays

        Returns:
            ProteinStructure instance
        """
        # Convert numpy arrays to jax arrays
        jax_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                jax_data[key] = jnp.array(value)
            else:
                jax_data[key] = value

        return cls(**jax_data)


class ProteinDataset:
    """Dataset class for protein structures.

    This class handles loading and preprocessing protein structure data
    for training and evaluation.
    """

    def __init__(
        self,
        data_dir_or_proteins=None,
        data_dir=None,
        max_seq_length: int = 128,
        backbone_atom_indices: list[int] | None = None,
        center_positions: bool = True,
        normalize_positions: bool = True,
        random_seed: int = 42,
        split: str = "train",
    ):
        """Initialize the protein dataset.

        Args:
            data_dir_or_proteins: Directory containing protein structure files
                                  or a list of protein data dictionaries
            data_dir: Alternative parameter name for directory containing protein files
                     (for backward compatibility)
            max_seq_length: Maximum sequence length to use
            backbone_atom_indices: Indices of backbone atoms to use
            center_positions: Whether to center atom positions
            normalize_positions: Whether to normalize atom positions
            random_seed: Random seed for data shuffling
            split: Dataset split ("train", "valid", or "test")
        """
        # Handle both parameter naming options
        if data_dir is not None:
            self.data_dir = data_dir
            self.proteins = None
        elif isinstance(data_dir_or_proteins, str):
            self.data_dir = data_dir_or_proteins
            self.proteins = None
        elif data_dir_or_proteins is not None:
            self.data_dir = ""
            self.proteins = data_dir_or_proteins
        else:
            self.data_dir = ""
            self.proteins = None

        self.max_seq_length = max_seq_length
        # Default to backbone atoms (N, CA, C, O)
        self.backbone_atom_indices = backbone_atom_indices or [0, 1, 2, 4]
        self.center_positions = center_positions
        self.normalize_positions = normalize_positions
        self.random_seed = random_seed
        self.split = split

        # Standard atom types in proteins
        self.atom_types = [
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
            "OXT",
        ]

        # Standard amino acid types
        self.aa_types = [
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

        # Create aa_type to index mapping
        self.aa_type_to_idx = {aa: i for i, aa in enumerate(self.aa_types)}

        # Load data
        self.structures = self._load_data()

    def _load_data(self) -> list[ProteinStructure]:
        """Load protein structure data from files or existing protein data.

        Returns:
            List of ProteinStructure instances
        """
        if self.proteins is not None:
            # Convert provided protein dictionaries to ProteinStructure objects
            return [ProteinStructure.from_numpy(p) for p in self.proteins]
        else:
            # This is a placeholder implementation
            # In a real implementation, this would load PDB files
            # For now, we'll return an empty list
            return []

    def _process_structure(self, structure: ProteinStructure) -> ProteinStructure:
        """Process a protein structure for use with diffusion models.

        Args:
            structure: Input protein structure

        Returns:
            Processed protein structure
        """
        # Get structure data
        atom_positions = structure.atom_positions
        atom_mask = structure.atom_mask

        # Extract backbone atoms if needed
        if self.backbone_atom_indices is not None:
            atom_positions = atom_positions[:, self.backbone_atom_indices]
            atom_mask = atom_mask[:, self.backbone_atom_indices]

        # Truncate sequence if needed
        seq_length = atom_positions.shape[0]
        if self.max_seq_length is not None and seq_length > self.max_seq_length:
            atom_positions = atom_positions[: self.max_seq_length]
            atom_mask = atom_mask[: self.max_seq_length]
            if structure.aatype is not None:
                aatype = structure.aatype[: self.max_seq_length]
            else:
                aatype = None
        else:
            aatype = structure.aatype

        # Center positions
        if self.center_positions:
            # Only use masked atoms for centering
            masked_positions = atom_positions * atom_mask[:, :, None]
            mask_sum = atom_mask.sum()
            if mask_sum > 0:
                # Compute center of mass for masked atoms
                center = masked_positions.sum(axis=(0, 1)) / mask_sum
                # Subtract center from all positions
                atom_positions = atom_positions - center

        # Normalize positions
        if self.normalize_positions:
            # Only use masked atoms for normalization
            masked_positions = atom_positions * atom_mask[:, :, None]
            if atom_mask.sum() > 0:
                # Compute std dev for masked atoms
                std = jnp.sqrt(
                    jnp.sum(jnp.sum(masked_positions**2, axis=-1) * atom_mask) / atom_mask.sum()
                )
                # Scale positions by std dev
                atom_positions = atom_positions / (std + 1e-6)

        # Create residue indices if not provided
        seq_length = atom_positions.shape[0]
        if structure.residue_index is None:
            residue_index = jnp.arange(seq_length)
        else:
            residue_index = structure.residue_index

        # Update structure
        return ProteinStructure(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=aatype,
            residue_index=residue_index,
            chain_index=structure.chain_index,
            b_factors=structure.b_factors,
            resolution=structure.resolution,
            seq_length=seq_length,
            sequence=structure.sequence,
            pdb_id=structure.pdb_id,
        )

    def __len__(self) -> int:
        """Get the number of structures in the dataset.

        Returns:
            Number of structures
        """
        return len(self.structures)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a protein structure by index.

        Args:
            idx: Structure index

        Returns:
            Dictionary with structure data
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} items")

        # Get structure
        structure = self.structures[idx]

        # Process structure
        processed = self._process_structure(structure)

        # Convert to dictionary
        return {
            "atom_positions": processed.atom_positions,
            "atom_mask": processed.atom_mask,
            "aatype": processed.aatype,
            "residue_index": processed.residue_index,
            "seq_length": processed.seq_length,
            "sequence": processed.sequence,
            "pdb_id": processed.pdb_id,
        }

    def collate_batch(self, examples: list[dict[str, Any]]) -> dict[str, jnp.ndarray]:
        """Collate a batch of examples.

        This function pads sequences to the same length and creates a batch.

        Args:
            examples: List of examples

        Returns:
            Batched data
        """
        batch_size = len(examples)

        # Find max sequence length in batch
        max_len = max(ex["seq_length"] for ex in examples)
        max_len = min(max_len, self.max_seq_length)

        # Get number of atom types
        num_atom_types = examples[0]["atom_positions"].shape[1]

        # Initialize batch arrays
        batch_positions = np.zeros((batch_size, max_len, num_atom_types, 3), dtype=np.float32)
        batch_mask = np.zeros((batch_size, max_len, num_atom_types), dtype=np.float32)
        batch_aatype = np.zeros((batch_size, max_len), dtype=np.int32)
        batch_residue_index = np.zeros((batch_size, max_len), dtype=np.int32)

        # Fill batch arrays
        for i, ex in enumerate(examples):
            seq_len = min(ex["seq_length"], max_len)

            # Copy positions and mask
            batch_positions[i, :seq_len] = ex["atom_positions"][:seq_len]
            batch_mask[i, :seq_len] = ex["atom_mask"][:seq_len]

            # Copy aatype if available
            if ex["aatype"] is not None:
                batch_aatype[i, :seq_len] = ex["aatype"][:seq_len]

            # Copy residue index if available
            if ex["residue_index"] is not None:
                batch_residue_index[i, :seq_len] = ex["residue_index"][:seq_len]

        # Convert to JAX arrays
        return {
            "atom_positions": jnp.array(batch_positions),
            "atom_mask": jnp.array(batch_mask),
            "aatype": jnp.array(batch_aatype),
            "residue_index": jnp.array(batch_residue_index),
        }

    def get_batch(
        self, indices: list[int], max_length: int | None = None
    ) -> dict[str, jnp.ndarray]:
        """Get a batch of protein structures by indices.

        Args:
            indices: List of structure indices
            max_length: Optional maximum sequence length for padding (default: max in batch)

        Returns:
            Dictionary with batched structure data
        """
        # Get items by indices
        examples = [self[idx] for idx in indices]

        # If max_length is provided, enforce it
        if max_length is not None:
            # Make sure all examples have seq_length attribute
            for example in examples:
                if "seq_length" not in example:
                    example["seq_length"] = example["atom_positions"].shape[0]

            # Create a new batch with custom max_length
            batch_size = len(examples)
            num_atom_types = examples[0]["atom_positions"].shape[1]

            # Initialize batch arrays
            batch_positions = np.zeros(
                (batch_size, max_length, num_atom_types, 3), dtype=np.float32
            )
            batch_mask = np.zeros((batch_size, max_length, num_atom_types), dtype=np.float32)
            batch_aatype = np.zeros((batch_size, max_length), dtype=np.int32)
            batch_residue_index = np.zeros((batch_size, max_length), dtype=np.int32)

            # Fill batch arrays
            for i, ex in enumerate(examples):
                seq_len = min(ex["seq_length"], max_length)

                # Copy positions and mask
                batch_positions[i, :seq_len] = ex["atom_positions"][:seq_len]
                batch_mask[i, :seq_len] = ex["atom_mask"][:seq_len]

                # Copy aatype if available
                if ex["aatype"] is not None:
                    batch_aatype[i, :seq_len] = ex["aatype"][:seq_len]

                # Copy residue index if available
                if ex["residue_index"] is not None:
                    batch_residue_index[i, :seq_len] = ex["residue_index"][:seq_len]

            # Convert to JAX arrays
            return {
                "atom_positions": jnp.array(batch_positions),
                "atom_mask": jnp.array(batch_mask),
                "aatype": jnp.array(batch_aatype),
                "residue_index": jnp.array(batch_residue_index),
            }
        else:
            # Use collate_batch for automatic padding
            return self.collate_batch(examples)

    def get_statistics(self) -> dict[str, float]:
        """Compute dataset statistics.

        Returns:
            Dictionary of statistics
        """
        # Get sequence lengths
        seq_lengths = [s.atom_positions.shape[0] for s in self.structures]

        # Compute statistics
        return {
            "num_examples": len(self.structures),
            "seq_length_mean": float(np.mean(seq_lengths)),
            "seq_length_std": float(np.std(seq_lengths)),
            "seq_length_min": int(min(seq_lengths)),
            "seq_length_max": int(max(seq_lengths)),
        }


def create_synthetic_protein_dataset(
    num_proteins: int = 100,
    min_seq_length: int = 20,
    max_seq_length: int = 128,
    num_atom_types: int = 4,  # N, CA, C, O
    random_seed: int = 42,
) -> ProteinDataset:
    """Create a synthetic protein dataset for testing.

    Args:
        num_proteins: Number of synthetic proteins to generate
        min_seq_length: Minimum sequence length
        max_seq_length: Maximum sequence length
        num_atom_types: Number of atom types per residue
        random_seed: Random seed

    Returns:
        ProteinDataset with synthetic data
    """
    rng = np.random.RandomState(random_seed)

    structures = []
    for i in range(num_proteins):
        # Generate random sequence length
        seq_length = rng.randint(min_seq_length, max_seq_length + 1)

        # Generate random amino acid types
        aatype = rng.randint(0, 20, size=seq_length)

        # Generate random backbone-like structure with realistic geometry
        # This is a simplified model but creates somewhat reasonable backbones
        # For each residue, place N, CA, C, O in a reasonable geometry
        atom_positions = np.zeros((seq_length, num_atom_types, 3), dtype=np.float32)

        # Start with a random position for the first N atom
        current_pos = rng.normal(0, 1, size=3)

        # Standard bond lengths and angles
        ca_n_length = 1.45  # Å
        c_ca_length = 1.52  # Å
        n_c_length = 1.33  # Å
        o_c_length = 1.23  # Å

        # Generate backbone atoms for each residue
        for res_idx in range(seq_length):
            # N atom
            atom_positions[res_idx, 0] = current_pos

            # CA atom (relative to N)
            # Random direction, but maintain bond length
            ca_dir = rng.normal(0, 1, size=3)
            ca_dir = ca_dir / np.linalg.norm(ca_dir) * ca_n_length
            ca_pos = current_pos + ca_dir
            atom_positions[res_idx, 1] = ca_pos

            # C atom (relative to CA)
            c_dir = rng.normal(0, 1, size=3)
            c_dir = c_dir / np.linalg.norm(c_dir) * c_ca_length
            c_pos = ca_pos + c_dir
            atom_positions[res_idx, 2] = c_pos

            # O atom (relative to C)
            o_dir = rng.normal(0, 1, size=3)
            o_dir = o_dir / np.linalg.norm(o_dir) * o_c_length
            o_pos = c_pos + o_dir
            atom_positions[res_idx, 3] = o_pos

            # Update current_pos for next residue N atom
            if res_idx < seq_length - 1:
                # N atom of next residue (relative to current C)
                next_n_dir = rng.normal(0, 1, size=3)
                next_n_dir = next_n_dir / np.linalg.norm(next_n_dir) * n_c_length
                current_pos = c_pos + next_n_dir

        # All atoms are valid in synthetic data
        atom_mask = np.ones((seq_length, num_atom_types), dtype=np.float32)

        # Create structure
        structure = ProteinStructure.from_numpy(
            {
                "atom_positions": atom_positions,
                "atom_mask": atom_mask,
                "aatype": aatype,
                "residue_index": np.arange(seq_length),
                "seq_length": seq_length,
                "pdb_id": f"synthetic_{i}",
                "sequence": "".join(["A" for _ in range(seq_length)]),
            }
        )

        structures.append(structure)

    # Create dataset
    dataset = ProteinDataset(
        data_dir="",  # Not used for synthetic data
        max_seq_length=max_seq_length,
        backbone_atom_indices=list(range(num_atom_types)),
        random_seed=random_seed,
    )

    # Set structures directly
    dataset.structures = structures

    return dataset
