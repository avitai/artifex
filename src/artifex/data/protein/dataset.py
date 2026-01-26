"""Protein dataset implementation."""

import os
import pickle  # nosec B403
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# Protein constants
ATOM_TYPES = [
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
BACKBONE_ATOM_INDICES = [0, 1, 2, 4]  # N, CA, C, O
MAX_SEQ_LENGTH = 128

NpExampleType = dict[str, np.ndarray]
BatchType = dict[str, jax.Array]


class ProteinDataset:
    """Dataset for protein structures.

    This dataset handles loading and preprocessing protein structures for
    use in generative models.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_seq_length: int = MAX_SEQ_LENGTH,
        backbone_only: bool = True,
        shuffle: bool = True,
    ):
        """Initialize the protein dataset.

        Args:
            data_path: Path to the dataset directory or file.
            max_seq_length: Maximum sequence length (number of residues).
            backbone_only: Whether to use only backbone atoms.
            shuffle: Whether to shuffle the dataset.
        """
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        self.backbone_only = backbone_only
        self.shuffle = shuffle

        # Load or create dataset
        self.data = self._load_data()

        # Get indices for backbone atoms if needed
        self.backbone_indices = BACKBONE_ATOM_INDICES if backbone_only else None

        # Shuffle dataset if needed
        if shuffle:
            np.random.shuffle(self.data)

    def _load_data(self) -> list[NpExampleType]:
        """Load protein data from the data path.

        Returns:
            List of protein examples.
        """
        # Check if path is a file or directory
        if self.data_path.is_file():
            # Load from a single file (assumed to be pickle)
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)  # nosec B301
        elif self.data_path.is_dir():
            # Load from directory of files
            data = []
            for file_path in self.data_path.glob("*.pkl"):
                with open(file_path, "rb") as f:
                    examples = pickle.load(f)  # nosec B301
                    if isinstance(examples, list):
                        data.extend(examples)
                    else:
                        data.append(examples)
        else:
            raise ValueError(f"Data path {self.data_path} does not exist")

        return data

    def __len__(self) -> int:
        """Get the number of protein examples.

        Returns:
            Number of examples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> NpExampleType:
        """Get a protein example by index.

        Args:
            idx: Index of the example.

        Returns:
            Protein example as a dictionary.
        """
        # Get example
        example = self.data[idx]

        # Process example (truncate/pad, normalize, etc.)
        processed = self._process_example(example)

        return processed

    def _process_example(self, example: NpExampleType) -> NpExampleType:
        """Process a protein example for use in the model.

        Args:
            example: Raw protein example.

        Returns:
            Processed protein example.
        """
        # Extract data
        atom_positions = example["atom_positions"]
        atom_mask = example["atom_mask"]
        residue_index = example["residue_index"]

        # Truncate or pad sequence to max_seq_length
        seq_length = atom_positions.shape[0]
        if seq_length > self.max_seq_length:
            # Truncate
            atom_positions = atom_positions[: self.max_seq_length]
            atom_mask = atom_mask[: self.max_seq_length]
            residue_index = residue_index[: self.max_seq_length]
        elif seq_length < self.max_seq_length:
            # Pad with zeros
            pad_length = self.max_seq_length - seq_length
            atom_positions = np.pad(
                atom_positions,
                ((0, pad_length), (0, 0), (0, 0)),
                mode="constant",
            )
            atom_mask = np.pad(
                atom_mask,
                ((0, pad_length), (0, 0)),
                mode="constant",
            )
            residue_index = np.pad(
                residue_index,
                ((0, pad_length),),
                mode="constant",
                constant_values=-1,
            )

        # Filter backbone atoms if needed
        if self.backbone_only:
            atom_positions = atom_positions[:, self.backbone_indices, :]
            atom_mask = atom_mask[:, self.backbone_indices]

        # Center coordinates on CA atoms (alpha carbons)
        if self.backbone_only:
            ca_idx = 1  # CA is at index 1 in backbone atoms
        else:
            ca_idx = 1  # CA is at index 1 in all atoms

        # Calculate center of mass using CA atoms
        center_of_mass = np.sum(
            atom_positions[:, ca_idx, :] * atom_mask[:, ca_idx, None],
            axis=0,
        ) / np.sum(atom_mask[:, ca_idx])

        # Center the protein
        atom_positions = atom_positions - center_of_mass

        # Return processed example
        return {
            "atom_positions": atom_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
        }

    def get_batch(self, indices: list[int]) -> BatchType:
        """Get a batch of protein examples.

        Args:
            indices: Indices of examples to include in the batch.

        Returns:
            Batch of protein examples as JAX arrays.
        """
        # Get examples
        examples = [self[idx] for idx in indices]

        # Stack examples into batch
        batch = {
            "atom_positions": np.stack([example["atom_positions"] for example in examples]),
            "atom_mask": np.stack([example["atom_mask"] for example in examples]),
            "residue_index": np.stack([example["residue_index"] for example in examples]),
        }

        # Convert to JAX arrays
        jax_batch = {k: jnp.array(v) for k, v in batch.items()}

        return jax_batch

    def create_synthetic_data(self, num_examples: int = 10, noise_level: float = 0.1) -> None:
        """Create synthetic protein data for testing.

        Args:
            num_examples: Number of synthetic examples to create.
            noise_level: Standard deviation of random noise to add.
        """
        # Create random seed
        rng = np.random.RandomState(42)

        data = []
        for _ in range(num_examples):
            # Create random protein structure
            seq_length = rng.randint(32, self.max_seq_length + 1)
            num_atoms = len(BACKBONE_ATOM_INDICES) if self.backbone_only else len(ATOM_TYPES)

            # Create atom positions with backbone geometry
            # Starting with a simple helix structure and adding noise
            atom_positions = np.zeros((seq_length, num_atoms, 3))
            atom_mask = np.ones((seq_length, num_atoms))

            # Create a helix backbone
            for i in range(seq_length):
                # CA positions along a helix
                t = i * 0.5
                atom_positions[i, 1, 0] = 3.0 * np.sin(t)
                atom_positions[i, 1, 1] = 3.0 * np.cos(t)
                atom_positions[i, 1, 2] = 1.5 * t

                # N positions (-0.5, 0, 0) relative to CA
                atom_positions[i, 0, :] = atom_positions[i, 1, :] + np.array([-1.45, 0, 0])

                # C positions (0.5, 0, 0) relative to CA
                atom_positions[i, 2, :] = atom_positions[i, 1, :] + np.array([1.52, 0, 0])

                # O positions (0, 0.5, 0) relative to C
                atom_positions[i, 3, :] = atom_positions[i, 2, :] + np.array([0, 1.23, 0])

            # Add random noise
            atom_positions += rng.normal(0, noise_level, atom_positions.shape)

            # Create residue indices
            residue_index = np.arange(seq_length)

            # Create example
            example = {
                "atom_positions": atom_positions,
                "atom_mask": atom_mask,
                "residue_index": residue_index,
            }

            data.append(example)

        self.data = data

    def save_synthetic_data(self, output_path: str | Path) -> None:
        """Save synthetic data to disk.

        Args:
            output_path: Path to save the data.
        """
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self.data, f)

        print(f"Saved synthetic data to {output_path}")


# Utility functions for data conversion


def pdb_to_protein_example(pdb_file: str) -> NpExampleType:
    """Convert a PDB file to a protein example.

    Args:
        pdb_file: Path to the PDB file.

    Returns:
        Protein example as a dictionary.
    """
    # Import here to avoid biopython dependency for the whole module
    try:
        from Bio.PDB import PDBParser  # type: ignore
    except ImportError as err:
        raise ImportError(
            "Biopython is required for PDB parsing. Install with 'pip install biopython'."
        ) from err

    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Get first model and chain
    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]

    # Get residues
    residues = list(chain.get_residues())
    seq_length = len(residues)

    # Initialize arrays
    atom_positions = np.zeros((seq_length, len(ATOM_TYPES), 3))
    atom_mask = np.zeros((seq_length, len(ATOM_TYPES)))
    residue_index = np.arange(seq_length)

    # Fill arrays
    for i, residue in enumerate(residues):
        for atom in residue:
            atom_name = atom.get_name()
            if atom_name in ATOM_TYPES:
                atom_idx = ATOM_TYPES.index(atom_name)
                atom_positions[i, atom_idx] = atom.get_coord()
                atom_mask[i, atom_idx] = 1.0

    # Create example
    example = {
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "residue_index": residue_index,
    }

    return example
