"""Protein dataset backed by datarax DataSourceModule.

Provides a DataSourceModule subclass for loading, processing, and serving
protein structure data. Supports loading from files (pickle/PDB)
and from in-memory protein dictionaries.
"""

import logging
import pickle  # nosec B403
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import struct


logger = logging.getLogger(__name__)

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
BACKBONE_ATOM_INDICES = (0, 1, 2, 4)  # N, CA, C, O
MAX_SEQ_LENGTH = 128

# Standard amino acid types (single-letter codes)
AA_TYPES = [
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
AA_TYPE_TO_IDX = {aa: i for i, aa in enumerate(AA_TYPES)}

# Type aliases
NpExampleValue = np.ndarray | int | str
NpExampleType = dict[str, NpExampleValue]
BatchType = dict[str, jax.Array]


@struct.dataclass
class ProteinStructure:
    """Protein structure data representation.

    Attributes:
        atom_positions: Atom positions with shape [num_res, num_atoms, 3].
        atom_mask: Binary mask for atom positions [num_res, num_atoms].
        aatype: Amino acid types [num_res].
        residue_index: Residue indices [num_res].
        chain_index: Chain indices [num_res].
        b_factors: B-factors for each atom [num_res, num_atoms].
        resolution: Structure resolution.
        seq_length: Sequence length (number of residues).
        sequence: Amino acid sequence as string.
        pdb_id: PDB identifier.
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
    def from_numpy(cls, data: dict[str, Any]) -> "ProteinStructure":
        """Create a ProteinStructure from numpy arrays.

        Args:
            data: Dictionary with numpy arrays.

        Returns:
            ProteinStructure instance.
        """
        jax_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                jax_data[key] = jnp.array(value)
            else:
                jax_data[key] = value
        return cls(**jax_data)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinDatasetConfig(StructuralConfig):
    """Configuration for ProteinDataset.

    Attributes:
        max_seq_length: Maximum sequence length to use.
        backbone_atom_indices: Indices of backbone atoms to extract.
        center_positions: Whether to center atom positions on center of mass.
        normalize_positions: Whether to normalize atom positions by std.
        split: Dataset split (train/valid/test).
    """

    max_seq_length: int = MAX_SEQ_LENGTH
    backbone_atom_indices: tuple[int, ...] = BACKBONE_ATOM_INDICES
    center_positions: bool = True
    normalize_positions: bool = True
    split: str = "train"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ProteinDataset(DataSourceModule):
    """Protein structure dataset backed by datarax DataSourceModule.

    Supports two initialization modes:
    - From file path: loads protein data from pickle files or directories
    - From protein list: accepts a list of protein data dictionaries

    Provides processing (centering, normalizing, backbone extraction),
    and integrates with datarax pipelines for batching and shuffling.

    Examples:
        From in-memory data::

            config = ProteinDatasetConfig(max_seq_length=64)
            dataset = ProteinDataset(config, list_of_protein_dicts)

        From files::

            config = ProteinDatasetConfig()
            dataset = ProteinDataset(config, data_dir="/path/to/proteins")

        With datarax pipeline::

            from datarax import Pipeline
            from flax import nnx

            pipeline = Pipeline(source=dataset, stages=[], batch_size=8, rngs=nnx.Rngs(0))
            for batch in pipeline:
                train_step(batch)
    """

    # Narrow config type for pyright
    config: ProteinDatasetConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    # NNX data annotation — structures stored as non-trainable data
    structures: list[ProteinStructure] = nnx.data()

    def __init__(
        self,
        config: ProteinDatasetConfig,
        data_or_path: list[NpExampleType] | str | Path | None = None,
        *,
        data_dir: str | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the protein dataset.

        Args:
            config: Dataset configuration.
            data_or_path: Either a list of protein data dictionaries,
                a path (str or Path) to a pickle file/directory, or None.
            data_dir: Alternative keyword for data directory path.
            rngs: Optional NNX Rngs for stochastic operations.
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name or "ProteinDataset")
        self.structures = _load_structures(data_or_path, data_dir)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed protein structures.

        Yields:
            Processed protein dictionaries.
        """
        for i in range(len(self.structures)):
            yield self[i]

    def __next__(self) -> dict[str, Any]:
        """Not used — iteration via __iter__ generator."""
        raise NotImplementedError("Use __iter__ for iteration")

    def __len__(self) -> int:
        """Get the number of structures in the dataset."""
        return len(self.structures)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a processed protein structure by index.

        Args:
            idx: Structure index.

        Returns:
            Dictionary with processed structure data.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} items")

        structure = self.structures[idx]
        processed = self._process_structure(structure)

        return {
            "atom_positions": processed.atom_positions,
            "atom_mask": processed.atom_mask,
            "aatype": processed.aatype,
            "residue_index": processed.residue_index,
            "seq_length": processed.seq_length,
            "sequence": processed.sequence,
            "pdb_id": processed.pdb_id,
        }

    def _process_structure(self, structure: ProteinStructure) -> ProteinStructure:
        """Process a protein structure for use with generative models.

        Applies backbone extraction, truncation, centering, and normalization
        based on the dataset configuration.

        Args:
            structure: Input protein structure.

        Returns:
            Processed protein structure.
        """
        atom_positions = structure.atom_positions
        atom_mask = structure.atom_mask
        cfg = self.config

        # Extract backbone atoms
        if cfg.backbone_atom_indices:
            indices = list(cfg.backbone_atom_indices)
            atom_positions = atom_positions[:, indices]
            atom_mask = atom_mask[:, indices]

        # Truncate sequence
        seq_length = atom_positions.shape[0]
        if seq_length > cfg.max_seq_length:
            atom_positions = atom_positions[: cfg.max_seq_length]
            atom_mask = atom_mask[: cfg.max_seq_length]
            aatype = (
                structure.aatype[: cfg.max_seq_length] if structure.aatype is not None else None
            )
        else:
            aatype = structure.aatype

        # Center positions on center of mass
        if cfg.center_positions:
            masked_positions = atom_positions * atom_mask[:, :, None]
            mask_sum = atom_mask.sum()
            if mask_sum > 0:
                center = masked_positions.sum(axis=(0, 1)) / mask_sum
                atom_positions = atom_positions - center

        # Normalize positions by standard deviation
        if cfg.normalize_positions:
            masked_positions = atom_positions * atom_mask[:, :, None]
            if atom_mask.sum() > 0:
                std = jnp.sqrt(
                    jnp.sum(jnp.sum(masked_positions**2, axis=-1) * atom_mask) / atom_mask.sum()
                )
                atom_positions = atom_positions / (std + 1e-6)

        # Create residue indices if not provided
        seq_length = atom_positions.shape[0]
        residue_index = (
            structure.residue_index
            if structure.residue_index is not None
            else jnp.arange(seq_length)
        )

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

    def get_batch(
        self,
        batch_size_or_indices: int | list[int],
        max_length: int | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Get a batch of protein structures.

        Supports two calling conventions:
        - ``get_batch(batch_size)`` — sequential batch convention used
          by datarax sources, returns next ``batch_size`` elements.
        - ``get_batch(indices)`` — direct index-based access.

        Args:
            batch_size_or_indices: Either an int (batch size for sequential
                access) or a list of ints (specific indices).
            max_length: Optional maximum sequence length for padding.
                If None, pads to the max length in the batch.

        Returns:
            Dictionary with batched structure data.
        """
        if isinstance(batch_size_or_indices, int):
            # Datarax sequential batch convention: get_batch(batch_size)
            batch_size = batch_size_or_indices
            start = getattr(self, "_batch_cursor", 0)
            indices = list(range(start, min(start + batch_size, len(self.structures))))
            object.__setattr__(self, "_batch_cursor", start + batch_size)
        else:
            indices = batch_size_or_indices

        examples = [self[idx] for idx in indices]
        return protein_collate_fn(
            examples,
            max_seq_length=self.config.max_seq_length,
            pad_to=max_length,
        )

    def get_statistics(self) -> dict[str, float]:
        """Compute dataset statistics.

        Returns:
            Dictionary of statistics (num_examples, seq_length_mean/std/min/max).
        """
        if not self.structures:
            return {
                "num_examples": 0,
                "seq_length_mean": 0.0,
                "seq_length_std": 0.0,
                "seq_length_min": 0,
                "seq_length_max": 0,
            }

        seq_lengths = [s.atom_positions.shape[0] for s in self.structures]

        return {
            "num_examples": len(self.structures),
            "seq_length_mean": float(np.mean(seq_lengths)),
            "seq_length_std": float(np.std(seq_lengths)),
            "seq_length_min": int(min(seq_lengths)),
            "seq_length_max": int(max(seq_lengths)),
        }

    def save_synthetic_data(self, output_path: str | Path) -> None:
        """Save structure data to disk as pickle.

        Args:
            output_path: Path to save the data.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(  # nosec B301
                [
                    {
                        "atom_positions": np.array(s.atom_positions),
                        "atom_mask": np.array(s.atom_mask),
                        "residue_index": np.array(s.residue_index)
                        if s.residue_index is not None
                        else np.arange(s.atom_positions.shape[0]),
                    }
                    for s in self.structures
                ],
                f,
            )

        logger.info("Saved data to %s", output_path)


# ---------------------------------------------------------------------------
# Collation (standalone for use with datarax DefaultBatcher)
# ---------------------------------------------------------------------------


def protein_collate_fn(
    examples: list[dict[str, Any]],
    max_seq_length: int = MAX_SEQ_LENGTH,
    pad_to: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Collate protein examples into a padded batch.

    Pads variable-length sequences to the maximum length in the batch
    (or ``max_seq_length``, whichever is smaller). If ``pad_to`` is given,
    pads to exactly that length instead.

    Can be used standalone or passed to ``DefaultBatcher(collate_fn=...)``.

    Args:
        examples: List of protein example dictionaries from ProteinDataset.
        max_seq_length: Upper bound for padding length.
        pad_to: If given, pad to exactly this length.

    Returns:
        Batched data as JAX arrays with shape [batch, seq_len, ...].
    """
    batch_size = len(examples)
    if pad_to is not None:
        max_len = pad_to
    else:
        max_len = min(
            max(ex["seq_length"] for ex in examples),
            max_seq_length,
        )
    num_atom_types = examples[0]["atom_positions"].shape[1]

    batch_positions = np.zeros((batch_size, max_len, num_atom_types, 3), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_len, num_atom_types), dtype=np.float32)
    batch_aatype = np.zeros((batch_size, max_len), dtype=np.int32)
    batch_residue_index = np.zeros((batch_size, max_len), dtype=np.int32)

    for i, ex in enumerate(examples):
        seq_len = min(ex["seq_length"], max_len)
        batch_positions[i, :seq_len] = ex["atom_positions"][:seq_len]
        batch_mask[i, :seq_len] = ex["atom_mask"][:seq_len]
        if ex["aatype"] is not None:
            batch_aatype[i, :seq_len] = ex["aatype"][:seq_len]
        if ex["residue_index"] is not None:
            batch_residue_index[i, :seq_len] = ex["residue_index"][:seq_len]

    return {
        "atom_positions": jnp.array(batch_positions),
        "atom_mask": jnp.array(batch_mask),
        "aatype": jnp.array(batch_aatype),
        "residue_index": jnp.array(batch_residue_index),
    }


# ---------------------------------------------------------------------------
# Data loading helpers (module-level, not methods)
# ---------------------------------------------------------------------------


def _load_structures(
    data_or_path: list[NpExampleType] | str | Path | None,
    data_dir: str | None,
) -> list[ProteinStructure]:
    """Load protein structure data from files or in-memory data.

    Args:
        data_or_path: List of protein dicts, file/dir path, or None.
        data_dir: Alternative keyword for data directory path.

    Returns:
        List of ProteinStructure instances.
    """
    # Resolve source
    if data_dir is not None:
        data_path = Path(data_dir)
        if data_path.exists():
            raw = _load_from_path(data_path)
            return [ProteinStructure.from_numpy(d) for d in raw]
        return []

    if isinstance(data_or_path, str | Path):
        data_path = Path(data_or_path)
        if data_path.exists():
            raw = _load_from_path(data_path)
            return [ProteinStructure.from_numpy(d) for d in raw]
        return []

    if isinstance(data_or_path, list):
        return [ProteinStructure.from_numpy(p) for p in data_or_path]

    return []


def _load_from_path(data_path: Path) -> list[NpExampleType]:
    """Load protein data from a file or directory.

    Args:
        data_path: Path to a pickle file or directory of pickle files.

    Returns:
        List of protein example dictionaries.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if data_path.is_file():
        with open(data_path, "rb") as f:
            data = pickle.load(f)  # nosec B301
        return data if isinstance(data, list) else [data]

    if data_path.is_dir():
        data: list[NpExampleType] = []
        for file_path in sorted(data_path.glob("*.pkl")):
            with open(file_path, "rb") as f:
                examples = pickle.load(f)  # nosec B301
                if isinstance(examples, list):
                    data.extend(examples)
                else:
                    data.append(examples)
        return data

    raise FileNotFoundError(f"Data path does not exist: {data_path}")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_synthetic_protein_dataset(
    num_proteins: int = 100,
    min_seq_length: int = 20,
    max_seq_length: int = 128,
    num_atom_types: int = 4,
    random_seed: int = 42,
) -> ProteinDataset:
    """Create a synthetic protein dataset for testing.

    Generates proteins with realistic bond geometries using standard
    bond lengths (N-CA: 1.45A, CA-C: 1.52A, C-N: 1.33A, C-O: 1.23A).

    Args:
        num_proteins: Number of synthetic proteins to generate.
        min_seq_length: Minimum sequence length.
        max_seq_length: Maximum sequence length.
        num_atom_types: Number of atom types per residue.
        random_seed: Random seed.

    Returns:
        ProteinDataset backed by DataSourceModule.
    """
    rng = np.random.RandomState(random_seed)

    # Standard bond lengths (Angstroms)
    ca_n_length = 1.45
    c_ca_length = 1.52
    n_c_length = 1.33
    o_c_length = 1.23

    proteins: list[NpExampleType] = []
    for i in range(num_proteins):
        seq_length = rng.randint(min_seq_length, max_seq_length + 1)
        aatype = rng.randint(0, 20, size=seq_length)
        atom_positions = np.zeros((seq_length, num_atom_types, 3), dtype=np.float32)

        current_pos = rng.normal(0, 1, size=3).astype(np.float32)

        for res_idx in range(seq_length):
            atom_positions[res_idx, 0] = current_pos

            ca_dir = rng.normal(0, 1, size=3)
            ca_dir = (ca_dir / np.linalg.norm(ca_dir) * ca_n_length).astype(np.float32)
            ca_pos = current_pos + ca_dir
            atom_positions[res_idx, 1] = ca_pos

            c_dir = rng.normal(0, 1, size=3)
            c_dir = (c_dir / np.linalg.norm(c_dir) * c_ca_length).astype(np.float32)
            c_pos = ca_pos + c_dir
            atom_positions[res_idx, 2] = c_pos

            if num_atom_types >= 4:
                o_dir = rng.normal(0, 1, size=3)
                o_dir = (o_dir / np.linalg.norm(o_dir) * o_c_length).astype(np.float32)
                atom_positions[res_idx, 3] = c_pos + o_dir

            if res_idx < seq_length - 1:
                next_n_dir = rng.normal(0, 1, size=3)
                next_n_dir = (next_n_dir / np.linalg.norm(next_n_dir) * n_c_length).astype(
                    np.float32
                )
                current_pos = c_pos + next_n_dir

        atom_mask = np.ones((seq_length, num_atom_types), dtype=np.float32)

        proteins.append(
            {
                "atom_positions": atom_positions,
                "atom_mask": atom_mask,
                "aatype": aatype,
                "residue_index": np.arange(seq_length),
                "seq_length": seq_length,
                "pdb_id": f"synthetic_{i}",
                "sequence": "A" * seq_length,
            }
        )

    config = ProteinDatasetConfig(
        max_seq_length=max_seq_length,
        backbone_atom_indices=tuple(range(num_atom_types)),
    )

    return ProteinDataset(config, proteins)


def pdb_to_protein_example(pdb_file: str) -> NpExampleType:
    """Convert a PDB file to a protein example.

    Args:
        pdb_file: Path to the PDB file.

    Returns:
        Protein example as a dictionary.

    Raises:
        ImportError: If Biopython is not installed.
    """
    try:
        from Bio.PDB import PDBParser  # type: ignore[import-untyped]
    except ImportError as err:
        raise ImportError(
            "Biopython is required for PDB parsing. Install with 'uv add biopython'."
        ) from err

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]

    residues = list(chain.get_residues())
    seq_length = len(residues)

    atom_positions = np.zeros((seq_length, len(ATOM_TYPES), 3))
    atom_mask = np.zeros((seq_length, len(ATOM_TYPES)))
    residue_index = np.arange(seq_length)

    for i, residue in enumerate(residues):
        for atom in residue:
            atom_name = atom.get_name()
            if atom_name in ATOM_TYPES:
                atom_idx = ATOM_TYPES.index(atom_name)
                atom_positions[i, atom_idx] = atom.get_coord()
                atom_mask[i, atom_idx] = 1.0

    return {
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "residue_index": residue_index,
    }
