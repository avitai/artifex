"""QM9 molecular dataset for SE(3)-equivariant molecular flows."""

from typing import Any, Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.datasets.base import DatasetProtocol
from artifex.generative_models.core.configuration import DataConfig


class QM9Dataset(DatasetProtocol):
    """QM9 molecular dataset for molecular conformation generation.

    The QM9 dataset contains 134,000 small organic molecules with up to 9 heavy atoms
    (excluding hydrogen), along with their geometric, energetic, electronic, and
    thermodynamic properties computed using DFT.

    For SE(3)-equivariant flows, we focus on molecular conformations and properties.
    """

    def __init__(
        self,
        data_path: str,
        config: DataConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize QM9 dataset.

        Args:
            data_path: Path to QM9 dataset files
            config: DataConfig instance with dataset settings.
                Configuration metadata should include:
                - max_atoms: Maximum number of atoms (default: 29)
                - batch_size: Batch size for data loading (default: 32)
                - num_conformations: Number of conformations per molecule (default: 100)
            rngs: Random number generators
        """
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        # Extract QM9-specific parameters from config metadata BEFORE calling super().__init__
        self.max_atoms = config.metadata.get("max_atoms", 29)
        self.batch_size = config.metadata.get("batch_size", 32)
        self.split = config.split
        self.num_conformations = config.metadata.get("num_conformations", 100)

        # QM9 atom types: H, C, N, O, F
        self.atom_types_map = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        self.num_atom_types = len(self.atom_types_map)

        # Initialize num_molecules for later use
        self.num_molecules = 1000  # Default, will be updated in _setup_mock_data

        # Now call parent init which will call _load_dataset
        super().__init__(data_path, config, rngs=rngs)

    def _load_dataset(self):
        """Load the dataset."""
        self._setup_mock_data()

    def _setup_mock_data(self):
        """Setup mock QM9 data for development and testing."""
        # Mock dataset size based on split
        if self.split == "train":
            self.num_molecules = 110000
        elif self.split == "val":
            self.num_molecules = 12000
        else:  # test
            self.num_molecules = 12000

        # For testing, we'll generate a smaller subset
        self.num_molecules = min(self.num_molecules, 1000)

    def __iter__(self) -> Iterator[dict[str, jnp.ndarray]]:
        """Iterate over batches of molecular data."""
        num_batches = self.num_molecules // self.batch_size

        for batch_idx in range(num_batches):
            yield self._generate_batch()

    def _generate_batch(self) -> dict[str, jnp.ndarray]:
        """Generate a batch of mock QM9 molecular data."""
        # Generate random number of atoms per molecule (3 to max_atoms)
        num_atoms_per_mol = jax.random.randint(
            self.rngs.params(),
            (self.batch_size,),
            minval=3,  # At least 3 atoms for a meaningful molecule
            maxval=self.max_atoms - 5,  # Leave room for reasonable molecules
        )

        # Initialize batch arrays
        coordinates = jnp.zeros((self.batch_size, self.max_atoms, 3))
        atom_types = jnp.zeros((self.batch_size, self.max_atoms), dtype=jnp.int32)
        atom_mask = jnp.zeros((self.batch_size, self.max_atoms), dtype=jnp.bool_)
        energies = jnp.zeros((self.batch_size,))
        forces = jnp.zeros((self.batch_size, self.max_atoms, 3))

        for mol_idx in range(self.batch_size):
            n_atoms = num_atoms_per_mol[mol_idx]

            # Generate realistic molecular coordinates
            mol_coords, mol_types = self._generate_realistic_molecule(n_atoms)

            # Pad and store in batch
            coordinates = coordinates.at[mol_idx, :n_atoms].set(mol_coords)
            atom_types = atom_types.at[mol_idx, :n_atoms].set(mol_types)
            atom_mask = atom_mask.at[mol_idx, :n_atoms].set(True)

            # Generate mock energy (typical range for small molecules)
            energies = energies.at[mol_idx].set(
                jax.random.normal(self.rngs.params()) * 100.0 - 500.0  # Hartree units
            )

            # Generate mock forces (small values for stable conformations)
            mol_forces = jax.random.normal(self.rngs.params(), (n_atoms, 3)) * 0.1
            forces = forces.at[mol_idx, :n_atoms].set(mol_forces)

        return {
            "coordinates": coordinates,
            "atom_types": atom_types,
            "atom_mask": atom_mask,
            "num_atoms": num_atoms_per_mol,
            "energies": energies,
            "forces": forces,
        }

    def _generate_realistic_molecule(self, n_atoms: int) -> tuple[jax.Array, jax.Array]:
        """Generate a realistic molecular structure.

        Args:
            n_atoms: Number of atoms in the molecule

        Returns:
            Tuple of (coordinates, atom_types)
        """
        # Start with a carbon atom at origin
        coordinates: list[jax.Array] = [jnp.array([0.0, 0.0, 0.0])]
        atom_types: list[int] = [1]  # Carbon

        # Typical bond lengths (in Angstroms)

        for i in range(1, n_atoms):
            # Choose atom type (weighted toward H and C for realistic organic molecules)
            atom_type_probs = jnp.array([0.5, 0.3, 0.1, 0.08, 0.02])  # H, C, N, O, F
            atom_type = jax.random.choice(self.rngs.params(), jnp.arange(5), p=atom_type_probs)
            atom_types.append(int(atom_type))

            # Place atom near a random existing atom with appropriate bond length
            parent_idx = jax.random.randint(self.rngs.params(), (), 0, len(coordinates))
            parent_pos = coordinates[parent_idx]

            # Get approximate bond length
            bond_length = 1.2 + jax.random.normal(self.rngs.params()) * 0.1

            # Random direction with some geometric constraints
            direction = jax.random.normal(self.rngs.params(), (3,))
            direction = direction / jnp.linalg.norm(direction)

            new_pos = parent_pos + direction * bond_length
            coordinates.append(new_pos)

        return jnp.stack(coordinates), jnp.array(atom_types, dtype=jnp.int32)

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        return {
            "num_molecules": self.num_molecules,
            "max_atoms": self.max_atoms,
            "num_atom_types": self.num_atom_types,
            "atom_types_map": self.atom_types_map,
            "split": self.split,
            "batch_size": self.batch_size,
        }

    def validate_structure(self) -> bool:
        """Validate dataset structure and integrity."""
        try:
            # Generate a test batch
            batch = next(iter(self))

            # Check batch structure
            required_keys = {
                "coordinates",
                "atom_types",
                "atom_mask",
                "num_atoms",
                "energies",
                "forces",
            }
            if not required_keys.issubset(batch.keys()):
                return False

            # Check shapes
            batch_size = batch["coordinates"].shape[0]
            if batch_size != self.batch_size:
                return False

            # Check coordinate ranges (reasonable for molecules)
            coords = batch["coordinates"]
            mask = batch["atom_mask"]
            masked_coords = coords[mask]

            if jnp.any(jnp.abs(masked_coords) > 20.0):  # Unreasonably large molecules
                return False

            # Check atom type ranges
            if jnp.any(batch["atom_types"] < 0) or jnp.any(
                batch["atom_types"] >= self.num_atom_types
            ):
                return False

            return True

        except Exception:
            return False

    def get_batch(self, batch_size: int | None = None) -> dict[str, jax.Array]:
        """Get a batch of data."""
        batch_size = batch_size or self.batch_size
        # For compatibility with DatasetProtocol, return a single batch
        return self._generate_batch()

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.config.dataset_name or "QM9",
            "modality": "molecular",
            "n_samples": self.num_molecules,
            "max_atoms": self.max_atoms,
            "num_atom_types": self.num_atom_types,
            "split": self.split,
            "data_shape": {
                "coordinates": (self.batch_size, self.max_atoms, 3),
                "atom_types": (self.batch_size, self.max_atoms),
                "atom_mask": (self.batch_size, self.max_atoms),
            },
        }

    def __len__(self) -> int:
        """Return number of batches in dataset."""
        return self.num_molecules // self.batch_size
