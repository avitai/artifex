"""Protein datasets for benchmarking.

This module provides synthetic protein datasets for benchmarking protein
generative models.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.datasets.base import DatasetProtocol
from artifex.generative_models.core.configuration import DataConfig


class SyntheticProteinDataset(DatasetProtocol):
    """Synthetic protein structure dataset for benchmarking.

    This dataset generates synthetic protein structures with random
    coordinates for testing.
    """

    def __init__(
        self,
        data_path: str,
        config: DataConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the synthetic protein dataset.

        Args:
            data_path: Path to dataset (not used for synthetic data but required by protocol)
            config: DataConfig instance with dataset settings.
                Configuration metadata should include:
                - num_samples: Number of samples in the dataset (default: 100)
                - num_residues: Number of residues per protein (default: 10)
                - num_atoms: Number of atoms per residue (default: 4)
                - seed: Random seed for reproducible dataset generation (default: 42)
                - batch_size: Batch size for data loading (default: 32)
            rngs: Random number generators
        """
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        # Extract protein-specific parameters from config metadata BEFORE calling super().__init__
        self.num_samples = config.metadata.get("num_samples", 100)
        self.num_residues = config.metadata.get("num_residues", 10)
        self.num_atoms = config.metadata.get("num_atoms", 4)
        self.seed = config.metadata.get("seed", 42)
        self.batch_size = config.metadata.get("batch_size", 32)

        # Now call parent init which will call _load_dataset
        super().__init__(data_path, config, rngs=rngs)

    def _generate_dataset(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate synthetic protein structures.

        Returns:
            Tuple of (coordinates, aatype).
        """
        key = jax.random.PRNGKey(self.seed)
        key, subkey = jax.random.split(key)

        # Generate random coordinates
        # Shape: [num_samples, num_residues, num_atoms, 3]
        coordinates = jax.random.normal(
            key, shape=(self.num_samples, self.num_residues, self.num_atoms, 3)
        )

        # Apply transformations to make coordinates more protein-like
        # For each residue, shift atoms to be close to each other
        # Simple model: place atoms along a "backbone" with slight variation
        coordinates = self._create_protein_like_coordinates(coordinates)

        # Generate random amino acid types (0-19 for 20 standard amino acids)
        # Shape: [num_samples, num_residues]
        aatype = jax.random.randint(
            subkey, shape=(self.num_samples, self.num_residues), minval=0, maxval=20
        )

        return coordinates, aatype

    def _create_protein_like_coordinates(self, coordinates: jnp.ndarray) -> jnp.ndarray:
        """Transform random coordinates to be more protein-like.

        Args:
            coordinates: Random coordinates with shape
                [num_samples, num_residues, num_atoms, 3].

        Returns:
            Transformed coordinates with protein-like structure.
        """
        # Create a regular backbone trace (simplified alpha helix)
        # Each residue is ~3.8Å apart along the helix
        t = jnp.arange(self.num_residues) * 1.5
        backbone_x = 2.5 * jnp.cos(t)
        backbone_y = 2.5 * jnp.sin(t)
        backbone_z = t

        # Combine into backbone coordinates
        # Shape: [num_residues, 3]
        backbone = jnp.stack([backbone_x, backbone_y, backbone_z], axis=1)

        # For each residue, place atoms around the backbone position
        # with appropriate bond distances
        bond_length = 1.5  # Typical bond length ~1.5Å

        # Create offsets for atoms within each residue
        # These are simplified offsets to mimic backbone and side chain atoms
        # Real proteins would have specific geometries based on atom types
        # Generate enough offsets for the requested number of atoms
        base_offsets: list[list[float]] = [
            [0.0, 0.0, 0.0],  # CA (alpha carbon)
            [bond_length, 0.0, 0.0],  # C (carbonyl carbon)
            [0.0, bond_length, 0.0],  # N (nitrogen)
            [0.0, 0.0, bond_length],  # CB (beta carbon)
        ]

        # If we need more atoms, generate additional offsets
        if self.num_atoms > 4:
            for i in range(4, self.num_atoms):
                # Add side chain atoms with varying positions
                angle = (i - 4) * 2.0 * jnp.pi / (self.num_atoms - 4)
                x = bond_length * jnp.cos(angle)
                y = bond_length * jnp.sin(angle)
                z = bond_length * 0.5
                base_offsets.append([x, y, z])

        offsets = jnp.array(base_offsets)[: self.num_atoms]

        # Add small random variations to make it realistic
        # Use the same random seed for reproducibility
        key = jax.random.PRNGKey(self.seed + 1)
        noise = 0.1 * jax.random.normal(key, shape=(self.num_residues, self.num_atoms, 3))

        # Create a template structure by adding offsets to backbone positions
        # and then adding small random variations
        template = backbone[:, None, :] + offsets[None, :, :] + noise

        # Broadcast the template to all samples
        # Shape: [1, num_residues, num_atoms, 3] -> [num_samples, ...]
        template = jnp.broadcast_to(
            template[None, ...], (self.num_samples, self.num_residues, self.num_atoms, 3)
        )

        # Add sample-specific variations to create different structures
        key = jax.random.PRNGKey(self.seed + 2)
        sample_noise = 0.3 * jax.random.normal(
            key, shape=(self.num_samples, self.num_residues, self.num_atoms, 3)
        )

        return template + sample_noise

    def _load_dataset(self):
        """Load the dataset."""
        # Generate the dataset
        coordinates, aatype = self._generate_dataset()

        # Store as regular attributes (datasets are NOT nnx.Module, so no nnx.data needed)
        self.coordinates = coordinates
        self.aatype = aatype

        # Store flattened coordinates for clustering algorithms
        flat_coords = self.flatten_coordinates(coordinates)
        self.flat_coordinates = flat_coords

    def flatten_coordinates(self, coordinates: jnp.ndarray) -> jnp.ndarray:
        """Flatten 4D protein coordinates to 2D for clustering algorithms.

        Args:
            coordinates: Protein coordinates with shape
                [num_samples, num_residues, num_atoms, 3]

        Returns:
            Flattened coordinates with shape [num_samples, num_residues * num_atoms * 3]
            suitable for clustering algorithms.
        """
        batch_size = coordinates.shape[0]

        # Reshape from [batch, residues, atoms, 3] to [batch, residues*atoms, 3]
        reshaped = coordinates.reshape(batch_size, -1, 3)

        # For clustering algorithms, flatten completely to [batch, residues*atoms*3]
        flattened = reshaped.reshape(batch_size, -1)

        return flattened

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> jnp.ndarray:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Protein structure coordinates.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.num_samples}")

        # Return flattened coordinates for compatibility with clustering algorithms
        return self.flat_coordinates[idx]

    def get_raw_coordinates(self, idx: int) -> jnp.ndarray:
        """Get the original 4D coordinates for a sample.

        Args:
            idx: Index of the sample.

        Returns:
            Original 4D protein coordinates [num_residues, num_atoms, 3]
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.num_samples}")

        return self.coordinates[idx]

    def get_sample_with_metadata(self, idx: int) -> dict[str, Any]:
        """Get a sample with full metadata.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with protein structure data.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.num_samples}")

        return {
            "coordinates": self.coordinates[idx],
            "flat_coordinates": self.flat_coordinates[idx],
            "aatype": self.aatype[idx],
        }

    def get_batch(self, batch_size: int | None = None) -> dict[str, jax.Array]:
        """Get a batch of protein structures.

        Args:
            batch_size: Batch size (uses config default if None)

        Returns:
            Batch dictionary containing coordinates, flat_coordinates, and aatype
        """
        batch_size = batch_size or self.batch_size

        # Simple random sampling
        indices = jax.random.choice(
            self.rngs.default(), self.num_samples, (batch_size,), replace=False
        )

        return {
            "coordinates": self.coordinates[indices],
            "flat_coordinates": self.flat_coordinates[indices],
            "aatype": self.aatype[indices],
        }

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.config.dataset_name or "SyntheticProtein",
            "modality": "protein",
            "num_samples": self.num_samples,
            "num_residues": self.num_residues,
            "num_atoms": self.num_atoms,
            "split": self.config.split,
            "data_shape": {
                "coordinates": (self.num_samples, self.num_residues, self.num_atoms, 3),
                "flat_coordinates": (self.num_samples, self.num_residues * self.num_atoms * 3),
                "aatype": (self.num_samples, self.num_residues),
            },
        }


def create_synthetic_protein_dataset(
    config: DataConfig, rngs: nnx.Rngs, data_path: str = "./protein_data"
) -> SyntheticProteinDataset:
    """Create a synthetic protein dataset for benchmarking.

    Args:
        config: DataConfig instance with dataset settings
        rngs: Random number generators
        data_path: Path to dataset directory

    Returns:
        A synthetic protein dataset.
    """
    return SyntheticProteinDataset(data_path=data_path, config=config, rngs=rngs)
