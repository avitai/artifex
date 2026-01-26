"""CrossDocked2020 dataset implementation.

This module provides a dataset interface for the CrossDocked2020 dataset,
which contains protein-ligand complexes for co-design benchmarks.
"""

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from artifex.benchmarks.datasets.base import DatasetProtocol
from artifex.generative_models.core.configuration import DataConfig


class CrossDockedDataset(DatasetProtocol):
    """CrossDocked2020 dataset for protein-ligand complexes.

    This dataset provides protein-ligand complexes with binding affinity data
    for co-design benchmarks. For now, it generates mock data that follows
    the same structure as real CrossDocked data.
    """

    def __init__(
        self,
        data_path: str,
        config: DataConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the CrossDocked dataset.

        Args:
            data_path: Path to CrossDocked data (mock for now)
            config: DataConfig instance with dataset settings.
                Configuration metadata should include:
                - max_protein_atoms: Maximum number of protein atoms (default: 1000)
                - max_ligand_atoms: Maximum number of ligand atoms (default: 50)
                - pocket_radius: Radius for pocket extraction in Angstroms (default: 10.0)
                - num_samples: Number of samples to generate for mock data (default: 1000)
                - batch_size: Batch size for data loading (default: 32)
            rngs: Random number generator keys
        """
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        # Extract CrossDocked-specific parameters from config metadata
        # BEFORE calling super().__init__
        self.max_protein_atoms = config.metadata.get("max_protein_atoms", 1000)
        self.max_ligand_atoms = config.metadata.get("max_ligand_atoms", 50)
        self.pocket_radius = config.metadata.get("pocket_radius", 10.0)
        self.num_samples = config.metadata.get("num_samples", 1000)
        self.batch_size = config.metadata.get("batch_size", 32)

        # Now call parent init which will call _load_dataset
        super().__init__(data_path, config, rngs=rngs)

    def _load_dataset(self):
        """Load and preprocess CrossDocked dataset.

        For now, this generates mock data with the correct structure.
        In a real implementation, this would load actual PDB files and
        binding affinity data.
        """
        print(f"Loading mock CrossDocked dataset from {self.data_path}")
        print(f"Generating {self.num_samples} protein-ligand complexes")

        # Mock data generation parameters
        self.protein_atom_types = np.arange(1, 21)  # 20 amino acid types
        self.ligand_atom_types = np.arange(1, 9)  # 8 common ligand atom types

        # Mock binding affinity range (kcal/mol)
        self.affinity_range = (-12.0, -2.0)

        print("Mock dataset loaded successfully")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single protein-ligand complex.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - protein_coords: Protein coordinates (N_protein, 3)
                - protein_types: Protein atom types (N_protein,)
                - ligand_coords: Ligand coordinates (N_ligand, 3)
                - ligand_types: Ligand atom types (N_ligand,)
                - binding_affinity: Binding affinity in kcal/mol
                - complex_id: Unique identifier for the complex
        """
        # Use index as seed for reproducible mock data
        key = jax.random.key(idx)
        keys = jax.random.split(key, 6)

        # Generate protein structure
        n_protein = jax.random.randint(keys[0], (), minval=50, maxval=self.max_protein_atoms)
        protein_coords = jax.random.normal(keys[1], (n_protein, 3)) * 5.0
        protein_types = jax.random.choice(keys[2], self.protein_atom_types, (n_protein,))

        # Generate ligand structure (positioned near protein center)
        n_ligand = jax.random.randint(keys[3], (), minval=5, maxval=self.max_ligand_atoms)
        # Center ligand near protein center with some offset
        protein_center = jnp.mean(protein_coords, axis=0)
        ligand_offset = jax.random.normal(keys[4], (3,)) * 2.0
        ligand_center = protein_center + ligand_offset
        ligand_coords = jax.random.normal(keys[5], (n_ligand, 3)) * 1.5 + ligand_center
        ligand_types = jax.random.choice(keys[5], self.ligand_atom_types, (n_ligand,))

        # Generate mock binding affinity
        affinity = jax.random.uniform(
            keys[0], (), minval=self.affinity_range[0], maxval=self.affinity_range[1]
        )

        return {
            "protein_coords": protein_coords,
            "protein_types": protein_types,
            "ligand_coords": ligand_coords,
            "ligand_types": ligand_types,
            "binding_affinity": affinity,
            "complex_id": f"mock_complex_{idx:06d}",
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def get_batch(self, batch_size: int, start_idx: int = 0) -> dict[str, Any]:
        """Get a batch of protein-ligand complexes.

        Args:
            batch_size: Number of samples in the batch
            start_idx: Starting index for the batch

        Returns:
            Batched data with consistent shapes (padded if necessary)
        """
        batch_samples = []

        for i in range(batch_size):
            idx = (start_idx + i) % len(self)
            sample = self[idx]
            batch_samples.append(sample)

        # Stack and pad sequences to consistent lengths
        return self._collate_batch(batch_samples)

    def _collate_batch(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of samples into a batch with consistent shapes.

        Args:
            samples: List of sample dictionaries

        Returns:
            Batched data with padded sequences
        """
        batch_size = len(samples)

        # Use configuration-specified maximum lengths for consistent padding
        max_protein_len = self.max_protein_atoms
        max_ligand_len = self.max_ligand_atoms

        # Initialize batch arrays
        protein_coords = jnp.zeros((batch_size, max_protein_len, 3))
        protein_types = jnp.zeros((batch_size, max_protein_len), dtype=jnp.int32)
        ligand_coords = jnp.zeros((batch_size, max_ligand_len, 3))
        ligand_types = jnp.zeros((batch_size, max_ligand_len), dtype=jnp.int32)
        binding_affinities = jnp.zeros((batch_size,))

        # Masks for actual data vs padding
        protein_masks = jnp.zeros((batch_size, max_protein_len), dtype=jnp.bool_)
        ligand_masks = jnp.zeros((batch_size, max_ligand_len), dtype=jnp.bool_)

        complex_ids = []

        # Fill batch arrays
        for i, sample in enumerate(samples):
            p_len = len(sample["protein_coords"])
            l_len = len(sample["ligand_coords"])

            protein_coords = protein_coords.at[i, :p_len].set(sample["protein_coords"])
            protein_types = protein_types.at[i, :p_len].set(sample["protein_types"])
            ligand_coords = ligand_coords.at[i, :l_len].set(sample["ligand_coords"])
            ligand_types = ligand_types.at[i, :l_len].set(sample["ligand_types"])
            binding_affinities = binding_affinities.at[i].set(sample["binding_affinity"])

            protein_masks = protein_masks.at[i, :p_len].set(True)
            ligand_masks = ligand_masks.at[i, :l_len].set(True)

            complex_ids.append(sample["complex_id"])

        return {
            "protein_coords": protein_coords,
            "protein_types": protein_types,
            "protein_masks": protein_masks,
            "ligand_coords": ligand_coords,
            "ligand_types": ligand_types,
            "ligand_masks": ligand_masks,
            "binding_affinities": binding_affinities,
            "complex_ids": complex_ids,
        }

    def extract_pocket(
        self,
        protein_coords: jnp.ndarray,
        ligand_coords: jnp.ndarray,
        radius: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract protein pocket atoms within radius of ligand.

        Args:
            protein_coords: Protein coordinates (N_protein, 3)
            ligand_coords: Ligand coordinates (N_ligand, 3)
            radius: Pocket extraction radius (default: self.pocket_radius)

        Returns:
            Tuple of (pocket_coords, pocket_indices)
        """
        if radius is None:
            radius = self.pocket_radius

        # Compute ligand center
        ligand_center = jnp.mean(ligand_coords, axis=0)

        # Compute distances from protein atoms to ligand center
        distances = jnp.linalg.norm(protein_coords - ligand_center[None, :], axis=1)

        # Find atoms within pocket radius
        pocket_mask = distances <= radius
        pocket_indices = jnp.where(pocket_mask)[0]
        pocket_coords = protein_coords[pocket_mask]

        return pocket_coords, pocket_indices

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.config.dataset_name or "CrossDocked2020",
            "modality": "molecular",
            "n_samples": self.num_samples,
            "max_protein_atoms": self.max_protein_atoms,
            "max_ligand_atoms": self.max_ligand_atoms,
            "pocket_radius": self.pocket_radius,
            "split": self.config.split,
            "data_shape": {
                "protein_coords": (self.max_protein_atoms, 3),
                "ligand_coords": (self.max_ligand_atoms, 3),
            },
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        # Sample a subset for statistics (to avoid loading everything)
        sample_size = min(100, self.num_samples)
        samples = [self[i] for i in range(sample_size)]

        protein_lengths = [len(s["protein_coords"]) for s in samples]
        ligand_lengths = [len(s["ligand_coords"]) for s in samples]
        affinities = [float(s["binding_affinity"]) for s in samples]

        return {
            "num_samples": self.num_samples,
            "protein_atoms": {
                "mean": np.mean(protein_lengths),
                "std": np.std(protein_lengths),
                "min": np.min(protein_lengths),
                "max": np.max(protein_lengths),
            },
            "ligand_atoms": {
                "mean": np.mean(ligand_lengths),
                "std": np.std(ligand_lengths),
                "min": np.min(ligand_lengths),
                "max": np.max(ligand_lengths),
            },
            "binding_affinity": {
                "mean": np.mean(affinities),
                "std": np.std(affinities),
                "min": np.min(affinities),
                "max": np.max(affinities),
            },
        }
