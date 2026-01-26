"""Molecular flows evaluation metrics for SE(3)-equivariant models."""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.metrics.core import MetricBase
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.distance import (
    _calculate_rmsd_matrix,
)


class MolecularFlowsMetrics(MetricBase):
    """Comprehensive metrics for evaluating molecular flows.

    This class implements metrics specifically designed for SE(3)-equivariant
    molecular flows, focusing on chemical validity, conformational diversity,
    and energy consistency.
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize molecular flows metrics.

        Args:
            rngs: Random number generators for stochastic computations
            config: Evaluation configuration (must be EvaluationConfig)
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # Chemical constraints for validity checking
        self.bond_length_thresholds = {
            # (atom_type_1, atom_type_2): (min_length, max_length) in Angstroms
            (0, 1): (0.8, 1.3),  # H-C
            (1, 1): (1.2, 1.8),  # C-C
            (1, 2): (1.2, 1.7),  # C-N
            (1, 3): (1.2, 1.7),  # C-O
            (1, 4): (1.1, 1.6),  # C-F
            (0, 2): (0.8, 1.2),  # H-N
            (0, 3): (0.8, 1.2),  # H-O
        }

        # Angle thresholds for chemical validity
        self.angle_thresholds = {
            "min_angle": 60.0,  # degrees
            "max_angle": 180.0,  # degrees
        }

    def chemical_validity(
        self,
        coordinates: jax.Array,
        atom_types: jax.Array,
        atom_mask: jax.Array,
    ) -> float:
        """Evaluate chemical validity of molecular conformations.

        Args:
            coordinates: Molecular coordinates [batch, max_atoms, 3]
            atom_types: Atom types [batch, max_atoms]
            atom_mask: Atom mask [batch, max_atoms]

        Returns:
            Chemical validity score (0.0 to 1.0)
        """
        batch_size = coordinates.shape[0]
        validity_scores = []

        for mol_idx in range(batch_size):
            mol_coords = coordinates[mol_idx]
            mol_types = atom_types[mol_idx]
            mol_mask = atom_mask[mol_idx]

            # Get actual atoms (remove padding)
            num_atoms = jnp.sum(mol_mask).astype(int)
            if num_atoms < 2:
                validity_scores.append(0.0)
                continue

            valid_coords = mol_coords[:num_atoms]
            valid_types = mol_types[:num_atoms]

            # Check bond lengths
            bond_validity = self._check_bond_lengths(valid_coords, valid_types)

            # Check bond angles (if enough atoms)
            if num_atoms >= 3:
                angle_validity = self._check_bond_angles(valid_coords, valid_types)
            else:
                angle_validity = 1.0

            # Check for reasonable molecular size
            size_validity = self._check_molecular_size(valid_coords)

            # Combined validity score
            mol_validity = (bond_validity + angle_validity + size_validity) / 3.0
            validity_scores.append(mol_validity)

        return float(jnp.mean(jnp.array(validity_scores)))

    def _check_bond_lengths(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> float:
        """Check if bond lengths are chemically reasonable."""
        num_atoms = coordinates.shape[0]
        valid_bonds = 0
        total_bonds = 0

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = jnp.linalg.norm(coordinates[i] - coordinates[j])

                # Check if this could be a bond (distance < 3 Angstroms)
                if distance < 3.0:
                    total_bonds += 1

                    # Get atom types
                    type_i, type_j = int(atom_types[i]), int(atom_types[j])
                    bond_type = tuple(sorted([type_i, type_j]))

                    # Check against known bond length constraints
                    if bond_type in self.bond_length_thresholds:
                        min_len, max_len = self.bond_length_thresholds[bond_type]
                        if min_len <= distance <= max_len:
                            valid_bonds += 1
                    else:
                        # For unknown bond types, use general constraints
                        if 0.5 <= distance <= 2.5:
                            valid_bonds += 1

        if total_bonds == 0:
            return 1.0  # No bonds to validate

        return valid_bonds / total_bonds

    def _check_bond_angles(self, coordinates: jax.Array, atom_types: jax.Array) -> float:
        """Check if bond angles are chemically reasonable."""
        num_atoms = coordinates.shape[0]
        valid_angles = 0
        total_angles = 0

        for i in range(num_atoms):
            for j in range(num_atoms):
                for k in range(num_atoms):
                    if i != j and j != k and i != k:
                        # Calculate angle at atom j
                        vec1 = coordinates[i] - coordinates[j]
                        vec2 = coordinates[k] - coordinates[j]

                        # Check if these form reasonable bonds
                        dist1 = jnp.linalg.norm(vec1)
                        dist2 = jnp.linalg.norm(vec2)

                        if dist1 < 2.5 and dist2 < 2.5:  # Reasonable bond distances
                            total_angles += 1

                            # Calculate angle
                            cos_angle = jnp.dot(vec1, vec2) / (dist1 * dist2)
                            cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
                            angle_deg = jnp.degrees(jnp.arccos(cos_angle))

                            # Check if angle is reasonable
                            if (
                                self.angle_thresholds["min_angle"]
                                <= angle_deg
                                <= self.angle_thresholds["max_angle"]
                            ):
                                valid_angles += 1

        if total_angles == 0:
            return 1.0  # No angles to validate

        return valid_angles / total_angles

    def _check_molecular_size(self, coordinates: jnp.ndarray) -> float:
        """Check if molecular size is reasonable."""
        if coordinates.shape[0] < 2:
            return 1.0

        # Calculate molecular diameter
        distances = jnp.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        max_distance = jnp.max(distances)

        # Reasonable molecular size (less than 20 Angstroms diameter)
        if max_distance > 20.0:
            return 0.0
        elif max_distance > 15.0:
            return 0.5
        else:
            return 1.0

    def conformational_diversity(
        self,
        coordinates: jnp.ndarray,
        atom_types: jnp.ndarray,
        atom_mask: jnp.ndarray,
        clustering_threshold: float = 0.5,
    ) -> float:
        """Evaluate conformational diversity using RMSD clustering.

        Args:
            coordinates: Molecular coordinates [batch, max_atoms, 3]
            atom_types: Atom types [batch, max_atoms]
            atom_mask: Atom mask [batch, max_atoms]
            clustering_threshold: RMSD threshold for clustering (Angstroms)

        Returns:
            Diversity score (0.0 to 1.0)
        """
        batch_size = coordinates.shape[0]

        # Calculate pairwise RMSD matrix
        rmsd_matrix = _calculate_rmsd_matrix(coordinates, atom_mask)

        # Perform clustering based on RMSD threshold
        clusters = self._cluster_conformations(rmsd_matrix, clustering_threshold)

        # Diversity score is ratio of unique clusters to total conformations
        num_clusters = len(set(clusters))
        diversity_score = num_clusters / batch_size

        return float(diversity_score)

    def _cluster_conformations(self, rmsd_matrix: jnp.ndarray, threshold: float) -> list[int]:
        """Simple clustering based on RMSD threshold."""
        n_conformations = rmsd_matrix.shape[0]
        clusters = [-1] * n_conformations  # -1 means unassigned
        cluster_id = 0

        for i in range(n_conformations):
            if clusters[i] == -1:  # Unassigned
                # Start new cluster
                clusters[i] = cluster_id

                # Find all conformations within threshold
                for j in range(i + 1, n_conformations):
                    if clusters[j] == -1 and rmsd_matrix[i, j] < threshold:
                        clusters[j] = cluster_id

                cluster_id += 1

        return clusters

    def energy_consistency(
        self,
        coordinates: jnp.ndarray,
        atom_types: jnp.ndarray,
        atom_mask: jnp.ndarray,
        reference_energies: jnp.ndarray,
    ) -> float:
        """Evaluate energy consistency using a simple force field approximation.

        Args:
            coordinates: Molecular coordinates [batch, max_atoms, 3]
            atom_types: Atom types [batch, max_atoms]
            atom_mask: Atom mask [batch, max_atoms]
            reference_energies: Reference energies [batch]

        Returns:
            Energy consistency RMSE (lower is better)
        """
        batch_size = coordinates.shape[0]
        predicted_energies = []

        for mol_idx in range(batch_size):
            mol_coords = coordinates[mol_idx]
            mol_types = atom_types[mol_idx]
            mol_mask = atom_mask[mol_idx]

            # Calculate approximate energy using simple force field
            energy = self._approximate_energy(mol_coords, mol_types, mol_mask)
            predicted_energies.append(energy)

        predicted_energies = jnp.array(predicted_energies, dtype=jnp.float32)

        # Calculate RMSE
        energy_rmse = jnp.sqrt(jnp.mean((predicted_energies - reference_energies) ** 2))

        return float(energy_rmse)

    def _approximate_energy(
        self,
        coordinates: jnp.ndarray,
        atom_types: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> float:
        """Approximate molecular energy using simple force field."""
        # Get actual atoms
        num_atoms = jnp.sum(atom_mask).astype(int)
        if num_atoms < 2:
            return 0.0

        valid_coords = coordinates[:num_atoms]
        atom_types[:num_atoms]

        total_energy = 0.0

        # Bond energy contribution
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = jnp.linalg.norm(valid_coords[i] - valid_coords[j])

                # Simple Lennard-Jones-like potential
                if distance < 5.0:  # Reasonable interaction distance
                    # Avoid division by zero
                    distance = jnp.maximum(distance, 0.1)

                    # Simple repulsion/attraction model
                    energy_contribution = 4.0 * ((1.0 / distance) ** 12 - (1.0 / distance) ** 6)
                    total_energy += energy_contribution

        # Add small random component to simulate quantum effects
        quantum_noise = jax.random.normal(self.rngs.params()) * 0.1

        return float(total_energy + quantum_noise)

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute all molecular flows metrics.

        Args:
            real_data: Real molecular data (not used, we evaluate generated)
            generated_data: Generated molecular data dict with keys:
                - coordinates: Molecular coordinates [batch, max_atoms, 3]
                - atom_types: Atom types [batch, max_atoms]
                - atom_mask: Atom mask [batch, max_atoms]
            **kwargs: Additional arguments including reference_energies

        Returns:
            Dictionary containing all computed metrics
        """
        # Extract data from generated_data
        coordinates = generated_data["coordinates"]
        atom_types = generated_data["atom_types"]
        atom_mask = generated_data["atom_mask"]
        reference_energies = kwargs.get("reference_energies")

        results: dict[str, float] = {}

        # Chemical validity
        results["chemical_validity"] = self.chemical_validity(coordinates, atom_types, atom_mask)

        # Conformational diversity
        results["conformational_diversity"] = self.conformational_diversity(
            coordinates, atom_types, atom_mask
        )

        # Energy consistency (if reference energies provided)
        if reference_energies is not None:
            results["energy_consistency"] = self.energy_consistency(
                coordinates, atom_types, atom_mask, reference_energies
            )

        return results

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Real molecular data
            generated_data: Generated molecular data dict

        Returns:
            True if inputs are valid for molecular metrics
        """
        if not isinstance(generated_data, dict):
            return False

        required_keys = {"coordinates", "atom_types", "atom_mask"}
        if not required_keys.issubset(generated_data.keys()):
            return False

        coords = generated_data["coordinates"]
        types = generated_data["atom_types"]
        mask = generated_data["atom_mask"]

        # Check shapes are compatible
        if len(coords.shape) != 3 or coords.shape[-1] != 3:
            return False
        if len(types.shape) != 2 or len(mask.shape) != 2:
            return False
        if coords.shape[:2] != types.shape or coords.shape[:2] != mask.shape:
            return False

        return True
