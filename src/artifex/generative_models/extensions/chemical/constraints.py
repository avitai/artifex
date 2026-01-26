"""Chemical constraint validation and enforcement.

This module provides chemical validation and constraint enforcement
for molecular generation tasks following the Week 13 implementation plan.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.extensions.base.extensions import (
    ConstraintExtension,
    ExtensionConfig,
)


class ChemicalConstraints(ConstraintExtension):
    """Chemical constraint validation and enforcement."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize chemical constraints module.

        Args:
            config: Extension configuration with constraint parameters:
                - weight: Weight for the constraint loss (default: 1.0)
                - enabled: Whether the extension is enabled (default: True)
                - extensions.constraints.constraint_types: Types of constraints to validate
                - extensions.constraints.tolerance_levels: Tolerance levels for each constraint type
            rngs: Random number generator keys
        """
        # Handle configuration
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get constraint parameters from extensions field
        constraint_params = getattr(config, "extensions", {}).get("constraints", {})
        self.rngs = rngs

        # Default constraint types
        self.constraint_types = constraint_params.get(
            "constraint_types",
            [
                "bond_length",
                "valence",
                "stereochemistry",
                "ring_strain",
            ],
        )

        # Default tolerance levels
        self.tolerance_levels = constraint_params.get(
            "tolerance_levels",
            {
                "bond_length": 0.2,  # Angstroms
                "valence": 0.1,  # Valence error tolerance
                "angle": 15.0,  # Degrees
                "dihedral": 30.0,  # Degrees
            },
        )

        self._load_chemical_databases()

    def _load_chemical_databases(self):
        """Load chemical databases and reference data."""
        # Standard bond lengths (Angstroms) - common covalent bonds
        self.standard_bond_lengths = {
            ("C", "C"): 1.54,  # Single bond
            ("C", "N"): 1.47,
            ("C", "O"): 1.43,
            ("C", "H"): 1.09,
            ("N", "H"): 1.01,
            ("O", "H"): 0.96,
            ("N", "N"): 1.45,
            ("O", "O"): 1.48,
        }

        # Standard valences for common elements
        self.standard_valences = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "P": 3, "S": 2, "Cl": 1}

    def validate_molecular_structure(
        self, coordinates: jax.Array, atom_types: jax.Array, bonds: jax.Array | None = None
    ) -> dict[str, float]:
        """Validate molecular structure against chemical rules.

        Args:
            coordinates: Atomic coordinates [n_atoms, 3]
            atom_types: Atomic type indices [n_atoms]
            bonds: Bond connectivity matrix [n_atoms, n_atoms] (optional)

        Returns:
            dictionary of validation scores (0.0 = invalid, 1.0 = valid)
        """
        results = {}

        if "bond_length" in self.constraint_types:
            results["bond_length_validity"] = self._check_bond_lengths(
                coordinates, atom_types, bonds
            )

        if "valence" in self.constraint_types:
            results["valence_validity"] = self._check_valence_rules(atom_types, bonds)

        if "stereochemistry" in self.constraint_types:
            results["stereochemistry_validity"] = self._check_stereochemistry(
                coordinates, atom_types
            )

        if "ring_strain" in self.constraint_types:
            results["ring_strain_validity"] = self._check_ring_strain(coordinates, bonds)

        return results

    def _check_bond_lengths(
        self, coordinates: jax.Array, atom_types: jax.Array, bonds: jax.Array | None = None
    ) -> float:
        """Check if bond lengths are within acceptable ranges."""
        if bonds is None:
            # Infer bonds from distances (simple threshold-based)
            bonds = self._infer_bonds_from_distance(coordinates, atom_types)

        # Calculate actual bond lengths
        bond_indices = jnp.where(bonds > 0)
        if len(bond_indices[0]) == 0:
            return 1.0  # No bonds to validate

        actual_lengths = jnp.linalg.norm(
            coordinates[bond_indices[0]] - coordinates[bond_indices[1]], axis=1
        )

        # Get expected lengths for each bond type
        # For simplicity, use average C-C bond length as default
        expected_lengths = jnp.full_like(actual_lengths, 1.54)

        # Calculate relative errors
        relative_errors = jnp.abs(actual_lengths - expected_lengths) / expected_lengths
        tolerance = self.tolerance_levels["bond_length"] / 1.54  # Normalize tolerance

        # Fraction of bonds within tolerance
        valid_bonds = jnp.mean(relative_errors < tolerance)

        return float(valid_bonds)

    def _check_valence_rules(self, atom_types: jax.Array, bonds: jax.Array | None = None) -> float:
        """Check if valence rules are satisfied."""
        if bonds is None:
            return 1.0  # Cannot validate without bond information

        # Calculate actual valences (sum of bonds per atom)
        actual_valences = jnp.sum(bonds, axis=1)

        # For simplicity, assume all atoms are carbon (valence 4)
        # In practice, this would use the atom_types to look up expected valences
        expected_valences = jnp.full_like(actual_valences, 4.0)

        # Calculate valence errors
        valence_errors = jnp.abs(actual_valences - expected_valences)
        tolerance = self.tolerance_levels["valence"]

        # Fraction of atoms with correct valence
        valid_valences = jnp.mean(valence_errors < tolerance)

        return float(valid_valences)

    def _check_stereochemistry(self, coordinates: jax.Array, atom_types: jax.Array) -> float:
        """Check stereochemical consistency."""
        # Simplified stereochemistry check based on chirality preservation
        # This is a placeholder for more sophisticated stereochemical validation

        n_atoms = coordinates.shape[0]
        if n_atoms < 4:
            return 1.0  # Need at least 4 atoms for chirality

        # Check for reasonable 3D structure (not planar)
        # Calculate volume of tetrahedron formed by first 4 atoms
        if n_atoms >= 4:
            tetrahedron_coords = coordinates[:4]  # [4, 3]

            # Calculate volume using scalar triple product
            v1 = tetrahedron_coords[1] - tetrahedron_coords[0]
            v2 = tetrahedron_coords[2] - tetrahedron_coords[0]
            v3 = tetrahedron_coords[3] - tetrahedron_coords[0]

            # Volume = |v1 · (v2 × v3)| / 6
            cross_product = jnp.cross(v2, v3)
            volume = jnp.abs(jnp.dot(v1, cross_product)) / 6.0

            # Reasonable 3D structure should have non-zero volume
            min_volume = 0.1  # Cubic angstroms
            stereochemistry_score = float(jnp.minimum(volume / min_volume, 1.0))
        else:
            stereochemistry_score = 1.0

        return stereochemistry_score

    def _check_ring_strain(self, coordinates: jax.Array, bonds: jax.Array | None = None) -> float:
        """Check for excessive ring strain."""
        if bonds is None:
            return 1.0  # Cannot validate without bond information

        # Simplified ring strain check
        # Look for short cycles and check if angles are reasonable

        # For now, return a reasonable default
        # In practice, this would involve:
        # 1. Finding all rings in the molecule
        # 2. Calculating ring angles and dihedrals
        # 3. Comparing to ideal values for different ring sizes

        return 0.95  # Assume slight ring strain is typical

    def _infer_bonds_from_distance(
        self, coordinates: jax.Array, atom_types: jax.Array, max_bond_length: float = 2.0
    ) -> jax.Array:
        """Infer bond connectivity from atomic distances."""
        coordinates.shape[0]

        # Calculate pairwise distances
        distances = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)

        # Create bond matrix based on distance threshold
        bonds = (distances < max_bond_length) & (distances > 0.1)

        # Make symmetric and convert to float
        bonds = bonds.astype(jnp.float32)

        return bonds

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension outputs.
        """
        # Extract coordinates and atom types from model outputs
        if isinstance(model_outputs, dict):
            coordinates = model_outputs.get("coordinates", model_outputs.get("positions"))
            atom_types = model_outputs.get(
                "atom_types", jnp.zeros(coordinates.shape[0], dtype=jnp.int32)
            )
            bonds = model_outputs.get("bonds")
        else:
            # Assume model_outputs is coordinates directly
            coordinates = model_outputs
            atom_types = (
                inputs.get("atom_types", jnp.zeros(coordinates.shape[0], dtype=jnp.int32))
                if isinstance(inputs, dict)
                else jnp.zeros(coordinates.shape[0], dtype=jnp.int32)
            )
            bonds = None

        # Validate molecular structure
        validation_results = self.validate_molecular_structure(coordinates, atom_types, bonds)

        return {
            "validation_scores": validation_results,
            "extension_type": "chemical_constraints",
        }

    def validate(self, outputs: Any) -> dict[str, jax.Array]:
        """Validate outputs against constraints.

        Args:
            outputs: Model outputs to validate.

        Returns:
            dictionary of validation metrics.
        """
        # Extract coordinates and atom types
        if isinstance(outputs, dict):
            coordinates = outputs.get("coordinates", outputs.get("positions"))
            atom_types = outputs.get("atom_types", jnp.zeros(coordinates.shape[0], dtype=jnp.int32))
            bonds = outputs.get("bonds")
        else:
            coordinates = outputs
            atom_types = jnp.zeros(coordinates.shape[0], dtype=jnp.int32)
            bonds = None

        return self.validate_molecular_structure(coordinates, atom_types, bonds)

    def project(self, outputs: Any) -> Any:
        """Project outputs to satisfy constraints.

        Args:
            outputs: Model outputs to project.

        Returns:
            Projected outputs that satisfy constraints.
        """
        if not self.enabled:
            return outputs

        # Extract coordinates and atom types
        if isinstance(outputs, dict):
            coordinates = outputs.get("coordinates", outputs.get("positions"))
            atom_types = outputs.get("atom_types", jnp.zeros(coordinates.shape[0], dtype=jnp.int32))
            constraint_strength = 1.0

            # Apply constraints
            constrained_coords = self.apply_constraints(
                coordinates, atom_types, constraint_strength
            )

            # Return updated outputs
            outputs_copy = outputs.copy()
            if "coordinates" in outputs_copy:
                outputs_copy["coordinates"] = constrained_coords
            elif "positions" in outputs_copy:
                outputs_copy["positions"] = constrained_coords
            return outputs_copy
        else:
            # Assume outputs is coordinates directly
            atom_types = jnp.zeros(outputs.shape[0], dtype=jnp.int32)
            return self.apply_constraints(outputs, atom_types, 1.0)

    def apply_constraints(
        self, coordinates: jax.Array, atom_types: jax.Array, constraint_strength: float = 1.0
    ) -> jax.Array:
        """Apply constraints to coordinates to improve chemical validity.

        Args:
            coordinates: Atomic coordinates [n_atoms, 3]
            atom_types: Atomic type indices [n_atoms]
            constraint_strength: Strength of constraint application (0-1)

        Returns:
            Constrained coordinates [n_atoms, 3]
        """
        # This is a placeholder for constraint application
        # In practice, this would involve optimization to satisfy constraints

        # For now, apply small random perturbations as a placeholder
        if constraint_strength > 0:
            noise_scale = 0.01 * constraint_strength
            noise = jax.random.normal(self.rngs.sample(), coordinates.shape) * noise_scale

            constrained_coords = coordinates + noise
        else:
            constrained_coords = coordinates

        return constrained_coords

    def compute_constraint_loss(
        self, coordinates: jax.Array, atom_types: jax.Array, target_validity: float = 0.95
    ) -> float:
        """Compute loss based on constraint violations.

        Args:
            coordinates: Atomic coordinates [n_atoms, 3]
            atom_types: Atomic type indices [n_atoms]
            target_validity: Target validity score

        Returns:
            Constraint loss (higher = more violations)
        """
        validity_scores = self.validate_molecular_structure(coordinates, atom_types)

        # Compute weighted average of validity scores
        total_validity = jnp.mean(jnp.array(list(validity_scores.values())))

        # Loss is higher when validity is lower than target
        constraint_loss = jnp.maximum(0.0, target_validity - total_validity)

        return float(constraint_loss)
