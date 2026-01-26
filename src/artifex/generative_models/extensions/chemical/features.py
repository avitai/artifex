"""Molecular feature computation and extraction.

This module provides molecular descriptor computation and feature extraction
for chemical property prediction and drug-likeness assessment.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import ModelExtension


class MolecularFeatures(ModelExtension):
    """Molecular feature computation and extraction."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize molecular features module.

        Args:
            config: Extension configuration with feature parameters:
                - weight: Weight for the extension (default: 1.0)
                - enabled: Whether the extension is enabled (default: True)
                - extensions.features.feature_types: Types of features to compute
                - extensions.features.include_3d_features: Whether to include 3D geometry features
            rngs: Random number generator keys
        """
        # Handle configuration
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get feature parameters from extensions field
        feature_params = getattr(config, "extensions", {}).get("features", {})
        self.rngs = rngs

        # Default feature types
        self.feature_types = feature_params.get(
            "feature_types",
            [
                "molecular_weight",
                "lipophilicity",
                "hydrogen_bonds",
                "polar_surface_area",
                "rotatable_bonds",
                "aromatic_rings",
            ],
        )

        self.include_3d_features = feature_params.get("include_3d_features", True)

        # Initialize atomic properties database
        self._initialize_atomic_properties()

    def _initialize_atomic_properties(self):
        """Initialize atomic properties database."""
        # Atomic masses (in atomic mass units)
        self.atomic_masses = {
            1: 1.008,  # H
            6: 12.011,  # C
            7: 14.007,  # N
            8: 15.999,  # O
            9: 18.998,  # F
            15: 30.974,  # P
            16: 32.065,  # S
            17: 35.453,  # Cl
        }

        # Van der Waals radii (in Angstroms)
        self.vdw_radii = {
            1: 1.20,  # H
            6: 1.70,  # C
            7: 1.55,  # N
            8: 1.52,  # O
            9: 1.47,  # F
            15: 1.80,  # P
            16: 1.80,  # S
            17: 1.75,  # Cl
        }

        # Electronegativity values (Pauling scale)
        self.electronegativities = {
            1: 2.20,  # H
            6: 2.55,  # C
            7: 3.04,  # N
            8: 3.44,  # O
            9: 3.98,  # F
            15: 2.19,  # P
            16: 2.58,  # S
            17: 3.16,  # Cl
        }

    def compute_descriptors(self, molecule_data: dict[str, jax.Array]) -> dict[str, float]:
        """Compute molecular descriptors.

        Args:
            molecule_data: Dictionary containing:
                - coordinates: [n_atoms, 3] atomic coordinates
                - atom_types: [n_atoms] atomic numbers
                - bonds: [n_atoms, n_atoms] bond matrix (optional)

        Returns:
            Dictionary of computed molecular descriptors
        """
        descriptors = {}

        coordinates = molecule_data["coordinates"]
        atom_types = molecule_data["atom_types"]
        bonds = molecule_data.get("bonds", None)

        if "molecular_weight" in self.feature_types:
            descriptors["molecular_weight"] = self._compute_molecular_weight(atom_types)

        if "lipophilicity" in self.feature_types:
            descriptors["lipophilicity"] = self._compute_logp(atom_types, bonds)

        if "hydrogen_bonds" in self.feature_types:
            hb_descriptors = self._count_hydrogen_bonds(atom_types, bonds)
            descriptors.update(hb_descriptors)

        if "polar_surface_area" in self.feature_types:
            descriptors["polar_surface_area"] = self._compute_polar_surface_area(atom_types, bonds)

        if "rotatable_bonds" in self.feature_types:
            descriptors["rotatable_bonds"] = self._count_rotatable_bonds(atom_types, bonds)

        if "aromatic_rings" in self.feature_types:
            descriptors["aromatic_rings"] = self._count_aromatic_rings(atom_types, bonds)

        # 3D geometry features
        if self.include_3d_features:
            geometry_descriptors = self._compute_3d_descriptors(coordinates, atom_types)
            descriptors.update(geometry_descriptors)

        return descriptors

    def _compute_molecular_weight(self, atom_types: jax.Array) -> float:
        """Compute molecular weight in atomic mass units."""
        total_weight = 0.0

        for i in range(len(atom_types)):
            atomic_number = int(atom_types[i])
            if atomic_number in self.atomic_masses:
                total_weight += self.atomic_masses[atomic_number]
            else:
                # Default to carbon if unknown
                total_weight += self.atomic_masses[6]

        return float(total_weight)

    def _compute_logp(self, atom_types: jax.Array, bonds: jax.Array | None = None) -> float:
        """Compute lipophilicity (LogP) estimate.

        This is a simplified Ghose-Crippen approach based on atom contributions.
        """
        # Simplified atom contributions to LogP
        logp_contributions = {
            1: -0.23,  # H (polar)
            6: 0.08,  # C (aliphatic)
            7: -0.78,  # N
            8: -0.42,  # O
            9: 0.06,  # F
            17: 0.06,  # Cl
        }

        total_logp: float = 0.0

        for i in range(len(atom_types)):
            atomic_number = int(atom_types[i])
            if atomic_number in logp_contributions:
                total_logp += logp_contributions[atomic_number]

        return float(total_logp)

    def _count_hydrogen_bonds(
        self, atom_types: jax.Array, bonds: jax.Array | None = None
    ) -> dict[str, float]:
        """Count hydrogen bond donors and acceptors."""
        # Hydrogen bond donors (N-H, O-H)
        donors: float = 0.0
        acceptors: float = 0.0

        for i in range(len(atom_types)):
            atomic_number = int(atom_types[i])

            if atomic_number == 7:  # Nitrogen
                # Count as both donor and acceptor (simplified)
                donors += 1.0
                acceptors += 1.0
            elif atomic_number == 8:  # Oxygen
                # Count as both donor and acceptor (simplified)
                donors += 1.0
                acceptors += 1.0

        return {"hydrogen_bond_donors": donors, "hydrogen_bond_acceptors": acceptors}

    def _compute_polar_surface_area(
        self, atom_types: jax.Array, bonds: jax.Array | None = None
    ) -> float:
        """Compute topological polar surface area (TPSA)."""
        # Simplified TPSA calculation based on polar atoms
        tpsa_contributions = {
            7: 23.79,  # N
            8: 23.06,  # O
        }

        total_tpsa: float = 0.0

        for i in range(len(atom_types)):
            atomic_number = int(atom_types[i])
            if atomic_number in tpsa_contributions:
                total_tpsa += tpsa_contributions[atomic_number]

        return float(total_tpsa)

    def _count_rotatable_bonds(
        self, atom_types: jax.Array, bonds: jax.Array | None = None
    ) -> float:
        """Count rotatable bonds (simplified estimate)."""
        if bonds is None:
            return 0.0

        # Count single bonds between heavy atoms (non-ring, non-terminal)
        # This is a simplified approximation

        n_atoms = len(atom_types)
        rotatable: float = 0.0

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if bonds[i, j] > 0:  # There's a bond
                    # Check if both atoms are heavy (not hydrogen)
                    if atom_types[i] > 1 and atom_types[j] > 1:
                        # Simplified: assume it's rotatable if both have >1 connections
                        if jnp.sum(bonds[i]) > 1 and jnp.sum(bonds[j]) > 1:
                            rotatable += 1.0

        return float(rotatable)

    def _count_aromatic_rings(self, atom_types: jax.Array, bonds: jax.Array | None = None) -> float:
        """Count aromatic rings (simplified estimate)."""
        if bonds is None:
            return 0.0

        # Simplified aromatic ring detection
        # Look for 6-membered cycles with all carbon atoms

        len(atom_types)
        aromatic_rings: float = 0.0

        # Very simplified: count carbon atoms that could be in aromatic rings
        carbon_count = jnp.sum(atom_types == 6)

        # Rough estimate: every 6 carbons might form one aromatic ring
        aromatic_rings = float(carbon_count // 6)

        return aromatic_rings

    def _compute_3d_descriptors(
        self, coordinates: jax.Array, atom_types: jax.Array
    ) -> dict[str, float]:
        """Compute 3D geometry-based descriptors."""
        descriptors: dict[str, float] = {}

        # Molecular volume (convex hull approximation)
        descriptors["molecular_volume"] = self._compute_molecular_volume(coordinates)

        # Surface area estimate
        descriptors["molecular_surface_area"] = self._compute_surface_area(coordinates)

        # Shape descriptors
        shape_descriptors = self._compute_shape_descriptors(coordinates)
        descriptors.update(shape_descriptors)

        # Gyration radius
        descriptors["radius_of_gyration"] = self._compute_gyration_radius(coordinates)

        return descriptors

    def _compute_molecular_volume(self, coordinates: jax.Array) -> float:
        """Compute molecular volume using convex hull approximation."""
        if len(coordinates) < 4:
            return 0.0

        # Simple approximation: volume of bounding box
        min_coords = jnp.min(coordinates, axis=0)
        max_coords = jnp.max(coordinates, axis=0)

        dimensions = max_coords - min_coords
        volume = jnp.prod(dimensions)

        return float(volume)

    def _compute_surface_area(self, coordinates: jax.Array) -> float:
        """Compute molecular surface area approximation."""
        if len(coordinates) < 2:
            return 0.0

        # Simple approximation: surface area of bounding box
        min_coords = jnp.min(coordinates, axis=0)
        max_coords = jnp.max(coordinates, axis=0)

        dimensions = max_coords - min_coords
        # Surface area of rectangular box: 2(xy + xz + yz)
        surface_area = 2 * (
            dimensions[0] * dimensions[1]
            + dimensions[0] * dimensions[2]
            + dimensions[1] * dimensions[2]
        )

        return float(surface_area)

    def _compute_shape_descriptors(self, coordinates: jax.Array) -> dict[str, float]:
        """Compute molecular shape descriptors."""
        if len(coordinates) < 2:
            return {"asphericity": 0.0, "eccentricity": 0.0}

        # Center coordinates
        center = jnp.mean(coordinates, axis=0)
        centered_coords = coordinates - center

        # Compute moment of inertia tensor
        I = jnp.zeros((3, 3))
        for i in range(len(centered_coords)):
            r = centered_coords[i]
            r_sq = jnp.dot(r, r)
            I = I + r_sq * jnp.eye(3) - jnp.outer(r, r)

        # Eigenvalues of moment of inertia tensor (symmetric matrix)
        # Use eigvalsh for symmetric matrices to get real eigenvalues
        eigenvals = jnp.linalg.eigvalsh(I)
        # eigvalsh returns sorted eigenvalues in ascending order

        # Shape descriptors
        if eigenvals[2] > 0:
            asphericity: float = eigenvals[2] - 0.5 * (eigenvals[0] + eigenvals[1])
            asphericity = asphericity / eigenvals[2]

            eccentricity = jnp.sqrt(1 - eigenvals[0] / eigenvals[2])
        else:
            asphericity: float = 0.0
            eccentricity: float = 0.0

        return {"asphericity": float(asphericity), "eccentricity": float(eccentricity)}

    def _compute_gyration_radius(self, coordinates: jax.Array) -> float:
        """Compute radius of gyration."""
        if len(coordinates) < 2:
            return 0.0

        # Center of mass (assuming equal masses)
        center = jnp.mean(coordinates, axis=0)

        # Squared distances from center
        distances_sq: jax.Array = jnp.sum((coordinates - center) ** 2, axis=1)

        # Radius of gyration
        rg_sq: jax.Array = jnp.mean(distances_sq)
        rg: jax.Array = jnp.sqrt(rg_sq)

        return float(rg)

    def compute_drug_likeness_score(self, descriptors: dict[str, float]) -> float:
        """Compute drug-likeness score based on Lipinski's Rule of Five."""
        violations: int = 0

        # Rule of Five criteria
        if descriptors.get("molecular_weight", 0) > 500:
            violations += 1

        if descriptors.get("lipophilicity", 0) > 5:
            violations += 1

        if descriptors.get("hydrogen_bond_donors", 0) > 5:
            violations += 1

        if descriptors.get("hydrogen_bond_acceptors", 0) > 10:
            violations += 1

        # Score: 1.0 - (violations / 4)
        drug_likeness: float = 1.0 - (violations / 4.0)

        return float(jnp.maximum(0.0, drug_likeness))

    def extract_fingerprint(
        self, molecule_data: dict[str, jax.Array], fingerprint_size: int = 1024
    ) -> jax.Array:
        """Extract molecular fingerprint for similarity comparisons."""
        molecule_data["coordinates"]
        atom_types = molecule_data["atom_types"]
        bonds: jax.Array | None = molecule_data.get("bonds", None)

        # Simple fingerprint based on atom counts and basic properties
        fingerprint: jax.Array = jnp.zeros(fingerprint_size)

        # Atom type counts (first 118 bits for elements)
        for i in range(min(len(atom_types), 118)):
            atomic_number = int(atom_types[i])
            if atomic_number < fingerprint_size:
                fingerprint: jax.Array = fingerprint.at[atomic_number].add(1)

        # Add some structural features
        if bonds is not None and fingerprint_size > 150:
            # Bond counts
            total_bonds: float = jnp.sum(bonds) / 2  # Divide by 2 for symmetric matrix
            fingerprint: jax.Array = fingerprint.at[120].set(total_bonds / 100.0)  # Normalized

            # Ring features (simplified)
            if fingerprint_size > 130:
                ring_estimate: float = self._count_aromatic_rings(atom_types, bonds)
                fingerprint: jax.Array = fingerprint.at[130].set(ring_estimate / 10.0)

        # Normalize fingerprint
        norm: float = jnp.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint: jax.Array = fingerprint / norm

        return fingerprint

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension outputs including computed features.
        """
        if not self.enabled:
            return {"extension_type": "molecular_features"}

        # Extract molecule data from model outputs
        if isinstance(model_outputs, dict):
            molecule_data = {
                "coordinates": model_outputs.get("coordinates", model_outputs.get("positions")),
                "atom_types": model_outputs.get(
                    "atom_types",
                    jnp.zeros(
                        model_outputs.get("coordinates", model_outputs.get("positions")).shape[0],
                        dtype=jnp.int32,
                    ),
                ),
                "bonds": model_outputs.get("bonds"),
            }
        else:
            # Assume model_outputs is coordinates directly
            molecule_data = {
                "coordinates": model_outputs,
                "atom_types": inputs.get(
                    "atom_types", jnp.zeros(model_outputs.shape[0], dtype=jnp.int32)
                )
                if isinstance(inputs, dict)
                else jnp.zeros(model_outputs.shape[0], dtype=jnp.int32),
                "bonds": inputs.get("bonds") if isinstance(inputs, dict) else None,
            }

        # Compute descriptors
        descriptors = self.compute_descriptors(molecule_data)

        # Compute drug-likeness score
        drug_likeness = self.compute_drug_likeness_score(descriptors)

        # Extract fingerprint
        fingerprint = self.extract_fingerprint(molecule_data)

        return {
            "descriptors": descriptors,
            "drug_likeness_score": drug_likeness,
            "molecular_fingerprint": fingerprint,
            "extension_type": "molecular_features",
        }
