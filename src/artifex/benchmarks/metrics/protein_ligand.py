"""Protein-ligand co-design metrics.

This module provides metrics for evaluating protein-ligand co-design models,
including binding affinity prediction and molecular validity assessment.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx


# Note: MetricProtocol will be defined later, using ABC for now


class BindingAffinityMetric(nnx.Module):
    """Metric for evaluating binding affinity prediction accuracy.

    This metric computes RMSE between predicted and experimental binding affinities,
    which is the primary target for Week 5-8 (RMSE <1.0 kcal/mol).
    """

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize binding affinity metric.

        Args:
            rngs: Random number generator keys
        """
        super().__init__()

    def compute(
        self, predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
    ) -> dict[str, float | int]:
        """Compute binding affinity metrics.

        Args:
            predictions: Predicted binding affinities (batch_size,)
            targets: True binding affinities (batch_size,)
            **kwargs: Additional parameters

        Returns:
            dictionary with metrics:
                - rmse: Root mean square error (kcal/mol)
                - mae: Mean absolute error (kcal/mol)
                - r2: R-squared correlation coefficient
                - pearson_r: Pearson correlation coefficient
        """
        # Compute errors
        errors = predictions - targets
        squared_errors = jnp.square(errors)
        abs_errors = jnp.abs(errors)

        # RMSE (primary target metric)
        rmse = jnp.sqrt(jnp.mean(squared_errors))

        # MAE
        mae = jnp.mean(abs_errors)

        # R-squared
        ss_res = jnp.sum(squared_errors)
        ss_tot = jnp.sum(jnp.square(targets - jnp.mean(targets)))
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add epsilon for numerical stability

        # Pearson correlation
        mean_pred = jnp.mean(predictions)
        mean_true = jnp.mean(targets)

        numerator = jnp.sum((predictions - mean_pred) * (targets - mean_true))
        denominator = jnp.sqrt(
            jnp.sum(jnp.square(predictions - mean_pred)) * jnp.sum(jnp.square(targets - mean_true))
        )
        pearson_r = numerator / (denominator + 1e-8)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "pearson_r": float(pearson_r),
        }


class MolecularValidityMetric(nnx.Module):
    """Metric for evaluating molecular validity of generated structures.

    This metric assesses chemical validity of generated molecules,
    targeting >95% validity as specified in Week 5-8 requirements.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize molecular validity metric.

        Args:
            rngs: Random number generator keys
        """
        super().__init__()

    def compute(
        self,
        coordinates: jnp.ndarray,
        atom_types: jnp.ndarray,
        masks: jnp.ndarray | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute molecular validity metrics.

        Args:
            coordinates: Molecular coordinates (batch_size, num_atoms, 3)
            atom_types: Atom types (batch_size, num_atoms)
            masks: Validity masks (batch_size, num_atoms)
            **kwargs: Additional parameters

        Returns:
            dictionary with validity metrics:
                - validity_rate: Fraction of valid molecules
                - bond_validity: Fraction with valid bond lengths
                - angle_validity: Fraction with valid bond angles
                - clash_free: Fraction without atomic clashes
        """
        batch_size = coordinates.shape[0]

        # Initialize validity arrays
        valid_molecules = jnp.zeros(batch_size, dtype=jnp.bool_)
        valid_bonds = jnp.zeros(batch_size, dtype=jnp.bool_)
        valid_angles = jnp.zeros(batch_size, dtype=jnp.bool_)
        clash_free = jnp.zeros(batch_size, dtype=jnp.bool_)

        # Check each molecule in the batch
        for i in range(batch_size):
            mol_coords = coordinates[i]
            mol_types = atom_types[i]

            # Ensure mask length matches coordinate length
            if masks is not None:
                mol_mask = masks[i]
                # Truncate or pad mask to match coordinate length
                coord_len = len(mol_coords)
                mask_len = len(mol_mask)
                if mask_len > coord_len:
                    mol_mask = mol_mask[:coord_len]
                elif mask_len < coord_len:
                    # Pad with False for the extra coordinates
                    padding = jnp.zeros(coord_len - mask_len, dtype=jnp.bool_)
                    mol_mask = jnp.concatenate([mol_mask, padding])
            else:
                mol_mask = jnp.ones(len(mol_coords), dtype=jnp.bool_)

            # Apply mask to get actual atoms
            actual_coords = mol_coords[mol_mask]
            actual_types = mol_types[mol_mask]

            if len(actual_coords) == 0:
                continue

            # Check bond lengths
            bond_valid = self._check_bond_validity(actual_coords, actual_types)
            valid_bonds = valid_bonds.at[i].set(bond_valid)

            # Check bond angles
            angle_valid = self._check_angle_validity(actual_coords, actual_types)
            valid_angles = valid_angles.at[i].set(angle_valid)

            # Check for atomic clashes
            no_clashes = self._check_no_clashes(actual_coords, actual_types)
            clash_free = clash_free.at[i].set(no_clashes)

            # Overall validity requires all checks to pass
            overall_valid = bond_valid and angle_valid and no_clashes
            valid_molecules = valid_molecules.at[i].set(overall_valid)

        return {
            "validity_rate": float(jnp.mean(valid_molecules)),
            "bond_validity": float(jnp.mean(valid_bonds)),
            "angle_validity": float(jnp.mean(valid_angles)),
            "clash_free": float(jnp.mean(clash_free)),
        }

    def _check_bond_validity(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check if bond lengths are within reasonable ranges.

        Args:
            coordinates: Atom coordinates (num_atoms, 3)
            atom_types: Atom types (num_atoms,)

        Returns:
            True if bond lengths are valid
        """
        # Compute pairwise distances
        num_atoms = len(coordinates)
        if num_atoms < 2:
            return True

        # Compute distance matrix
        dist_matrix = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)

        # Mask diagonal (self-distances)
        mask = ~jnp.eye(num_atoms, dtype=jnp.bool_)
        distances = dist_matrix[mask]

        # Simple bond length validation (in Angstroms)
        # Typical covalent bonds: 0.7-3.0 Angstroms
        min_bond_length = 0.7
        max_bond_length = 3.0

        # Find potential bonds (atoms closer than max_bond_length)
        bond_mask = distances < max_bond_length
        bond_distances = distances[bond_mask]

        if len(bond_distances) == 0:
            return True  # No bonds found

        # Check if all bond distances are within valid range
        valid_bonds = (bond_distances >= min_bond_length) & (bond_distances <= max_bond_length)
        return bool(jnp.all(valid_bonds))

    def _check_angle_validity(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check if bond angles are within reasonable ranges.

        Args:
            coordinates: Atom coordinates (num_atoms, 3)
            atom_types: Atom types (num_atoms,)

        Returns:
            True if bond angles are valid
        """
        # Simplified angle validation
        # In a real implementation, this would check specific angle constraints
        # based on atom types and molecular topology

        num_atoms = len(coordinates)
        if num_atoms < 3:
            return True

        # For now, just check that atoms aren't collinear (angle not ~0 or ~180)
        # This is a basic geometric constraint

        # Sample some triplets of atoms
        max_triplets = min(10, num_atoms * (num_atoms - 1) * (num_atoms - 2) // 6)

        for i in range(0, num_atoms - 2, max(1, num_atoms // max_triplets)):
            for j in range(i + 1, num_atoms - 1, max(1, num_atoms // max_triplets)):
                for k in range(j + 1, num_atoms, max(1, num_atoms // max_triplets)):
                    # Compute angle at atom j
                    v1 = coordinates[i] - coordinates[j]
                    v2 = coordinates[k] - coordinates[j]

                    # Normalize vectors
                    v1_norm = jnp.linalg.norm(v1)
                    v2_norm = jnp.linalg.norm(v2)

                    if v1_norm < 1e-6 or v2_norm < 1e-6:
                        continue

                    v1_unit = v1 / v1_norm
                    v2_unit = v2 / v2_norm

                    # Compute angle
                    cos_angle = jnp.clip(jnp.dot(v1_unit, v2_unit), -1.0, 1.0)
                    angle = jnp.arccos(cos_angle)

                    # Check if angle is reasonable (not too close to 0 or π)
                    min_angle = 0.2  # ~11 degrees
                    max_angle = jnp.pi - 0.2  # ~169 degrees

                    if angle < min_angle or angle > max_angle:
                        return False

        return True

    def _check_no_clashes(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check for atomic clashes (atoms too close together).

        Args:
            coordinates: Atom coordinates (num_atoms, 3)
            atom_types: Atom types (num_atoms,)

        Returns:
            True if no atomic clashes are detected
        """
        num_atoms = len(coordinates)
        if num_atoms < 2:
            return True

        # Compute pairwise distances
        dist_matrix = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)

        # Mask diagonal
        mask = ~jnp.eye(num_atoms, dtype=jnp.bool_)
        distances = dist_matrix[mask]

        # Check for clashes (atoms too close)
        # Typical van der Waals radii: ~0.5-2.0 Angstroms
        # Minimum allowed distance: 0.5 Angstroms
        min_distance = 0.5

        clashes = distances < min_distance
        return not jnp.any(clashes)


class DrugLikenessMetric(nnx.Module):
    """Metric for evaluating drug-likeness of generated ligands.

    This metric assesses pharmaceutical properties like QED score,
    targeting >0.7 QED as specified in the benchmark requirements.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize drug-likeness metric.

        Args:
            rngs: Random number generator keys
        """
        super().__init__()

    def compute(
        self,
        coordinates: jnp.ndarray,
        atom_types: jnp.ndarray,
        masks: jnp.ndarray | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute drug-likeness metrics.

        Args:
            coordinates: Ligand coordinates (batch_size, num_atoms, 3)
            atom_types: Atom types (batch_size, num_atoms)
            masks: Validity masks (batch_size, num_atoms)
            **kwargs: Additional parameters

        Returns:
            dictionary with drug-likeness metrics:
                - qed_score: Quantitative Estimate of Drug-likeness
                - lipinski_compliance: Fraction following Lipinski's rule
                - molecular_weight: Average molecular weight
                - num_rotatable_bonds: Average number of rotatable bonds
        """
        batch_size = coordinates.shape[0]

        qed_scores = []
        lipinski_compliant = []
        mol_weights = []
        rotatable_bonds = []

        for i in range(batch_size):
            mol_coords = coordinates[i]
            mol_types = atom_types[i]

            # Ensure mask length matches coordinate length
            if masks is not None:
                mol_mask = masks[i]
                # Truncate or pad mask to match coordinate length
                coord_len = len(mol_coords)
                mask_len = len(mol_mask)
                if mask_len > coord_len:
                    mol_mask = mol_mask[:coord_len]
                elif mask_len < coord_len:
                    # Pad with False for the extra coordinates
                    padding = jnp.zeros(coord_len - mask_len, dtype=jnp.bool_)
                    mol_mask = jnp.concatenate([mol_mask, padding])
            else:
                mol_mask = jnp.ones(len(mol_coords), dtype=jnp.bool_)

            # Apply mask
            actual_coords = mol_coords[mol_mask]
            actual_types = mol_types[mol_mask]

            if len(actual_coords) == 0:
                continue

            # Compute mock drug-likeness properties
            # In a real implementation, this would use cheminformatics libraries

            # Mock QED score (0-1, higher is better)
            mock_qed = self._compute_mock_qed(actual_coords, actual_types)
            qed_scores.append(mock_qed)

            # Mock Lipinski compliance
            mock_lipinski = self._check_lipinski_compliance(actual_coords, actual_types)
            lipinski_compliant.append(mock_lipinski)

            # Mock molecular weight (based on number of atoms)
            mock_mw = len(actual_coords) * 12.0  # Rough estimate
            mol_weights.append(mock_mw)

            # Mock rotatable bonds
            mock_rot_bonds = max(0, len(actual_coords) - 10)  # Rough estimate
            rotatable_bonds.append(mock_rot_bonds)

        if not qed_scores:
            return {
                "qed_score": 0.0,
                "lipinski_compliance": 0.0,
                "molecular_weight": 0.0,
                "num_rotatable_bonds": 0.0,
            }

        return {
            "qed_score": float(np.mean(qed_scores)),
            "lipinski_compliance": float(np.mean(lipinski_compliant)),
            "molecular_weight": float(np.mean(mol_weights)),
            "num_rotatable_bonds": float(np.mean(rotatable_bonds)),
        }

    def _compute_mock_qed(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> float:
        """Compute mock QED score based on basic properties.

        Args:
            coordinates: Atom coordinates (num_atoms, 3)
            atom_types: Atom types (num_atoms,)

        Returns:
            Mock QED score (0-1)
        """
        # Mock QED computation based on simple properties
        num_atoms = len(coordinates)

        # Prefer molecules with 10-50 atoms (drug-like size)
        size_penalty = 1.0
        if num_atoms < 10:
            size_penalty = num_atoms / 10.0
        elif num_atoms > 50:
            size_penalty = 50.0 / num_atoms

        # Mock complexity score based on coordinate variance
        coord_var = float(jnp.var(coordinates))
        complexity_score = jnp.clip(coord_var / 10.0, 0.0, 1.0)

        # Combine factors
        mock_qed = size_penalty * complexity_score * 0.8  # Base score
        return float(jnp.clip(mock_qed, 0.0, 1.0))

    def _check_lipinski_compliance(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check mock Lipinski rule compliance.

        Args:
            coordinates: Atom coordinates (num_atoms, 3)
            atom_types: Atom types (num_atoms,)

        Returns:
            True if molecule passes mock Lipinski checks
        """
        num_atoms = len(coordinates)

        # Mock Lipinski rule checks
        # Real implementation would compute MW, LogP, HBD, HBA

        # Molecular weight < 500 Da (roughly < 40 atoms)
        mw_ok = num_atoms < 40

        # LogP < 5 (mock: based on coordinate spread)
        coord_spread = float(jnp.max(coordinates) - jnp.min(coordinates))
        logp_ok = coord_spread < 15.0

        # HBD < 5 and HBA < 10 (mock: based on atom count)
        hb_ok = num_atoms < 30  # Rough approximation

        return mw_ok and logp_ok and hb_ok
