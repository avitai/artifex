"""Protein-ligand co-design metrics.

This module provides metrics for evaluating protein-ligand co-design models,
including binding affinity prediction and molecular validity assessment.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from artifex.benchmarks.metrics.core import _init_metric_from_config, MetricBase
from artifex.benchmarks.runtime_guards import demo_mode_from_mapping, require_demo_mode
from artifex.generative_models.core.configuration import EvaluationConfig


class BindingAffinityMetric(MetricBase):
    """Metric for evaluating binding affinity prediction accuracy.

    Computes RMSE between predicted and experimental binding affinities.
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize binding affinity metric.

        Args:
            rngs: Random number generator keys
            config: Evaluation configuration
        """
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="binding_affinity",
            modality="protein_ligand",
            higher_is_better=False,
        )

    def compute(self, predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs) -> dict[str, float]:
        """Compute binding affinity metrics.

        Args:
            predictions: Predicted binding affinities (batch_size,)
            targets: True binding affinities (batch_size,)
            **kwargs: Additional parameters

        Returns:
            Dictionary with rmse, mae, r2, pearson_r
        """
        errors = predictions - targets
        squared_errors = jnp.square(errors)
        abs_errors = jnp.abs(errors)

        rmse = jnp.sqrt(jnp.mean(squared_errors))
        mae = jnp.mean(abs_errors)

        ss_res = jnp.sum(squared_errors)
        ss_tot = jnp.sum(jnp.square(targets - jnp.mean(targets)))
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

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

    def validate_inputs(self, predictions, targets) -> None:
        """Validate input data compatibility.

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(predictions, jnp.ndarray) or not isinstance(targets, jnp.ndarray):
            raise ValueError("Both inputs must be jax arrays")
        if predictions.shape != targets.shape:
            raise ValueError("Input shapes must match")
        if predictions.ndim != 1:
            raise ValueError("Inputs must be 1D arrays")


class MolecularValidityMetric(MetricBase):
    """Metric for evaluating molecular validity of generated structures.

    Assesses chemical validity of generated molecules.
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize molecular validity metric.

        Args:
            rngs: Random number generator keys
            config: Evaluation configuration
        """
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="molecular_validity",
            modality="protein_ligand",
            higher_is_better=True,
        )

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
            Dictionary with validity_rate, bond_validity, angle_validity, clash_free
        """
        batch_size = coordinates.shape[0]

        valid_molecules = jnp.zeros(batch_size, dtype=jnp.bool_)
        valid_bonds = jnp.zeros(batch_size, dtype=jnp.bool_)
        valid_angles = jnp.zeros(batch_size, dtype=jnp.bool_)
        clash_free = jnp.zeros(batch_size, dtype=jnp.bool_)

        for i in range(batch_size):
            mol_coords = coordinates[i]
            mol_types = atom_types[i]

            if masks is not None:
                mol_mask = masks[i]
                coord_len = len(mol_coords)
                mask_len = len(mol_mask)
                if mask_len > coord_len:
                    mol_mask = mol_mask[:coord_len]
                elif mask_len < coord_len:
                    padding = jnp.zeros(coord_len - mask_len, dtype=jnp.bool_)
                    mol_mask = jnp.concatenate([mol_mask, padding])
            else:
                mol_mask = jnp.ones(len(mol_coords), dtype=jnp.bool_)

            actual_coords = mol_coords[mol_mask]
            actual_types = mol_types[mol_mask]

            if len(actual_coords) == 0:
                continue

            bond_valid = self._check_bond_validity(actual_coords, actual_types)
            valid_bonds = valid_bonds.at[i].set(bond_valid)

            angle_valid = self._check_angle_validity(actual_coords, actual_types)
            valid_angles = valid_angles.at[i].set(angle_valid)

            no_clashes = self._check_no_clashes(actual_coords, actual_types)
            clash_free = clash_free.at[i].set(no_clashes)

            overall_valid = bond_valid and angle_valid and no_clashes
            valid_molecules = valid_molecules.at[i].set(overall_valid)

        return {
            "validity_rate": float(jnp.mean(valid_molecules)),
            "bond_validity": float(jnp.mean(valid_bonds)),
            "angle_validity": float(jnp.mean(valid_angles)),
            "clash_free": float(jnp.mean(clash_free)),
        }

    def validate_inputs(self, coordinates, atom_types) -> None:
        """Validate input data compatibility.

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(coordinates, jnp.ndarray):
            raise ValueError("Coordinates must be a jax array")
        if not isinstance(atom_types, jnp.ndarray):
            raise ValueError("Atom types must be a jax array")
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("Coordinates must be 3D with last dim = 3")

    def _check_bond_validity(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check if bond lengths are within reasonable ranges."""
        num_atoms = len(coordinates)
        if num_atoms < 2:
            return True

        dist_matrix = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        mask = ~jnp.eye(num_atoms, dtype=jnp.bool_)
        distances = dist_matrix[mask]

        min_bond_length = 0.7
        max_bond_length = 3.0

        bond_mask = distances < max_bond_length
        bond_distances = distances[bond_mask]

        if len(bond_distances) == 0:
            return True

        valid_bonds = (bond_distances >= min_bond_length) & (bond_distances <= max_bond_length)
        return bool(jnp.all(valid_bonds))

    def _check_angle_validity(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check if bond angles are within reasonable ranges."""
        num_atoms = len(coordinates)
        if num_atoms < 3:
            return True

        max_triplets = min(10, num_atoms * (num_atoms - 1) * (num_atoms - 2) // 6)

        for i in range(0, num_atoms - 2, max(1, num_atoms // max_triplets)):
            for j in range(i + 1, num_atoms - 1, max(1, num_atoms // max_triplets)):
                for k in range(j + 1, num_atoms, max(1, num_atoms // max_triplets)):
                    v1 = coordinates[i] - coordinates[j]
                    v2 = coordinates[k] - coordinates[j]

                    v1_norm = jnp.linalg.norm(v1)
                    v2_norm = jnp.linalg.norm(v2)

                    if v1_norm < 1e-6 or v2_norm < 1e-6:
                        continue

                    v1_unit = v1 / v1_norm
                    v2_unit = v2 / v2_norm

                    cos_angle = jnp.clip(jnp.dot(v1_unit, v2_unit), -1.0, 1.0)
                    angle = jnp.arccos(cos_angle)

                    min_angle = 0.2
                    max_angle = jnp.pi - 0.2

                    if angle < min_angle or angle > max_angle:
                        return False

        return True

    def _check_no_clashes(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check for atomic clashes (atoms too close together)."""
        num_atoms = len(coordinates)
        if num_atoms < 2:
            return True

        dist_matrix = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        mask = ~jnp.eye(num_atoms, dtype=jnp.bool_)
        distances = dist_matrix[mask]

        min_distance = 0.5
        clashes = distances < min_distance
        return not jnp.any(clashes)


class DrugLikenessMetric(MetricBase):
    """Metric for evaluating drug-likeness of generated ligands.

    Assesses pharmaceutical properties like QED score.
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize drug-likeness metric.

        Args:
            rngs: Random number generator keys
            config: Evaluation configuration
        """
        metric_params = _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="drug_likeness",
            modality="protein_ligand",
            higher_is_better=True,
        )
        self.demo_mode = demo_mode_from_mapping(metric_params)
        require_demo_mode(
            enabled=self.demo_mode,
            component="DrugLikenessMetric",
            detail=(
                "This retained drug-likeness path still uses mock QED and Lipinski heuristics "
                "instead of a benchmark-grade chemistry backend."
            ),
        )

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
            Dictionary with qed_score, lipinski_compliance, molecular_weight, num_rotatable_bonds
        """
        batch_size = coordinates.shape[0]

        qed_scores = []
        lipinski_compliant = []
        mol_weights = []
        rotatable_bonds = []

        for i in range(batch_size):
            mol_coords = coordinates[i]
            mol_types = atom_types[i]

            if masks is not None:
                mol_mask = masks[i]
                coord_len = len(mol_coords)
                mask_len = len(mol_mask)
                if mask_len > coord_len:
                    mol_mask = mol_mask[:coord_len]
                elif mask_len < coord_len:
                    padding = jnp.zeros(coord_len - mask_len, dtype=jnp.bool_)
                    mol_mask = jnp.concatenate([mol_mask, padding])
            else:
                mol_mask = jnp.ones(len(mol_coords), dtype=jnp.bool_)

            actual_coords = mol_coords[mol_mask]
            actual_types = mol_types[mol_mask]

            if len(actual_coords) == 0:
                continue

            mock_qed = self._compute_mock_qed(actual_coords, actual_types)
            qed_scores.append(mock_qed)

            mock_lipinski = self._check_lipinski_compliance(actual_coords, actual_types)
            lipinski_compliant.append(mock_lipinski)

            mock_mw = len(actual_coords) * 12.0
            mol_weights.append(mock_mw)

            mock_rot_bonds = max(0, len(actual_coords) - 10)
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

    def validate_inputs(self, coordinates, atom_types) -> None:
        """Validate input data compatibility.

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(coordinates, jnp.ndarray):
            raise ValueError("Coordinates must be a jax array")
        if not isinstance(atom_types, jnp.ndarray):
            raise ValueError("Atom types must be a jax array")
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("Coordinates must be 3D with last dim = 3")

    def _compute_mock_qed(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> float:
        """Compute mock QED score based on basic properties."""
        num_atoms = len(coordinates)

        size_penalty = 1.0
        if num_atoms < 10:
            size_penalty = num_atoms / 10.0
        elif num_atoms > 50:
            size_penalty = 50.0 / num_atoms

        coord_var = float(jnp.var(coordinates))
        complexity_score = jnp.clip(coord_var / 10.0, 0.0, 1.0)

        mock_qed = size_penalty * complexity_score * 0.8
        return float(jnp.clip(mock_qed, 0.0, 1.0))

    def _check_lipinski_compliance(self, coordinates: jnp.ndarray, atom_types: jnp.ndarray) -> bool:
        """Check mock Lipinski rule compliance."""
        num_atoms = len(coordinates)

        mw_ok = num_atoms < 40
        coord_spread = float(jnp.max(coordinates) - jnp.min(coordinates))
        logp_ok = coord_spread < 15.0
        hb_ok = num_atoms < 30

        return mw_ok and logp_ok and hb_ok
