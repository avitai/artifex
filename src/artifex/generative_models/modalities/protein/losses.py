"""Protein-specific loss builders backed by pure JAX functions."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from artifex.generative_models.extensions.protein.constraints import (
    BOND_ANGLES,
    BOND_LENGTHS,
    calculate_bond_angles,
    calculate_bond_lengths,
    calculate_dihedral_angles,
    DIHEDRAL_ANGLES,
)


ProteinLossFunction = Callable[[dict[str, Any], dict[str, Any]], jax.Array]
ProteinLossDictFunction = Callable[[dict[str, Any], dict[str, Any]], dict[str, jax.Array]]


def _extract_atom_positions(
    batch: dict[str, Any], outputs: dict[str, Any]
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Extract protein coordinates from batch and model outputs."""
    target_pos = batch.get("atom_positions")
    pred_pos = outputs.get("atom_positions")
    atom_mask = batch.get("atom_mask")

    if target_pos is None or pred_pos is None:
        raise ValueError("Protein losses require atom positions in both batch and outputs")

    return target_pos, pred_pos, atom_mask


def _mean_over_valid(values: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    """Average values over valid masked entries."""
    if mask is None:
        return jnp.mean(values)

    masked_values = values * mask
    normalizer = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(masked_values) / normalizer


def _squared_periodic_difference(values: jax.Array, targets: jax.Array) -> jax.Array:
    """Square the shortest signed angular difference."""
    delta = jnp.arctan2(jnp.sin(values - targets), jnp.cos(values - targets))
    return jnp.square(delta)


def create_rmsd_loss() -> ProteinLossFunction:
    """Build RMSD loss for protein coordinates."""

    def rmsd_loss(batch: dict[str, Any], outputs: dict[str, Any], **kwargs: Any) -> jax.Array:
        del kwargs
        target_pos, pred_pos, atom_mask = _extract_atom_positions(batch, outputs)

        squared_diff = jnp.sum(jnp.square(pred_pos - target_pos), axis=-1)
        rmsd = jnp.sqrt(_mean_over_valid(squared_diff, atom_mask))
        return rmsd

    return rmsd_loss


def create_backbone_loss() -> ProteinLossFunction:
    """Build backbone geometry loss for protein coordinates."""
    ideal_lengths = {
        name: jnp.asarray(value)
        for name, value in BOND_LENGTHS.items()
        if name in {"N-CA", "CA-C", "C-N"}
    }
    ideal_angles = {
        name: jnp.asarray(value)
        for name, value in BOND_ANGLES.items()
        if name in {"N-CA-C", "CA-C-N"}
    }

    def backbone_loss(batch: dict[str, Any], outputs: dict[str, Any], **kwargs: Any) -> jax.Array:
        del kwargs
        _, pred_pos, atom_mask = _extract_atom_positions(batch, outputs)

        bond_lengths = calculate_bond_lengths(pred_pos, atom_mask)
        bond_angles = calculate_bond_angles(pred_pos, atom_mask)

        n_mask = ca_mask = c_mask = None
        peptide_mask = angle_mask = None
        if atom_mask is not None:
            n_mask = atom_mask[..., 0]
            ca_mask = atom_mask[..., 1]
            c_mask = atom_mask[..., 2]
            peptide_mask = jnp.zeros_like(n_mask)
            peptide_mask = peptide_mask.at[..., :-1].set(c_mask[..., :-1] * n_mask[..., 1:])
            angle_mask = jnp.zeros_like(n_mask)
            angle_mask = angle_mask.at[..., :-1].set(
                ca_mask[..., :-1] * c_mask[..., :-1] * n_mask[..., 1:]
            )

        n_ca_mask = None if n_mask is None or ca_mask is None else n_mask * ca_mask
        ca_c_mask = None if ca_mask is None or c_mask is None else ca_mask * c_mask
        n_ca_c_mask = (
            None
            if n_mask is None or ca_mask is None or c_mask is None
            else n_mask * ca_mask * c_mask
        )

        n_ca_loss = _mean_over_valid(
            jnp.square(bond_lengths["N-CA"] - ideal_lengths["N-CA"]),
            n_ca_mask,
        )
        ca_c_loss = _mean_over_valid(
            jnp.square(bond_lengths["CA-C"] - ideal_lengths["CA-C"]),
            ca_c_mask,
        )
        c_n_loss = _mean_over_valid(
            jnp.square(bond_lengths["C-N"] - ideal_lengths["C-N"]),
            peptide_mask,
        )
        n_ca_c_loss = _mean_over_valid(
            jnp.square(bond_angles["N-CA-C"] - ideal_angles["N-CA-C"]),
            n_ca_c_mask,
        )
        ca_c_n_loss = _mean_over_valid(
            jnp.square(bond_angles["CA-C-N"] - ideal_angles["CA-C-N"]),
            angle_mask,
        )

        return n_ca_loss + ca_c_loss + c_n_loss + n_ca_c_loss + ca_c_n_loss

    return backbone_loss


def create_dihedral_loss(
    *,
    target_secondary_structure: str = "alpha_helix",
    phi_weight: float = 1.0,
    psi_weight: float = 1.0,
    ideal_phi: float | None = None,
    ideal_psi: float | None = None,
) -> ProteinLossFunction:
    """Build backbone dihedral loss for protein coordinates."""
    defaults = DIHEDRAL_ANGLES.get(target_secondary_structure, DIHEDRAL_ANGLES["alpha_helix"])
    target_phi = jnp.asarray(defaults["phi"] if ideal_phi is None else ideal_phi)
    target_psi = jnp.asarray(defaults["psi"] if ideal_psi is None else ideal_psi)

    def dihedral_loss(batch: dict[str, Any], outputs: dict[str, Any], **kwargs: Any) -> jax.Array:
        del kwargs
        _, pred_pos, atom_mask = _extract_atom_positions(batch, outputs)

        dihedrals = calculate_dihedral_angles(pred_pos, atom_mask)
        phi_sq_error = _squared_periodic_difference(dihedrals["phi"], target_phi)
        psi_sq_error = _squared_periodic_difference(dihedrals["psi"], target_psi)

        phi_mask = psi_mask = None
        if atom_mask is not None:
            n_mask = atom_mask[..., 0]
            ca_mask = atom_mask[..., 1]
            c_mask = atom_mask[..., 2]

            phi_mask = jnp.zeros_like(dihedrals["phi"])
            phi_mask = phi_mask.at[..., 1:].set(
                c_mask[..., :-1] * n_mask[..., 1:] * ca_mask[..., 1:] * c_mask[..., 1:]
            )

            psi_mask = jnp.zeros_like(dihedrals["psi"])
            psi_mask = psi_mask.at[..., :-1].set(
                n_mask[..., :-1] * ca_mask[..., :-1] * c_mask[..., :-1] * n_mask[..., 1:]
            )

        phi_loss = _mean_over_valid(phi_sq_error, phi_mask)
        psi_loss = _mean_over_valid(psi_sq_error, psi_mask)
        return phi_weight * phi_loss + psi_weight * psi_loss

    return dihedral_loss


def create_protein_structure_loss(
    rmsd_weight: float = 1.0,
    backbone_weight: float = 0.5,
    dihedral_weight: float = 0.3,
) -> ProteinLossDictFunction:
    """Build a canonical protein structure loss dict."""
    rmsd_loss = create_rmsd_loss()
    backbone_loss = create_backbone_loss()
    dihedral_loss = create_dihedral_loss()

    def protein_structure_loss(
        batch: dict[str, Any], outputs: dict[str, Any], **kwargs: Any
    ) -> dict[str, jax.Array]:
        rmsd_value = rmsd_loss(batch, outputs, **kwargs)
        backbone_value = backbone_loss(batch, outputs, **kwargs)
        dihedral_value = dihedral_loss(batch, outputs, **kwargs)

        total_loss = (
            rmsd_weight * rmsd_value
            + backbone_weight * backbone_value
            + dihedral_weight * dihedral_value
        )
        return {
            "total_loss": total_loss,
            "rmsd_loss": rmsd_value,
            "backbone_loss": backbone_value,
            "dihedral_loss": dihedral_value,
        }

    return protein_structure_loss
