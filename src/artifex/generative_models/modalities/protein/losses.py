"""Protein-specific loss functions for generative models.

This module provides composable loss functions for protein structure generation
and diffusion models.
"""

from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp


class ProteinLossFunction(Protocol):
    """Protocol for protein loss functions."""

    def __call__(self, batch: dict[str, Any], outputs: dict[str, Any], **kwargs) -> jax.Array:
        """Compute loss.

        Args:
            batch: Batch of data with ground truth
            outputs: Model outputs
            **kwargs: Additional keyword arguments

        Returns:
            Loss value
        """
        ...


class CompositeLoss:
    """Composable loss function that combines multiple loss terms.

    This class allows combining multiple loss functions with weights for
    protein structure generation tasks.
    """

    def __init__(
        self,
        loss_terms: dict[str, tuple[ProteinLossFunction, float]],
    ):
        """Initialize the composite loss.

        Args:
            loss_terms: Dictionary mapping loss names to (loss_fn, weight) tuples
        """
        self.loss_terms = loss_terms

    def __call__(
        self, batch: dict[str, Any], outputs: dict[str, Any], **kwargs
    ) -> dict[str, jax.Array]:
        """Compute the combined loss.

        Args:
            batch: Batch of data with ground truth
            outputs: Model outputs
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with individual and total loss values
        """
        losses: dict[str, jax.Array] = {}

        # Calculate each loss term
        for name, (loss_fn, weight) in self.loss_terms.items():
            loss_value = loss_fn(batch, outputs, **kwargs)
            weighted_loss = weight * loss_value
            losses[name] = weighted_loss

        # Calculate total loss
        if losses:
            total_loss = sum(losses.values())
            losses["total"] = jnp.asarray(total_loss)
        else:
            losses["total"] = jnp.array(0.0)

        return losses


def create_rmsd_loss() -> ProteinLossFunction:
    """Create a loss function for protein RMSD.

    Returns:
        RMSD loss function
    """

    def rmsd_loss(batch: dict[str, Any], outputs: dict[str, Any], **kwargs) -> jax.Array:
        """Calculate RMSD between predicted and ground truth protein structures.

        Args:
            batch: Batch with ground truth atom_positions
            outputs: Model outputs with predicted atom_positions
            **kwargs: Additional keyword arguments

        Returns:
            RMSD loss
        """
        # Get target and predicted positions
        target_pos = batch.get("atom_positions")
        pred_pos = outputs.get("atom_positions")

        if target_pos is None or pred_pos is None:
            # Return zero loss if positions not available
            return jnp.array(0.0)

        # Get atom mask if available
        atom_mask = batch.get("atom_mask")

        if atom_mask is not None:
            # Apply mask to consider only valid atoms
            mask_3d = atom_mask[:, :, :, None]  # [batch, res, atoms, 1]
            # Calculate masked MSE
            squared_diff = ((pred_pos - target_pos) ** 2) * mask_3d
            # Sum over the coordinate dimensions (last dimension)
            squared_diff_sum = jnp.sum(squared_diff, axis=-1)  # [batch, res, atoms]
            # Calculate mean over all atoms considering the mask
            valid_atoms = jnp.sum(atom_mask)
            # Avoid division by zero
            valid_atoms = jnp.maximum(valid_atoms, 1.0)
            mse = jnp.sum(squared_diff_sum) / valid_atoms
        else:
            # Calculate unmasked MSE
            mse = jnp.mean(jnp.sum((pred_pos - target_pos) ** 2, axis=-1))

        # RMSD is the square root of MSE
        rmsd = jnp.sqrt(mse)

        return rmsd

    return rmsd_loss


def create_backbone_loss() -> ProteinLossFunction:
    """Create a loss function for protein backbone geometry.

    Enforces correct bond lengths and angles in the protein backbone.

    Returns:
        Backbone geometry loss function
    """

    def backbone_loss(batch: dict[str, Any], outputs: dict[str, Any], **kwargs) -> jax.Array:
        """Calculate backbone geometry loss.

        Args:
            batch: Batch of data with ground truth
            outputs: Model outputs
            **kwargs: Additional keyword arguments

        Returns:
            Backbone geometry loss
        """
        # Get predicted positions
        pred_pos = outputs.get("atom_positions")

        if pred_pos is None:
            # Return zero loss if positions not available
            return jnp.array(0.0)

        # Get atom mask if available
        atom_mask = batch.get("atom_mask")

        # Standard protein backbone bond lengths in Angstroms
        # N-CA: ~1.45Å, CA-C: ~1.52Å, C-N+1: ~1.33Å
        ideal_lengths = jnp.array([1.45, 1.52, 1.33])

        # Calculate bond length loss
        # Extract backbone atoms N(0), CA(1), C(2), O(3)
        n_pos = pred_pos[:, :, 0]  # [batch, res, 3]
        ca_pos = pred_pos[:, :, 1]  # [batch, res, 3]
        c_pos = pred_pos[:, :, 2]  # [batch, res, 3]

        # Calculate bond lengths
        n_ca_len = jnp.sqrt(jnp.sum((n_pos - ca_pos) ** 2, axis=-1))  # [batch, res]
        ca_c_len = jnp.sqrt(jnp.sum((ca_pos - c_pos) ** 2, axis=-1))  # [batch, res]

        # Calculate C-N+1 lengths (for peptide bonds)
        # For each residue i, calculate distance from C_i to N_{i+1}
        c_to_next_n = jnp.zeros_like(n_ca_len)
        c_res = c_pos[:, :-1]  # All C atoms except the last residue
        n_next = n_pos[:, 1:]  # All N atoms except the first residue
        c_n_next_dist = jnp.sqrt(jnp.sum((c_res - n_next) ** 2, axis=-1))
        # Pad to original shape
        c_to_next_n = c_to_next_n.at[:, :-1].set(c_n_next_dist)

        # Apply mask if available
        if atom_mask is not None:
            n_mask = atom_mask[:, :, 0]
            ca_mask = atom_mask[:, :, 1]
            c_mask = atom_mask[:, :, 2]

            # For peptide bonds, create mask valid only where both atoms exist
            peptide_mask = n_mask[:, 1:] * c_mask[:, :-1]  # [batch, res-1]
            peptide_mask_full = jnp.zeros_like(n_mask)
            peptide_mask_full = peptide_mask_full.at[:, :-1].set(peptide_mask)

            # Calculate masked squared errors
            n_ca_error = jnp.square(n_ca_len - ideal_lengths[0]) * n_mask * ca_mask
            ca_c_error = jnp.square(ca_c_len - ideal_lengths[1]) * ca_mask * c_mask
            c_n_error = jnp.square(c_to_next_n - ideal_lengths[2]) * peptide_mask_full

            # Count valid bonds
            valid_count = (
                jnp.sum(n_mask * ca_mask) + jnp.sum(ca_mask * c_mask) + jnp.sum(peptide_mask)
            )
            # Avoid division by zero
            valid_count = jnp.maximum(valid_count, 1.0)

            # Sum errors and normalize
            bond_loss = (
                jnp.sum(n_ca_error) + jnp.sum(ca_c_error) + jnp.sum(c_n_error)
            ) / valid_count
        else:
            # Unmasked calculation
            n_ca_error = jnp.square(n_ca_len - ideal_lengths[0])
            ca_c_error = jnp.square(ca_c_len - ideal_lengths[1])
            c_n_error = jnp.square(c_to_next_n - ideal_lengths[2])

            # Mean over all bonds
            bond_loss = jnp.mean(n_ca_error) + jnp.mean(ca_c_error) + jnp.mean(c_n_error)

        # Placeholder for bond angle loss (to be implemented)
        angle_loss = jnp.array(0.0)

        # Combine bond length and angle losses
        return bond_loss + angle_loss

    return backbone_loss


def create_dihedral_loss() -> ProteinLossFunction:
    """Create a loss function for protein dihedral angles.

    Returns:
        Dihedral angle loss function
    """

    def dihedral_loss(batch: dict[str, Any], outputs: dict[str, Any], **kwargs) -> jax.Array:
        """Calculate dihedral angle loss for protein structures.

        Args:
            batch: Batch of data with ground truth
            outputs: Model outputs
            **kwargs: Additional keyword arguments

        Returns:
            Dihedral angle loss
        """
        # Get predicted positions
        pred_pos = outputs.get("atom_positions")

        if pred_pos is None:
            # Return zero loss if positions not available
            return jnp.array(0.0)

        # Placeholder for dihedral angle calculation and loss
        # This should assess phi/psi angles using Ramachandran plot preferences
        return jnp.array(0.0)

    return dihedral_loss


def create_protein_structure_loss(
    rmsd_weight: float = 1.0,
    backbone_weight: float = 0.5,
    dihedral_weight: float = 0.3,
) -> CompositeLoss:
    """Create a composite loss for protein structure generation.

    Args:
        rmsd_weight: Weight for RMSD loss
        backbone_weight: Weight for backbone geometry loss
        dihedral_weight: Weight for dihedral angle loss

    Returns:
        Composite loss function
    """
    loss_terms = {
        "rmsd": (create_rmsd_loss(), rmsd_weight),
        "backbone": (create_backbone_loss(), backbone_weight),
        "dihedral": (create_dihedral_loss(), dihedral_weight),
    }

    return CompositeLoss(loss_terms)


class LossRegistry:
    """Registry for protein loss functions."""

    _losses: dict[str, Callable[..., ProteinLossFunction]] = {
        "rmsd": create_rmsd_loss,
        "backbone": create_backbone_loss,
        "dihedral": create_dihedral_loss,
    }

    _composite_losses: dict[str, Callable[..., CompositeLoss]] = {
        "protein_structure": create_protein_structure_loss,
    }

    @classmethod
    def get_loss(cls, name: str, **kwargs) -> ProteinLossFunction | CompositeLoss:
        """Get a loss function by name.

        Args:
            name: Loss function name
            **kwargs: Additional arguments to pass to the loss constructor

        Returns:
            Loss function
        """
        if name in cls._losses:
            return cls._losses[name](**kwargs)
        elif name in cls._composite_losses:
            return cls._composite_losses[name](**kwargs)
        else:
            raise ValueError(f"Unknown loss function: {name}")

    @classmethod
    def register_loss(cls, name: str, loss_fn: Callable[..., ProteinLossFunction]) -> None:
        """Register a new loss function.

        Args:
            name: Loss function name
            loss_fn: Loss function constructor
        """
        cls._losses[name] = loss_fn

    @classmethod
    def register_composite_loss(cls, name: str, loss_fn: Callable[..., CompositeLoss]) -> None:
        """Register a new composite loss function.

        Args:
            name: Composite loss function name
            loss_fn: Composite loss function constructor
        """
        cls._composite_losses[name] = loss_fn
