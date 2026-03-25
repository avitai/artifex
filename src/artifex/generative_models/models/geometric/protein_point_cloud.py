"""Protein-specific point cloud model implementation."""

import logging
from typing import Any

import jax
from flax import nnx


logger = logging.getLogger(__name__)

from artifex.generative_models.core.configuration import (
    ProteinPointCloudConfig,
)
from artifex.generative_models.extensions.protein import create_protein_extensions
from artifex.generative_models.extensions.protein.constraints import (
    ProteinBackboneConstraint,
    ProteinDihedralConstraint,
)
from artifex.generative_models.models.geometric.point_cloud import PointCloudModel


class ProteinPointCloudModel(PointCloudModel):
    """Protein-specific point cloud model.

    This model extends the base PointCloudModel with protein-specific
    features like backbone constraints and amino acid awareness.
    """

    def __init__(
        self,
        config: ProteinPointCloudConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the protein point cloud model.

        Args:
            config: Protein point cloud model configuration.
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not a ProteinPointCloudConfig
        """
        if not isinstance(config, ProteinPointCloudConfig):
            raise TypeError(f"config must be ProteinPointCloudConfig, got {type(config).__name__}")

        # Store protein-specific configuration from config
        self.num_residues = config.num_residues
        self.num_atoms = config.num_atoms_per_residue
        self.backbone_indices = list(config.backbone_indices)

        extensions_dict = None
        if config.extensions is not None:
            built_extensions = create_protein_extensions(config.extensions, rngs=rngs)
            if len(built_extensions):
                extensions_dict = built_extensions

        # Initialize base PointCloudModel with extensions (wrap in nnx.Dict for Flax 0.12.0+)
        super().__init__(config, extensions=extensions_dict, rngs=rngs)

        # Store references to specific extension types for easier access
        # (Don't initialize to None - only create when extension exists)
        if hasattr(self, "extension_modules"):
            for name, extension in self.extension_modules.items():
                if isinstance(extension, ProteinBackboneConstraint):
                    self.backbone_constraint = extension
                elif isinstance(extension, ProteinDihedralConstraint):
                    self.dihedral_constraint = extension

    def __call__(
        self,
        x: jax.Array | dict[str, jax.Array] | None = None,
        *,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Forward pass through protein point cloud model.

        Args:
            x: Input data, either point cloud array or dictionary with keys.
            deterministic: Whether to run in deterministic mode.

        Returns:
            Dictionary of outputs.
        """
        # Process amino acid types if provided
        if isinstance(x, dict) and "aatype" in x:
            aatype = x["aatype"]
            batch_size, seq_length = aatype.shape

            # Encode amino acid types
            aa_embeddings = self._encode_aatype(aatype)  # [batch, residues, embed_dim]

            # If atom positions are provided in protein format [batch, residues, atoms, 3]
            if "atom_positions" in x:
                atom_positions = x["atom_positions"]  # [batch, residues, atoms, 3]

                # Check if we need to reshape to [batch, residues*atoms, 3]
                if len(atom_positions.shape) == 4:
                    batch_size, num_residues, num_atoms, coords_dim = atom_positions.shape
                    # Reshape to [batch, residues*atoms, 3]
                    atom_positions = atom_positions.reshape(
                        batch_size, num_residues * num_atoms, coords_dim
                    )
                    x["positions"] = atom_positions

                # Broadcast aa_embeddings to match atom positions
                if len(aa_embeddings.shape) == 3:  # [batch, residues, embed_dim]
                    # Add atom dimension
                    aa_embeddings = aa_embeddings[:, :, None, :]  # [batch, residues, 1, embed_dim]
                    # Repeat along atom dimension
                    aa_embeddings = aa_embeddings.repeat(
                        self.num_atoms, axis=2
                    )  # [batch, residues, atoms, embed_dim]
                    # Reshape to flatten residues and atoms dimensions
                    aa_embeddings = aa_embeddings.reshape(
                        batch_size, seq_length * self.num_atoms, self.embed_dim
                    )  # [batch, residues*atoms, embed_dim]

                # Add to input dict
                if x is None:
                    x = {}
                elif not isinstance(x, dict):
                    x = {"positions": x}

                x["features"] = aa_embeddings

        # Call parent class
        outputs = super().__call__(x, deterministic=deterministic)

        # Reshape positions to protein-specific format [batch, residues, atoms, 3]
        if "positions" in outputs:
            positions = outputs["positions"]
            # Check if reshaping is needed
            if len(positions.shape) == 3:  # [batch, num_points, 3]
                batch_size = positions.shape[0]
                num_points = positions.shape[1]

                # Make sure we can reshape
                if num_points == self.num_residues * self.num_atoms:
                    # Reshape to [batch, residues, atoms, 3]
                    positions = positions.reshape(batch_size, self.num_residues, self.num_atoms, 3)
                    outputs["positions"] = positions

            if len(positions.shape) == 4:
                outputs["atom_positions"] = positions

        # Return outputs
        return outputs

    def sample(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate samples from the model.

        Args:
            n_samples: Number of samples to generate.
            rngs: Optional random number generator keys.

        Returns:
            Generated samples with shape [n_samples, num_residues, num_atoms, 3].
        """
        # Use the provided rngs or fall back to the model's rngs
        if rngs is None and hasattr(self, "rngs"):
            rngs = self.rngs

        # Sample from base model
        outputs = super().sample(n_samples, rngs=rngs)

        # Reshape to protein structure format
        if outputs.ndim == 3:  # [n_samples, num_points, 3]
            # Assuming samples are flattened (num_residues * num_atoms, 3)
            outputs = outputs.reshape(n_samples, self.num_residues, self.num_atoms, 3)

        return outputs

    def generate(self, batch_size: int = 1, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate samples from the model.

        This is an alias for sample to maintain consistent API with other models.

        Args:
            batch_size: Number of samples to generate.
            rngs: Optional random number generator keys.

        Returns:
            Generated samples with shape [batch_size, num_residues, num_atoms, 3].
        """
        return self.sample(n_samples=batch_size, rngs=rngs)

    def _encode_aatype(self, aatype: jax.Array) -> jax.Array:
        """Encode amino acid types as features.

        Args:
            aatype: Amino acid types with shape [batch, num_residues].

        Returns:
            Encoded features with shape [batch, num_residues, embed_dim].
        """
        batch_size, num_residues = aatype.shape

        # One-hot encode amino acid types (20 standard amino acids)
        one_hot = jax.nn.one_hot(aatype, num_classes=20)  # [batch, num_residues, 20]

        # Project to embedding dimension
        if not hasattr(self, "aa_proj"):
            # Create projection layer if not already created
            self.aa_proj = nnx.Linear(
                in_features=20,
                out_features=self.embed_dim,
                rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
            )

        # Apply projection
        aa_embeddings = self.aa_proj(one_hot)  # [batch, num_residues, embed_dim]

        # Note: The actual broadcasting to atoms is now handled in __call__
        # to ensure proper shape matching with the positional embeddings

        return aa_embeddings

    def get_loss_fn(self, auxiliary: dict[str, Any] | None = None) -> Any:
        """Get the canonical point-cloud loss function for protein batches."""

        def protein_loss_fn(
            batch: dict[str, Any], outputs: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return self.loss_fn(batch, outputs, **kwargs)

        return protein_loss_fn

    def loss_fn(
        self, batch: dict[str, Any], outputs: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Calculate the canonical point-cloud loss for protein batches."""

        def flatten_protein_positions(value: Any) -> Any:
            if value is None or value.ndim != 4:
                return value

            batch_size, num_residues, num_atoms, coord_dim = value.shape
            return value.reshape(batch_size, num_residues * num_atoms, coord_dim)

        normalized_batch = dict(batch)
        normalized_outputs = dict(outputs)

        batch_coords = normalized_batch.get("positions", normalized_batch.get("atom_positions"))
        if batch_coords is not None:
            normalized_batch["positions"] = flatten_protein_positions(batch_coords)

        output_coords = normalized_outputs.get(
            "atom_positions",
            normalized_outputs.get("positions"),
        )
        if output_coords is not None:
            normalized_outputs["atom_positions"] = output_coords
            normalized_outputs["positions"] = flatten_protein_positions(output_coords)

        return super().get_loss_fn()(normalized_batch, normalized_outputs, **kwargs)
