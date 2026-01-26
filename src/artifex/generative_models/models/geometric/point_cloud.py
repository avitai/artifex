"""Point cloud model implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    PointCloudConfig,
)
from artifex.generative_models.core.layers.transformers import (
    TransformerEncoderBlock,
)
from artifex.generative_models.models.geometric.base import GeometricModel


class PointCloudModel(GeometricModel):
    """Model for generating point cloud data.

    This model can be used for generating arbitrary point cloud data,
    including protein structures when extended with protein-specific
    extensions.
    """

    def __init__(
        self,
        config: PointCloudConfig,
        *,
        extensions: dict[str, nnx.Module] | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the point cloud model.

        Args:
            config: PointCloudConfig dataclass with model parameters.
            extensions: Optional dictionary of extension modules.
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not a PointCloudConfig
        """
        super().__init__(config, extensions=extensions, rngs=rngs)

        # Extract configuration parameters from dataclass config
        self.embed_dim = config.network.embed_dim
        self.num_points = config.num_points
        self.num_layers = config.network.num_layers
        self.num_heads = config.network.num_heads
        self.dropout = config.dropout_rate

        # Ensure rngs has both params and dropout keys
        if not hasattr(rngs, "dropout") and hasattr(rngs, "params"):
            # Create dropout key from params key
            params_key = rngs.params()
            dropout_key = jax.random.fold_in(params_key, 0)
            # Create a new rngs with both keys
            rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

        # Initialize transformer blocks using nnx.List for proper pytree handling
        self.transformer_blocks = nnx.List(
            [
                TransformerEncoderBlock(
                    hidden_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout,
                    rngs=rngs,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Initialize coordinate projection
        self.coord_proj = nnx.Linear(
            in_features=self.embed_dim, out_features=3, rngs=rngs or nnx.Rngs()
        )  # project to 3D coordinates

        # Initialize embeddings
        self.pos_embedding = nnx.Embed(
            num_embeddings=self.num_points,
            features=self.embed_dim,
            rngs=rngs or nnx.Rngs(),
        )

    def __call__(
        self,
        x: jax.Array | dict[str, jax.Array] | None = None,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> dict[str, jax.Array]:
        """Forward pass through the model.

        Args:
            x: Optional input array or dict.
            deterministic: Whether to run in deterministic mode.
            rngs: Random number generator keys for stochastic operations.

        Returns:
            dictionary with model outputs.
        """
        # If x is a direct array, convert it to a dict with "positions" key
        if isinstance(x, jax.Array) and not isinstance(x, dict):
            x = {"positions": x}

        # Get batch size
        batch_size = 1
        if x is not None:
            if isinstance(x, dict):
                # Try to extract batch size from positions or features
                if "positions" in x and x["positions"] is not None:
                    batch_size = x["positions"].shape[0]
                elif "features" in x and x["features"] is not None:
                    batch_size = x["features"].shape[0]
            else:
                # Direct array input (though we should have converted it by now)
                batch_size = x.shape[0]

        # Special handling for protein model inputs
        if x is not None and isinstance(x, dict) and ("atom_positions" in x or "atom_mask" in x):
            # If we have protein-specific inputs, reshape them
            if "atom_positions" in x:
                positions = x["atom_positions"]
                if len(positions.shape) == 4:  # [batch, residues, atoms, 3]
                    batch_size, num_residues, num_atoms, _ = positions.shape
                    # Reshape to [batch, residues*atoms, 3]
                    positions = positions.reshape(batch_size, num_residues * num_atoms, 3)
                    x["positions"] = positions

            if "atom_mask" in x and "mask" not in x:
                mask = x["atom_mask"]
                if len(mask.shape) == 3:  # [batch, residues, atoms]
                    batch_size, num_residues, num_atoms = mask.shape
                    # Reshape to [batch, residues*atoms]
                    mask = mask.reshape(batch_size, num_residues * num_atoms)
                    x["mask"] = mask

        # Create position indices for embeddings
        pos_indices = jnp.arange(self.num_points)

        # Get positional embeddings
        embeddings = self.pos_embedding(pos_indices)  # [num_points, embed_dim]

        # Add batch dimension
        embeddings = embeddings[None, :, :].repeat(
            batch_size, axis=0
        )  # [batch, num_points, embed_dim]

        # Add input features if provided
        if x is not None and "features" in x:
            # Make sure features match the same shape as the embeddings
            features = x["features"]

            # Handle potential shape mismatch
            if features.shape[1] != self.num_points:
                # If the features have a different number of points, we'll need to adjust
                if features.shape[1] < self.num_points:
                    # We need to pad the features
                    padding = [(0, 0), (0, self.num_points - features.shape[1]), (0, 0)]
                    features = jnp.pad(features, padding)
                else:
                    # We need to truncate the features
                    features = features[:, : self.num_points, :]

            embeddings = embeddings + features

        # Apply mask if provided
        mask = None
        attention_mask = None  # Initialize to None by default
        if x is not None and isinstance(x, dict) and "mask" in x:
            mask = x["mask"]

            # Handle potential shape mismatch
            if mask.shape[1] != self.num_points:
                # If the mask has a different number of points, we'll need to adjust
                if mask.shape[1] < self.num_points:
                    # We need to pad the mask
                    padding = [(0, 0), (0, self.num_points - mask.shape[1])]
                    mask = jnp.pad(mask, padding, constant_values=0)
                else:
                    # We need to truncate the mask
                    mask = mask[:, : self.num_points]

            # Convert mask to attention mask for transformer blocks
            # For self-attention, create [batch, 1, 1, sequence_length] mask
            # which will broadcast correctly in the attention calculation
            if mask is not None:
                # Convert to boolean mask where True = keep, False = mask out
                bool_mask = mask > 0

                # Prepare for attention with correct broadcasting dimensions
                # [batch, points] -> [batch, 1, 1, points]
                attention_mask = bool_mask[:, None, None, :]

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            embeddings = block(
                embeddings,
                mask=attention_mask,
                deterministic=deterministic,
                rngs=rngs,
            )

        # Project to 3D coordinates
        coords = self.coord_proj(embeddings)  # [batch, num_points, 3]

        # Apply extensions (constraints) if any
        if self.extension_modules:
            processed_coords, extension_outputs = self.apply_extensions(
                x or {}, coords, deterministic=deterministic
            )
        else:
            processed_coords = coords
            extension_outputs = {}

        # Return results
        return {
            "positions": processed_coords,
            "embeddings": embeddings,
            "extension_outputs": extension_outputs,
        }

    def sample(self, n_samples: int, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate point cloud samples.

        Args:
            n_samples: Number of samples to generate.
            rngs: Optional random number generator keys.

        Returns:
            Generated point clouds with shape [n_samples, num_points, 3]
        """
        # Use the provided rngs or create a default one
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Get the params key for random number generation
        key = rngs.params()

        # Create random normal vectors
        shape = (n_samples, self.num_points, self.embed_dim)
        random_vectors = jax.random.normal(key, shape=shape)

        # Process all samples as a batch through transformer blocks
        embedded = random_vectors
        for block in self.transformer_blocks:
            embedded = block(embedded, deterministic=True)

        # Project to 3D coordinates
        return self.coord_proj(embedded)

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Generate point cloud samples. Alias for sample.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generator keys
            **kwargs: Additional keyword arguments passed to sample

        Returns:
            Generated point clouds with shape [n_samples, num_points, 3]
        """
        # Call sample with the appropriate parameters
        return self.sample(n_samples=n_samples, rngs=rngs)

    def get_loss_fn(self, auxiliary: dict[str, Any] | None = None) -> Any:
        """Get the loss function for the model.

        Args:
            auxiliary: Optional auxiliary outputs to use in the loss.

        Returns:
            Point cloud loss function.
        """

        def loss_fn(batch: dict[str, jax.Array], outputs: Any, **kwargs) -> dict[str, jax.Array]:
            # Get target point cloud
            target = None

            # Try to get target from various possible keys
            if "positions" in batch:
                target = batch["positions"]
            elif "target" in batch:
                target = batch["target"]
            elif "atom_positions" in batch:
                # Protein-specific format
                target = batch["atom_positions"]
                # Handle reshaping if needed
                if len(target.shape) == 4:  # [batch, residue, atoms, 3]
                    batch_size, num_residues, num_atoms, _ = target.shape
                    target = target.reshape(batch_size, num_residues * num_atoms, 3)

            if target is None:
                raise ValueError(
                    "Batch must contain 'positions', 'atom_positions', "
                    "or 'target' for loss calculation"
                )

            # Get predicted point cloud
            if isinstance(outputs, dict):
                predicted = outputs.get("positions", outputs)
            else:
                predicted = outputs

            # Check if shapes match, reshape if needed
            if predicted.shape != target.shape:
                if len(predicted.shape) == 4 and len(target.shape) == 3:
                    # Predicted is [batch, residues, atoms, 3], target is [batch, num_points, 3]
                    batch_size = predicted.shape[0]
                    num_residues = predicted.shape[1]
                    num_atoms = predicted.shape[2]
                    # Reshape predicted to match target
                    predicted = predicted.reshape(batch_size, num_residues * num_atoms, 3)
                elif len(predicted.shape) == 3 and len(target.shape) == 4:
                    # Predicted is [batch, num_points, 3], target is [batch, residues, atoms, 3]
                    batch_size = target.shape[0]
                    num_residues = target.shape[1]
                    num_atoms = target.shape[2]
                    # Reshape target to match predicted
                    target = target.reshape(batch_size, num_residues * num_atoms, 3)

            # Apply mask if available
            mask = batch.get("mask", None)
            if mask is None and "atom_mask" in batch:
                # Try to use atom_mask if available
                atom_mask = batch["atom_mask"]
                if len(atom_mask.shape) == 3:  # [batch, residues, atoms]
                    batch_size = atom_mask.shape[0]
                    num_residues = atom_mask.shape[1]
                    num_atoms = atom_mask.shape[2]
                    mask = atom_mask.reshape(batch_size, num_residues * num_atoms)

            # Adapt mask shape if needed
            if mask is not None and len(mask.shape) < len(target.shape):
                # Expand mask for coordinate dimensions
                if len(mask.shape) == 2 and len(target.shape) == 3:
                    # [B, N] -> [B, N, 1]
                    expanded_mask = mask[:, :, None]
                else:
                    # This will raise an error, but we'll handle it properly
                    expanded_mask = mask

                # Calculate masked MSE loss
                try:
                    sq_diff = jnp.square(predicted - target) * expanded_mask
                    mse_loss = jnp.sum(sq_diff) / (jnp.sum(mask) * 3 + 1e-8)
                except Exception as e:
                    # If broadcasting fails, print useful debug info and re-raise
                    print(f"Error during loss calculation: {e}")
                    print(f"predicted shape: {predicted.shape}, target shape: {target.shape}")
                    print(f"mask shape: {mask.shape}, expanded_mask shape: {expanded_mask.shape}")
                    raise
            else:
                # Calculate simple MSE loss
                mse_loss = jnp.mean(jnp.square(predicted - target))

            # Get extension losses if any
            extension_losses = {}
            if hasattr(self, "extension_modules") and self.extension_modules:
                # Check if we have any extension modules
                extension_losses = self.get_extension_losses(batch, outputs, **kwargs)

            # Combine losses
            total_loss = mse_loss
            for name, loss in extension_losses.items():
                total_loss = total_loss + loss

            # Return loss dict
            return {
                "total_loss": total_loss,
                "mse_loss": mse_loss,
                **extension_losses,
            }

        return loss_fn


class TransformerBlock(nnx.Module):
    """Transformer block for point cloud modeling."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the transformer block.

        Args:
            embed_dim: Dimension of embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            rngs: Random number generator keys.
        """
        super().__init__(rngs=rngs or nnx.Rngs())

        # Self-attention
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=embed_dim,
            rngs=rngs,
        )

        # Feed-forward network components
        self.ffn_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ffn_linear1 = nnx.Linear(in_features=embed_dim, out_features=embed_dim * 4, rngs=rngs)
        self.ffn_dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.ffn_linear2 = nnx.Linear(in_features=embed_dim * 4, out_features=embed_dim, rngs=rngs)
        self.ffn_dropout2 = nnx.Dropout(dropout, rngs=rngs)

        # Layer normalization
        self.layer_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: jax.Array | None = None,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor with shape [batch, num_points, embed_dim] or [num_points, embed_dim].
            mask: Optional attention mask with shape [batch, 1, 1, num_points].
            deterministic: Whether to run in deterministic mode.
            rngs: Optional random number generators for stochastic operations.

        Returns:
            Processed tensor with same shape as input.
        """
        # rngs parameter accepted for API compatibility but not currently used
        # (dropout uses deterministic flag instead)
        _ = rngs
        # Self-attention
        attention_output = self.attention(
            x,
            x,
            x,
            mask=mask,
            deterministic=deterministic,
        )

        # Add & norm
        x = x + attention_output
        x = self.layer_norm(x)

        # Feed-forward network
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn_linear1(ffn_input)
        ffn_output = nnx.gelu(ffn_output)
        ffn_output = self.ffn_dropout1(ffn_output, deterministic=deterministic)
        ffn_output = self.ffn_linear2(ffn_output)
        ffn_output = self.ffn_dropout2(ffn_output, deterministic=deterministic)

        # Add & norm
        x = x + ffn_output
        x = self.layer_norm(x)

        return x
