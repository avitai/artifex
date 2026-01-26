"""Graph model implementation with E(n) Equivariant Graph Neural Networks (EGNN).

This module implements a generic graph model with E(n) equivariance support,
which is useful for generating and processing molecular structures,
including proteins.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    GraphConfig,
)
from artifex.generative_models.core.layers.egnn import EGNNLayer
from artifex.generative_models.models.geometric.base import GeometricModel


class GraphModel(GeometricModel):
    """Graph model with E(n) equivariance support.

    This model can work with arbitrary graphs represented as nodes,
    edges, and positional data in 3D space. It maintains E(n) equivariance,
    making it suitable for molecular modeling tasks like protein diffusion.
    """

    def __init__(
        self,
        config: GraphConfig,
        *,
        extensions: dict[str, nnx.Module] | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the graph model.

        Args:
            config: GraphConfig dataclass with model parameters.
            extensions: Optional dictionary of extension modules.
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not a GraphConfig
        """
        super().__init__(config, extensions=extensions, rngs=rngs)

        # Extract configuration parameters from dataclass config
        self.node_dim = config.network.node_features_dim
        self.edge_dim = config.network.edge_features_dim
        self.hidden_dim = config.network.hidden_dims[0]
        self.num_layers = config.network.num_layers
        self.num_mlp_layers = config.network.num_mlp_layers
        self.dropout = config.dropout_rate
        self.use_attention = config.network.use_attention
        self.norm_coordinates = config.network.norm_coordinates
        self.residual = config.network.residual

        # Ensure we have all required RNG keys
        if not hasattr(rngs, "dropout") and hasattr(rngs, "params"):
            params_key = rngs.params()
            dropout_key = jax.random.fold_in(params_key, 0)
            rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

        # Initialize EGNN layers
        self.egnn_layers = nnx.List(
            [
                EGNNLayer(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    hidden_dim=self.hidden_dim,
                    num_mlp_layers=self.num_mlp_layers,
                    dropout_rate=self.dropout,
                    use_attention=self.use_attention,
                    residual=self.residual,
                    rngs=rngs,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Node feature embedding (if input features need projection)
        self.node_embedding = nnx.Linear(
            in_features=self.node_dim,
            out_features=self.hidden_dim,
            rngs=rngs,
        )

        # Edge feature embedding (if input features need projection)
        self.edge_embedding = nnx.Linear(
            in_features=self.edge_dim,
            out_features=self.hidden_dim,
            rngs=rngs,
        )

        # Output projections
        self.node_proj = nnx.Linear(
            in_features=self.hidden_dim,
            out_features=self.node_dim,
            rngs=rngs,
        )

        # Coordinate refinement layer
        self.coord_refinement = nnx.Linear(
            in_features=self.hidden_dim,
            out_features=1,  # One scalar per dimension for scaling
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array | dict[str, jax.Array] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool = False,
        batch_size: int = 1,
    ) -> dict[str, jax.Array]:
        """Forward pass through the model.

        Args:
            x: Input dictionary with:
                - node_features: Node features [batch, num_nodes, node_dim]
                - edge_features: Edge features [batch, num_nodes, num_nodes, edge_dim]
                - coordinates: Node coordinates [batch, num_nodes, 3]
                - adjacency: Adjacency matrix [batch, num_nodes, num_nodes]
                - mask: Node mask [batch, num_nodes]
            rngs: Optional random number generator keys
            deterministic: Whether to run in deterministic mode
            batch_size: Batch size for generated inputs when x is None

        Returns:
            dictionary with model outputs:
                - node_features: Updated node features
                - coordinates: Updated node coordinates
                - edge_features: Updated edge features
        """
        if rngs is None and hasattr(self, "rngs"):
            rngs = self.rngs

        # Default empty inputs if None provided
        if x is None:
            # Create dummy inputs for sampling
            # Use max_nodes from the config
            num_nodes = self.config.max_nodes

            # Generate random initial positions
            if rngs is not None and hasattr(rngs, "params"):
                key = rngs.params()
            else:
                key = jax.random.PRNGKey(0)

            coordinates = jax.random.normal(key, shape=(batch_size, num_nodes, 3))

            # Important: Use self.node_dim and self.edge_dim to ensure compatibility
            # This will match the dimensions expected by the model's linear layers
            node_features = jnp.zeros((batch_size, num_nodes, self.node_dim))

            # Create a fully connected graph (adjacency matrix with all ones)
            adjacency = jnp.ones((batch_size, num_nodes, num_nodes))

            # Create empty edge features with correct dimensions
            edge_features = jnp.zeros((batch_size, num_nodes, num_nodes, self.edge_dim))

            # All nodes are valid (no mask)
            mask = jnp.ones((batch_size, num_nodes))

            # Create input dictionary
            x = {
                "node_features": node_features,
                "edge_features": edge_features,
                "coordinates": coordinates,
                "adjacency": adjacency,
                "mask": mask,
            }

        # Extract inputs from dictionary
        if isinstance(x, dict):
            node_features = x.get("node_features")
            edge_features = x.get("edge_features")
            coordinates = x.get("coordinates")
            adjacency = x.get("adjacency")
            mask = x.get("mask")
        else:
            # If x is not a dictionary, assume it's the coordinates
            coordinates = x
            # Create dummy node/edge features (will be overridden by extensions if needed)
            batch_size, num_nodes = coordinates.shape[0], coordinates.shape[1]
            node_features = jnp.zeros((batch_size, num_nodes, self.node_dim))
            # Create fully connected adjacency matrix (all nodes connected)
            adjacency = jnp.ones((batch_size, num_nodes, num_nodes))
            # Create dummy edge features
            edge_features = jnp.zeros((batch_size, num_nodes, num_nodes, self.edge_dim))
            # No mask (all nodes are valid)
            mask = jnp.ones((batch_size, num_nodes))

        # Apply node feature embedding
        h = self.node_embedding(node_features)

        # Apply edge feature embedding
        if edge_features is not None:
            e = self.edge_embedding(edge_features)
        else:
            # If no edge features provided, create zero vectors
            batch_size, num_nodes = node_features.shape[0], node_features.shape[1]
            e = jnp.zeros((batch_size, num_nodes, num_nodes, self.hidden_dim))

        # Normalize coordinates if needed
        if self.norm_coordinates and coordinates is not None:
            # Compute centroid for each graph in batch
            if mask is not None:
                # Only use valid nodes for centroid calculation
                mask_sum = jnp.sum(mask, axis=1, keepdims=True)
                mask_sum = jnp.maximum(mask_sum, 1.0)  # Avoid division by zero
                mask_for_mean = mask[:, :, None]  # [batch, num_nodes, 1]

                # Compute centroid with masked nodes
                centroid = (
                    jnp.sum(coordinates * mask_for_mean, axis=1, keepdims=True)
                    / mask_sum[:, :, None]
                )
            else:
                # Use all nodes for centroid calculation
                centroid = jnp.mean(coordinates, axis=1, keepdims=True)

            # Center coordinates
            coordinates = coordinates - centroid

            # Scale coordinates if needed
            if mask is not None:
                # Compute std with masked nodes
                squared_dist = jnp.sum((coordinates**2) * mask_for_mean, axis=1)
                std = jnp.sqrt(jnp.mean(squared_dist))
            else:
                # Compute std with all nodes
                std = jnp.std(coordinates)

            # Avoid division by zero
            std = jnp.maximum(std, 1e-6)

            # Scale to unit standard deviation
            coordinates = coordinates / std

        # Apply EGNN layers
        for layer in self.egnn_layers:
            h, coordinates, e = layer(
                h,
                coordinates,
                edge_index=adjacency,
                edge_features=e,
                mask=mask,
                deterministic=deterministic,
            )

        # Project node features back to original dimensions
        node_features_out = self.node_proj(h)

        # Prepare output dictionary
        outputs = {
            "node_features": node_features_out,
            "coordinates": coordinates,
            "edge_features": e,
        }

        # Apply extensions if available
        if hasattr(self, "extension_modules") and self.extension_modules:
            processed_outputs, extension_outputs = self.apply_extensions(x, outputs)
            outputs.update(processed_outputs)
            outputs["extension_outputs"] = extension_outputs

        return outputs

    def sample(self, n_samples: int, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate samples from the model.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generator keys

        Returns:
            Generated node positions with shape [n_samples, num_nodes, 3]
        """
        if rngs is None and hasattr(self, "rngs"):
            rngs = self.rngs
        elif rngs is None:
            # Create new RNGs if not provided
            key = jax.random.PRNGKey(0)
            rngs = nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 1))

        # Generate random initial positions (standard normal)
        if hasattr(rngs, "params"):
            key = rngs.params()
        else:
            key = jax.random.PRNGKey(0)

        # Process through the model directly (the __call__ method will handle input creation)
        # Pass None as input to trigger the automatic input generation
        # Pass n_samples as batch_size to generate the correct number of samples
        outputs = self(None, rngs=rngs, deterministic=True, batch_size=n_samples)

        # Return the generated coordinates
        return outputs["coordinates"]

    def get_loss_fn(self, auxiliary: dict[str, Any] | None = None) -> Any:
        """Get the loss function for the model.

        Args:
            auxiliary: Optional auxiliary outputs to use in the loss

        Returns:
            Loss function
        """
        auxiliary = auxiliary or {}

        def loss_fn(batch: dict[str, jax.Array], outputs: Any, **kwargs) -> dict[str, jax.Array]:
            """Calculate loss for graph generation.

            Args:
                batch: Batch of data with ground truth
                outputs: Model outputs
                **kwargs: Additional keyword arguments

            Returns:
                dictionary with losses
            """
            # Extract target values
            target_coords = batch.get("coordinates")
            target_node_features = batch.get("node_features")
            mask = batch.get("mask")

            # Extract model outputs
            pred_coords = outputs["coordinates"]
            pred_node_features = outputs["node_features"]

            losses = {}

            # Coordinate reconstruction loss
            if target_coords is not None and pred_coords is not None:
                if mask is not None:
                    # Apply mask to consider only valid nodes
                    mask_for_coords = mask[:, :, None]  # [batch, num_nodes, 1]
                    coord_error = ((pred_coords - target_coords) ** 2) * mask_for_coords
                    # Normalize by number of valid nodes
                    valid_count = jnp.sum(mask)
                    valid_count = jnp.maximum(valid_count, 1.0)  # Avoid division by zero
                    losses["coord_loss"] = jnp.sum(coord_error) / valid_count
                else:
                    # No mask, use MSE over all nodes
                    losses["coord_loss"] = jnp.mean((pred_coords - target_coords) ** 2)

            # Node feature reconstruction loss
            if target_node_features is not None and pred_node_features is not None:
                if mask is not None:
                    # Apply mask to consider only valid nodes
                    mask_for_features = mask[:, :, None]  # [batch, num_nodes, 1]
                    feat_error = (
                        (pred_node_features - target_node_features) ** 2
                    ) * mask_for_features
                    # Normalize by number of valid nodes
                    valid_count = jnp.sum(mask)
                    valid_count = jnp.maximum(valid_count, 1.0)  # Avoid division by zero
                    losses["feat_loss"] = jnp.sum(feat_error) / valid_count
                else:
                    # No mask, use MSE over all nodes
                    losses["feat_loss"] = jnp.mean((pred_node_features - target_node_features) ** 2)

            # Get extension losses
            extension_losses = self.get_extension_losses(batch, outputs, **kwargs)
            losses.update(extension_losses)

            # Compute total loss (sum of all losses)
            losses["total_loss"] = sum(
                loss for name, loss in losses.items() if name != "total_loss"
            )

            return losses

        return loss_fn
