"""E(n) Equivariant Graph Neural Network layers.

This module implements EGNN layers that maintain E(n) equivariance by
separating processing of scalar features and coordinate features.
Node features are updated using only scalar quantities (distances, dot
products) while coordinates are updated using direction vectors, ensuring
the output transforms correctly under rotations, reflections, and
translations.

Reference:
    Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021)
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx


logger = logging.getLogger(__name__)


class EGNNBlock(nnx.Module):
    """MLP block used within EGNN layers.

    Replaces the mixed nnx.List pattern (containing nnx.Linear,
    nnx.LayerNorm, bare callables, and string sentinels) with a proper
    module that handles all sub-operations internally.

    Attributes:
        layers: Sequence of linear layers.
        norms: Sequence of layer norm modules (one fewer than layers).
        dropout: Dropout module, or None if dropout_rate is 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        num_layers: int = 2,
        *,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the EGNN block.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            hidden_features: Hidden layer dimension.
            num_layers: Number of linear layers.
            dropout_rate: Dropout rate applied between layers.
            use_layer_norm: Whether to apply layer normalization.
            rngs: Random number generator keys.
        """
        super().__init__()

        layers = []
        norms = []
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            out_dim = out_features if i == num_layers - 1 else hidden_features
            layers.append(nnx.Linear(in_features=in_dim, out_features=out_dim, rngs=rngs))
            # Add norm for all layers except the last
            if i < num_layers - 1 and use_layer_norm:
                norms.append(nnx.LayerNorm(hidden_features, rngs=rngs))

        self.layers = nnx.List(layers)
        self.norms = nnx.List(norms)
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers

        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass through the MLP block.

        Args:
            x: Input tensor.
            deterministic: If True, disable dropout.

        Returns:
            Output tensor after MLP transformation.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply norm + activation + dropout for all except last layer
            if i < self.num_layers - 1:
                if self.use_layer_norm and i < len(self.norms):
                    x = self.norms[i](x)
                x = nnx.relu(x)
                if self.dropout is not None:
                    x = self.dropout(x, deterministic=deterministic)
        return x


class EGNNLayer(nnx.Module):
    """E(n) Equivariant Graph Neural Network Layer.

    This layer maintains E(n) equivariance by separating processing of
    scalar features and coordinate features.

    Architecture:
        edge_mlp:    [h_i, h_j, d^2_ij, e_ij] -> m_ij   (edge messages)
        attention:   [h_i, h_j, d^2_ij, e_ij] -> a_ij   (optional weights)
        node_update: [h_i, sum_j m_ij] -> h'_i           (node feature update)
        coord_mlp:   sum_j m_ij -> delta_x_i             (coordinate update)

    Equivariance guarantee:
        - Node features updated using only scalar quantities (distances)
        - Coordinates updated using only direction vectors (r_ij / |r_ij|)
        - No absolute positions enter scalar computations

    Attributes:
        edge_mlp: MLP for computing edge messages.
        node_update: MLP for updating node features from aggregated messages.
        coord_mlp: MLP for computing coordinate update scalars.
        attention: Optional linear layer for attention weights.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_mlp_layers: int = 2,
        *,
        dropout_rate: float = 0.1,
        use_attention: bool = True,
        residual: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the EGNN layer.

        Args:
            node_dim: Dimension of node features.
            edge_dim: Dimension of edge features.
            hidden_dim: Dimension of hidden layers.
            num_mlp_layers: Number of layers in each MLP block.
            dropout_rate: Dropout rate for MLP blocks.
            use_attention: Whether to use attention for message passing.
            residual: Whether to use residual connections for node features.
            rngs: Random number generator keys.
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.residual = residual

        # Edge MLP: [h_i, h_j, d^2, e_ij] -> messages
        edge_in_dim = hidden_dim * 2 + 1 + edge_dim
        self.edge_mlp = EGNNBlock(
            in_features=edge_in_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            num_layers=num_mlp_layers,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Coordinate MLP: aggregated messages -> scalar updates
        self.coord_mlp = EGNNBlock(
            in_features=hidden_dim,
            out_features=1,
            hidden_features=hidden_dim,
            num_layers=num_mlp_layers,
            dropout_rate=0.0,
            rngs=rngs,
        )

        # Node update MLP: [h_i, aggregated_messages] -> updated features
        self.node_update = EGNNBlock(
            in_features=hidden_dim * 2,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            num_layers=num_mlp_layers,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Attention mechanism
        if use_attention:
            self.attention = nnx.Linear(
                in_features=edge_in_dim,
                out_features=1,
                rngs=rngs,
            )

    def __call__(
        self,
        node_features: jax.Array,
        coordinates: jax.Array,
        edge_index: jax.Array,
        edge_features: jax.Array | None = None,
        mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass through the EGNN layer.

        Args:
            node_features: Node features [batch, num_nodes, hidden_dim].
            coordinates: Node coordinates [batch, num_nodes, 3].
            edge_index: Adjacency matrix [batch, num_nodes, num_nodes].
            edge_features: Edge features
                [batch, num_nodes, num_nodes, edge_dim] or None.
            mask: Node mask [batch, num_nodes] or None.
            deterministic: Whether to run in deterministic mode.

        Returns:
            Tuple of (node_features, coordinates, edge_features) with
            updated values.
        """
        num_nodes = node_features.shape[1]

        # Step 1: Compute relative positions and squared distances
        # [batch, N, 1, 3] - [batch, 1, N, 3] = [batch, N, N, 3]
        rel_pos = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        squared_dist = jnp.sum(rel_pos**2, axis=-1, keepdims=True)  # [batch, N, N, 1]

        # Step 2: Build edge inputs
        node_i = node_features[:, :, None, :].repeat(num_nodes, axis=2)
        node_j = node_features[:, None, :, :].repeat(num_nodes, axis=1)

        if edge_features is not None:
            edge_inputs = jnp.concatenate([node_i, node_j, squared_dist, edge_features], axis=-1)
        else:
            batch_size = node_features.shape[0]
            zero_edge = jnp.zeros((batch_size, num_nodes, num_nodes, self.edge_dim))
            edge_inputs = jnp.concatenate([node_i, node_j, squared_dist, zero_edge], axis=-1)

        # Step 3: Compute edge messages
        edge_messages = self.edge_mlp(edge_inputs, deterministic=deterministic)
        new_edge_features = edge_messages

        # Step 4: Apply attention if enabled
        if self.use_attention:
            attention_weights = jax.nn.sigmoid(self.attention(edge_inputs))
            edge_messages = edge_messages * attention_weights

        # Mask by adjacency
        edge_messages = edge_messages * edge_index[:, :, :, None]

        # Mask by valid nodes
        if mask is not None:
            message_mask = mask[:, :, None] * mask[:, None, :]
            edge_messages = edge_messages * message_mask[:, :, :, None]

        # Step 5: Aggregate messages per node
        node_messages = jnp.sum(edge_messages, axis=2)

        # Step 6: Update node features
        node_inputs = jnp.concatenate([node_features, node_messages], axis=-1)
        new_node_features = self.node_update(node_inputs, deterministic=deterministic)

        # Step 7: Compute coordinate updates
        coord_messages = self.coord_mlp(node_messages, deterministic=deterministic)

        # Normalize direction vectors
        norm = jnp.sqrt(jnp.sum(rel_pos**2, axis=-1, keepdims=True) + 1e-8)
        normalized_rel_pos = rel_pos / norm

        # Compute coordinate update direction
        if self.use_attention:
            coord_update_dir = jnp.sum(
                normalized_rel_pos * attention_weights * edge_index[:, :, :, None],
                axis=2,
            )
        else:
            coord_update_dir = jnp.sum(normalized_rel_pos * edge_index[:, :, :, None], axis=2)

        coord_updates = coord_update_dir * coord_messages

        if mask is not None:
            coord_updates = coord_updates * mask[:, :, None]

        new_coordinates = coordinates + coord_updates

        # Apply residual connection for node features
        if self.residual:
            new_node_features = node_features + new_node_features

        return new_node_features, new_coordinates, new_edge_features
