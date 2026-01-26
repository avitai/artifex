"""Multi-modal representations and processors.

This module provides processors for combining and aligning multiple modalities.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MultiModalProcessor(nnx.Module):
    """Basic multi-modal processor that combines multiple modalities."""

    def __init__(
        self,
        modalities: list[str],
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-modal processor.

        Args:
            modalities: List of modality names
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        super().__init__()
        self.modalities = modalities
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Create modality-specific encoders using nnx.Dict of nnx.Sequential
        encoders = {}
        for modality in modalities:
            input_dim = self._get_modality_input_dim(modality)
            layers = []

            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(
                    nnx.Linear(
                        in_features=input_dim if i == 0 else hidden_dims[i - 1],
                        out_features=hidden_dim,
                        rngs=rngs,
                    )
                )

            encoders[modality] = nnx.Sequential(*layers)

        self.encoders = nnx.Dict(encoders)

        # Fusion layer
        fusion_input_dim = len(modalities) * hidden_dims[-1]
        self.fusion = nnx.Linear(
            in_features=fusion_input_dim,
            out_features=output_dim,
            rngs=rngs,
        )

        # Activation
        self.activation = getattr(nnx, activation)

        # Dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout: nnx.Dropout | None = None  # type: ignore

    def _get_modality_input_dim(self, modality: str) -> int:
        """Get input dimension for a modality.

        Args:
            modality: Modality name

        Returns:
            Input dimension
        """
        # Simplified - in practice this would be configurable
        dim_map = {
            "image": 3072,  # 32x32x3 flattened
            "text": 100,  # Embedding dimension
            "audio": 256,  # Feature dimension
        }
        return dim_map.get(modality, 256)

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Process multi-modal inputs.

        Args:
            inputs: Dictionary of modality inputs
            deterministic: Whether to apply dropout

        Returns:
            Fused representation
        """
        encoded = []

        for modality in self.modalities:
            if modality in inputs:
                x = inputs[modality]

                # Flatten if needed
                if x.ndim > 1:
                    x = x.reshape(-1)

                # Apply modality-specific encoder
                x = self.encoders[modality](x)
                x = self.activation(x)

                if self.dropout is not None and not deterministic:
                    x = self.dropout(x, deterministic=deterministic)

                encoded.append(x)

        # Concatenate encoded modalities
        fused = jnp.concatenate(encoded, axis=-1)

        # Apply fusion layer
        output = self.fusion(fused)
        output = self.activation(output)

        return output


class CrossModalProcessor(nnx.Module):
    """Processor for cross-modal alignment."""

    def __init__(
        self,
        source_modality: str,
        target_modality: str,
        alignment_dim: int,
        use_attention: bool = True,
        hidden_dims: list[int] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize cross-modal processor.

        Args:
            source_modality: Source modality name
            target_modality: Target modality name
            alignment_dim: Dimension of aligned space
            use_attention: Whether to use attention mechanism
            hidden_dims: Hidden layer dimensions
            rngs: Random number generators
        """
        super().__init__()
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.alignment_dim = alignment_dim
        self.use_attention = use_attention

        if hidden_dims is None:
            hidden_dims = [512]

        # Source and target projectors
        self.source_projector = nnx.Sequential(
            *[
                nnx.Linear(
                    in_features=512 if i == 0 else hidden_dims[i - 1],
                    out_features=dim,
                    rngs=rngs,
                )
                for i, dim in enumerate(hidden_dims)
            ],
            nnx.Linear(
                in_features=hidden_dims[-1],
                out_features=alignment_dim,
                rngs=rngs,
            ),
        )

        self.target_projector = nnx.Sequential(
            *[
                nnx.Linear(
                    in_features=512 if i == 0 else hidden_dims[i - 1],
                    out_features=dim,
                    rngs=rngs,
                )
                for i, dim in enumerate(hidden_dims)
            ],
            nnx.Linear(
                in_features=hidden_dims[-1],
                out_features=alignment_dim,
                rngs=rngs,
            ),
        )

        # Optional attention mechanism
        if use_attention:
            self.attention = CrossModalAttention(
                query_dim=alignment_dim,
                key_dim=alignment_dim,
                value_dim=alignment_dim,
                num_heads=8,
                rngs=rngs,
            )

    def __call__(
        self,
        source_features: jax.Array,
        target_features: jax.Array,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Align source and target modality features.

        Args:
            source_features: Source modality features
            target_features: Target modality features
            deterministic: Whether to apply dropout

        Returns:
            Tuple of aligned source and target features
        """
        # Project to alignment space
        source_aligned = self.source_projector(source_features)
        target_aligned = self.target_projector(target_features)

        # Apply attention if enabled
        if self.use_attention and hasattr(self, "attention"):
            # Source attends to target
            source_aligned = self.attention(
                query=source_aligned.reshape(1, -1),
                key=target_aligned.reshape(1, -1),
                value=target_aligned.reshape(1, -1),
                deterministic=deterministic,
            ).reshape(-1)

        return source_aligned, target_aligned


class ModalityFusionProcessor(nnx.Module):
    """Advanced modality fusion processor."""

    def __init__(
        self,
        modalities: list[str],
        fusion_method: str = "attention",
        output_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize modality fusion processor.

        Args:
            modalities: List of modality names
            fusion_method: Fusion method ("attention", "gated", "concatenate")
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            rngs: Random number generators
        """
        super().__init__()
        self.modalities = modalities
        self.fusion_method = fusion_method
        self.output_dim = output_dim

        # Modality embeddings using nnx.Dict
        embeddings = {}
        for modality in modalities:
            embeddings[modality] = nnx.Param(jax.random.normal(rngs.params(), (hidden_dim,)))
        self.modality_embeddings = nnx.Dict(embeddings)

        # Fusion components based on method
        if fusion_method == "attention":
            self.attention = nnx.MultiHeadAttention(
                in_features=hidden_dim,
                num_heads=num_heads,
                qkv_features=hidden_dim,
                out_features=output_dim,
                rngs=rngs,
            )
        elif fusion_method == "gated":
            # Gating network
            self.gate_network = nnx.Sequential(
                nnx.Linear(len(modalities) * hidden_dim, hidden_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(hidden_dim, len(modalities), rngs=rngs),
                nnx.sigmoid,
            )

        # Output projection
        if fusion_method == "concatenate":
            input_dim = len(modalities) * hidden_dim
        elif fusion_method == "attention":
            # Attention output has same dimension as output_dim
            input_dim = output_dim
        else:
            input_dim = hidden_dim

        self.output_projection = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Fuse multiple modalities.

        Args:
            inputs: Dictionary of modality features
            deterministic: Whether to apply dropout

        Returns:
            Fused representation
        """
        # Add modality embeddings
        features = []
        for modality in self.modalities:
            if modality in inputs:
                feature = inputs[modality]
                # Add modality embedding
                feature = feature + self.modality_embeddings[modality]
                features.append(feature)

        if not features:
            return jnp.zeros((self.output_dim,))

        # Stack features
        stacked = jnp.stack(features)  # [num_modalities, hidden_dim]

        # Apply fusion
        if self.fusion_method == "attention":
            # Self-attention across modalities
            # Shape: [1, num_modalities, hidden_dim]
            attended = self.attention(
                stacked.reshape(1, len(features), -1),
                decode=False,
                deterministic=deterministic,
            )
            # attended shape: [1, num_modalities, output_dim]
            # Average pool across modalities
            fused = jnp.mean(attended, axis=1).squeeze(0)  # Shape: [output_dim]
        elif self.fusion_method == "gated":
            # Gated fusion
            concatenated = jnp.concatenate(features)
            gates = self.gate_network(concatenated)
            # Apply gates
            weighted = []
            for i, feature in enumerate(features):
                weighted.append(feature * gates[i])
            fused = jnp.sum(jnp.stack(weighted), axis=0)
        else:
            # Simple concatenation
            fused = jnp.concatenate(features)

        # Output projection (skip if attention already produces correct size)
        if self.fusion_method == "attention" and fused.shape[-1] == self.output_dim:
            return fused
        else:
            output = self.output_projection(fused)
            return output


class CrossModalAttention(nnx.Module):
    """Cross-modal attention mechanism."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize cross-modal attention.

        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            value_dim: Value dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        super().__init__()
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Multi-head attention
        self.attention = nnx.MultiHeadAttention(
            in_features=query_dim,
            num_heads=num_heads,
            qkv_features=value_dim,
            out_features=query_dim,
            rngs=rngs,
        )

        # Dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout: nnx.Dropout | None = None  # type: ignore

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply cross-modal attention.

        Args:
            query: Query features from one modality
            key: Key features from another modality
            value: Value features
            deterministic: Whether to apply dropout

        Returns:
            Attended features
        """
        # Ensure batch dimension
        if query.ndim == 2:
            query = query[jnp.newaxis, ...]
        if key.ndim == 2:
            key = key[jnp.newaxis, ...]
        if value.ndim == 2:
            value = value[jnp.newaxis, ...]

        # Apply attention
        attended = self.attention(
            query,
            key,
            value,
            decode=False,
            deterministic=deterministic,
        )

        # Apply dropout if needed
        if self.dropout is not None and not deterministic:
            attended = self.dropout(attended, deterministic=deterministic)

        # Remove batch dimension if it was added
        if attended.shape[0] == 1:
            attended = attended[0]

        return attended


class HierarchicalFusion(nnx.Module):
    """Hierarchical fusion of modalities.

    This class follows NNX's eager initialization philosophy - all parameters
    are created in __init__ rather than lazily on first call. This requires
    specifying input dimensions for each modality upfront.
    """

    def __init__(
        self,
        modality_groups: list[list[str]],
        group_fusion_dims: list[int],
        final_fusion_dim: int,
        modality_dims: dict[str, int],
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize hierarchical fusion.

        Args:
            modality_groups: Groups of modalities to fuse hierarchically
            group_fusion_dims: Fusion dimensions for each group
            final_fusion_dim: Final output dimension
            modality_dims: Dictionary mapping modality names to their input dimensions
            rngs: Random number generators
        """
        super().__init__()
        self.modality_groups = modality_groups
        self.group_fusion_dims = group_fusion_dims
        self.final_fusion_dim = final_fusion_dim

        # Eagerly create all projectors in __init__ (NNX philosophy)
        projectors: dict[str, nnx.Linear] = {}
        for group, fusion_dim in zip(modality_groups, group_fusion_dims):
            for modality in group:
                if modality in modality_dims:
                    input_dim = modality_dims[modality]
                    projectors[modality] = nnx.Linear(input_dim, fusion_dim, rngs=rngs)
        self.group_projectors = nnx.Dict(projectors)

        # Group fusion layers (post-projection) using nnx.List
        fusers = []
        for fusion_dim in group_fusion_dims:
            # Each group gets projected to a common dimension first
            fuser = nnx.Sequential(
                nnx.Linear(fusion_dim, fusion_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(fusion_dim, fusion_dim, rngs=rngs),
            )
            fusers.append(fuser)
        self.group_fusers = nnx.List(fusers)

        # Final fusion
        total_group_dim = sum(group_fusion_dims)
        self.final_fuser = nnx.Sequential(
            nnx.Linear(total_group_dim, final_fusion_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(final_fusion_dim, final_fusion_dim, rngs=rngs),
        )

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply hierarchical fusion.

        Args:
            inputs: Dictionary of modality features
            deterministic: Whether to apply dropout (unused, for interface compatibility)

        Returns:
            Hierarchically fused representation
        """
        del deterministic  # Unused

        group_outputs = []

        # Fuse each group
        for group, fuser in zip(self.modality_groups, self.group_fusers):
            group_features = []
            for modality in group:
                if modality in inputs:
                    # Project to group's fusion dimension using eagerly-initialized projector
                    # Use list() to iterate nnx.Dict keys (avoids membership check issues)
                    if modality in list(self.group_projectors):
                        projected = self.group_projectors[modality](inputs[modality])
                    else:
                        # Fallback if modality wasn't in modality_dims
                        projected = inputs[modality]
                    group_features.append(projected)

            if group_features:
                # Average the projected features (since they're all same dim now)
                if len(group_features) > 1:
                    group_input = jnp.mean(jnp.stack(group_features), axis=0)
                else:
                    group_input = group_features[0]

                group_output = fuser(group_input)
                group_outputs.append(group_output)

        # Final fusion
        if group_outputs:
            final_input = jnp.concatenate(group_outputs, axis=-1)
            output = self.final_fuser(final_input)
        else:
            output = jnp.zeros((self.final_fusion_dim,))

        return output


class ModalityDropout(nnx.Module):
    """Modality dropout for robustness."""

    def __init__(
        self,
        modalities: list[str],
        dropout_rate: float = 0.5,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize modality dropout.

        Args:
            modalities: List of modality names
            dropout_rate: Probability of dropping each modality
            rngs: Random number generators
        """
        super().__init__()
        self.modalities = modalities
        self.dropout_rate = dropout_rate
        self.rngs = rngs

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        deterministic: bool = False,
    ) -> dict[str, jax.Array]:
        """Apply modality dropout.

        Args:
            inputs: Dictionary of modality features
            deterministic: Whether to apply dropout

        Returns:
            Dictionary with some modalities potentially dropped
        """
        if deterministic or self.dropout_rate == 0:
            return inputs

        # Generate dropout mask for each modality
        outputs: dict[str, jax.Array] = {}
        for modality in self.modalities:
            if modality in inputs:
                # Randomly decide whether to keep this modality
                keep_prob = 1.0 - self.dropout_rate
                if "dropout" in self.rngs:
                    key = self.rngs.dropout()
                else:
                    key = jax.random.key(0)

                keep = jax.random.bernoulli(key, keep_prob)

                if keep:
                    # Scale by 1/keep_prob to maintain expected value
                    outputs[modality] = inputs[modality] / keep_prob

        # Ensure at least one modality is kept
        if not outputs and inputs:
            # Keep a random modality
            modality = list(inputs.keys())[0]
            outputs[modality] = inputs[modality]

        return outputs
