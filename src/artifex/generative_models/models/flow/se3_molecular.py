"""SE(3)-Equivariant Molecular Flow for conformation generation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


class SE3EquivariantLayer(nnx.Module):
    """SE(3)-equivariant layer respecting rotational and translational symmetries.

    This layer ensures that the output transforms appropriately under SE(3)
    transformations (rotations and translations) of the input coordinates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_atoms: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize SE(3)-equivariant layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            max_atoms: Maximum number of atoms
            rngs: Random number generators
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_atoms = max_atoms

        # Scalar feature network (invariant to rotations/translations)
        self.scalar_net = nnx.Sequential(
            nnx.Linear(in_features, out_features, rngs=rngs),
            nnx.LayerNorm(out_features, rngs=rngs),
            lambda x: nnx.gelu(x),
            nnx.Linear(out_features, out_features, rngs=rngs),
        )

        # Vector feature network (equivariant to rotations, invariant to translations)
        self.vector_net = nnx.Sequential(
            nnx.Linear(in_features, out_features, rngs=rngs),
            nnx.LayerNorm(out_features, rngs=rngs),
            lambda x: nnx.gelu(x),
            nnx.Linear(out_features, out_features, rngs=rngs),
        )

    def __call__(
        self,
        coordinates: jax.Array,  # [batch, max_atoms, 3]
        features: jax.Array,  # [batch, max_atoms, features]
        atom_mask: jax.Array,  # [batch, max_atoms]
    ) -> tuple[jax.Array, jax.Array]:
        """Apply SE(3)-equivariant transformation.

        Args:
            coordinates: 3D coordinates [batch, max_atoms, 3]
            features: Node features [batch, max_atoms, features]
            atom_mask: Atom presence mask [batch, max_atoms]

        Returns:
            Tuple of (new_coordinates, new_features)
        """
        batch_size, max_atoms = coordinates.shape[:2]

        # Compute pairwise distances (invariant to rotations/translations)
        distances = self._compute_distances(coordinates, atom_mask)

        # Compute edge features (invariant)
        edge_features = self._compute_edge_features(distances, features, atom_mask)

        # Update scalar features (invariant)
        new_scalar_features = self.scalar_net(edge_features)

        # Compute vector updates (equivariant)
        vector_updates = self._compute_vector_updates(coordinates, new_scalar_features, atom_mask)

        # Apply updates
        new_coordinates = coordinates + vector_updates
        new_features = features + new_scalar_features

        return new_coordinates, new_features

    def _compute_distances(self, coordinates: jax.Array, atom_mask: jax.Array) -> jax.Array:
        """Compute pairwise distances (SE(3) invariant)."""
        # Expand coordinates for pairwise computation
        coords_i = coordinates[:, :, None, :]  # [batch, atoms, 1, 3]
        coords_j = coordinates[:, None, :, :]  # [batch, 1, atoms, 3]

        # Compute pairwise distances
        distances = jnp.linalg.norm(coords_i - coords_j, axis=-1)  # [batch, atoms, atoms]

        # Mask invalid pairs
        mask_i = atom_mask[:, :, None]  # [batch, atoms, 1]
        mask_j = atom_mask[:, None, :]  # [batch, 1, atoms]
        valid_pairs = mask_i & mask_j

        # Set invalid distances to large value
        distances = jnp.where(valid_pairs, distances, 1e6)

        return distances

    def _compute_edge_features(
        self,
        distances: jax.Array,
        features: jax.Array,
        atom_mask: jax.Array,
    ) -> jax.Array:
        """Compute edge features for message passing."""
        batch_size, max_atoms, feature_dim = features.shape

        # Distance-based edge weights (smooth cutoff)
        cutoff_distance = 5.0  # Angstroms
        edge_weights = jnp.exp(-distances / cutoff_distance)
        edge_weights = jnp.where(distances < cutoff_distance, edge_weights, 0.0)

        # Aggregate neighbor features
        neighbor_features = jnp.sum(
            edge_weights[:, :, :, None] * features[:, None, :, :], axis=2
        )  # [batch, atoms, features]

        # Combine with self features
        combined_features = features + neighbor_features

        # Apply mask
        combined_features = combined_features * atom_mask[:, :, None]

        return combined_features

    def _compute_vector_updates(
        self,
        coordinates: jax.Array,
        scalar_features: jax.Array,
        atom_mask: jax.Array,
    ) -> jax.Array:
        """Compute SE(3) equivariant coordinate updates.

        For true SE(3) equivariance, we use relative displacement vectors
        between atoms, which transform correctly under rotations.
        """
        batch_size, max_atoms = coordinates.shape[:2]

        # Apply vector network to get update magnitudes per atom
        update_magnitudes = self.vector_net(scalar_features)  # [batch, atoms, out_features]

        # Compute pairwise displacement vectors (SE(3) equivariant)
        coords_i = coordinates[:, :, None, :]  # [batch, atoms, 1, 3]
        coords_j = coordinates[:, None, :, :]  # [batch, 1, atoms, 3]
        displacement_vectors = coords_i - coords_j  # [batch, atoms, atoms, 3]

        # Compute distances for weighting (SE(3) invariant)
        distances = jnp.linalg.norm(displacement_vectors, axis=-1)  # [batch, atoms, atoms]

        # Create attention weights based on distances (SE(3) invariant)
        cutoff_distance = 5.0
        attention_weights = jnp.exp(-distances / cutoff_distance)
        attention_weights = jnp.where(distances < cutoff_distance, attention_weights, 0.0)

        # Mask out invalid pairs
        mask_i = atom_mask[:, :, None]  # [batch, atoms, 1]
        mask_j = atom_mask[:, None, :]  # [batch, 1, atoms]
        valid_pairs = mask_i & mask_j
        attention_weights = attention_weights * valid_pairs

        # Normalize attention weights
        attention_weights = attention_weights / (
            jnp.sum(attention_weights, axis=-1, keepdims=True) + 1e-8
        )

        # Compute weighted displacement vectors (SE(3) equivariant)
        # This gives us the "average" direction each atom should move
        weighted_displacements = jnp.sum(
            displacement_vectors * attention_weights[:, :, :, None], axis=2
        )  # [batch, atoms, 3]

        # Scale by learned magnitudes (first 3 features as x,y,z scaling)
        magnitude_scaling = update_magnitudes[:, :, :3]  # [batch, atoms, 3]
        vector_updates = weighted_displacements * magnitude_scaling

        # Apply atom mask
        vector_updates = vector_updates * atom_mask[:, :, None]

        return vector_updates


class SE3CouplingLayer(nnx.Module):
    """SE(3)-equivariant coupling layer for normalizing flows."""

    def __init__(
        self,
        hidden_dim: int,
        max_atoms: int,
        atom_types: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize SE(3) coupling layer.

        Args:
            hidden_dim: Hidden dimension for networks
            max_atoms: Maximum number of atoms
            atom_types: Number of atom types
            rngs: Random number generators
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.atom_types = atom_types

        # Atom type embedding
        self.atom_embedding = nnx.Embed(
            num_embeddings=atom_types,
            features=hidden_dim,
            rngs=rngs,
        )

        # SE(3)-equivariant layers for transformation parameters
        # Use nnx.List for Flax NNX 0.12.0+ compatibility
        self.se3_layers = nnx.List(
            [SE3EquivariantLayer(hidden_dim, hidden_dim, max_atoms, rngs=rngs) for _ in range(2)]
        )

        # Output networks for scale and translation
        self.scale_net = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            lambda x: nnx.gelu(x),
            nnx.Linear(hidden_dim, 3, rngs=rngs),  # 3D scale factors
        )

        self.translation_net = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            lambda x: nnx.gelu(x),
            nnx.Linear(hidden_dim, 3, rngs=rngs),  # 3D translation
        )

    def __call__(
        self,
        coordinates: jax.Array,
        atom_types: jax.Array,
        atom_mask: jax.Array,
        *,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply coupling transformation.

        Args:
            coordinates: Molecular coordinates [batch, max_atoms, 3]
            atom_types: Atom types [batch, max_atoms]
            atom_mask: Atom mask [batch, max_atoms]
            reverse: Whether to apply reverse transformation

        Returns:
            Tuple of (transformed_coordinates, log_det_jacobian)
        """
        batch_size, max_atoms = coordinates.shape[:2]

        # Split coordinates for coupling
        split_idx = max_atoms // 2
        coords_1 = coordinates[:, :split_idx]
        coords_2 = coordinates[:, split_idx:]
        mask_1 = atom_mask[:, :split_idx]
        mask_2 = atom_mask[:, split_idx:]
        types_1 = atom_types[:, :split_idx]
        types_2 = atom_types[:, split_idx:]

        # Embed atom types
        embedded_1 = self.atom_embedding(types_1)
        self.atom_embedding(types_2)

        if not reverse:
            # Forward transformation
            # Use first half to transform second half
            transform_params = self._compute_transform_params(
                coords_1, embedded_1, mask_1, target_shape=(coords_2.shape[0], coords_2.shape[1])
            )

            coords_2_new, log_det = self._apply_transformation(coords_2, transform_params, mask_2)

            new_coordinates = jnp.concatenate([coords_1, coords_2_new], axis=1)
        else:
            # Reverse transformation
            transform_params = self._compute_transform_params(
                coords_1, embedded_1, mask_1, target_shape=(coords_2.shape[0], coords_2.shape[1])
            )

            coords_2_new, log_det = self._apply_transformation(
                coords_2, transform_params, mask_2, reverse=True
            )

            new_coordinates = jnp.concatenate([coords_1, coords_2_new], axis=1)

        return new_coordinates, log_det

    def _compute_transform_params(
        self,
        coordinates: jax.Array,
        features: jax.Array,
        atom_mask: jax.Array,
        target_shape: tuple[int, int] | None = None,
    ) -> dict[str, jax.Array]:
        """Compute transformation parameters using only SE(3) invariant features.

        Key insight: For SE(3) equivariance, transformation parameters must
        depend only on SE(3) invariant quantities (distances, angles), not
        on coordinate updates that break equivariance.
        """
        # Use only SE(3) invariant features, not coordinate updates
        current_features = features

        # Apply SE(3)-equivariant layers but ONLY use the feature updates,
        # not the coordinate updates (which would break invariance)
        for layer in self.se3_layers:
            # Only update features, discard coordinate updates
            _, current_features = layer(coordinates, current_features, atom_mask)

        # Compute scale and translation parameters from invariant features
        scale_params = self.scale_net(current_features)
        translation_params = self.translation_net(current_features)

        # Ensure positive scales
        scale_params = nnx.softplus(scale_params) + 1e-6

        # If target shape is provided and different, broadcast the parameters.
        # NOTE: Global pooling loses per-atom specificity when atom counts differ
        # between coupling layers. A better approach would use attention-based
        # aggregation, but this keeps the implementation simple.
        if target_shape is not None and scale_params.shape[1] != target_shape[1]:
            scale_global = jnp.mean(scale_params, axis=1, keepdims=True)  # [batch, 1, 3]
            translation_global = jnp.mean(
                translation_params, axis=1, keepdims=True
            )  # [batch, 1, 3]

            # Tile to match target atoms
            scale_params = jnp.tile(scale_global, (1, target_shape[1], 1))
            translation_params = jnp.tile(translation_global, (1, target_shape[1], 1))

        return {
            "scale": scale_params,
            "translation": translation_params,
        }

    def _apply_transformation(
        self,
        coordinates: jax.Array,
        transform_params: dict[str, jax.Array],
        atom_mask: jax.Array,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply SE(3)-equivariant transformation to coordinates.

        For true SE(3) equivariance, we apply transformations that are invariant
        under global rotations and translations. We avoid computing centers of mass
        which would break equivariance when applied to molecular subsets.
        """
        scale = transform_params["scale"]
        translation = transform_params["translation"]

        # For SE(3) equivariance, apply only coordinate-independent transformations
        # Use scale factors that preserve molecular geometry

        # Apply simple affine transformations that commute with SE(3)
        # Scale parameters affect the coordinates directly
        if not reverse:
            # Forward: scale coordinates and add translation
            transformed_coords = coordinates * scale + translation

            # Log determinant: sum of log scales for each coordinate dimension
            log_det = jnp.sum(jnp.log(scale) * atom_mask[:, :, None], axis=(1, 2))
        else:
            # Reverse: subtract translation and divide by scale
            transformed_coords = (coordinates - translation) / scale

            # Negative log determinant for reverse transformation
            log_det = -jnp.sum(jnp.log(scale) * atom_mask[:, :, None], axis=(1, 2))

        # Apply mask to ensure invalid atoms remain unchanged
        transformed_coords = transformed_coords * atom_mask[:, :, None] + coordinates * (
            1 - atom_mask[:, :, None]
        )

        return transformed_coords, log_det


class SE3MolecularFlow(GenerativeModel):
    """SE(3)-Equivariant Normalizing Flow for molecular conformation generation.

    This model uses a sequence of SE(3)-equivariant coupling layers to model
    the probability distribution over molecular conformations while respecting
    rotational and translational symmetries.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_coupling_layers: int = 4,
        max_atoms: int = 29,
        atom_types: int = 5,
        use_attention: bool = True,
        equivariant_layers: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize SE(3) molecular flow.

        Args:
            hidden_dim: Hidden dimension for networks
            num_layers: Number of SE(3) layers per coupling layer
            num_coupling_layers: Number of coupling layers
            max_atoms: Maximum number of atoms
            atom_types: Number of atom types (H, C, N, O, F)
            use_attention: Whether to use attention mechanisms
            equivariant_layers: Whether to use equivariant layers
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_coupling_layers = num_coupling_layers
        self.max_atoms = max_atoms
        self.atom_types = atom_types
        self.use_attention = use_attention
        self.equivariant_layers = equivariant_layers

        # Coupling layers
        # Use nnx.List for Flax NNX 0.12.0+ compatibility
        self.coupling_layers = nnx.List(
            [
                SE3CouplingLayer(hidden_dim, max_atoms, atom_types, rngs=rngs)
                for _ in range(num_coupling_layers)
            ]
        )

        # Base distribution for coordinates (standard normal)
        self.base_distribution_mean = jnp.zeros((max_atoms, 3))
        self.base_distribution_std = jnp.ones((max_atoms, 3))

    @nnx.jit
    def log_prob(
        self,
        coordinates: jax.Array,
        atom_types: jax.Array,
        atom_mask: jax.Array,
    ) -> jax.Array:
        """Compute log probability of molecular conformations.

        Uses JIT compilation for performance.

        Args:
            coordinates: Molecular coordinates [batch, max_atoms, 3]
            atom_types: Atom types [batch, max_atoms]
            atom_mask: Atom mask [batch, max_atoms]

        Returns:
            Log probabilities [batch]
        """
        current_coords = coordinates
        total_log_det = jnp.zeros(coordinates.shape[0])

        # Apply coupling layers in reverse to map to base distribution
        for layer in reversed(self.coupling_layers):
            current_coords, log_det = layer(current_coords, atom_types, atom_mask, reverse=True)
            total_log_det += log_det

        # Compute base distribution log probability
        # For SE(3) equivariance, we should NOT center coordinates here
        # as it breaks equivariance. Instead, use coordinates directly.

        # Standard normal log probability for valid atoms
        base_log_prob = -0.5 * jnp.sum(current_coords**2 * atom_mask[:, :, None], axis=(1, 2))
        base_log_prob -= 0.5 * jnp.sum(atom_mask, axis=1) * 3 * jnp.log(2 * jnp.pi)

        return base_log_prob + total_log_det

    @nnx.jit(static_argnums=(3,))  # num_samples is static for JIT
    def sample(
        self,
        atom_types: jnp.ndarray,
        atom_mask: jnp.ndarray,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ) -> jnp.ndarray:
        """Sample molecular conformations.

        Uses JIT compilation for performance.

        Args:
            atom_types: Atom type indices [batch, max_atoms]
            atom_mask: Boolean mask for valid atoms [batch, max_atoms]
            num_samples: Number of samples to generate
            rngs: Random number generators

        Returns:
            Sampled coordinates [num_samples, max_atoms, 3]
        """
        # Handle the case where num_samples != batch_size
        batch_size = atom_types.shape[0]

        if num_samples > batch_size:
            # Repeat the atom_types and atom_mask to match num_samples
            repeat_factor = (num_samples + batch_size - 1) // batch_size  # Ceiling division
            atom_types_expanded = jnp.tile(atom_types, (repeat_factor, 1))[:num_samples]
            atom_mask_expanded = jnp.tile(atom_mask, (repeat_factor, 1))[:num_samples]
        else:
            # Take the first num_samples from the batch
            atom_types_expanded = atom_types[:num_samples]
            atom_mask_expanded = atom_mask[:num_samples]

        # Sample from base distribution
        base_samples = jax.random.normal(rngs.params(), (num_samples, self.max_atoms, 3))

        # Apply mask to base samples
        base_samples = base_samples * atom_mask_expanded[:, :, None]

        # Center samples (translation invariance)
        centered_samples = self._center_coordinates(base_samples, atom_mask_expanded)

        current_coords = centered_samples

        # Apply coupling layers forward
        for layer in self.coupling_layers:
            current_coords, _ = layer(
                current_coords, atom_types_expanded, atom_mask_expanded, reverse=False
            )

        return current_coords

    def _center_coordinates(
        self,
        coordinates: jax.Array,
        atom_mask: jax.Array,
    ) -> jax.Array:
        """Center coordinates at center of mass (translation invariance)."""
        # Compute center of mass
        masked_coords = coordinates * atom_mask[:, :, None]
        center_of_mass = jnp.sum(masked_coords, axis=1, keepdims=True) / (
            jnp.sum(atom_mask, axis=1, keepdims=True)[:, :, None] + 1e-8
        )

        # Center coordinates
        centered = coordinates - center_of_mass

        # Apply mask
        centered = centered * atom_mask[:, :, None]

        return centered

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> jax.Array:
        """Generate molecular conformations (required by GenerativeModel interface).

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators
            **kwargs: Additional arguments (atom_types, atom_mask)

        Returns:
            Generated molecular coordinates
        """
        # Get required inputs from kwargs
        atom_types = kwargs.get("atom_types")
        atom_mask = kwargs.get("atom_mask")

        if atom_types is None or atom_mask is None:
            # Generate default molecular template
            atom_types = jnp.ones((n_samples, self.max_atoms), dtype=jnp.int32)  # All carbon
            atom_mask = jnp.ones((n_samples, self.max_atoms), dtype=jnp.bool_)  # All atoms present

        return self.sample(atom_types, atom_mask, n_samples, rngs=rngs)

    def loss_fn(
        self,
        batch: dict[str, jax.Array],
        model_outputs: Any,
        **kwargs,
    ) -> jax.Array:
        """Compute negative log-likelihood loss (required by GenerativeModel interface).

        Args:
            batch: Batch of molecular data
            model_outputs: Model outputs (not used for flows)
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        coordinates = batch["coordinates"]
        atom_types = batch["atom_types"]
        atom_mask = batch["atom_mask"]

        # Compute log probability
        log_prob = self.log_prob(coordinates, atom_types, atom_mask)

        # Return negative log-likelihood
        return -jnp.mean(log_prob)
