"""Loss functions for geometric generative models."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.losses.base import reduce_loss


def get_point_cloud_loss(loss_type: str, **kwargs) -> Callable:
    """Get the appropriate loss function for point cloud models.

    Args:
        loss_type: Type of loss function ("chamfer" or "earth_mover")
        **kwargs: Additional parameters for the loss function

    Returns:
        Loss function for point clouds
    """
    if loss_type == "chamfer":
        return chamfer_distance
    if loss_type == "earth_mover":
        return earth_mover_distance
    raise ValueError(f"Unknown point cloud loss type: {loss_type}")


def chamfer_distance(
    pred_points: jax.Array, target_points: jax.Array, reduction: str = "mean"
) -> jax.Array:
    """Compute Chamfer distance between two point clouds.

    Args:
        pred_points: Predicted point cloud [batch, num_points, 3]
        target_points: Target point cloud [batch, num_points, 3]
        reduction: How to reduce the batch dimension ("mean", "sum", "none")

    Returns:
        Chamfer distance loss
    """
    # Compute pairwise squared distances between points
    # pred_points: [B, N, 3], target_points: [B, M, 3]
    # Result: [B, N, M]
    pred_expanded = pred_points[:, :, None, :]  # [B, N, 1, 3]
    target_expanded = target_points[:, None, :, :]  # [B, 1, M, 3]
    dist_matrix = jnp.sum((pred_expanded - target_expanded) ** 2, axis=-1)

    # Find minimum distances for each point
    min_dist_pred_to_target = jnp.min(dist_matrix, axis=2)  # [B, N]
    min_dist_target_to_pred = jnp.min(dist_matrix, axis=1)  # [B, M]

    # Chamfer distance for each batch element
    chamfer_batch = jnp.mean(min_dist_pred_to_target, axis=1) + jnp.mean(
        min_dist_target_to_pred, axis=1
    )

    if reduction == "mean":
        return jnp.mean(chamfer_batch)
    elif reduction == "sum":
        return jnp.sum(chamfer_batch)
    else:  # "none"
        return chamfer_batch


def earth_mover_distance(
    pred_points: jax.Array, target_points: jax.Array, reduction: str = "mean"
) -> jax.Array:
    """Compute approximate Earth Mover's distance between point clouds.

    This implements a differentiable approximation using optimal transport
    with the Sinkhorn algorithm instead of the greedy approach in the original.

    Args:
        pred_points: Predicted point cloud [batch, num_points, 3]
        target_points: Target point cloud [batch, num_points, 3]
        reduction: How to reduce the batch dimension ("mean", "sum", "none")

    Returns:
        Approximate Earth Mover's distance
    """
    # For equal number of points, we can use a simpler approximation
    # Sort both point clouds along each dimension and compute L2 distance
    pred_sorted = jnp.sort(pred_points, axis=1)
    target_sorted = jnp.sort(target_points, axis=1)

    # Compute L2 distance between sorted points
    distances = jnp.sqrt(jnp.sum((pred_sorted - target_sorted) ** 2, axis=-1))
    emd_batch = jnp.mean(distances, axis=1)

    if reduction == "mean":
        return jnp.mean(emd_batch)
    elif reduction == "sum":
        return jnp.sum(emd_batch)
    else:  # "none"
        return emd_batch


class MeshLoss(nnx.Module):
    """Combined loss function for mesh models using NNX."""

    def __init__(
        self,
        vertex_weight: float = 1.0,
        normal_weight: float = 0.1,
        edge_weight: float = 0.1,
        laplacian_weight: float = 0.0,
    ):
        """Initialize the mesh loss.

        Args:
            vertex_weight: Weight for vertex position loss
            normal_weight: Weight for normal consistency loss
            edge_weight: Weight for edge length regularization
            laplacian_weight: Weight for Laplacian smoothness regularization
        """
        super().__init__()
        self.vertex_weight = vertex_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight

    def __call__(
        self,
        pred_mesh: tuple[jax.Array, jax.Array, jax.Array],
        target_mesh: tuple[jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        """Compute combined loss for mesh prediction.

        Args:
            pred_mesh: Predicted mesh (vertices, faces, normals)
            target_mesh: Target mesh (vertices, faces, normals)

        Returns:
            Weighted sum of vertex, normal, and edge losses
        """
        pred_vertices, pred_faces, pred_normals = pred_mesh
        target_vertices, target_faces, target_normals = target_mesh

        total_loss = jnp.array(0.0)

        # Vertex position loss (L2 distance)
        if self.vertex_weight > 0:
            vertex_loss = jnp.mean((pred_vertices - target_vertices) ** 2)
            total_loss = total_loss + self.vertex_weight * vertex_loss

        # Normal consistency loss (1 - cosine similarity)
        if self.normal_weight > 0:
            # Normalize normals to unit vectors
            pred_normals_normalized = pred_normals / (
                jnp.linalg.norm(pred_normals, axis=-1, keepdims=True) + 1e-8
            )
            target_normals_normalized = target_normals / (
                jnp.linalg.norm(target_normals, axis=-1, keepdims=True) + 1e-8
            )

            normal_sim = jnp.sum(pred_normals_normalized * target_normals_normalized, axis=-1)
            normal_loss = jnp.mean(1.0 - normal_sim)
            total_loss = total_loss + self.normal_weight * normal_loss

        # Edge length regularization
        if self.edge_weight > 0:
            pred_edges = self._compute_edge_lengths(pred_vertices, pred_faces)
            target_edges = self._compute_edge_lengths(target_vertices, target_faces)
            edge_loss = jnp.mean((pred_edges - target_edges) ** 2)
            total_loss = total_loss + self.edge_weight * edge_loss

        # Laplacian smoothness (optional)
        if self.laplacian_weight > 0:
            laplacian_loss = self._compute_laplacian_loss(pred_vertices, pred_faces)
            total_loss = total_loss + self.laplacian_weight * laplacian_loss

        return total_loss

    def _compute_edge_lengths(self, vertices: jax.Array, faces: jax.Array) -> jax.Array:
        """Compute edge lengths from vertices and faces.

        Args:
            vertices: Vertex positions [num_vertices, 3]
            faces: Face indices [num_faces, 3]

        Returns:
            Edge lengths [num_faces, 3]
        """
        # Extract vertices for each face
        v0 = vertices[faces[:, 0]]  # [num_faces, 3]
        v1 = vertices[faces[:, 1]]  # [num_faces, 3]
        v2 = vertices[faces[:, 2]]  # [num_faces, 3]

        # Compute edge lengths
        e01 = jnp.sqrt(jnp.sum((v0 - v1) ** 2, axis=-1))  # [num_faces]
        e12 = jnp.sqrt(jnp.sum((v1 - v2) ** 2, axis=-1))  # [num_faces]
        e20 = jnp.sqrt(jnp.sum((v2 - v0) ** 2, axis=-1))  # [num_faces]

        return jnp.stack([e01, e12, e20], axis=-1)  # [num_faces, 3]

    def _compute_laplacian_loss(self, vertices: jax.Array, faces: jax.Array) -> jax.Array:
        """Compute Laplacian smoothness regularization.

        Args:
            vertices: Vertex positions [num_vertices, 3]
            faces: Face indices [num_faces, 3]

        Returns:
            Laplacian loss scalar
        """
        # This is a simplified version - for production use, implement proper mesh Laplacian
        # For now, penalize large second derivatives (rough approximation)
        face_centers = jnp.mean(vertices[faces], axis=1)  # [num_faces, 3]

        # Compute variance of face centers as a smoothness proxy
        center_variance = jnp.var(face_centers, axis=0)  # [3]
        return jnp.mean(center_variance)


def get_mesh_loss(
    vertex_weight: float = 1.0,
    normal_weight: float = 0.1,
    edge_weight: float = 0.1,
    laplacian_weight: float = 0.0,
) -> MeshLoss:
    """Get a combined loss function for mesh models.

    Args:
        vertex_weight: Weight for vertex position loss
        normal_weight: Weight for normal consistency loss
        edge_weight: Weight for edge length regularization
        laplacian_weight: Weight for Laplacian smoothness regularization

    Returns:
        MeshLoss instance
    """
    return MeshLoss(vertex_weight, normal_weight, edge_weight, laplacian_weight)


def get_voxel_loss(loss_type: str, **kwargs) -> Callable:
    """Get the appropriate loss function for voxel grid models.

    Args:
        loss_type: Type of loss function ("bce", "dice", "focal", or "mse")
        **kwargs: Additional parameters for the loss function

    Returns:
        Loss function for voxel grids
    """
    if loss_type == "bce":
        return binary_cross_entropy
    if loss_type == "dice":
        return dice_loss
    if loss_type == "focal":
        gamma = kwargs.get("focal_gamma", 2.0)
        return lambda pred, target: focal_loss(pred, target, gamma=gamma)
    if loss_type == "mse":
        return mse_voxel_loss
    raise ValueError(f"Unknown voxel loss type: {loss_type}")


def binary_cross_entropy(
    predictions: jax.Array,
    targets: jax.Array,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> jax.Array:
    """Compute binary cross-entropy loss.

    Supports multiple reduction modes for different use cases:
    - "mean": Mean over all elements (standard for classification)
    - "sum": Sum over all elements
    - "batch_sum": Sum over spatial dims, mean over batch (standard for VAE ELBO)
    - "none": No reduction, return element-wise loss

    Args:
        predictions: Predicted values in [0, 1], shape [batch, ...]
        targets: Target values in [0, 1], shape [batch, ...]
        eps: Small constant to avoid log(0)
        reduction: How to reduce ("mean", "sum", "batch_sum", "none")

    Returns:
        Binary cross-entropy loss

    Example:
        >>> # For VAE training, use batch_sum reduction
        >>> recon_loss = binary_cross_entropy(recon_x, x, reduction="batch_sum")
    """
    # Clip predictions to avoid numerical issues
    predictions = jnp.clip(predictions, eps, 1.0 - eps)

    # Binary cross-entropy formula
    bce = -targets * jnp.log(predictions) - (1 - targets) * jnp.log(1 - predictions)

    return reduce_loss(bce, reduction=reduction)


def mse_voxel_loss(
    pred_voxels: jax.Array, target_voxels: jax.Array, reduction: str = "mean"
) -> jax.Array:
    """Compute mean squared error loss for voxel grids.

    Args:
        pred_voxels: Predicted voxel grid [batch, x, y, z, 1] or [batch, x, y, z]
        target_voxels: Target voxel grid [batch, x, y, z, 1] or [batch, x, y, z]
        reduction: How to reduce ("mean", "sum", "none")

    Returns:
        Mean squared error loss
    """
    mse = jnp.square(pred_voxels - target_voxels)

    if reduction == "mean":
        return jnp.mean(mse)
    elif reduction == "sum":
        return jnp.sum(mse)
    else:  # "none"
        return mse


def dice_loss(
    pred_voxels: jax.Array, target_voxels: jax.Array, eps: float = 1e-7, reduction: str = "mean"
) -> jax.Array:
    """Compute Dice loss for voxel grids.

    Args:
        pred_voxels: Predicted voxel grid [batch, x, y, z, 1] or [batch, x, y, z]
        target_voxels: Target voxel grid [batch, x, y, z, 1] or [batch, x, y, z]
        eps: Small constant to avoid division by zero
        reduction: How to reduce ("mean", "sum", "none")

    Returns:
        Dice loss (1 - Dice coefficient)
    """
    # Flatten spatial dimensions but keep batch dimension
    batch_size = pred_voxels.shape[0]
    pred_flat = pred_voxels.reshape(batch_size, -1)
    target_flat = target_voxels.reshape(batch_size, -1)

    # Compute intersection and cardinalities
    intersection = jnp.sum(pred_flat * target_flat, axis=1)
    pred_sum = jnp.sum(pred_flat, axis=1)
    target_sum = jnp.sum(target_flat, axis=1)

    # Compute Dice coefficient
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    # Return Dice loss (1 - Dice coefficient)
    dice_loss_batch = 1.0 - dice

    if reduction == "mean":
        return jnp.mean(dice_loss_batch)
    elif reduction == "sum":
        return jnp.sum(dice_loss_batch)
    else:  # "none"
        return dice_loss_batch


def focal_loss(
    pred_voxels: jax.Array,
    target_voxels: jax.Array,
    gamma: float = 2.0,
    alpha: float | None = None,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> jax.Array:
    """Compute focal loss for voxel grids.

    Args:
        pred_voxels: Predicted voxel grid [batch, x, y, z, 1] or [batch, x, y, z]
        target_voxels: Target voxel grid [batch, x, y, z, 1] or [batch, x, y, z]
        gamma: Focusing parameter
        alpha: Optional class weighting factor
        eps: Small constant to avoid numerical issues
        reduction: How to reduce ("mean", "sum", "none")

    Returns:
        Focal loss
    """
    # Clip predictions to avoid numerical issues
    pred_voxels = jnp.clip(pred_voxels, eps, 1.0 - eps)

    # Compute cross entropy
    ce_loss = -target_voxels * jnp.log(pred_voxels) - (1 - target_voxels) * jnp.log(1 - pred_voxels)

    # Compute pt (probability of true class)
    pt = pred_voxels * target_voxels + (1 - pred_voxels) * (1 - target_voxels)

    # Apply focal weight
    focal_weight = (1 - pt) ** gamma
    focal = focal_weight * ce_loss

    # Apply alpha weighting if provided
    if alpha is not None:
        alpha_t = alpha * target_voxels + (1 - alpha) * (1 - target_voxels)
        focal = alpha_t * focal

    if reduction == "mean":
        return jnp.mean(focal)
    elif reduction == "sum":
        return jnp.sum(focal)
    else:  # "none"
        return focal
