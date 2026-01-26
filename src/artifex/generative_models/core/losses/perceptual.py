"""
Perceptual losses module.

This module provides loss functions that compare features extracted from
neural networks, rather than direct pixel-wise comparisons. These losses
are especially useful for image generation tasks.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.losses.base import reduce_loss
from artifex.generative_models.core.losses.reconstruction import mse_loss


def feature_reconstruction_loss(
    predictions: jax.Array | list[jax.Array] | dict[str, jax.Array],
    targets: jax.Array | list[jax.Array] | dict[str, jax.Array],
    weights: list[float] | dict[str, float] | None = None,
    reduction: str = "mean",
    distance_fn: Callable = mse_loss,
) -> jax.Array:
    """
    Feature reconstruction loss between predicted and target features.

    Computes the distance between features extracted from generated and target images,
    typically using a pre-trained network as a feature extractor.

    Args:
        predictions: Features extracted from generated/predicted images
        targets: Features extracted from target/real images
        weights: Optional weights for each feature layer
        reduction: Reduction method ('none', 'mean', 'sum')
        distance_fn: Function to compute distance between features
            (default: MSE loss). Signature: fn(predictions, targets) -> loss

    Returns:
        Feature reconstruction loss

    Example:
        >>> feat_pred = {"conv1": jnp.zeros((2, 16, 16, 64)),
        ...              "conv2": jnp.zeros((2, 8, 8, 128))}
        >>> feat_target = {"conv1": jnp.ones((2, 16, 16, 64)),
        ...                "conv2": jnp.ones((2, 8, 8, 128))}
        >>> feature_reconstruction_loss(feat_pred, feat_target,
        ...                             weights={"conv1": 1.0, "conv2": 0.5})
    """
    # Initialize total loss
    total_loss = jnp.array(0.0)

    # Handle dictionary input
    if isinstance(predictions, dict) and isinstance(targets, dict):
        if weights is None:
            weights = {k: 1.0 for k in targets.keys()}

        # Ensure weights is a dictionary
        weights_dict = weights if isinstance(weights, dict) else {k: 1.0 for k in targets.keys()}

        for key in targets.keys():
            if key in predictions and key in weights_dict:
                layer_loss = distance_fn(predictions[key], targets[key], reduction="mean")
                total_loss = total_loss + weights_dict[key] * layer_loss

    # Handle list input
    elif isinstance(predictions, list) and isinstance(targets, list):
        if weights is None:
            weights = [1.0] * len(targets)

        # Ensure weights is a list
        weights_list = weights if isinstance(weights, list) else [1.0] * len(targets)

        for i, (feat_pred, feat_target) in enumerate(zip(predictions, targets)):
            if i < len(weights_list):
                layer_loss = distance_fn(feat_pred, feat_target, reduction="mean")
                total_loss = total_loss + weights_list[i] * layer_loss

    # Handle single array input
    else:
        total_loss = distance_fn(predictions, targets, reduction=reduction)
        if weights is not None:
            total_loss = total_loss * jnp.asarray(weights).mean()

    # Apply reduction across layers if needed
    if reduction == "mean" and (isinstance(predictions, (dict, list))):
        if isinstance(weights, dict):
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                total_loss = total_loss / weight_sum
        elif isinstance(weights, list) and len(weights) > 0:
            weight_sum = sum(weights)
            if weight_sum > 0:
                total_loss = total_loss / weight_sum

    return total_loss


def style_loss(
    predictions: jax.Array | list[jax.Array] | dict[str, jax.Array],
    targets: jax.Array | list[jax.Array] | dict[str, jax.Array],
    weights: list[float] | dict[str, float] | None = None,
    reduction: str = "mean",
) -> jax.Array:
    """
    Style loss based on Gram matrices.

    Computes the distance between Gram matrices of features extracted from
    predicted and target images, capturing style/texture information.

    Args:
        predictions: Features extracted from generated/predicted images
        targets: Features extracted from target/real images
        weights: Optional weights for each feature layer
        reduction: Reduction method ('none', 'mean', 'sum')

    Returns:
        Style loss

    Example:
        >>> feat_pred = {"conv1": jnp.zeros((2, 16, 16, 64))}
        >>> feat_target = {"conv1": jnp.ones((2, 16, 16, 64))}
        >>> style_loss(feat_pred, feat_target, weights={"conv1": 1.0})
    """

    def _compute_gram_matrix(features):
        """Compute Gram matrix from features."""
        batch_size, height, width, channels = features.shape
        features_reshaped = jnp.reshape(features, (batch_size, height * width, channels))
        # Compute gram matrix: F*F^T / (H*W)
        gram = jnp.matmul(features_reshaped, jnp.transpose(features_reshaped, (0, 2, 1)))
        return gram / (height * width)

    # Initialize total loss
    total_loss = jnp.array(0.0)

    # Handle dictionary input
    if isinstance(predictions, dict) and isinstance(targets, dict):
        if weights is None:
            weights = {k: 1.0 for k in targets.keys()}

        # Ensure weights is a dictionary
        weights_dict = weights if isinstance(weights, dict) else {k: 1.0 for k in targets.keys()}

        for key in targets.keys():
            if key in predictions and key in weights_dict:
                # Compute gram matrices
                gram_pred = _compute_gram_matrix(predictions[key])
                gram_target = _compute_gram_matrix(targets[key])

                # Compute MSE between gram matrices
                layer_loss = jnp.mean(jnp.square(gram_pred - gram_target))
                total_loss = total_loss + weights_dict[key] * layer_loss

    # Handle list input
    elif isinstance(predictions, list) and isinstance(targets, list):
        if weights is None:
            weights = [1.0] * len(targets)

        # Ensure weights is a list
        weights_list = weights if isinstance(weights, list) else [1.0] * len(targets)

        for i, (feat_pred, feat_target) in enumerate(zip(predictions, targets)):
            if i < len(weights_list):
                # Compute gram matrices
                gram_pred = _compute_gram_matrix(feat_pred)
                gram_target = _compute_gram_matrix(feat_target)

                # Compute MSE between gram matrices
                layer_loss = jnp.mean(jnp.square(gram_pred - gram_target))
                total_loss = total_loss + weights_list[i] * layer_loss

    # Handle single array input
    else:
        gram_pred = _compute_gram_matrix(predictions)
        gram_target = _compute_gram_matrix(targets)
        total_loss = jnp.mean(jnp.square(gram_pred - gram_target))
        if weights is not None:
            total_loss = total_loss * jnp.asarray(weights).mean()

    # Apply reduction across layers if needed
    if reduction == "mean" and (isinstance(predictions, (dict, list))):
        if isinstance(weights, dict):
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                total_loss = total_loss / weight_sum
        elif isinstance(weights, list) and len(weights) > 0:
            weight_sum = sum(weights)
            if weight_sum > 0:
                total_loss = total_loss / weight_sum

    return total_loss


def contextual_loss(
    predictions: jax.Array,
    targets: jax.Array,
    band_width: float = 0.1,
    eps: float = 1e-5,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    max_samples: int = 1024,  # Limit samples to avoid memory issues
) -> jax.Array:
    """
    Contextual loss for feature similarity that is robust to misalignments.

    This implementation is memory-efficient and fully vectorized for JAX compatibility.
    Measures the similarity between feature distributions rather than
    spatially aligned features, making it useful for style transfer and
    image-to-image translation tasks.

    Args:
        predictions: Features from generated/predicted images [B, H, W, C]
        targets: Features from target/real images [B, H, W, C]
        band_width: Controls the band width of the similarity kernel
        eps: Small constant for numerical stability
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for batch elements
        max_samples: Maximum number of samples to use per image (for memory efficiency)

    Returns:
        Contextual loss after specified reduction

    Example:
        >>> feat_pred = jnp.zeros((2, 16, 16, 64))
        >>> feat_target = jnp.ones((2, 16, 16, 64))
        >>> contextual_loss(feat_pred, feat_target)
    """
    batch_size, h_pred, w_pred, channels = predictions.shape
    h_target, w_target = targets.shape[1:3]

    # Reshape features to [B, HW, C]
    features_pred_flat = predictions.reshape(batch_size, h_pred * w_pred, channels)
    features_target_flat = targets.reshape(batch_size, h_target * w_target, channels)

    # Subsample if too many spatial locations (for memory efficiency)
    pred_samples = features_pred_flat.shape[1]
    target_samples = features_target_flat.shape[1]

    if pred_samples > max_samples:
        # Use regular sampling to maintain spatial distribution
        step = pred_samples // max_samples
        indices = jnp.arange(0, pred_samples, step)[:max_samples]
        features_pred_flat = features_pred_flat[:, indices, :]

    if target_samples > max_samples:
        step = target_samples // max_samples
        indices = jnp.arange(0, target_samples, step)[:max_samples]
        features_target_flat = features_target_flat[:, indices, :]

    # Normalize features to unit vectors
    features_pred_norm = features_pred_flat / (
        jnp.linalg.norm(features_pred_flat, axis=-1, keepdims=True) + eps
    )
    features_target_norm = features_target_flat / (
        jnp.linalg.norm(features_target_flat, axis=-1, keepdims=True) + eps
    )

    # Process each batch element separately to avoid large memory allocation
    def compute_contextual_loss_single(feat_pred, feat_target):
        """Compute contextual loss for a single batch element."""
        # Compute cosine similarity matrix
        cosine_sim = jnp.matmul(feat_pred, feat_target.T)  # [N_pred, N_target]

        # Convert to cosine distance
        cosine_dist = 1.0 - cosine_sim

        # Compute affinity using gaussian kernel
        affinity = jnp.exp(-cosine_dist / band_width)

        # Find best matches for each predicted feature
        max_affinity_per_pred = jnp.max(affinity, axis=1)  # [N_pred]

        # Normalize by maximum affinity to get relative similarities
        max_affinity_global = jnp.max(affinity)
        relative_affinity = max_affinity_per_pred / (max_affinity_global + eps)

        # Compute contextual loss: -log(mean(max_relative_affinities))
        mean_relative_affinity = jnp.mean(relative_affinity)
        return -jnp.log(mean_relative_affinity + eps)

    # Use vmap to process all batch elements
    contextual_loss_batch = nnx.vmap(compute_contextual_loss_single)(
        features_pred_norm, features_target_norm
    )

    return reduce_loss(contextual_loss_batch, reduction, weights)


class PerceptualLoss(nnx.Module):
    """
    Perceptual loss module that combines feature reconstruction and style losses.

    This module can be used as a standalone loss or combined with other losses.
    """

    def __init__(
        self,
        feature_extractor: nnx.Module | None = None,
        layer_weights: dict[str, float] | None = None,
        content_weight: float = 1.0,
        style_weight: float = 0.0,
        contextual_weight: float = 0.0,
        normalize_features: bool = True,
        max_contextual_samples: int = 512,  # Reduce for memory efficiency
    ):
        """Initialize the perceptual loss module.

        Args:
            feature_extractor: Pre-trained network for feature extraction
            layer_weights: Weights for different layers
            content_weight: Weight for content (feature reconstruction) loss
            style_weight: Weight for style loss
            contextual_weight: Weight for contextual loss
            normalize_features: Whether to normalize features before computing loss
            max_contextual_samples: Max samples for contextual loss (memory control)
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {}
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.contextual_weight = contextual_weight
        self.normalize_features = normalize_features
        self.max_contextual_samples = max_contextual_samples

    def __call__(
        self,
        pred_images: jax.Array,
        target_images: jax.Array,
        features_pred: dict[str, jax.Array] | None = None,
        features_target: dict[str, jax.Array] | None = None,
    ) -> jax.Array:
        """Compute perceptual loss.

        Args:
            pred_images: Predicted images [B, H, W, C]
            target_images: Target images [B, H, W, C]
            features_pred: Pre-computed features for predicted images
            features_target: Pre-computed features for target images

        Returns:
            Combined perceptual loss
        """
        # Extract features if not provided
        if features_pred is None or features_target is None:
            if self.feature_extractor is None:
                raise ValueError("Must provide either feature_extractor or pre-computed features")

            # Use the feature extractor as a callable
            if hasattr(self.feature_extractor, "__call__"):
                features_pred = self.feature_extractor(pred_images)
                features_target = self.feature_extractor(target_images)
            else:
                raise ValueError("Feature extractor must be callable")

        # Initialize total loss
        total_loss = jnp.array(0.0)

        # Content loss (feature reconstruction)
        if self.content_weight > 0 and features_target is not None and features_pred is not None:
            content_loss = feature_reconstruction_loss(
                features_pred, features_target, weights=self.layer_weights
            )
            total_loss = total_loss + self.content_weight * content_loss

        # Style loss
        if self.style_weight > 0 and features_target is not None and features_pred is not None:
            style_loss_value = style_loss(
                features_pred, features_target, weights=self.layer_weights
            )
            total_loss = total_loss + self.style_weight * style_loss_value

        # Contextual loss (computed on a single representative layer)
        if self.contextual_weight > 0:
            # Skip contextual loss if features are None
            if features_target is None or features_pred is None:
                return total_loss

            # Use the last layer or a specified layer for contextual loss
            if isinstance(features_target, dict):
                # Use the last layer in the dictionary
                key = list(features_target.keys())[-1]
                feat_target = features_target[key]
                feat_pred = features_pred[key]
            else:
                feat_target = features_target
                feat_pred = features_pred

            contextual_loss_value = contextual_loss(
                feat_pred, feat_target, max_samples=self.max_contextual_samples
            )
            total_loss = total_loss + self.contextual_weight * contextual_loss_value

        return total_loss


def create_vgg_perceptual_loss(
    layers: list[str] = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
    weights: list[float] | None = None,
    content_weight: float = 1.0,
    style_weight: float = 0.0,
) -> PerceptualLoss:
    """Create a VGG-based perceptual loss.

    Args:
        layers: list of VGG layer names to use
        weights: Weights for each layer
        content_weight: Weight for content loss
        style_weight: Weight for style loss

    Returns:
        PerceptualLoss module configured for VGG features
    """
    if weights is None:
        weights = [1.0] * len(layers)

    layer_weights = dict(zip(layers, weights))

    # Note: In practice, you would load a pre-trained VGG model here
    # For this example, we return the module without the feature extractor
    return PerceptualLoss(
        feature_extractor=None,  # Would be VGG model
        layer_weights=layer_weights,
        content_weight=content_weight,
        style_weight=style_weight,
    )
