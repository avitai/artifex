"""Loss functions for generative models.

This module provides various loss functions that are used to train generative models.
The API centers on pure functional loss primitives and lightweight helpers for
combining them explicitly inside model objectives and trainers.

Modules:
 - adversarial.py      # GAN losses (vanilla, LSGAN, WGAN, hinge)
 - reconstruction.py   # Reconstruction losses (MSE, MAE, Huber, etc.)
 - regularization.py   # Regularization losses (L1, L2, TV, etc.)
 - divergence.py       # KL, JS, Wasserstein, MMD divergences
 - perceptual.py       # Perceptual losses using neural network features
 - geometric.py        # 3D geometry-specific losses (point clouds, meshes, voxels)
 - base.py            # Minimal reduction utilities

Example usage:

    # Functional approach
    from artifex.generative_models.core.losses import mse_loss, l1_regularization

    content_loss = mse_loss(predictions, targets)
    reg_loss = l1_regularization(model_params, scale=0.01)
    total_loss = content_loss + reg_loss

    # Explicit composition stays simple and JAX-native
    perceptual_loss = create_vgg_perceptual_loss()
    schedule_weight = min(1.0, step / 1000)
    total_loss = content_loss + 0.1 * schedule_weight * perceptual_loss(predictions, targets)

    # Adversarial objectives stay explicit
    from artifex.generative_models.core.losses import (
        least_squares_discriminator_loss,
        least_squares_generator_loss,
    )
"""

# Base utilities
# Adversarial losses
from artifex.generative_models.core.losses.adversarial import (
    hinge_discriminator_loss,
    hinge_generator_loss,
    least_squares_discriminator_loss,
    least_squares_generator_loss,
    ns_vanilla_discriminator_loss,
    ns_vanilla_generator_loss,
    vanilla_discriminator_loss,
    vanilla_generator_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)
from artifex.generative_models.core.losses.base import reduce_loss

# Divergence losses
from artifex.generative_models.core.losses.divergence import (
    energy_distance,
    gaussian_kl_divergence,
    js_divergence,
    kl_divergence,
    maximum_mean_discrepancy,
    reverse_kl_divergence,
    wasserstein_distance,
)

# Geometric losses
from artifex.generative_models.core.losses.geometric import (
    binary_cross_entropy,
    chamfer_distance,
    dice_loss,
    earth_mover_distance,
    focal_loss,
    get_mesh_loss,
    get_point_cloud_loss,
    get_voxel_loss,
    hausdorff_distance,
    MeshLoss,
)

# Metric learning losses (re-exported from calibrax)
from artifex.generative_models.core.losses.metric_learning import (
    ArcFaceLoss,
    ContrastiveLoss,
    CosFaceLoss,
    MetricLearningLoss,
    NTXentLoss,
    ProxyAnchorLoss,
    ProxyNCALoss,
    TripletMarginLoss,
)

# Perceptual losses
from artifex.generative_models.core.losses.perceptual import (
    contextual_loss,
    create_vgg_perceptual_loss,
    feature_reconstruction_loss,
    PerceptualLoss,
    style_loss,
)

# Reconstruction losses
from artifex.generative_models.core.losses.reconstruction import (
    charbonnier_loss,
    huber_loss,
    mae_loss,
    mse_loss,
    psnr_loss,
)

# Regularization losses
from artifex.generative_models.core.losses.regularization import (
    gradient_penalty,
    l1_regularization,
    l2_regularization,
    orthogonal_regularization,
    spectral_norm_regularization,
    total_variation_loss,
)


__all__ = [
    # Base utilities
    "reduce_loss",
    # Reconstruction losses
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "charbonnier_loss",
    "psnr_loss",
    # Divergence losses
    "kl_divergence",
    "reverse_kl_divergence",
    "js_divergence",
    "wasserstein_distance",
    "maximum_mean_discrepancy",
    "energy_distance",
    "gaussian_kl_divergence",
    # Regularization losses
    "l1_regularization",
    "l2_regularization",
    "spectral_norm_regularization",
    "orthogonal_regularization",
    "total_variation_loss",
    "gradient_penalty",
    # Adversarial losses
    "vanilla_generator_loss",
    "vanilla_discriminator_loss",
    "ns_vanilla_generator_loss",
    "ns_vanilla_discriminator_loss",
    "least_squares_generator_loss",
    "least_squares_discriminator_loss",
    "wasserstein_generator_loss",
    "wasserstein_discriminator_loss",
    "hinge_generator_loss",
    "hinge_discriminator_loss",
    # Perceptual losses
    "feature_reconstruction_loss",
    "style_loss",
    "contextual_loss",
    "PerceptualLoss",
    "create_vgg_perceptual_loss",
    # Geometric losses
    "chamfer_distance",
    "earth_mover_distance",
    "hausdorff_distance",
    "get_point_cloud_loss",
    "MeshLoss",
    "get_mesh_loss",
    "binary_cross_entropy",
    "dice_loss",
    "focal_loss",
    "get_voxel_loss",
    # Metric learning losses
    "MetricLearningLoss",
    "ContrastiveLoss",
    "TripletMarginLoss",
    "NTXentLoss",
    "ArcFaceLoss",
    "CosFaceLoss",
    "ProxyNCALoss",
    "ProxyAnchorLoss",
]


# Version info
__version__ = "0.2.0"
__author__ = "Artifex Team"
__description__ = "Functional loss primitives for generative models using JAX and Flax NNX"
