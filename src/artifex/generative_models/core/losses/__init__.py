"""
Loss functions for generative models.

This module provides various loss functions that are used to train generative models.
The API supports both functional and modular (Flax NNX) approaches for maximum flexibility.

Modules:
 - adversarial.py      # GAN losses (vanilla, LSGAN, WGAN, hinge)
 - reconstruction.py   # Reconstruction losses (MSE, MAE, Huber, etc.)
 - regularization.py   # Regularization losses (L1, L2, TV, etc.)
 - divergence.py       # KL, JS, Wasserstein, MMD divergences
 - perceptual.py       # Perceptual losses using neural network features
 - geometric.py        # 3D geometry-specific losses (point clouds, meshes, voxels)
 - composable.py       # Framework for composing multiple losses
 - base.py            # Base utilities and collection classes

Example usage:

    # Functional approach
    from artifex.generative_models.core.losses import mse_loss, l1_regularization

    content_loss = mse_loss(predictions, targets)
    reg_loss = l1_regularization(model_params, scale=0.01)
    total_loss = content_loss + reg_loss

    # Modular approach with NNX
    from artifex.generative_models.core.losses import CompositeLoss, WeightedLoss
    from artifex.generative_models.core.losses.perceptual import PerceptualLoss

    # Create individual loss modules
    content_loss = WeightedLoss(mse_loss, weight=1.0, name="content")
    perceptual_loss = PerceptualLoss(content_weight=0.1, style_weight=0.01)

    # Combine them
    total_loss = CompositeLoss([content_loss, perceptual_loss], return_components=True)
    loss_value, loss_dict = total_loss(predictions, targets)

    # Advanced composition with scheduling
    from artifex.generative_models.core.losses import ScheduledLoss, create_loss_suite

    # Progressive training schedule
    schedule_fn = lambda step: min(1.0, step / 1000)  # Ramp up over 1000 steps
    scheduled_loss = ScheduledLoss(perceptual_loss, schedule_fn)

    # Create a complete loss suite
    loss_suite = create_loss_suite(content_loss, scheduled_loss)
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
from artifex.generative_models.core.losses.base import (
    LossCollection,
    LossMetrics,
    LossScheduler,
    reduce_loss,
)

# Composable framework (NNX-based)
from artifex.generative_models.core.losses.composable import (
    CompositeLoss,
    create_loss_suite,
    create_weighted_loss,
    Loss,
    ScheduledLoss,
    WeightedLoss,
)

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
    MeshLoss,
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
    "LossCollection",
    "reduce_loss",
    "LossMetrics",
    "LossScheduler",
    # Composable framework
    "Loss",
    "CompositeLoss",
    "WeightedLoss",
    "ScheduledLoss",
    "create_weighted_loss",
    "create_loss_suite",
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
    "get_point_cloud_loss",
    "MeshLoss",
    "get_mesh_loss",
    "binary_cross_entropy",
    "dice_loss",
    "focal_loss",
    "get_voxel_loss",
]


# Version info
__version__ = "0.2.0"
__author__ = "Artifex Team"
__description__ = "Composable loss functions for generative models using JAX and Flax NNX"


# Convenience functions for common use cases
def create_gan_loss_suite(
    generator_loss_type: str = "vanilla",
    discriminator_loss_type: str = "vanilla",
    gradient_penalty_weight: float = 10.0,
    **kwargs,
) -> tuple[Loss, Loss]:
    """Create a standard GAN loss suite.

    Args:
        generator_loss_type: Type of generator loss ('vanilla', 'lsgan', 'wgan', 'hinge')
        discriminator_loss_type: Type of discriminator loss
        gradient_penalty_weight: Weight for gradient penalty (WGAN-GP)
        **kwargs: Additional arguments for loss functions

    Returns:
        Tuple of (generator_loss, discriminator_loss) modules
    """
    # Generator loss mapping
    gen_loss_map = {
        "vanilla": vanilla_generator_loss,
        "lsgan": least_squares_generator_loss,
        "wgan": wasserstein_generator_loss,
        "hinge": hinge_generator_loss,
    }

    # Discriminator loss mapping
    disc_loss_map = {
        "vanilla": vanilla_discriminator_loss,
        "lsgan": least_squares_discriminator_loss,
        "wgan": wasserstein_discriminator_loss,
        "hinge": hinge_discriminator_loss,
    }

    gen_loss_fn = gen_loss_map.get(generator_loss_type)
    disc_loss_fn = disc_loss_map.get(discriminator_loss_type)

    if gen_loss_fn is None:
        raise ValueError(f"Unknown generator loss type: {generator_loss_type}")
    if disc_loss_fn is None:
        raise ValueError(f"Unknown discriminator loss type: {discriminator_loss_type}")

    generator_loss = WeightedLoss(gen_loss_fn, name=f"{generator_loss_type}_generator")

    # Add gradient penalty for WGAN
    if discriminator_loss_type == "wgan" and gradient_penalty_weight > 0:
        base_disc_loss = WeightedLoss(disc_loss_fn, name=f"{discriminator_loss_type}_discriminator")
        gp_loss = WeightedLoss(
            gradient_penalty, weight=gradient_penalty_weight, name="gradient_penalty"
        )
        discriminator_loss = CompositeLoss([base_disc_loss, gp_loss])
    else:
        discriminator_loss = WeightedLoss(
            disc_loss_fn, name=f"{discriminator_loss_type}_discriminator"
        )

    return generator_loss, discriminator_loss


def create_vae_loss_suite(
    reconstruction_loss_type: str = "mse", kl_weight: float = 1.0, **kwargs
) -> CompositeLoss:
    """Create a standard VAE loss suite.

    Args:
        reconstruction_loss_type: Type of reconstruction loss ('mse', 'mae', 'bce')
        kl_weight: Weight for KL divergence term
        **kwargs: Additional arguments for loss functions

    Returns:
        CompositeLoss module combining reconstruction and KL losses
    """
    # Reconstruction loss mapping
    recon_loss_map = {
        "mse": mse_loss,
        "mae": mae_loss,
        "bce": binary_cross_entropy,  # For voxel/binary data
    }

    recon_loss_fn = recon_loss_map.get(reconstruction_loss_type)
    if recon_loss_fn is None:
        raise ValueError(f"Unknown reconstruction loss type: {reconstruction_loss_type}")

    reconstruction_loss = WeightedLoss(recon_loss_fn, weight=1.0, name="reconstruction")
    kl_loss = WeightedLoss(kl_divergence, weight=kl_weight, name="kl_divergence")

    return CompositeLoss([reconstruction_loss, kl_loss], return_components=True)


def create_image_generation_loss_suite(
    content_weight: float = 1.0,
    perceptual_weight: float = 0.1,
    style_weight: float = 0.01,
    tv_weight: float = 0.001,
    **kwargs,
) -> CompositeLoss:
    """Create a comprehensive image generation loss suite.

    Args:
        content_weight: Weight for pixel-wise content loss
        perceptual_weight: Weight for perceptual feature loss
        style_weight: Weight for style loss
        tv_weight: Weight for total variation regularization
        **kwargs: Additional arguments

    Returns:
        CompositeLoss module for image generation
    """
    losses = []

    # Content loss
    if content_weight > 0:
        content_loss = WeightedLoss(mse_loss, weight=content_weight, name="content")
        losses.append(content_loss)

    # Perceptual loss
    if perceptual_weight > 0:
        perceptual_loss = PerceptualLoss(
            content_weight=perceptual_weight, style_weight=style_weight
        )
        losses.append(perceptual_loss)

    # Total variation regularization
    if tv_weight > 0:
        tv_loss = WeightedLoss(total_variation_loss, weight=tv_weight, name="total_variation")
        losses.append(tv_loss)

    return CompositeLoss(losses, return_components=True)
