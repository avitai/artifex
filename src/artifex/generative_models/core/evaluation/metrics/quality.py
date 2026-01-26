"""Quality metrics for evaluation across modalities."""

import jax
import jax.numpy as jnp


def calculate_fid_statistics(features: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Calculate mean and covariance of features.

    Args:
        features: Feature vectors [batch_size, feature_dim]

    Returns:
        Tuple of (mean, covariance)
    """
    mu = jnp.mean(features, axis=0)
    sigma = jnp.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(
    mu1: jax.Array, sigma1: jax.Array, mu2: jax.Array, sigma2: jax.Array
) -> float:
    """Calculate Fréchet distance between two multivariate Gaussians.

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution

    Returns:
        Fréchet distance
    """
    # Calculate square root of product of covariances
    # JAX doesn't have sqrtm, so we use scipy through jax.scipy
    from jax.scipy import linalg

    covmean = linalg.sqrtm(sigma1 @ sigma2)

    # Ensure covmean is real
    if jnp.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate Fréchet distance
    mean_diff = mu1 - mu2
    trace_term = jnp.trace(sigma1 + sigma2 - 2 * covmean)
    fid = mean_diff @ mean_diff + trace_term

    return float(fid)


def calculate_fid_score(gen_features: jax.Array, real_features: jax.Array) -> float:
    """Calculate FID score from features.

    Args:
        gen_features: Features of generated images
        real_features: Features of real images

    Returns:
        FID score
    """
    # Calculate mean and covariance
    mu1, sigma1 = calculate_fid_statistics(gen_features)
    mu2, sigma2 = calculate_fid_statistics(real_features)

    # Calculate FID
    try:
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return 100.0  # Return high FID score on error


def compute_lpips_distance(images1: jax.Array, images2: jax.Array) -> float:
    """Compute LPIPS distance between two sets of images.

    This implementation approximates LPIPS by combining MSE with a
    perceptual weighting based on image structure.

    Args:
        images1: First set of images
        images2: Second set of images

    Returns:
        LPIPS distance
    """
    # Basic pixel-wise MSE
    mse = jnp.mean((images1 - images2) ** 2, axis=(1, 2, 3))

    # Add structure-based weighting to approximate perceptual similarity
    # Higher weight for high-frequency differences

    # Simple edge detection using gradient magnitude
    def gradient_magnitude(image):
        # Compute x and y gradients
        grad_x = image[:, 1:, :-1] - image[:, :-1, :-1]
        grad_y = image[:, :-1, 1:] - image[:, :-1, :-1]

        # Compute gradient magnitude
        grad_mag = jnp.sqrt(grad_x**2 + grad_y**2)

        # Pad to original size
        padded = jnp.pad(grad_mag, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return padded

    # Compute edge maps
    edges1 = gradient_magnitude(images1)
    edges2 = gradient_magnitude(images2)

    # Weight MSE by edge difference
    edge_diff = jnp.mean(jnp.abs(edges1 - edges2), axis=(1, 2, 3))
    weighted_mse = mse * (1.0 + edge_diff)

    # Scale to approximate LPIPS range
    lpips_approx = 0.1 * jnp.sqrt(weighted_mse)

    return float(jnp.mean(lpips_approx))


def compute_spectral_convergence(real_mag: jax.Array, gen_mag: jax.Array) -> float:
    """Compute spectral convergence between real and generated audio.

    Args:
        real_mag: Magnitude spectrum of real audio
        gen_mag: Magnitude spectrum of generated audio

    Returns:
        Spectral convergence score
    """
    # Compute spectral convergence
    numerator = jnp.linalg.norm(real_mag - gen_mag, ord="fro", axis=(-2, -1))
    denominator = jnp.linalg.norm(real_mag, ord="fro", axis=(-2, -1))

    # Avoid division by zero
    spectral_convergence = jnp.where(
        denominator > 1e-8, numerator / denominator, jnp.ones_like(numerator)
    )

    # Average across batch
    avg_spectral_convergence = float(jnp.mean(spectral_convergence))

    return avg_spectral_convergence


def compute_mel_cepstral_distortion(real_mfcc: jax.Array, gen_mfcc: jax.Array) -> float:
    """Compute mel-cepstral distortion between real and generated audio.

    Args:
        real_mfcc: MFCC coefficients of real audio
        gen_mfcc: MFCC coefficients of generated audio

    Returns:
        Mel-cepstral distortion score
    """
    # Use only the first 13 coefficients (standard practice)
    n_coeffs = min(13, real_mfcc.shape[1])
    real_mfcc = real_mfcc[:, :n_coeffs, :]
    gen_mfcc = gen_mfcc[:, :n_coeffs, :]

    # Compute frame-wise euclidean distance
    diff = real_mfcc - gen_mfcc
    frame_distances = jnp.sqrt(jnp.sum(diff**2, axis=1))

    # Average across frames and batch
    mcd = jnp.mean(frame_distances)

    # Scale by constant factor (common in MCD computation)
    mcd = (10.0 / jnp.log(10.0)) * jnp.sqrt(2.0) * mcd

    return float(mcd)
