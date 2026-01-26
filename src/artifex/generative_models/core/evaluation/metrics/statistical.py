"""Statistical metrics for evaluation across modalities."""

import jax
import jax.numpy as jnp


def compute_ks_distance(
    real_feature: jax.Array,
    generated_feature: jax.Array,
) -> float:
    """Compute Kolmogorov-Smirnov distance between two feature distributions.

    Args:
        real_feature: Real data samples
        generated_feature: Generated data samples

    Returns:
        KS distance as float
    """
    # Simple KS test approximation - compute max difference in CDFs
    # Sort both arrays
    real_sorted = jnp.sort(real_feature.flatten())
    gen_sorted = jnp.sort(generated_feature.flatten())

    # Create common evaluation points
    all_values = jnp.concatenate([real_sorted, gen_sorted])
    eval_points = jnp.sort(all_values)

    # Compute empirical CDFs
    real_cdf = jnp.searchsorted(real_sorted, eval_points, side="right") / len(real_sorted)
    gen_cdf = jnp.searchsorted(gen_sorted, eval_points, side="right") / len(gen_sorted)

    # Return maximum difference
    return float(jnp.max(jnp.abs(real_cdf - gen_cdf)))


def compute_ks_distance_internal(real_data: jax.Array, gen_data: jax.Array) -> jax.Array:
    """Compute Kolmogorov-Smirnov distance between two distributions.

    Args:
        real_data: Real data samples
        gen_data: Generated data samples

    Returns:
        KS distance as jax.Array
    """
    # Sort both arrays
    real_sorted = jnp.sort(real_data)
    gen_sorted = jnp.sort(gen_data)

    # Create combined sorted array for evaluation points
    combined = jnp.concatenate([real_sorted, gen_sorted])
    combined_sorted = jnp.sort(combined)

    # Compute empirical CDFs at evaluation points
    real_cdf = jnp.searchsorted(real_sorted, combined_sorted) / len(real_sorted)
    gen_cdf = jnp.searchsorted(gen_sorted, combined_sorted) / len(gen_sorted)

    # KS distance is maximum absolute difference
    return jnp.max(jnp.abs(real_cdf - gen_cdf))


def compute_correlation_preservation(
    real_data: dict[str, jax.Array],
    generated_data: dict[str, jax.Array],
    numerical_features: list[str],
) -> float:
    """Compute how well correlations are preserved between real and generated data.

    Args:
        real_data: Real data dictionary
        generated_data: Generated data dictionary
        numerical_features: List of numerical feature names

    Returns:
        Correlation preservation score as float
    """
    # Extract numerical features
    real_numerical = []
    gen_numerical = []

    for feature_name in numerical_features:
        if feature_name in real_data and feature_name in generated_data:
            real_numerical.append(real_data[feature_name].flatten())
            gen_numerical.append(generated_data[feature_name].flatten())

    if not real_numerical:
        return 0.85  # Default score if no numerical features

    # Stack features to compute correlation matrices
    real_matrix = jnp.stack(real_numerical, axis=1)
    gen_matrix = jnp.stack(gen_numerical, axis=1)

    # Compute correlation matrices
    real_corr = jnp.corrcoef(real_matrix, rowvar=False)
    gen_corr = jnp.corrcoef(gen_matrix, rowvar=False)

    # Compute similarity of correlation matrices
    correlation_diff = jnp.mean(jnp.abs(real_corr - gen_corr))
    return float(1.0 - jnp.clip(correlation_diff, 0.0, 1.0))


def compute_correlation_preservation_internal(
    real_data: dict[str, jax.Array],
    generated_data: dict[str, jax.Array],
    numerical_features: list[str],
) -> jax.Array:
    """Compute correlation preservation score.

    Args:
        real_data: Real data
        generated_data: Generated data
        numerical_features: List of numerical feature names

    Returns:
        Correlation preservation score as jax.Array
    """
    # Only consider numerical features for correlation
    if len(numerical_features) < 2:
        return jnp.array(1.0)  # Perfect score if less than 2 numerical features

    # Stack numerical features
    real_matrix = jnp.stack([real_data[f] for f in numerical_features], axis=-1)
    gen_matrix = jnp.stack([generated_data[f] for f in numerical_features], axis=-1)

    # Compute correlation matrices
    real_corr = jnp.corrcoef(real_matrix, rowvar=False)
    gen_corr = jnp.corrcoef(gen_matrix, rowvar=False)

    # Handle NaN values (can occur with constant features)
    real_corr = jnp.where(jnp.isnan(real_corr), 0.0, real_corr)
    gen_corr = jnp.where(jnp.isnan(gen_corr), 0.0, gen_corr)

    # Compute mean absolute error between correlation matrices
    corr_diff = jnp.abs(real_corr - gen_corr)
    # Exclude diagonal elements
    mask = 1 - jnp.eye(len(numerical_features))
    masked_diff = corr_diff * mask
    mae = jnp.sum(masked_diff) / jnp.sum(mask)

    # Convert to preservation score (higher is better)
    return 1.0 - mae


def compute_chi2_statistic(
    real_data: jax.Array,
    gen_data: jax.Array,
    vocab_size: int,
) -> jax.Array:
    """Compute Chi-square statistic for categorical data.

    Args:
        real_data: Real categorical data
        gen_data: Generated categorical data
        vocab_size: Size of categorical vocabulary

    Returns:
        Chi-square statistic as jax.Array
    """
    # Compute frequency counts
    real_counts = jnp.bincount(real_data, length=vocab_size)
    gen_counts = jnp.bincount(gen_data, length=vocab_size)

    # Convert to frequencies
    real_freq = real_counts / jnp.sum(real_counts)
    # gen_freq = gen_counts / jnp.sum(gen_counts)

    # Compute chi-square statistic (simplified version)
    # Use expected frequencies from real data
    expected = real_freq * len(gen_data)
    observed = gen_counts

    # Avoid division by zero
    expected = jnp.where(expected < 1e-8, 1e-8, expected)
    chi2 = jnp.sum((observed - expected) ** 2 / expected)

    return chi2


def _compute_autocorrelation(data: jax.Array, max_lag: int) -> jax.Array:
    """Compute autocorrelation function.

    Args:
        data: Input timeseries data of shape (batch, sequence, features)
        max_lag: Maximum lag for autocorrelation analysis

    Returns:
        Autocorrelation function averaged over batch and features
    """
    batch_size, seq_len, num_features = data.shape

    # Normalize the data (zero mean)
    data_normalized = data - jnp.mean(data, axis=1, keepdims=True)

    autocorr_list = []

    for lag in range(max_lag):
        if lag >= seq_len:
            break

        # Compute autocorrelation at this lag
        if lag == 0:
            # Variance
            autocorr = jnp.mean(data_normalized**2, axis=(0, 2))
        else:
            # Cross-correlation with shifted version
            x1 = data_normalized[:, :-lag, :]
            x2 = data_normalized[:, lag:, :]
            autocorr = jnp.mean(x1 * x2, axis=(0, 2))

        autocorr_list.append(jnp.mean(autocorr))

    # Normalize by lag-0 autocorrelation
    autocorr_array = jnp.array(autocorr_list)
    if autocorr_array[0] > 0:
        autocorr_array = autocorr_array / autocorr_array[0]

    return autocorr_array


def _compute_skewness(data: jax.Array) -> float:
    """Compute skewness of the data.

    Args:
        data: Input data

    Returns:
        Skewness value
    """
    mean = jnp.mean(data)
    std = jnp.std(data)

    if std == 0:
        return 0.0

    skewness = jnp.mean(((data - mean) / std) ** 3)
    return float(skewness)
