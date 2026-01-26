"""JAX-compatible operations for metrics computation.

This module provides JAX implementations of common algorithms needed for metrics,
replacing numpy and sklearn dependencies to ensure compatibility with NNX modules.
"""

import jax
import jax.numpy as jnp


def nearest_neighbors(query: jax.Array, data: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """JAX implementation of k-nearest neighbors search.

    Args:
        query: Query points of shape (n_queries, feature_dim)
        data: Data points of shape (n_data, feature_dim)
        k: Number of nearest neighbors to find

    Returns:
        Tuple of (distances, indices) where:
            - distances: Shape (n_queries, k), distances to k nearest neighbors
            - indices: Shape (n_queries, k), indices of k nearest neighbors
    """
    # Compute pairwise squared distances
    # Using broadcasting: (n_queries, 1, feature_dim) - (1, n_data, feature_dim)
    query_expanded = query[:, None, :]
    data_expanded = data[None, :, :]
    squared_distances = jnp.sum((query_expanded - data_expanded) ** 2, axis=-1)

    # Get k nearest indices
    k_actual = min(k, data.shape[0])
    indices = jnp.argsort(squared_distances, axis=1)[:, :k_actual]

    # Get corresponding distances
    distances = jnp.sqrt(jnp.take_along_axis(squared_distances, indices, axis=1))

    return distances, indices


def pairwise_distances(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute pairwise Euclidean distances between two sets of points.

    Args:
        x: First set of points, shape (n_x, feature_dim)
        y: Second set of points, shape (n_y, feature_dim)

    Returns:
        Pairwise distances of shape (n_x, n_y)
    """
    # Expand dimensions for broadcasting
    x_expanded = x[:, None, :]  # (n_x, 1, feature_dim)
    y_expanded = y[None, :, :]  # (1, n_y, feature_dim)

    # Compute squared distances
    squared_distances = jnp.sum((x_expanded - y_expanded) ** 2, axis=-1)

    # Return Euclidean distances
    return jnp.sqrt(squared_distances)


def compute_cdf(data: jax.Array, eval_points: jax.Array) -> jax.Array:
    """Compute empirical cumulative distribution function.

    Args:
        data: Data points (1D array)
        eval_points: Points at which to evaluate CDF

    Returns:
        CDF values at eval_points
    """
    # Sort data
    data_sorted = jnp.sort(data)

    # Compute CDF using searchsorted
    # searchsorted returns the index where eval_points would be inserted
    indices = jnp.searchsorted(data_sorted, eval_points, side="right")

    # Convert to probabilities
    cdf_values = indices / len(data)

    return cdf_values


def compute_ks_distance(data1: jax.Array, data2: jax.Array) -> jax.Array:
    """Compute Kolmogorov-Smirnov distance between two distributions.

    Args:
        data1: First dataset (1D array)
        data2: Second dataset (1D array)

    Returns:
        KS distance (maximum difference between CDFs)
    """
    # Combine and sort all unique values
    all_values = jnp.concatenate([data1, data2])
    eval_points = jnp.unique(all_values)

    # Compute CDFs
    cdf1 = compute_cdf(data1, eval_points)
    cdf2 = compute_cdf(data2, eval_points)

    # KS distance is maximum absolute difference
    ks_distance = jnp.max(jnp.abs(cdf1 - cdf2))

    return ks_distance


def bincount(data: jax.Array, length: int) -> jax.Array:
    """JAX implementation of bincount for categorical data.

    Args:
        data: Integer array of values
        length: Size of output array (vocabulary size)

    Returns:
        Count of each value in data
    """
    # Create one-hot encoding and sum
    one_hot = jax.nn.one_hot(data, length)
    counts = jnp.sum(one_hot, axis=0)

    return counts


def corrcoef(data: jax.Array, rowvar: bool = True) -> jax.Array:
    """JAX implementation of correlation coefficient matrix.

    Args:
        data: Input data array
        rowvar: If True, rows are variables; if False, columns are variables

    Returns:
        Correlation coefficient matrix
    """
    # Transpose if needed
    if not rowvar and data.ndim == 2:
        data = data.T

    # Center the data
    data_centered = data - jnp.mean(data, axis=1, keepdims=True)

    # Compute covariance
    cov = jnp.dot(data_centered, data_centered.T) / (data.shape[1] - 1)

    # Compute standard deviations
    std = jnp.sqrt(jnp.diag(cov))

    # Avoid division by zero
    std = jnp.where(std == 0, 1, std)

    # Compute correlation
    corr = cov / (std[:, None] * std[None, :])

    return corr


def matrix_sqrtm(matrix: jax.Array) -> jax.Array:
    """Compute matrix square root using eigendecomposition.

    This is more stable than jnp.linalg.sqrtm for symmetric positive
    semi-definite matrices like covariance matrices.

    Args:
        matrix: Symmetric positive semi-definite matrix

    Returns:
        Matrix square root
    """
    # Compute eigendecomposition
    eigenvals, eigenvecs = jnp.linalg.eigh(matrix)

    # Ensure eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, 1e-8)

    # Compute square root
    sqrt_eigenvals = jnp.sqrt(eigenvals)

    # Reconstruct matrix square root
    matrix_sqrt = eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.T

    return matrix_sqrt
