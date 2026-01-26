"""Distance-based metrics for evaluation across modalities."""

import jax
import jax.numpy as jnp


def _compute_dcr_score(
    real_data: dict[str, jax.Array],
    generated_data: dict[str, jax.Array],
    numerical_features: list[str],
) -> jax.Array:
    """Compute Distance to Closest Record (DCR) score.

    Args:
        real_data: Real data
        generated_data: Generated data
        numerical_features: List of numerical feature names

    Returns:
        DCR score (higher is better for privacy)
    """
    # Flatten all features for distance computation
    # For simplicity, use only numerical features
    if not numerical_features:
        return jnp.array(1.0)  # Default score if no numerical features

    real_matrix = jnp.stack([real_data[f] for f in numerical_features], axis=-1)
    gen_matrix = jnp.stack([generated_data[f] for f in numerical_features], axis=-1)

    # Normalize features to [0, 1] for fair distance computation
    combined = jnp.concatenate([real_matrix, gen_matrix], axis=0)
    min_vals = jnp.min(combined, axis=0)
    max_vals = jnp.max(combined, axis=0)
    ranges = max_vals - min_vals + 1e-8  # Avoid division by zero

    real_normalized = (real_matrix - min_vals) / ranges
    gen_normalized = (gen_matrix - min_vals) / ranges

    # Compute minimum distance from each generated sample to any real sample
    min_distances = []
    for gen_sample in gen_normalized:
        distances = jnp.sum((real_normalized - gen_sample) ** 2, axis=-1)
        min_dist = jnp.min(distances)
        min_distances.append(min_dist)

    # Return mean minimum distance
    return jnp.mean(jnp.array(min_distances))


def _compute_memorization_score(
    real_data: dict[str, jax.Array],
    generated_data: dict[str, jax.Array],
    categorical_features: list[str],
    ordinal_features: list[str],
    binary_features: list[str],
) -> jax.Array:
    """Compute memorization score (proportion of exact matches).

    Args:
        real_data: Real data
        generated_data: Generated data
        categorical_features: List of categorical feature names
        ordinal_features: List of ordinal feature names
        binary_features: List of binary feature names

    Returns:
        Memorization score (lower is better)
    """
    # Check for exact matches in categorical and binary features
    exact_matches = 0
    total_generated = len(next(iter(generated_data.values())))

    # Check each generated sample against all real samples
    for i in range(total_generated):
        gen_sample = {feature: data[i] for feature, data in generated_data.items()}

        # Check if this generated sample exactly matches any real sample
        for j in range(len(next(iter(real_data.values())))):
            real_sample = {feature: data[j] for feature, data in real_data.items()}

            # Check exact match for categorical, ordinal, and binary features
            match = True
            for feature in categorical_features + ordinal_features + binary_features:
                if gen_sample[feature] != real_sample[feature]:
                    match = False
                    break

            if match:
                exact_matches += 1
                break

    return jnp.array(exact_matches / total_generated)


def _calculate_rmsd_matrix(coordinates: jnp.ndarray, atom_mask: jnp.ndarray) -> jnp.ndarray:
    """Calculate pairwise RMSD matrix between conformations."""
    batch_size = coordinates.shape[0]
    rmsd_matrix = jnp.zeros((batch_size, batch_size))

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            rmsd = _calculate_rmsd(coordinates[i], coordinates[j], atom_mask[i], atom_mask[j])
            rmsd_matrix = rmsd_matrix.at[i, j].set(rmsd)
            rmsd_matrix = rmsd_matrix.at[j, i].set(rmsd)

    return rmsd_matrix


def _calculate_rmsd(
    coords1: jnp.ndarray,
    coords2: jnp.ndarray,
    mask1: jnp.ndarray,
    mask2: jnp.ndarray,
) -> float:
    """Calculate RMSD between two conformations."""
    # Ensure same number of atoms
    common_mask = mask1 & mask2
    if jnp.sum(common_mask) == 0:
        return float("inf")

    # Extract common atoms
    c1 = coords1[common_mask]
    c2 = coords2[common_mask]

    # Center coordinates
    c1_centered = c1 - jnp.mean(c1, axis=0)
    c2_centered = c2 - jnp.mean(c2, axis=0)

    # Calculate RMSD
    squared_diffs = jnp.sum((c1_centered - c2_centered) ** 2, axis=1)
    rmsd = jnp.sqrt(jnp.mean(squared_diffs))

    return rmsd
