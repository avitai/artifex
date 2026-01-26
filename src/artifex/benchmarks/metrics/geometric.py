"""Geometric metrics for point cloud and 3D shape evaluation."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.benchmarks.metrics.core import MetricBase
from artifex.generative_models.core.configuration import EvaluationConfig


class PointCloudMetrics(MetricBase):
    """Point cloud metrics including 1-NN accuracy, coverage, and Chamfer distance.

    This class implements comprehensive evaluation metrics for point cloud
    generation models following the MetricProtocol interface.
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize point cloud metrics.

        Args:
            rngs: NNX Rngs for stochastic operations
            config: Evaluation configuration (must be EvaluationConfig)
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size
        self.metric_name = "point_cloud_metrics"

        # Point cloud parameters from config
        pc_params = config.metric_params.get("point_cloud", {})
        self.coverage_threshold = pc_params.get("coverage_threshold", 0.1)
        self.metric_weights = pc_params.get(
            "metric_weights",
            {
                "1nn_accuracy": 0.4,
                "coverage": 0.3,
                "geometric_fidelity": 0.2,
                "chamfer_distance": 0.1,
            },
        )

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute point cloud metrics required by MetricProtocol.

        Args:
            real_data: Real point clouds
            generated_data: Generated point clouds
            **kwargs: Additional parameters

        Returns:
            dictionary of computed metric values
        """
        return self.compute_metrics(generated_data, real_data, **kwargs)

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Real point clouds
            generated_data: Generated point clouds

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check that both have same number of dimensions (should be 3D: batch x points x coords)
        if len(real_data.shape) != 3 or len(generated_data.shape) != 3:
            return False

        # Check that last dimension is 3 (for 3D coordinates)
        if real_data.shape[-1] != 3 or generated_data.shape[-1] != 3:
            return False

        return True

    def compute_metric(self, generated: jax.Array, real: jax.Array, **kwargs) -> float:
        """Compute comprehensive point cloud metrics.

        Args:
            generated: Generated point clouds [batch_size, num_points, 3]
            real: Real point clouds [batch_size, num_points, 3]
            **kwargs: Additional metric parameters

        Returns:
            Combined metric score
        """
        metrics = self.compute_metrics(generated, real, **kwargs)

        # Weighted combination of metrics
        weights = self.metric_weights

        combined_score = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                if metric_name == "chamfer_distance":
                    # Lower is better for distance metrics
                    score = 1.0 / (1.0 + metrics[metric_name])
                else:
                    score = metrics[metric_name]
                combined_score += weight * score
                total_weight += weight

        return combined_score / total_weight if total_weight > 0 else 0.0

    def compute_metrics(self, generated: jax.Array, real: jax.Array, **kwargs) -> dict[str, float]:
        """Compute multiple point cloud metrics.

        Args:
            generated: Generated point clouds [batch_size, num_points, 3]
            real: Real point clouds [batch_size, num_points, 3]
            **kwargs: Additional parameters

        Returns:
            dictionary with metric scores
        """
        metrics = {}

        # 1-NN Accuracy
        metrics["1nn_accuracy"] = self._compute_1nn_accuracy(generated, real)

        # Coverage
        metrics["coverage"] = self._compute_coverage(generated, real, **kwargs)

        # Chamfer Distance
        metrics["chamfer_distance"] = self._compute_chamfer_distance(generated, real)

        # Geometric Fidelity
        metrics["geometric_fidelity"] = self._compute_geometric_fidelity(generated, real)

        # Earth Mover's Distance (approximated)
        metrics["earth_movers_distance"] = self._compute_emd_approximation(generated, real)

        return metrics

    def _compute_1nn_accuracy(self, generated: jax.Array, real: jax.Array) -> float:
        """Compute 1-NN accuracy between generated and real point clouds.

        This metric measures how well the generated distribution matches
        the real distribution by checking if the nearest neighbor of each
        generated sample is from the correct distribution.

        Args:
            generated: Generated point clouds
            real: Real point clouds

        Returns:
            1-NN accuracy score
        """
        # Flatten point clouds to compute distances between entire clouds
        gen_flat = generated.reshape(generated.shape[0], -1)
        real_flat = real.reshape(real.shape[0], -1)

        # Combine datasets
        all_data = jnp.concatenate([gen_flat, real_flat], axis=0)
        labels = jnp.concatenate(
            [
                jnp.zeros(generated.shape[0]),  # 0 for generated
                jnp.ones(real.shape[0]),  # 1 for real
            ]
        )

        # Compute pairwise distances
        distances = self._compute_pairwise_distances(all_data, all_data)

        # For each sample, find nearest neighbor (excluding itself)
        distances = distances.at[jnp.diag_indices_from(distances)].set(jnp.inf)
        nn_indices = jnp.argmin(distances, axis=1)
        nn_labels = labels[nn_indices]

        # Compute accuracy
        correct_predictions = jnp.sum(labels == nn_labels)
        total_samples = len(labels)

        return float(correct_predictions / total_samples)

    def _compute_coverage(
        self, generated: jax.Array, real: jax.Array, k: int = 5, **kwargs
    ) -> float:
        """Compute coverage metric.

        Coverage measures what fraction of the real data is "covered" by
        the generated data, using k-nearest neighbors.

        Args:
            generated: Generated point clouds
            real: Real point clouds
            k: Number of nearest neighbors to consider
            **kwargs: Additional parameters including coverage_threshold

        Returns:
            Coverage score
        """
        # Flatten point clouds
        gen_flat = generated.reshape(generated.shape[0], -1)
        real_flat = real.reshape(real.shape[0], -1)

        # For each real sample, find k nearest neighbors in generated set
        distances = self._compute_pairwise_distances(real_flat, gen_flat)

        # Get k nearest neighbors for each real sample
        # Adjust k if it's larger than available samples
        actual_k = min(k, distances.shape[1])
        knn_distances, _ = jax.lax.top_k(-distances, actual_k)  # Negative for top-k

        # Count how many real samples have at least one close generated neighbor
        # Use a threshold based on the dataset scale
        threshold = self.coverage_threshold
        covered = jnp.any(-knn_distances <= threshold, axis=1)

        coverage = jnp.mean(covered)
        return float(coverage)

    def _compute_chamfer_distance(self, generated: jax.Array, real: jax.Array) -> float:
        """Compute Chamfer distance between point clouds.

        Args:
            generated: Generated point clouds [batch_size, num_points, 3]
            real: Real point clouds [batch_size, num_points, 3]

        Returns:
            Average Chamfer distance
        """

        def chamfer_distance_single(gen_cloud, real_cloud):
            # Distance from each generated point to nearest real point
            gen_to_real = self._compute_pairwise_distances(
                gen_cloud[None, :, :], real_cloud[None, :, :]
            )[0]
            gen_to_real_min = jnp.min(gen_to_real, axis=1)

            # Distance from each real point to nearest generated point
            real_to_gen = self._compute_pairwise_distances(
                real_cloud[None, :, :], gen_cloud[None, :, :]
            )[0]
            real_to_gen_min = jnp.min(real_to_gen, axis=1)

            # Chamfer distance is sum of both directions
            return jnp.mean(gen_to_real_min) + jnp.mean(real_to_gen_min)

        # Compute for each pair in the batch
        chamfer_distances = jax.vmap(chamfer_distance_single)(generated, real)

        return float(jnp.mean(chamfer_distances))

    def _compute_geometric_fidelity(self, generated: jax.Array, real: jax.Array) -> float:
        """Compute geometric fidelity metric.

        This measures how well the generated point clouds preserve
        geometric properties like local density and curvature.

        Args:
            generated: Generated point clouds
            real: Real point clouds

        Returns:
            Geometric fidelity score
        """
        # Compute local density statistics
        gen_densities = self._compute_local_densities(generated)
        real_densities = self._compute_local_densities(real)

        # Compare density distributions using KL divergence approximation
        density_similarity = self._compute_distribution_similarity(gen_densities, real_densities)

        # Compute surface normal consistency (if applicable)
        # For now, use a simplified geometric consistency measure
        geometric_consistency = self._compute_geometric_consistency(generated, real)

        # Combine metrics
        fidelity = 0.7 * density_similarity + 0.3 * geometric_consistency

        return float(fidelity)

    def _compute_local_densities(self, point_clouds: jax.Array, k: int = 10) -> jax.Array:
        """Compute local point densities.

        Args:
            point_clouds: Point clouds [batch_size, num_points, 3]
            k: Number of neighbors for density computation

        Returns:
            Local densities for each point
        """

        def compute_density_single(cloud):
            # Compute pairwise distances within the cloud
            distances = self._compute_pairwise_distances(cloud[None, :, :], cloud[None, :, :])[0]

            # Set diagonal to infinity to exclude self
            distances = distances.at[jnp.diag_indices_from(distances)].set(jnp.inf)

            # Get k nearest neighbors
            actual_k = min(k, distances.shape[1])
            knn_distances, _ = jax.lax.top_k(-distances, actual_k)

            # Local density is inverse of average distance to k nearest neighbors
            avg_distances = jnp.mean(-knn_distances, axis=1)
            densities = 1.0 / (avg_distances + 1e-8)

            return densities

        # Compute for each cloud in batch
        densities = jax.vmap(compute_density_single)(point_clouds)

        return densities.flatten()

    def _compute_distribution_similarity(self, dist1: jax.Array, dist2: jax.Array) -> float:
        """Compute similarity between two distributions.

        Args:
            dist1: First distribution
            dist2: Second distribution

        Returns:
            Similarity score (higher is better)
        """
        # Compute histograms
        min_val = jnp.min(jnp.concatenate([dist1, dist2]))
        max_val = jnp.max(jnp.concatenate([dist1, dist2]))

        bins = jnp.linspace(min_val, max_val, 50)

        hist1 = jnp.histogram(dist1, bins)[0]
        hist2 = jnp.histogram(dist2, bins)[0]

        # Normalize histograms
        hist1 = hist1 / (jnp.sum(hist1) + 1e-8)
        hist2 = hist2 / (jnp.sum(hist2) + 1e-8)

        # Compute similarity using histogram intersection
        intersection = jnp.sum(jnp.minimum(hist1, hist2))

        return float(intersection)

    def _compute_geometric_consistency(self, generated: jax.Array, real: jax.Array) -> float:
        """Compute geometric consistency between generated and real clouds.

        Args:
            generated: Generated point clouds
            real: Real point clouds

        Returns:
            Geometric consistency score
        """

        # Simplified geometric consistency based on pairwise distance distributions
        def get_distance_stats(clouds):
            def stats_single(cloud):
                distances = self._compute_pairwise_distances(cloud[None, :, :], cloud[None, :, :])[
                    0
                ]
                # Remove diagonal (self-distances)
                distances = distances[~jnp.eye(distances.shape[0], dtype=bool)]
                return jnp.array([jnp.mean(distances), jnp.std(distances)])

            return jax.vmap(stats_single)(clouds)

        gen_stats = get_distance_stats(generated)
        real_stats = get_distance_stats(real)

        # Compare mean and std of distances
        mean_diff = jnp.mean(jnp.abs(gen_stats[:, 0] - real_stats[:, 0]))
        std_diff = jnp.mean(jnp.abs(gen_stats[:, 1] - real_stats[:, 1]))

        # Convert to similarity score (lower difference = higher similarity)
        consistency = jnp.exp(-(mean_diff + std_diff))

        return float(consistency)

    def _compute_emd_approximation(self, generated: jax.Array, real: jax.Array) -> float:
        """Compute approximation of Earth Mover's Distance.

        Args:
            generated: Generated point clouds
            real: Real point clouds

        Returns:
            Approximated EMD score
        """

        # Simplified EMD approximation using sorted distances
        def emd_single(gen_cloud, real_cloud):
            # Compute all pairwise distances
            distances = self._compute_pairwise_distances(
                gen_cloud[None, :, :], real_cloud[None, :, :]
            )[0]

            # Sort distances for approximation
            sorted_distances = jnp.sort(distances.flatten())

            # Use median as a simple EMD approximation
            return jnp.median(sorted_distances)

        # Compute for each pair in batch
        emds = jax.vmap(emd_single)(generated, real)

        return float(jnp.mean(emds))

    def _compute_pairwise_distances(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute pairwise Euclidean distances between point sets.

        Args:
            x: First point set [num_points_x, features] or [batch_size, num_points_x, features]
            y: Second point set [num_points_y, features] or [batch_size, num_points_y, features]

        Returns:
            Distance matrix [num_points_x, num_points_y] or [batch_size, num_points_x, num_points_y]
        """
        if len(x.shape) == 2:
            # Handle 2D case: [num_points, features]
            x_expanded = x[:, None, :]  # [num_x, 1, features]
            y_expanded = y[None, :, :]  # [1, num_y, features]

            # Compute squared differences
            diff = x_expanded - y_expanded  # [num_x, num_y, features]

            # Compute Euclidean distances
            distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))

        else:
            # Handle 3D case: [batch_size, num_points, features]
            x_expanded = x[:, :, None, :]  # [batch, num_x, 1, features]
            y_expanded = y[:, None, :, :]  # [batch, 1, num_y, features]

            # Compute squared differences
            diff = x_expanded - y_expanded  # [batch, num_x, num_y, features]

            # Compute Euclidean distances
            distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))

        return distances


# Factory functions for convenient metric creation
def create_point_cloud_metric(
    *,
    rngs: nnx.Rngs,
    coverage_threshold: float = 0.1,
    metric_weights: dict[str, float] | None = None,
    batch_size: int = 32,
    config_name: str = "point_cloud_metric",
) -> PointCloudMetrics:
    """Create point cloud metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        coverage_threshold: Threshold for coverage computation
        metric_weights: Weights for combining metrics
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured PointCloudMetrics instance
    """
    if metric_weights is None:
        metric_weights = {
            "1nn_accuracy": 0.4,
            "coverage": 0.3,
            "geometric_fidelity": 0.2,
            "chamfer_distance": 0.1,
        }

    config = EvaluationConfig(
        name=config_name,
        metrics=["point_cloud"],
        metric_params={
            "point_cloud": {
                "higher_is_better": True,
                "coverage_threshold": coverage_threshold,
                "metric_weights": metric_weights,
            }
        },
        eval_batch_size=batch_size,
    )

    return PointCloudMetrics(rngs=rngs, config=config)
