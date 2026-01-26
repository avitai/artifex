"""Test geometric metrics with unified configuration system.

Following TDD principles - write tests first, then implement.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class TestGeometricMetricsUnifiedConfig:
    """Test geometric metrics with new unified configuration system."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def test_point_clouds(self):
        """Create test point cloud data."""
        batch_size = 10
        num_points = 100

        # Generate random point clouds
        key = jax.random.key(42)
        key1, key2 = jax.random.split(key)

        # Real point clouds - samples from a unit sphere
        real_points = jax.random.normal(key1, (batch_size, num_points, 3))
        real_points = real_points / jnp.linalg.norm(real_points, axis=-1, keepdims=True)

        # Generated point clouds - slightly perturbed version
        noise = jax.random.normal(key2, (batch_size, num_points, 3)) * 0.1
        generated_points = real_points + noise
        generated_points = generated_points / jnp.linalg.norm(
            generated_points, axis=-1, keepdims=True
        )

        return real_points, generated_points

    def test_point_cloud_metric_requires_evaluation_config(self, rngs):
        """Test that PointCloudMetrics requires EvaluationConfig."""
        from artifex.benchmarks.metrics.geometric import PointCloudMetrics

        # Should only accept EvaluationConfig
        config = EvaluationConfig(
            name="point_cloud_metric",
            metrics=["point_cloud"],
            metric_params={
                "point_cloud": {
                    "higher_is_better": True,
                    "metric_weights": {
                        "1nn_accuracy": 0.4,
                        "coverage": 0.3,
                        "geometric_fidelity": 0.2,
                        "chamfer_distance": 0.1,
                    },
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = PointCloudMetrics(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.eval_batch_size == 32

        # This should NOT work - no backward compatibility
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            PointCloudMetrics(rngs=rngs, config={"name": "point_cloud"})

    def test_point_cloud_computation(self, rngs, test_point_clouds):
        """Test point cloud metric computation with typed config."""
        from artifex.benchmarks.metrics.geometric import PointCloudMetrics

        config = EvaluationConfig(
            name="point_cloud_test",
            metrics=["point_cloud"],
            metric_params={
                "point_cloud": {
                    "coverage_threshold": 0.2,
                }
            },
            eval_batch_size=16,
        )

        metric = PointCloudMetrics(rngs=rngs, config=config)
        real_points, generated_points = test_point_clouds

        result = metric.compute(real_points, generated_points)

        # Check that all expected metrics are present
        expected_metrics = [
            "1nn_accuracy",
            "coverage",
            "chamfer_distance",
            "geometric_fidelity",
            "earth_movers_distance",
        ]

        for metric_name in expected_metrics:
            assert metric_name in result
            assert isinstance(result[metric_name], float)

    def test_geometric_metric_factory_functions(self, rngs):
        """Test geometric metric factory functions."""
        from artifex.benchmarks.metrics.geometric import create_point_cloud_metric

        # Point cloud factory
        pc_metric = create_point_cloud_metric(
            rngs=rngs,
            coverage_threshold=0.15,
            metric_weights={
                "1nn_accuracy": 0.5,
                "coverage": 0.3,
                "geometric_fidelity": 0.1,
                "chamfer_distance": 0.1,
            },
            batch_size=64,
        )
        assert isinstance(pc_metric.config, EvaluationConfig)
        assert pc_metric.config.eval_batch_size == 64
        assert pc_metric.metric_weights["1nn_accuracy"] == 0.5

    def test_geometric_metrics_inherit_from_base(self, rngs):
        """Test that all geometric metrics inherit from MetricBase."""
        from artifex.benchmarks.metrics.core import MetricBase
        from artifex.benchmarks.metrics.geometric import PointCloudMetrics

        config = EvaluationConfig(
            name="test_inheritance", metrics=["point_cloud"], metric_params={"point_cloud": {}}
        )

        pc_metric = PointCloudMetrics(rngs=rngs, config=config)
        assert isinstance(pc_metric, MetricBase)

        # All should have required methods
        assert hasattr(pc_metric, "compute")
        assert hasattr(pc_metric, "validate_inputs")
        assert hasattr(pc_metric, "rngs")

    def test_validation_inputs_for_geometric_metrics(self, rngs, test_point_clouds):
        """Test input validation for geometric metrics."""
        from artifex.benchmarks.metrics.geometric import PointCloudMetrics

        config = EvaluationConfig(
            name="validation_test", metrics=["point_cloud"], metric_params={"point_cloud": {}}
        )

        metric = PointCloudMetrics(rngs=rngs, config=config)
        real_points, generated_points = test_point_clouds

        # Valid inputs
        assert metric.validate_inputs(real_points, generated_points)

        # Invalid inputs - not arrays
        assert not metric.validate_inputs([1, 2, 3], generated_points)

        # Invalid inputs - wrong dimensions (needs 3D)
        assert not metric.validate_inputs(real_points[0], generated_points[0])

        # Invalid inputs - last dimension not 3
        wrong_dim = jnp.ones((10, 100, 2))
        assert not metric.validate_inputs(wrong_dim, generated_points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
