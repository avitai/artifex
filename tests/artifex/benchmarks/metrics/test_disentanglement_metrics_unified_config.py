"""Test disentanglement metrics with unified configuration system.

Following TDD principles - write tests first, then implement.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class TestDisentanglementMetricsUnifiedConfig:
    """Test disentanglement metrics with new unified configuration system."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def test_data(self):
        """Create test latent representations and ground truth factors."""
        batch_size = 100
        latent_dim = 10
        num_factors = 5

        # Simulated latent representations
        key = jax.random.key(42)
        key1, key2 = jax.random.split(key)
        latents = jax.random.normal(key1, (batch_size, latent_dim))

        # Simulated ground truth factors (some binary, some continuous)
        factors = jnp.zeros((batch_size, num_factors))
        # Binary factors
        factors = factors.at[:, 0].set(jax.random.bernoulli(key2, p=0.5, shape=(batch_size)))
        factors = factors.at[:, 1].set(jax.random.bernoulli(key2, p=0.3, shape=(batch_size)))
        # Continuous factors
        factors = factors.at[:, 2].set(
            jax.random.uniform(key2, shape=(batch_size), minval=0, maxval=1)
        )
        factors = factors.at[:, 3].set(jax.random.normal(key2, shape=(batch_size)))
        factors = factors.at[:, 4].set(
            jax.random.uniform(key2, shape=(batch_size), minval=-1, maxval=1)
        )

        return factors, latents

    def test_mig_metric_requires_evaluation_config(self, rngs):
        """Test that MIG metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.disentanglement import MutualInformationGapMetric

        # Should only accept EvaluationConfig
        config = EvaluationConfig(
            name="mig_metric",
            metrics=["mig"],
            metric_params={
                "mig": {
                    "higher_is_better": True,
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = MutualInformationGapMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.eval_batch_size == 32

        # This should NOT work - no backward compatibility
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            MutualInformationGapMetric(rngs=rngs, config={"name": "mig"})

    def test_mig_computation(self, rngs, test_data):
        """Test MIG metric computation with typed config."""
        from artifex.benchmarks.metrics.disentanglement import MutualInformationGapMetric

        config = EvaluationConfig(
            name="mig_test", metrics=["mig"], metric_params={"mig": {}}, eval_batch_size=16
        )

        metric = MutualInformationGapMetric(rngs=rngs, config=config)
        factors, latents = test_data

        result = metric.compute(factors, latents)

        assert "mig_score" in result
        assert isinstance(result["mig_score"], float)
        assert 0 <= result["mig_score"] <= 1

    def test_sap_metric_requires_evaluation_config(self, rngs):
        """Test that SAP metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.disentanglement import SeparationMetric

        config = EvaluationConfig(
            name="sap_metric",
            metrics=["sap"],
            metric_params={
                "sap": {
                    "higher_is_better": True,
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = SeparationMetric(rngs=rngs, config=config)
        assert metric.config == config

        # No dict allowed
        with pytest.raises(TypeError):
            SeparationMetric(rngs=rngs, config={"name": "sap"})

    def test_dci_metric_requires_evaluation_config(self, rngs):
        """Test that DCI metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.disentanglement import DisentanglementMetric

        config = EvaluationConfig(
            name="dci_metric",
            metrics=["dci"],
            metric_params={
                "dci": {
                    "weights": {
                        "disentanglement": 0.4,
                        "completeness": 0.4,
                        "informativeness": 0.2,
                    }
                }
            },
            eval_batch_size=64,
        )

        # This should work
        metric = DisentanglementMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.eval_batch_size == 64

    def test_disentanglement_metric_factory_functions(self, rngs):
        """Test disentanglement metric factory functions."""
        from artifex.benchmarks.metrics.disentanglement import (
            create_dci_metric,
            create_mig_metric,
            create_sap_metric,
        )

        # MIG factory
        mig = create_mig_metric(rngs=rngs, batch_size=64)
        assert isinstance(mig.config, EvaluationConfig)
        assert mig.config.eval_batch_size == 64

        # SAP factory
        sap = create_sap_metric(rngs=rngs, batch_size=32)
        assert sap.config.eval_batch_size == 32

        # DCI factory
        dci = create_dci_metric(
            rngs=rngs,
            disentanglement_weight=0.5,
            completeness_weight=0.3,
            informativeness_weight=0.2,
            batch_size=128,
        )
        assert dci.config.eval_batch_size == 128
        weights = dci.config.metric_params["dci"]["weights"]
        assert weights["disentanglement"] == 0.5
        assert weights["completeness"] == 0.3
        assert weights["informativeness"] == 0.2

    def test_disentanglement_metrics_inherit_from_base(self, rngs):
        """Test that all disentanglement metrics inherit from MetricBase."""
        from artifex.benchmarks.metrics.core import MetricBase
        from artifex.benchmarks.metrics.disentanglement import MutualInformationGapMetric

        config = EvaluationConfig(
            name="test_inheritance", metrics=["mig"], metric_params={"mig": {}}
        )

        mig = MutualInformationGapMetric(rngs=rngs, config=config)
        assert isinstance(mig, MetricBase)

        # All should have required methods
        assert hasattr(mig, "compute")
        assert hasattr(mig, "validate_inputs")
        assert hasattr(mig, "rngs")

    def test_validation_inputs_for_disentanglement_metrics(self, rngs, test_data):
        """Test input validation for disentanglement metrics."""
        from artifex.benchmarks.metrics.disentanglement import MutualInformationGapMetric

        config = EvaluationConfig(
            name="validation_test", metrics=["mig"], metric_params={"mig": {}}
        )

        metric = MutualInformationGapMetric(rngs=rngs, config=config)
        factors, latents = test_data

        # Valid inputs
        assert metric.validate_inputs(factors, latents)

        # Invalid inputs - not arrays
        assert not metric.validate_inputs([1, 2, 3], latents)

        # Invalid inputs - wrong dimensions
        assert not metric.validate_inputs(factors[0], latents[0])

        # Invalid inputs - batch size mismatch
        assert not metric.validate_inputs(factors[:50], latents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
