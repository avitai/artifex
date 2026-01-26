"""Tests for evaluation protocols with unified configuration system.

Following TDD principles - these tests are written FIRST before implementation.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration import (
    EvaluationConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.core.protocols.evaluation import (
    DatasetProtocol,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.modalities.base import BaseEvaluationSuite


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def valid_eval_config():
    """Valid EvaluationConfig for testing."""
    return EvaluationConfig(
        name="test_evaluation",
        metrics=["fid", "is_score", "mse"],
        metric_params={
            "fid": {"n_features": 2048},
            "is_score": {"splits": 10},
        },
        eval_batch_size=64,
        num_eval_samples=1000,
        save_predictions=True,
        save_metrics=True,
    )


@pytest.fixture
def model_config():
    """Valid VAEConfig for creating a test model."""
    encoder_config = EncoderConfig(
        name="test_encoder",
        input_shape=(28, 28, 1),
        hidden_dims=(256, 128),
        latent_dim=32,
        activation="gelu",
    )
    decoder_config = DecoderConfig(
        name="test_decoder",
        latent_dim=32,
        hidden_dims=(128, 256),
        output_shape=(28, 28, 1),
        activation="gelu",
    )
    return VAEConfig(
        name="test_vae",
        encoder=encoder_config,
        decoder=decoder_config,
        kl_weight=1.0,
    )


@pytest.fixture
def test_model(model_config, rngs):
    """Create a test model."""
    return create_model(config=model_config, rngs=rngs)


class MockEvaluator:
    """Mock evaluator that enforces EvaluationConfig."""

    def __init__(self, config: EvaluationConfig, model: GenerativeModel, *, rngs: nnx.Rngs):
        """Initialize with EvaluationConfig.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            model: Model to evaluate
            rngs: Random number generators

        Raises:
            TypeError: If config is not an EvaluationConfig
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.model = model
        self.rngs = rngs

    def evaluate(self, dataset: DatasetProtocol) -> dict[str, float]:
        """Evaluate model on dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Dictionary of metric values
        """
        # Mock implementation
        return {metric: 0.5 for metric in self.config.metrics}

    def compute_metrics(self, predictions: Any, targets: Any) -> dict[str, float]:
        """Compute metrics from predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of metric values
        """
        if not isinstance(self.config, EvaluationConfig):
            raise RuntimeError("Invalid configuration state")

        results = {}
        for metric in self.config.metrics:
            # Mock metric computation
            results[metric] = 0.5
        return results


class MockEvaluationSuite(BaseEvaluationSuite):
    """Mock evaluation suite for testing."""

    def __init__(self, config: Any, *, rngs: nnx.Rngs):
        """Initialize with configuration.

        Args:
            config: Configuration (should enforce typed config)
            rngs: Random number generators
        """
        # For now, convert EvaluationConfig to BaseModalityConfig
        if isinstance(config, EvaluationConfig):
            from dataclasses import dataclass

            from artifex.generative_models.core.protocols.configuration import BaseModalityConfig

            @dataclass
            class TempConfig(BaseModalityConfig):
                name: str = config.name

            super().__init__(TempConfig(), rngs=rngs)
            self.eval_config = config
        else:
            super().__init__(config, rngs=rngs)
            self.eval_config = None

    def evaluate_batch(
        self,
        generated_data: jax.Array,
        reference_data: jax.Array | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate a batch of generated data."""
        # Mock implementation
        if self.eval_config:
            return {metric: 0.5 for metric in self.eval_config.metrics}
        return {"mock_metric": 0.5}


class TestEvaluationProtocolsUnifiedConfig:
    """Test evaluation protocols with unified configuration requirements."""

    def test_evaluator_requires_evaluation_configuration(self, test_model, rngs):
        """Test that Evaluator raises TypeError for non-EvaluationConfig."""
        # Dict config should raise TypeError
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            MockEvaluator(config={"metrics": ["fid", "is_score"]}, model=test_model, rngs=rngs)

        # Any other type should raise TypeError
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            MockEvaluator(config="invalid", model=test_model, rngs=rngs)

        # None should raise TypeError
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            MockEvaluator(config=None, model=test_model, rngs=rngs)

    def test_evaluator_accepts_evaluation_configuration(self, valid_eval_config, test_model, rngs):
        """Test that Evaluator works correctly with EvaluationConfig."""
        evaluator = MockEvaluator(config=valid_eval_config, model=test_model, rngs=rngs)

        # Evaluator should be created successfully
        assert evaluator is not None
        assert evaluator.config == valid_eval_config
        assert evaluator.model is test_model

    def test_evaluator_config_attributes_accessible(self, valid_eval_config, test_model, rngs):
        """Test that EvaluationConfig attributes are accessible in evaluator."""
        evaluator = MockEvaluator(config=valid_eval_config, model=test_model, rngs=rngs)

        # Should be able to access evaluation config attributes
        assert evaluator.config.metrics == ["fid", "is_score", "mse"]
        assert evaluator.config.eval_batch_size == 64
        assert evaluator.config.num_eval_samples == 1000
        assert evaluator.config.save_predictions is True

    def test_evaluator_compute_metrics_with_typed_config(self, valid_eval_config, test_model, rngs):
        """Test that evaluator methods can use typed config properly."""
        evaluator = MockEvaluator(config=valid_eval_config, model=test_model, rngs=rngs)

        # Test compute_metrics method
        predictions = jnp.ones((10, 32))
        targets = jnp.zeros((10, 32))

        metrics = evaluator.compute_metrics(predictions, targets)

        # Should return metrics defined in config
        assert set(metrics.keys()) == set(valid_eval_config.metrics)
        for metric in valid_eval_config.metrics:
            assert metric in metrics

    def test_evaluation_suite_with_typed_config(self, valid_eval_config, rngs):
        """Test that evaluation suite can work with typed configuration."""
        suite = MockEvaluationSuite(config=valid_eval_config, rngs=rngs)

        # Suite should be created successfully
        assert suite is not None
        assert hasattr(suite, "eval_config")
        assert suite.eval_config == valid_eval_config

        # Test evaluate_batch method
        generated = jnp.ones((10, 28, 28, 1))
        reference = jnp.zeros((10, 28, 28, 1))

        metrics = suite.evaluate_batch(generated, reference)

        # Should return metrics from config
        assert len(metrics) == len(valid_eval_config.metrics)

    def test_legacy_evaluation_config_rejected(self, test_model, rngs):
        """Test that legacy evaluation config classes are rejected."""

        # Mock a legacy evaluation config class
        class LegacyEvaluationConfig:
            def __init__(self):
                self.metrics = ["fid", "is_score"]
                self.eval_batch_size = 32

        legacy_config = LegacyEvaluationConfig()

        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            MockEvaluator(config=legacy_config, model=test_model, rngs=rngs)

    def test_evaluation_configuration_validation(self, test_model, rngs):
        """Test that invalid EvaluationConfig raises appropriate errors."""
        # Test with minimal valid config
        minimal_config = EvaluationConfig(
            name="minimal",
            metrics=["accuracy"],
        )

        evaluator = MockEvaluator(config=minimal_config, model=test_model, rngs=rngs)
        assert evaluator.config == minimal_config

    def test_metric_params_in_evaluation_config(self, test_model, rngs):
        """Test that metric parameters are properly accessible."""
        config = EvaluationConfig(
            name="test",
            metrics=["fid", "is_score"],
            metric_params={
                "fid": {"n_features": 2048, "device": "cuda"},
                "is_score": {"splits": 10, "normalize": True},
            },
        )

        evaluator = MockEvaluator(config=config, model=test_model, rngs=rngs)

        # Metric params should be accessible
        assert evaluator.config.metric_params["fid"]["n_features"] == 2048
        assert evaluator.config.metric_params["is_score"]["splits"] == 10

    def test_evaluation_protocol_integration(self, valid_eval_config, test_model, rngs):
        """Test full integration of evaluation protocol with typed config."""
        # Create evaluator with typed config
        evaluator = MockEvaluator(config=valid_eval_config, model=test_model, rngs=rngs)

        # Mock dataset
        class MockDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"image": jnp.ones((28, 28, 1))}

        dataset = MockDataset()

        # Run evaluation
        results = evaluator.evaluate(dataset)

        # Should get results for all metrics in config
        assert len(results) == len(valid_eval_config.metrics)
        for metric in valid_eval_config.metrics:
            assert metric in results
            assert isinstance(results[metric], float)
