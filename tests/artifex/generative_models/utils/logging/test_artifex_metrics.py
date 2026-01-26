"""
Tests for the metrics logger implementation.

These tests verify the metrics logging functionality in Artifex library.
"""

import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from artifex.generative_models.core.evaluation.metrics.base import MetricModule as Metric
from artifex.generative_models.utils.logging import (
    ConsoleLogger,
    FileLogger,
    MetricsLogger,
)


class DummyMetric(Metric):
    """Dummy metric for testing."""

    def __init__(self, name="dummy_metric", batch_size=32):
        """Initialize the dummy metric."""
        super().__init__(name=name, batch_size=batch_size)

    def __call__(self, real_data, generated_data, **kwargs):
        """Compute the dummy metric."""
        # Just return the mean difference as a dummy metric
        return {
            "value": float(np.mean(np.abs(real_data - generated_data))),
            "min": float(np.min(np.abs(real_data - generated_data))),
            "max": float(np.max(np.abs(real_data - generated_data))),
        }


@pytest.fixture
def dummy_metric():
    """Create a dummy metric for testing."""
    return DummyMetric()


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def console_logger():
    """Create a console logger for testing."""
    logger = ConsoleLogger(name="test_metrics_console")
    yield logger
    logger.close()


@pytest.fixture
def file_logger(temp_log_dir):
    """Create a file logger for testing."""
    logger = FileLogger(name="test_metrics_file", log_dir=temp_log_dir)
    yield logger
    logger.close()


@pytest.fixture
def test_data():
    """Create test data."""
    rng = np.random.RandomState(42)
    real_data = jnp.array(rng.normal(0, 1, (10, 5)))
    generated_data = jnp.array(rng.normal(0.2, 1.1, (10, 5)))
    return real_data, generated_data


def test_metrics_logger_creation(console_logger, dummy_metric):
    """Test MetricsLogger creation."""
    metrics_logger = MetricsLogger(
        logger=console_logger,
        metrics={"dummy": dummy_metric},
        prefix="test/",
        compute_frequency=10,
    )
    assert metrics_logger.logger == console_logger
    assert "dummy" in metrics_logger.metrics
    assert metrics_logger.prefix == "test/"
    assert metrics_logger.compute_frequency == 10


def test_metrics_logger_empty_creation(console_logger):
    """Test MetricsLogger creation with no metrics."""
    metrics_logger = MetricsLogger(logger=console_logger)
    assert metrics_logger.logger == console_logger
    assert not metrics_logger.metrics
    assert metrics_logger.prefix == ""
    assert metrics_logger.compute_frequency == 100


def test_add_remove_metric(console_logger, dummy_metric):
    """Test adding and removing metrics."""
    metrics_logger = MetricsLogger(logger=console_logger)
    assert not metrics_logger.metrics

    # Add metric
    metrics_logger.add_metric("dummy1", dummy_metric)
    assert "dummy1" in metrics_logger.metrics
    assert metrics_logger.metrics["dummy1"] == dummy_metric

    # Add another metric
    metrics_logger.add_metric("dummy2", DummyMetric(name="dummy2"))
    assert "dummy2" in metrics_logger.metrics

    # Remove metric
    assert metrics_logger.remove_metric("dummy1")
    assert "dummy1" not in metrics_logger.metrics
    assert "dummy2" in metrics_logger.metrics

    # Try removing non-existent metric
    assert not metrics_logger.remove_metric("nonexistent")


def test_compute_metrics(console_logger, dummy_metric, test_data):
    """Test computing metrics."""
    real_data, generated_data = test_data
    metrics_logger = MetricsLogger(
        logger=console_logger,
        metrics={"dummy": dummy_metric},
    )

    # Compute metrics
    results = metrics_logger.compute_metrics(
        real_data=real_data,
        generated_data=generated_data,
        step=0,
    )

    assert "dummy" in results
    assert "value" in results["dummy"]
    assert "min" in results["dummy"]
    assert "max" in results["dummy"]
    assert isinstance(results["dummy"]["value"], float)


def test_compute_frequency(console_logger, dummy_metric, test_data):
    """Test metric computation frequency."""
    real_data, generated_data = test_data
    metrics_logger = MetricsLogger(
        logger=console_logger,
        metrics={"dummy": dummy_metric},
        compute_frequency=10,
    )

    # First computation should happen
    results1 = metrics_logger.compute_metrics(
        real_data=real_data,
        generated_data=generated_data,
        step=0,
    )
    assert results1

    # This should be skipped due to frequency
    results2 = metrics_logger.compute_metrics(
        real_data=real_data,
        generated_data=generated_data,
        step=5,
    )
    assert not results2

    # This should compute again
    results3 = metrics_logger.compute_metrics(
        real_data=real_data,
        generated_data=generated_data,
        step=10,
    )
    assert results3


def test_should_compute(console_logger):
    """Test should_compute method."""
    metrics_logger = MetricsLogger(
        logger=console_logger,
        compute_frequency=10,
    )
    metrics_logger.last_computed_step = 0

    assert not metrics_logger.should_compute(5)
    assert metrics_logger.should_compute(10)
    assert metrics_logger.should_compute(15)


def test_log_training_metrics(file_logger, temp_log_dir):
    """Test logging training metrics."""
    metrics_logger = MetricsLogger(logger=file_logger)
    metrics = {"loss": 0.5, "accuracy": 0.8}
    metrics_logger.log_training_metrics(metrics, step=10)

    # Check that the metrics file contains the prefixed metrics
    with open(file_logger.metrics_file, "r") as f:
        content = f.read()
        assert "train/loss" in content
        assert "train/accuracy" in content


def test_log_validation_metrics(file_logger, temp_log_dir):
    """Test logging validation metrics."""
    metrics_logger = MetricsLogger(logger=file_logger)
    metrics = {"loss": 0.6, "accuracy": 0.75}
    metrics_logger.log_validation_metrics(metrics, step=10)

    # Check that the metrics file contains the prefixed metrics
    with open(file_logger.metrics_file, "r") as f:
        content = f.read()
        assert "val/loss" in content
        assert "val/accuracy" in content


def test_log_test_metrics(file_logger, temp_log_dir):
    """Test logging test metrics."""
    metrics_logger = MetricsLogger(logger=file_logger)
    metrics = {"loss": 0.7, "accuracy": 0.7}
    metrics_logger.log_test_metrics(metrics, step=10)

    # Check that the metrics file contains the prefixed metrics
    with open(file_logger.metrics_file, "r") as f:
        content = f.read()
        assert "test/loss" in content
        assert "test/accuracy" in content


def test_log_generated_samples(console_logger):
    """Test logging generated samples."""
    metrics_logger = MetricsLogger(logger=console_logger)
    samples = jnp.zeros((5, 32, 32, 3))

    # This should not raise any exceptions
    metrics_logger.log_generated_samples(samples, step=10)


def test_log_comparison(console_logger):
    """Test logging sample comparisons."""
    metrics_logger = MetricsLogger(logger=console_logger)
    real = jnp.zeros((5, 32, 32, 3))
    gen = jnp.ones((5, 32, 32, 3))

    # This should not raise any exceptions
    metrics_logger.log_comparison(real, gen, step=10)
