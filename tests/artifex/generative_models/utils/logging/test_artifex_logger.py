"""
Tests for the logger implementation.

These tests verify the logging functionality in Artifex library.
Including JAX compliance tests to ensure that logging modules properly handle
JAX arrays when used within NNX modules.
"""

import glob
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from artifex.generative_models.utils.logging import (
    ConsoleLogger,
    create_logger,
    FileLogger,
)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_console_logger_creation():
    """Test ConsoleLogger creation."""
    logger = ConsoleLogger(name="test_console_logger")
    assert logger.name == "test_console_logger"
    assert logger.log_dir is None
    logger.close()


def test_file_logger_creation(temp_log_dir):
    """Test FileLogger creation."""
    logger = FileLogger(name="test_file_logger", log_dir=temp_log_dir)
    assert logger.name == "test_file_logger"
    assert logger.log_dir == temp_log_dir
    assert os.path.exists(logger.log_file)
    # metrics_file is only created when logging actual metrics
    logger.close()


def test_logger_factory(temp_log_dir):
    """Test logger factory function."""
    # Console logger
    logger = create_logger(
        name="test_factory_console",
        log_to_console=True,
        log_to_file=False,
    )
    assert isinstance(logger, ConsoleLogger)
    logger.close()

    # File logger
    logger = create_logger(
        name="test_factory_file",
        log_dir=temp_log_dir,
        log_to_console=True,
        log_to_file=True,
    )
    assert isinstance(logger, FileLogger)
    assert logger.log_dir == temp_log_dir
    logger.close()


def test_console_logger_log_scalar():
    """Test logging a scalar value with ConsoleLogger."""
    logger = ConsoleLogger(name="test_scalar")
    logger.log_scalar("test_metric", 0.5, step=10)
    # No exception should be raised
    logger.close()


def test_console_logger_log_scalars():
    """Test logging multiple scalar values with ConsoleLogger."""
    logger = ConsoleLogger(name="test_scalars")
    metrics = {"metric1": 0.5, "metric2": 0.7}
    logger.log_scalars(metrics, step=10)
    # No exception should be raised
    logger.close()


def test_file_logger_log_scalar(temp_log_dir):
    """Test logging a scalar value with FileLogger."""
    logger = FileLogger(name="test_scalar", log_dir=temp_log_dir)
    logger.log_scalar("test_metric", 0.5, step=10)

    # Check that the metrics file contains the scalar
    with open(logger.metrics_file, "r") as f:
        content = f.read()
        assert "test_metric" in content
        assert "0.5" in content

    logger.close()


def test_file_logger_log_scalars(temp_log_dir):
    """Test logging multiple scalar values with FileLogger."""
    logger = FileLogger(name="test_scalars", log_dir=temp_log_dir)
    metrics = {"metric1": 0.5, "metric2": 0.7}
    logger.log_scalars(metrics, step=10)

    # Check that the metrics file contains the scalars
    with open(logger.metrics_file, "r") as f:
        content = f.read()
        assert "metric1" in content
        assert "0.5" in content
        assert "metric2" in content
        assert "0.7" in content

    logger.close()


def test_console_logger_log_image():
    """Test logging an image with ConsoleLogger."""
    logger = ConsoleLogger(name="test_image")

    # Create a test image
    image = np.zeros((32, 32, 3))
    logger.log_image("test_image", image, step=10)

    # Create a list of images
    images = [np.zeros((32, 32, 3)) for _ in range(3)]
    logger.log_image("test_images", images, step=10)

    # No exception should be raised
    logger.close()


def test_file_logger_log_image(temp_log_dir):
    """Test logging an image with FileLogger."""
    try:
        # Import matplotlib only when needed for this test
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing

        logger = FileLogger(name="test_image", log_dir=temp_log_dir)

        # Create a test image
        image = np.zeros((32, 32, 3))
        logger.log_image("test_image", image, step=10)

        # Check that the image directory exists
        images_dir = os.path.join(temp_log_dir, "images")
        assert os.path.exists(images_dir)

        # Check that an image file was created
        image_files = list(Path(images_dir).glob("test_image_*.png"))
        assert len(image_files) > 0

        logger.close()
    except ImportError:
        pytest.skip("Matplotlib not installed, skipping image test")


def test_logger_log_hyperparams(temp_log_dir):
    """Test logging hyperparameters."""
    logger = FileLogger(name="test_hyperparams", log_dir=temp_log_dir)
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "VAE",
    }
    logger.log_hyperparams(params)

    # Check that the hyperparams file exists
    hyperparams_files = list(Path(temp_log_dir).glob("hyperparams_*.txt"))
    assert len(hyperparams_files) > 0

    # Check file contents
    with open(hyperparams_files[0], "r") as f:
        content = f.read()
        assert "learning_rate: 0.001" in content
        assert "batch_size: 32" in content
        assert "model_type: VAE" in content

    logger.close()


def test_logger_log_text(temp_log_dir):
    """Test logging text."""
    logger = FileLogger(name="test_text", log_dir=temp_log_dir)
    text = "This is a test message."
    logger.log_text("test_text", text, step=10)

    # Check that the texts directory exists
    texts_dir = os.path.join(temp_log_dir, "texts")
    assert os.path.exists(texts_dir)

    # Check that a text file was created
    text_files = list(Path(texts_dir).glob("test_text_*.txt"))
    assert len(text_files) > 0

    # Check file contents
    with open(text_files[0], "r") as f:
        content = f.read()
        assert text in content

    logger.close()


# ========================================================================
# JAX Compliance Tests (merged from test_logger_jax_compliance.py)
# ========================================================================


class TestLoggerJAXCompliance:
    """Test that logger module can handle JAX arrays properly."""

    def test_console_logger_scalar_conversion(self):
        """Test that ConsoleLogger properly converts JAX scalars."""
        logger = ConsoleLogger("test_logger")

        # Test logging JAX scalar
        jax_scalar = jnp.array(0.5)

        # This should not raise an error
        logger.log_scalar("loss", float(jax_scalar), step=0)
        logger.close()

    def test_file_logger_with_jax_arrays(self, temp_log_dir):
        """Test that FileLogger handles JAX arrays correctly."""
        logger = FileLogger("test_logger", log_dir=temp_log_dir)

        # Test logging JAX scalar values
        jax_loss = jnp.array(0.123)
        jax_accuracy = jnp.array(0.987)

        # Convert to Python floats before logging
        logger.log_scalar("loss", float(jax_loss), step=0)
        logger.log_scalar("accuracy", float(jax_accuracy), step=1)

        # Check that the CSV file was created
        csv_files = glob.glob(os.path.join(temp_log_dir, "test_logger_*_metrics.csv"))
        assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"

        logger.close()


class TestLoggerUsageInNNXModule:
    """Test logger usage patterns within NNX modules."""

    def test_nnx_module_with_logging(self):
        """Test that we can use loggers properly within NNX modules."""

        class TrainingModule(nnx.Module):
            """Example NNX module that demonstrates proper logging."""

            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.dense = nnx.Linear(10, 1, rngs=rngs)

            def compute_loss(self, x: jax.Array, y: jax.Array) -> jax.Array:
                """Compute loss using JAX operations."""
                pred = self.dense(x)
                # Use JAX operations only
                loss = jnp.mean((pred - y) ** 2)
                return loss

            def train_step(self, x: jax.Array, y: jax.Array, logger: Any, step: int) -> jax.Array:
                """Training step with proper JAX array handling."""
                loss = self.compute_loss(x, y)

                # Convert JAX array to Python float for logging
                # This conversion happens OUTSIDE the computational graph
                loss_value = float(loss)
                logger.log_scalar("train/loss", loss_value, step=step)

                return loss

        # Create module
        rngs = nnx.Rngs(42)
        module = TrainingModule(rngs=rngs)
        logger = ConsoleLogger("training")

        # Run training step
        x = jnp.ones((4, 10))
        y = jnp.ones((4, 1))

        loss = module.train_step(x, y, logger, step=0)

        # Verify loss is still a JAX array
        assert isinstance(loss, jax.Array)
        logger.close()


class TestNumpyUsagePatterns:
    """Test to identify where numpy is incorrectly used in NNX context."""

    def test_identify_numpy_usage_in_logger(self):
        """Verify that numpy is not imported at module level in logger."""
        from artifex.generative_models.utils.logging import logger as logger_module

        # Check that numpy is NOT imported at module level
        assert not hasattr(logger_module, "np")

        # Check that JAX is imported instead
        assert hasattr(logger_module, "jax")
        assert hasattr(logger_module, "jnp")

        # The logger now uses JAX for:
        # 1. Type annotations (jax.Array)
        # 2. Array conversion (jnp.asarray)
        # Numpy is only imported locally when needed for matplotlib

    def test_histogram_logging_needs_jax_conversion(self, temp_log_dir):
        """Test that histogram logging needs JAX-compatible implementation."""
        logger = FileLogger("test_logger", log_dir=temp_log_dir)

        # Logger now accepts JAX arrays directly
        jax_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Can pass JAX array directly - conversion happens internally
        logger.log_histogram("values", jax_values, step=0)
        logger.close()


class TestProperJAXPatterns:
    """Test and demonstrate proper JAX usage patterns."""

    def test_proper_scalar_extraction(self):
        """Test proper way to extract scalars from JAX arrays."""
        # JAX array
        jax_loss = jnp.array(0.5)

        # Proper conversion for logging
        python_scalar = float(jax_loss)  # or jax_loss.item()

        assert isinstance(python_scalar, float)
        assert python_scalar == 0.5

    def test_proper_array_conversion(self):
        """Test proper conversion of JAX arrays when needed."""
        # JAX array
        jax_array = jnp.array([1.0, 2.0, 3.0])

        # If numpy array is absolutely needed (e.g., for external libraries)
        # This should happen OUTSIDE NNX modules
        numpy_array = np.asarray(jax_array)

        assert isinstance(numpy_array, np.ndarray)
        assert np.array_equal(numpy_array, [1.0, 2.0, 3.0])
