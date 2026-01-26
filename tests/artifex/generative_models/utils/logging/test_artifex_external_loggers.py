"""
Tests for the external logger implementations (MLflow, Wandb).

These tests verify the external logger integrations in Artifex library.
The tests use mocking to avoid actual dependencies on external packages.
"""

import tempfile
from unittest import mock

import pytest


def test_import_mlflow_logger():
    """Test importing MLFlowLogger."""
    # Create a mock for mlflow module
    mlflow_mock = mock.MagicMock()

    # Mock the import of mlflow inside the MLFlowLogger class
    with mock.patch.dict("sys.modules", {"mlflow": mlflow_mock}):
        from artifex.generative_models.utils.logging import MLFlowLogger

        # If we can import it, the test passes
        assert MLFlowLogger is not None


def test_import_wandb_logger():
    """Test importing WandbLogger."""
    # Create a mock for wandb module
    wandb_mock = mock.MagicMock()

    # Mock the import of wandb inside the WandbLogger class
    with mock.patch.dict("sys.modules", {"wandb": wandb_mock}):
        from artifex.generative_models.utils.logging import WandbLogger

        # If we can import it, the test passes
        assert WandbLogger is not None


def test_mlflow_logger_basic():
    """Test basic MLflow logger functionality with mocked dependencies."""
    # Create a mock for mlflow module
    mlflow_mock = mock.MagicMock()

    # Mock the import of mlflow inside the MLFlowLogger class
    with mock.patch.dict("sys.modules", {"mlflow": mlflow_mock}):
        # Now we can safely import the logger
        from artifex.generative_models.utils.logging import MLFlowLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the active run
            mock_run = mock.MagicMock()
            mock_run.info.run_id = "test_run_id"
            mlflow_mock.start_run.return_value = mock_run

            # Create the logger
            logger = MLFlowLogger(
                name="test_mlflow",
                log_dir=temp_dir,
                experiment_name="test_experiment",
                run_name="test_run",
            )

            # Test basic logging functions
            logger.log_scalar("test_metric", 0.5, step=10)

            # Verify method was called
            mlflow_mock.log_metric.assert_called_with("test_metric", 0.5, step=10)

            logger.log_scalars({"metric1": 0.5, "metric2": 0.7}, step=10)

            # Verify method was called
            mlflow_mock.log_metrics.assert_called_with({"metric1": 0.5, "metric2": 0.7}, step=10)

            logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32})

            # Verify method was called
            mlflow_mock.log_params.assert_called_with({"learning_rate": 0.001, "batch_size": 32})

            # Clean up
            logger.close()

            # Verify end_run was called
            mlflow_mock.end_run.assert_called_once()


def test_wandb_logger_basic():
    """Test basic Wandb logger functionality with mocked dependencies."""
    # Create a mock for wandb module
    wandb_mock = mock.MagicMock()

    # Mock the import of wandb inside the WandbLogger class
    with mock.patch.dict("sys.modules", {"wandb": wandb_mock}):
        # Now we can safely import the logger
        from artifex.generative_models.utils.logging import WandbLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the wandb run
            mock_run = mock.MagicMock()
            mock_run.name = "test_run"
            mock_run.id = "test_run_id"
            wandb_mock.init.return_value = mock_run

            # Create the logger
            logger = WandbLogger(
                name="test_wandb",
                project="test_project",
                log_dir=temp_dir,
            )

            # Test basic logging functions
            logger.log_scalar("test_metric", 0.5, step=10)

            # For Weights & Biases, step is passed as part of the dictionary
            wandb_mock.log.assert_called_with({"test_metric": 0.5}, step=10)

            logger.log_scalars({"metric1": 0.5, "metric2": 0.7}, step=10)

            # Verify method was called with the correct merged dictionary
            wandb_mock.log.assert_called_with({"metric1": 0.5, "metric2": 0.7}, step=10)

            logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32})

            # Clean up
            logger.close()

            # Verify finish was called
            wandb_mock.finish.assert_called_once()


# Helper test for verifying that the loggers handle ImportError gracefully
def test_mlflow_logger_import_error():
    """Test MLFlowLogger handles ImportError gracefully."""
    # Create a context where mlflow import fails
    with mock.patch.dict("sys.modules", {"mlflow": None}):
        from artifex.generative_models.utils.logging import MLFlowLogger

        with pytest.raises(ImportError):
            MLFlowLogger(name="test_mlflow_error")
