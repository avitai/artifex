"""Tests for logging callbacks.

These callbacks integrate with the training loop to log metrics, images,
and other training artifacts using various logging backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleModel(nnx.Module):
    """Simple model for testing."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


@dataclass
class SimpleTrainer:
    """Simple trainer fixture for testing callbacks."""

    model: nnx.Module
    steps_per_epoch: int = 100


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel(rngs=nnx.Rngs(0))


@pytest.fixture
def simple_trainer(simple_model: SimpleModel) -> SimpleTrainer:
    """Create a simple trainer for testing."""
    return SimpleTrainer(model=simple_model)


# =============================================================================
# LoggerCallback Base Tests
# =============================================================================


class TestLoggerCallback:
    """Tests for LoggerCallback base class."""

    def test_logger_callback_init_with_logger(self) -> None:
        """Test LoggerCallback initialization with a logger instance."""
        from artifex.generative_models.training.callbacks.logging import LoggerCallback
        from artifex.generative_models.utils.logging.logger import ConsoleLogger

        logger = ConsoleLogger(name="test")
        callback = LoggerCallback(logger=logger)

        assert callback.logger is logger

    def test_logger_callback_init_with_config(self) -> None:
        """Test LoggerCallback initialization with config."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        config = LoggerCallbackConfig(log_every_n_steps=50, log_on_epoch_end=True)
        callback = LoggerCallback(logger=MagicMock(), config=config)

        assert callback.config.log_every_n_steps == 50
        assert callback.config.log_on_epoch_end is True

    def test_logger_callback_default_config(self) -> None:
        """Test LoggerCallback uses default config when not provided."""
        from artifex.generative_models.training.callbacks.logging import LoggerCallback

        callback = LoggerCallback(logger=MagicMock())

        assert callback.config is not None
        assert callback.config.log_every_n_steps == 1  # Default value

    def test_on_batch_end_logs_metrics(self, simple_trainer: SimpleTrainer) -> None:
        """Test on_batch_end logs metrics when step matches log interval."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        mock_logger = MagicMock()
        config = LoggerCallbackConfig(log_every_n_steps=10)
        callback = LoggerCallback(logger=mock_logger, config=config)

        logs = {"loss": 0.5, "accuracy": 0.9}
        callback.on_batch_end(simple_trainer, batch=10, logs=logs)

        mock_logger.log_scalars.assert_called_once()

    def test_on_batch_end_skips_when_not_log_step(self, simple_trainer: SimpleTrainer) -> None:
        """Test on_batch_end skips logging when step doesn't match interval."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        mock_logger = MagicMock()
        config = LoggerCallbackConfig(log_every_n_steps=10)
        callback = LoggerCallback(logger=mock_logger, config=config)

        logs = {"loss": 0.5}
        callback.on_batch_end(simple_trainer, batch=5, logs=logs)

        mock_logger.log_scalars.assert_not_called()

    def test_on_epoch_end_logs_when_enabled(self, simple_trainer: SimpleTrainer) -> None:
        """Test on_epoch_end logs metrics when enabled."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        mock_logger = MagicMock()
        config = LoggerCallbackConfig(log_on_epoch_end=True)
        callback = LoggerCallback(logger=mock_logger, config=config)

        logs = {"val_loss": 0.3, "val_accuracy": 0.95}
        callback.on_epoch_end(simple_trainer, epoch=1, logs=logs)

        mock_logger.log_scalars.assert_called_once()

    def test_on_epoch_end_skips_when_disabled(self, simple_trainer: SimpleTrainer) -> None:
        """Test on_epoch_end skips logging when disabled."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        mock_logger = MagicMock()
        config = LoggerCallbackConfig(log_on_epoch_end=False)
        callback = LoggerCallback(logger=mock_logger, config=config)

        logs = {"val_loss": 0.3}
        callback.on_epoch_end(simple_trainer, epoch=1, logs=logs)

        mock_logger.log_scalars.assert_not_called()

    def test_on_train_end_closes_logger(self, simple_trainer: SimpleTrainer) -> None:
        """Test on_train_end closes the logger."""
        from artifex.generative_models.training.callbacks.logging import LoggerCallback

        mock_logger = MagicMock()
        callback = LoggerCallback(logger=mock_logger)

        callback.on_train_end(simple_trainer)

        mock_logger.close.assert_called_once()


# =============================================================================
# WandbLoggerCallback Tests
# =============================================================================


class TestWandbLoggerCallback:
    """Tests for WandbLoggerCallback."""

    def test_wandb_callback_init_with_config(self) -> None:
        """Test WandbLoggerCallback initialization with config."""
        from artifex.generative_models.training.callbacks.logging import (
            WandbLoggerCallback,
            WandbLoggerConfig,
        )

        config = WandbLoggerConfig(
            project="test-project",
            entity="test-entity",
            name="test-run",
            tags=["test", "experiment"],
        )

        # Mock wandb to avoid actual API calls
        with patch("wandb.init") as mock_init:
            mock_init.return_value = MagicMock(name="test-run", id="abc123")
            callback = WandbLoggerCallback(config=config)

            assert callback.config.project == "test-project"
            assert callback.config.entity == "test-entity"

    def test_wandb_callback_logs_hyperparams_on_train_begin(
        self, simple_trainer: SimpleTrainer
    ) -> None:
        """Test WandbLoggerCallback logs hyperparams on train begin."""
        from artifex.generative_models.training.callbacks.logging import (
            WandbLoggerCallback,
            WandbLoggerConfig,
        )

        config = WandbLoggerConfig(
            project="test-project",
            config={"learning_rate": 0.001, "batch_size": 32},
        )

        with patch("wandb.init") as mock_init:
            mock_run = MagicMock(name="test-run", id="abc123")
            mock_run.config = {}
            mock_init.return_value = mock_run

            callback = WandbLoggerCallback(config=config)
            callback.on_train_begin(simple_trainer)

            # Hyperparams should be logged
            assert mock_run.config.get("learning_rate") == 0.001 or "learning_rate" in str(
                mock_run.config
            )

    def test_wandb_callback_finishes_on_train_end(self, simple_trainer: SimpleTrainer) -> None:
        """Test WandbLoggerCallback finishes run on train end."""
        from artifex.generative_models.training.callbacks.logging import (
            WandbLoggerCallback,
            WandbLoggerConfig,
        )

        config = WandbLoggerConfig(project="test-project")

        with patch("wandb.init") as mock_init, patch("wandb.finish") as mock_finish:
            mock_run = MagicMock(name="test-run", id="abc123")
            mock_init.return_value = mock_run

            callback = WandbLoggerCallback(config=config)
            callback.on_train_end(simple_trainer)

            mock_finish.assert_called_once()

    def test_wandb_callback_logs_metrics(self, simple_trainer: SimpleTrainer) -> None:
        """Test WandbLoggerCallback logs metrics on batch end."""
        from artifex.generative_models.training.callbacks.logging import (
            WandbLoggerCallback,
            WandbLoggerConfig,
        )

        config = WandbLoggerConfig(project="test-project", log_every_n_steps=1)

        with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log:
            mock_run = MagicMock(name="test-run", id="abc123")
            mock_init.return_value = mock_run

            callback = WandbLoggerCallback(config=config)
            logs = {"loss": 0.5, "accuracy": 0.9}
            callback.on_batch_end(simple_trainer, batch=1, logs=logs)

            mock_log.assert_called()

    def test_wandb_callback_disabled_mode(self) -> None:
        """Test WandbLoggerCallback in disabled mode."""
        from artifex.generative_models.training.callbacks.logging import (
            WandbLoggerCallback,
            WandbLoggerConfig,
        )

        config = WandbLoggerConfig(project="test-project", mode="disabled")

        with patch("wandb.init") as mock_init:
            mock_run = MagicMock(name="test-run", id="abc123")
            mock_init.return_value = mock_run

            _ = WandbLoggerCallback(config=config)

            # mode should be passed to wandb.init
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("mode") == "disabled"


# =============================================================================
# TensorBoardLoggerCallback Tests
# =============================================================================


# Check if TensorBoard is available
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


@pytest.mark.skipif(not HAS_TENSORBOARD, reason="TensorBoard not installed")
class TestTensorBoardLoggerCallback:
    """Tests for TensorBoardLoggerCallback."""

    def test_tensorboard_callback_init(self, tmp_path) -> None:
        """Test TensorBoardLoggerCallback initialization."""
        from artifex.generative_models.training.callbacks.logging import (
            TensorBoardLoggerCallback,
            TensorBoardLoggerConfig,
        )

        config = TensorBoardLoggerConfig(log_dir=str(tmp_path / "tensorboard"))
        callback = TensorBoardLoggerCallback(config=config)
        assert callback.config.log_dir == str(tmp_path / "tensorboard")

    def test_tensorboard_callback_logs_scalars(
        self, simple_trainer: SimpleTrainer, tmp_path
    ) -> None:
        """Test TensorBoardLoggerCallback logs scalars."""
        from artifex.generative_models.training.callbacks.logging import (
            TensorBoardLoggerCallback,
            TensorBoardLoggerConfig,
        )

        config = TensorBoardLoggerConfig(log_dir=str(tmp_path), log_every_n_steps=1)

        with patch("torch.utils.tensorboard.SummaryWriter") as MockWriter:
            mock_writer_instance = MagicMock()
            MockWriter.return_value = mock_writer_instance

            callback = TensorBoardLoggerCallback(config=config)
            callback.on_train_begin(simple_trainer)

            logs = {"loss": 0.5, "accuracy": 0.9}
            callback.on_batch_end(simple_trainer, batch=1, logs=logs)

            # Should call add_scalar for each metric
            assert mock_writer_instance.add_scalar.call_count >= 1

    def test_tensorboard_callback_closes_writer(
        self, simple_trainer: SimpleTrainer, tmp_path
    ) -> None:
        """Test TensorBoardLoggerCallback closes writer on train end."""
        from artifex.generative_models.training.callbacks.logging import (
            TensorBoardLoggerCallback,
            TensorBoardLoggerConfig,
        )

        config = TensorBoardLoggerConfig(log_dir=str(tmp_path))

        with patch("torch.utils.tensorboard.SummaryWriter") as MockWriter:
            mock_writer_instance = MagicMock()
            MockWriter.return_value = mock_writer_instance

            callback = TensorBoardLoggerCallback(config=config)
            callback.on_train_begin(simple_trainer)
            callback.on_train_end(simple_trainer)

            mock_writer_instance.close.assert_called_once()

    def test_tensorboard_callback_flush_interval(
        self, simple_trainer: SimpleTrainer, tmp_path
    ) -> None:
        """Test TensorBoardLoggerCallback flush interval configuration."""
        from artifex.generative_models.training.callbacks.logging import (
            TensorBoardLoggerCallback,
            TensorBoardLoggerConfig,
        )

        config = TensorBoardLoggerConfig(log_dir=str(tmp_path), flush_secs=60)

        with patch("torch.utils.tensorboard.SummaryWriter") as MockWriter:
            callback = TensorBoardLoggerCallback(config=config)
            callback.on_train_begin(simple_trainer)

            # flush_secs should be passed to SummaryWriter
            MockWriter.assert_called()


# =============================================================================
# ProgressBarCallback Tests
# =============================================================================


class TestProgressBarCallback:
    """Tests for ProgressBarCallback."""

    def test_progress_bar_callback_init(self) -> None:
        """Test ProgressBarCallback initialization."""
        from artifex.generative_models.training.callbacks.logging import (
            ProgressBarCallback,
            ProgressBarConfig,
        )

        config = ProgressBarConfig(refresh_rate=10, show_eta=True)
        callback = ProgressBarCallback(config=config)

        assert callback.config.refresh_rate == 10
        assert callback.config.show_eta is True

    def test_progress_bar_default_config(self) -> None:
        """Test ProgressBarCallback with default config."""
        from artifex.generative_models.training.callbacks.logging import ProgressBarCallback

        callback = ProgressBarCallback()

        assert callback.config is not None
        assert callback.config.refresh_rate == 10  # Default value

    def test_progress_bar_on_train_begin_creates_progress(
        self, simple_trainer: SimpleTrainer
    ) -> None:
        """Test ProgressBarCallback creates progress on train begin."""
        from artifex.generative_models.training.callbacks.logging import ProgressBarCallback

        with patch("rich.progress.Progress") as MockProgress:
            mock_progress = MagicMock()
            MockProgress.return_value = mock_progress

            callback = ProgressBarCallback()
            callback.on_train_begin(simple_trainer)

            mock_progress.start.assert_called_once()

    def test_progress_bar_on_epoch_begin_adds_task(self, simple_trainer: SimpleTrainer) -> None:
        """Test ProgressBarCallback adds task on epoch begin."""
        from artifex.generative_models.training.callbacks.logging import ProgressBarCallback

        with patch("rich.progress.Progress") as MockProgress:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            MockProgress.return_value = mock_progress

            callback = ProgressBarCallback()
            callback.on_train_begin(simple_trainer)
            callback.on_epoch_begin(simple_trainer, epoch=1)

            mock_progress.add_task.assert_called()

    def test_progress_bar_on_batch_end_advances(self, simple_trainer: SimpleTrainer) -> None:
        """Test ProgressBarCallback advances on batch end."""
        from artifex.generative_models.training.callbacks.logging import ProgressBarCallback

        with patch("rich.progress.Progress") as MockProgress:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            MockProgress.return_value = mock_progress

            callback = ProgressBarCallback()
            callback.on_train_begin(simple_trainer)
            callback.on_epoch_begin(simple_trainer, epoch=1)
            callback.on_batch_end(simple_trainer, batch=1, logs={"loss": 0.5})

            mock_progress.advance.assert_called()

    def test_progress_bar_on_train_end_stops(self, simple_trainer: SimpleTrainer) -> None:
        """Test ProgressBarCallback stops progress on train end."""
        from artifex.generative_models.training.callbacks.logging import ProgressBarCallback

        with patch("rich.progress.Progress") as MockProgress:
            mock_progress = MagicMock()
            MockProgress.return_value = mock_progress

            callback = ProgressBarCallback()
            callback.on_train_begin(simple_trainer)
            callback.on_train_end(simple_trainer)

            mock_progress.stop.assert_called_once()

    def test_progress_bar_displays_metrics(self, simple_trainer: SimpleTrainer) -> None:
        """Test ProgressBarCallback displays metrics in description."""
        from artifex.generative_models.training.callbacks.logging import (
            ProgressBarCallback,
            ProgressBarConfig,
        )

        config = ProgressBarConfig(show_metrics=True)

        with patch("rich.progress.Progress") as MockProgress:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            MockProgress.return_value = mock_progress

            callback = ProgressBarCallback(config=config)
            callback.on_train_begin(simple_trainer)
            callback.on_epoch_begin(simple_trainer, epoch=1)
            callback.on_batch_end(simple_trainer, batch=1, logs={"loss": 0.5, "acc": 0.9})

            # Should update task description with metrics
            mock_progress.update.assert_called()


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoggingCallbacksIntegration:
    """Integration tests for logging callbacks."""

    def test_multiple_loggers_can_be_combined(self) -> None:
        """Test multiple logging callbacks can be used together."""
        from artifex.generative_models.training.callbacks.base import CallbackList
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            ProgressBarCallback,
        )

        mock_logger = MagicMock()
        logger_callback = LoggerCallback(logger=mock_logger)

        with patch("rich.progress.Progress"):
            progress_callback = ProgressBarCallback()

            callback_list = CallbackList([logger_callback, progress_callback])

            assert len(callback_list) == 2

    def test_jax_array_conversion_in_logs(self, simple_trainer: SimpleTrainer) -> None:
        """Test that JAX arrays in logs are handled correctly."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        mock_logger = MagicMock()
        config = LoggerCallbackConfig(log_every_n_steps=1)
        callback = LoggerCallback(logger=mock_logger, config=config)

        # Use JAX arrays in logs
        logs = {
            "loss": jnp.array(0.5),
            "accuracy": jnp.array(0.9),
        }
        callback.on_batch_end(simple_trainer, batch=1, logs=logs)

        # Should still call log_scalars (values should be converted internally)
        mock_logger.log_scalars.assert_called_once()

    def test_empty_logs_handled_gracefully(self, simple_trainer: SimpleTrainer) -> None:
        """Test that empty logs are handled gracefully."""
        from artifex.generative_models.training.callbacks.logging import (
            LoggerCallback,
            LoggerCallbackConfig,
        )

        mock_logger = MagicMock()
        config = LoggerCallbackConfig(log_every_n_steps=1)
        callback = LoggerCallback(logger=mock_logger, config=config)

        # Empty logs
        callback.on_batch_end(simple_trainer, batch=1, logs={})

        # Should not raise, may or may not call log_scalars
        # (implementation decision: skip if empty or log empty dict)


# =============================================================================
# Config Tests
# =============================================================================


class TestLoggingConfigs:
    """Tests for logging configuration dataclasses."""

    def test_logger_callback_config_defaults(self) -> None:
        """Test LoggerCallbackConfig default values."""
        from artifex.generative_models.training.callbacks.logging import LoggerCallbackConfig

        config = LoggerCallbackConfig()

        assert config.log_every_n_steps == 1
        assert config.log_on_epoch_end is True

    def test_wandb_logger_config_defaults(self) -> None:
        """Test WandbLoggerConfig default values."""
        from artifex.generative_models.training.callbacks.logging import WandbLoggerConfig

        config = WandbLoggerConfig(project="test")

        assert config.project == "test"
        assert config.entity is None
        assert config.mode == "online"
        assert config.log_every_n_steps == 1

    def test_tensorboard_logger_config_defaults(self) -> None:
        """Test TensorBoardLoggerConfig default values."""
        from artifex.generative_models.training.callbacks.logging import TensorBoardLoggerConfig

        config = TensorBoardLoggerConfig()

        assert config.log_dir == "logs/tensorboard"
        assert config.flush_secs == 120

    def test_progress_bar_config_defaults(self) -> None:
        """Test ProgressBarConfig default values."""
        from artifex.generative_models.training.callbacks.logging import ProgressBarConfig

        config = ProgressBarConfig()

        assert config.refresh_rate == 10
        assert config.show_eta is True
        assert config.show_metrics is True
