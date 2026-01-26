"""Tests for ModelCheckpoint callback.

Following TDD principles - these tests define the expected behavior
for the ModelCheckpoint callback.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest
from flax import nnx


class SimpleModel(nnx.Module):
    """Simple NNX model for testing."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class SimpleTrainer:
    """Simple trainer-like object that satisfies TrainerLike protocol."""

    def __init__(self, model: nnx.Module):
        self._model = model

    @property
    def model(self) -> nnx.Module:
        return self._model


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_config_exists(self):
        """CheckpointConfig should be importable."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        assert CheckpointConfig is not None

    def test_config_default_values(self):
        """CheckpointConfig should have sensible defaults."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        config = CheckpointConfig()
        assert config.monitor == "val_loss"
        assert config.mode == "min"
        assert config.save_top_k == 3
        assert config.save_last is True
        assert config.every_n_epochs == 1
        assert config.save_weights_only is False
        assert config.dirpath == "checkpoints"
        assert "{epoch" in config.filename  # Contains epoch placeholder

    def test_config_custom_values(self):
        """CheckpointConfig should accept custom values."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        config = CheckpointConfig(
            dirpath="/custom/path",
            filename="best-{epoch:03d}",
            monitor="accuracy",
            mode="max",
            save_top_k=5,
            save_last=False,
            every_n_epochs=2,
            save_weights_only=True,
        )
        assert config.dirpath == "/custom/path"
        assert config.filename == "best-{epoch:03d}"
        assert config.monitor == "accuracy"
        assert config.mode == "max"
        assert config.save_top_k == 5
        assert config.save_last is False
        assert config.every_n_epochs == 2
        assert config.save_weights_only is True


class TestModelCheckpointBasic:
    """Test basic ModelCheckpoint functionality."""

    def test_model_checkpoint_exists(self):
        """ModelCheckpoint should be importable."""
        from artifex.generative_models.training.callbacks import ModelCheckpoint

        assert ModelCheckpoint is not None

    def test_model_checkpoint_inherits_base_callback(self):
        """ModelCheckpoint should inherit from BaseCallback."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir))
            assert isinstance(callback, BaseCallback)

    def test_model_checkpoint_initializes_state(self):
        """ModelCheckpoint should initialize tracking state."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir))
            assert callback.best_score is None
            assert callback.best_checkpoint_path is None


class TestModelCheckpointSaving:
    """Test ModelCheckpoint saving behavior."""

    def test_creates_checkpoint_directory(self):
        """ModelCheckpoint should create checkpoint directory if it doesn't exist."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "new_checkpoints"
            ModelCheckpoint(CheckpointConfig(dirpath=str(checkpoint_dir)))

            # Directory should be created on initialization
            assert checkpoint_dir.exists()

    def test_saves_checkpoint_on_improvement_min_mode(self):
        """ModelCheckpoint should save when metric improves in min mode."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))

            # Create mock trainer with model state
            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # First epoch - establishes baseline, should save
            callback.on_epoch_end(trainer, 0, {"loss": 1.0})
            assert callback.best_score == 1.0

            # Second epoch - improvement, should save
            callback.on_epoch_end(trainer, 1, {"loss": 0.8})
            assert callback.best_score == 0.8

    def test_saves_checkpoint_on_improvement_max_mode(self):
        """ModelCheckpoint should save when metric improves in max mode."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(dirpath=tmpdir, monitor="accuracy", mode="max")
            )

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # First epoch
            callback.on_epoch_end(trainer, 0, {"accuracy": 0.8})
            assert callback.best_score == 0.8

            # Improvement
            callback.on_epoch_end(trainer, 1, {"accuracy": 0.9})
            assert callback.best_score == 0.9

    def test_does_not_update_best_on_worse_metric(self):
        """ModelCheckpoint should not update best_score on worse metric."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            callback.on_epoch_end(trainer, 0, {"loss": 1.0})
            callback.on_epoch_end(trainer, 1, {"loss": 1.5})

            # Best score should still be from epoch 0
            assert callback.best_score == 1.0


class TestModelCheckpointTopK:
    """Test save_top_k functionality."""

    def test_saves_top_k_checkpoints(self):
        """ModelCheckpoint should keep only top-k best checkpoints."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    monitor="loss",
                    mode="min",
                    save_top_k=2,
                )
            )

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # Simulate 5 epochs with varying losses
            losses = [1.0, 0.8, 0.9, 0.6, 0.7]
            for epoch, loss in enumerate(losses):
                callback.on_epoch_end(trainer, epoch, {"loss": loss})

            # Should have at most save_top_k checkpoints tracked
            assert len(callback.saved_checkpoints) <= 2

    def test_save_top_k_minus_one_saves_all(self):
        """save_top_k=-1 should save all checkpoints."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    monitor="loss",
                    mode="min",
                    save_top_k=-1,
                )
            )

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # Save 5 checkpoints
            for epoch in range(5):
                callback.on_epoch_end(trainer, epoch, {"loss": 1.0 - epoch * 0.1})

            # All should be tracked
            assert len(callback.saved_checkpoints) == 5

    def test_save_top_k_zero_saves_none(self):
        """save_top_k=0 should not save any checkpoints (but still track best)."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    monitor="loss",
                    mode="min",
                    save_top_k=0,
                )
            )

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            callback.on_epoch_end(trainer, 0, {"loss": 1.0})

            # No checkpoints should be saved
            assert len(callback.saved_checkpoints) == 0


class TestModelCheckpointEveryNEpochs:
    """Test every_n_epochs functionality."""

    def test_saves_every_n_epochs(self):
        """ModelCheckpoint should only check for saving every n epochs."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    monitor="loss",
                    mode="min",
                    every_n_epochs=2,
                    save_top_k=-1,  # Save all to count
                )
            )

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # Simulate 5 epochs
            for epoch in range(5):
                callback.on_epoch_end(trainer, epoch, {"loss": 1.0 - epoch * 0.1})

            # Should only save on epochs 0, 2, 4 (every 2 epochs starting from 0)
            assert len(callback.saved_checkpoints) == 3


class TestModelCheckpointMissingMetric:
    """Test behavior when monitored metric is missing."""

    def test_missing_metric_does_not_crash(self):
        """Missing metric should not crash, just skip saving."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="val_loss"))

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # Metric not present
            callback.on_epoch_end(trainer, 0, {"loss": 1.0})
            assert callback.best_score is None


class TestModelCheckpointJaxArrays:
    """Test ModelCheckpoint with JAX arrays."""

    def test_works_with_jax_arrays(self):
        """Should work with JAX array metric values."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            callback.on_epoch_end(trainer, 0, {"loss": jnp.array(1.0)})
            callback.on_epoch_end(trainer, 1, {"loss": jnp.array(0.9)})

            assert callback.best_score == pytest.approx(0.9)


class TestModelCheckpointFilename:
    """Test checkpoint filename formatting."""

    def test_filename_formatting_with_epoch(self):
        """Filename should be formatted with epoch number."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    filename="model-{epoch:03d}",
                    monitor="loss",
                )
            )

            # Test filename formatting
            filename = callback._format_filename(epoch=5, logs={"loss": 0.123})
            assert "005" in filename

    def test_filename_formatting_with_metric(self):
        """Filename should be formatted with metric value."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    filename="model-{epoch:02d}-{loss:.4f}",
                    monitor="loss",
                )
            )

            filename = callback._format_filename(epoch=3, logs={"loss": 0.1234})
            assert "03" in filename
            assert "0.1234" in filename


class TestModelCheckpointOverhead:
    """Test that ModelCheckpoint has minimal overhead when not saving."""

    def test_overhead_is_minimal_when_not_saving(self):
        """ModelCheckpoint overhead should be minimal when not saving."""
        import time

        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                CheckpointConfig(
                    dirpath=tmpdir,
                    monitor="loss",
                    mode="min",
                    save_top_k=1,  # Only save best
                )
            )

            model = SimpleModel(rngs=nnx.Rngs(0))
            trainer = SimpleTrainer(model)

            # First call establishes baseline (may save)
            callback.on_epoch_end(trainer, 0, {"loss": 0.5})

            # Warmup
            for i in range(100):
                # Worse loss, should not trigger save
                callback.on_epoch_end(trainer, i + 1, {"loss": 1.0})

            # Benchmark (no saves should happen)
            iterations = 10_000
            start = time.perf_counter()
            for i in range(iterations):
                callback.on_epoch_end(trainer, i + 101, {"loss": 1.0})
            elapsed = time.perf_counter() - start

            # Should be < 10 microseconds per call when not saving
            avg_time_us = (elapsed / iterations) * 1_000_000
            assert avg_time_us < 10.0, f"ModelCheckpoint overhead too high: {avg_time_us:.3f}us"
