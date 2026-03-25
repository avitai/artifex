"""Tests for ModelCheckpoint callback.

Following TDD principles - these tests define the expected behavior
for the ModelCheckpoint callback.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest
from flax import nnx

from tests.artifex.generative_models.training.timing_utils import best_average_us_per_call


class _FakeCheckpointManager:
    """Minimal fake Orbax manager for callback unit tests."""

    def __init__(self, *, max_to_keep, best_fn, best_mode):
        self._max_to_keep = max_to_keep
        self._best_fn = best_fn
        self._best_mode = best_mode
        self._metrics_by_step: dict[int, dict[str, float]] = {}

    def save(self, step: int, metrics: dict[str, float]) -> None:
        self._metrics_by_step[step] = metrics
        if self._max_to_keep is not None and len(self._metrics_by_step) > self._max_to_keep:
            sorted_steps = sorted(
                self._metrics_by_step,
                key=lambda s: self._best_fn(self._metrics_by_step[s]),
                reverse=self._best_mode == "max",
            )
            kept_steps = set(sorted_steps[: self._max_to_keep])
            self._metrics_by_step = {
                step: metrics
                for step, metrics in self._metrics_by_step.items()
                if step in kept_steps
            }

    def all_steps(self) -> list[int]:
        return sorted(self._metrics_by_step)

    def best_step(self) -> int | None:
        if not self._metrics_by_step:
            return None
        return sorted(
            self._metrics_by_step,
            key=lambda s: self._best_fn(self._metrics_by_step[s]),
            reverse=self._best_mode == "max",
        )[0]


@pytest.fixture(autouse=True)
def _patch_checkpoint_backend(monkeypatch):
    """Patch checkpoint backend so callback tests stay unit-scoped."""

    def setup_checkpoint_manager(_dirpath, *, max_to_keep=5, best_fn=None, best_mode="max"):
        manager = _FakeCheckpointManager(
            max_to_keep=max_to_keep,
            best_fn=best_fn or (lambda _metrics: 0.0),
            best_mode=best_mode,
        )
        return manager, str(_dirpath)

    def save_checkpoint(manager, _model, step, *, metrics=None):
        manager.save(step, dict(metrics or {}))
        return manager

    monkeypatch.setattr(
        "artifex.generative_models.training.callbacks.checkpoint.setup_checkpoint_manager",
        setup_checkpoint_manager,
    )
    monkeypatch.setattr(
        "artifex.generative_models.training.callbacks.checkpoint.save_checkpoint",
        save_checkpoint,
    )


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
        assert config.every_n_epochs == 1
        assert config.dirpath == "checkpoints"

    def test_config_custom_values(self):
        """CheckpointConfig should accept custom values."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        config = CheckpointConfig(
            dirpath="/custom/path",
            monitor="accuracy",
            mode="max",
            save_top_k=5,
            every_n_epochs=2,
        )
        assert config.dirpath == "/custom/path"
        assert config.monitor == "accuracy"
        assert config.mode == "max"
        assert config.save_top_k == 5
        assert config.every_n_epochs == 2


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
            assert callback.best_checkpoint_step is None
            assert callback.saved_checkpoint_steps == []


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
            assert callback.best_checkpoint_step == 0

            # Second epoch - improvement, should save
            callback.on_epoch_end(trainer, 1, {"loss": 0.8})
            assert callback.best_score == 0.8
            assert callback.best_checkpoint_step == 1

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
            assert callback.best_checkpoint_step == 0

            # Improvement
            callback.on_epoch_end(trainer, 1, {"accuracy": 0.9})
            assert callback.best_score == 0.9
            assert callback.best_checkpoint_step == 1

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
            assert len(callback.saved_checkpoint_steps) <= 2
            assert callback.best_checkpoint_step == 3

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
            assert callback.saved_checkpoint_steps == [0, 1, 2, 3, 4]
            assert callback.best_checkpoint_step == 4

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
            assert callback.saved_checkpoint_steps == []


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
            assert callback.saved_checkpoint_steps == [0, 2, 4]


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


class TestModelCheckpointBestStepTracking:
    """Test best-step tracking on top of Orbax-managed checkpoints."""

    def test_tracks_best_step_for_min_mode(self):
        """The best checkpoint should be tracked by step for min-mode metrics."""
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

            callback.on_epoch_end(trainer, 0, {"loss": 1.0})
            callback.on_epoch_end(trainer, 1, {"loss": 0.8})
            callback.on_epoch_end(trainer, 2, {"loss": 0.9})

            assert callback.best_checkpoint_step == 1


class TestModelCheckpointOverhead:
    """Test that ModelCheckpoint has minimal overhead when not saving."""

    def test_overhead_is_minimal_when_not_saving(self):
        """ModelCheckpoint overhead should be minimal when not saving."""
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

            logs = {"loss": 1.0}

            def baseline_dispatch() -> None:
                current = float(logs[callback.config.monitor])
                callback._is_improvement(current)

            iterations = 20_000
            baseline_us = best_average_us_per_call(baseline_dispatch, iterations=iterations)
            avg_time_us = best_average_us_per_call(
                lambda: callback.on_epoch_end(trainer, 101, logs),
                iterations=iterations,
            )

            overhead_us = avg_time_us - baseline_us
            assert overhead_us < 3.0, f"ModelCheckpoint overhead too high: {overhead_us:.3f}us"
