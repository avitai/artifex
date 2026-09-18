"""Tests for ModelCheckpoint callback.

Following TDD principles - these tests define the expected behavior
for the ModelCheckpoint callback.
"""

import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest
from flax import nnx

from tests.artifex.generative_models.training.timing_utils import best_average_us_per_call


STEPS_PER_EPOCH = 10


class _FakeCheckpointStore:
    """Minimal fake of substrax's format-3 store for callback unit tests.

    It keeps what ``save`` receives, retains by recency as Orbax's ``max_to_keep``
    does, and ``best_step`` reads the metric the callback records in ``metrics``.
    """

    def __init__(self, checkpoint_dir, *, max_to_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.saved: dict[int, dict[str, Any]] = {}
        self.closed = False

    def save(
        self,
        step,
        items,
        *,
        epoch=None,
        metrics=None,
        producer=None,
        extra=None,
        overwrite=False,
    ):
        self.saved[step] = {
            "items": dict(items),
            "epoch": epoch,
            "metrics": dict(metrics or {}),
            "producer": producer,
            "extra": dict(extra or {}),
        }
        if self.max_to_keep is not None:
            for old in sorted(self.saved)[: -self.max_to_keep or None]:
                del self.saved[old]
        return Path(str(step))

    def list_steps(self) -> list[int]:
        return sorted(self.saved)

    def best_step(self, metric: str, *, mode: str = "min") -> int | None:
        scored = {
            step: record["metrics"][metric]
            for step, record in self.saved.items()
            if metric in record["metrics"]
        }
        if not scored:
            return None
        chooser = min if mode == "min" else max
        return chooser(scored, key=lambda step: scored[step])

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _patch_checkpoint_backend(monkeypatch):
    """Patch the store so callback tests stay unit-scoped."""
    monkeypatch.setattr(
        "artifex.generative_models.training.callbacks.checkpoint.OrbaxCheckpointStore",
        _FakeCheckpointStore,
    )


class SimpleModel(nnx.Module):
    """Simple NNX model for testing."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class SimpleTrainer:
    """A trainer the callback can read: the model and the global step."""

    def __init__(self, model: nnx.Module):
        self._model = model
        self.step = 0

    @property
    def model(self) -> nnx.Module:
        return self._model


def end_epoch(callback, trainer: SimpleTrainer, epoch: int, logs: dict[str, Any]) -> None:
    """Advance the trainer to the epoch's last global step, then end the epoch."""
    trainer.step = (epoch + 1) * STEPS_PER_EPOCH
    callback.on_epoch_end(trainer, epoch, logs)


def step_of(epoch: int) -> int:
    """The global step ``end_epoch`` gives an epoch."""
    return (epoch + 1) * STEPS_PER_EPOCH


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_config_exists(self):
        """CheckpointConfig should be importable."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        assert CheckpointConfig is not None

    def test_config_default_values(self, tmp_path):
        """The directory is the caller's to name; everything else has a default."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        config = CheckpointConfig(dirpath=tmp_path)
        assert config.monitor == "val_loss"
        assert config.mode == "min"
        assert config.save_top_k == 3
        assert config.every_n_epochs == 1
        assert config.dirpath == tmp_path

    def test_the_directory_is_required(self):
        """No directory means no callback, never a default under the working directory."""
        from artifex.generative_models.training.callbacks import CheckpointConfig

        with pytest.raises(TypeError, match="dirpath"):
            CheckpointConfig()  # type: ignore[call-arg]

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

    def test_nothing_is_created_before_the_first_save(self):
        """The store creates the directory on its first save; the callback creates nothing."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "new_checkpoints"
            ModelCheckpoint(CheckpointConfig(dirpath=str(checkpoint_dir)))

            assert not checkpoint_dir.exists()

    def test_a_trainer_without_a_step_is_refused_when_a_save_is_due(self):
        """The global step is the checkpoint's address; a trainer without one cannot be saved."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        class StepLess:
            def __init__(self, model: nnx.Module) -> None:
                self.model = model

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))
            trainer = StepLess(SimpleModel(rngs=nnx.Rngs(0)))

            # No metric: nothing is due, so nothing asks for the step.
            callback.on_epoch_end(trainer, 0, {})
            with pytest.raises(TypeError, match="step"):
                callback.on_epoch_end(trainer, 0, {"loss": 1.0})

    def test_saves_the_model_item_at_the_global_step_with_the_metric(self):
        """A save is the model item at the trainer's step, the metric in the record's metrics."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            end_epoch(callback, trainer, 2, {"loss": 0.5})

            store = callback._store
            assert store is not None
            record = store.saved[step_of(2)]
            assert set(record["items"]) == {"model"}
            assert record["metrics"] == {"loss": 0.5}
            assert record["epoch"] == 2
            assert callback.saved_checkpoint_steps == [step_of(2)]

    def test_saves_checkpoint_on_improvement_min_mode(self):
        """ModelCheckpoint should save when metric improves in min mode."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            # First epoch - establishes baseline, should save
            end_epoch(callback, trainer, 0, {"loss": 1.0})
            assert callback.best_score == 1.0
            assert callback.best_checkpoint_step == step_of(0)

            # Second epoch - improvement, should save
            end_epoch(callback, trainer, 1, {"loss": 0.8})
            assert callback.best_score == 0.8
            assert callback.best_checkpoint_step == step_of(1)

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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            end_epoch(callback, trainer, 0, {"accuracy": 0.8})
            assert callback.best_score == 0.8
            assert callback.best_checkpoint_step == step_of(0)

            end_epoch(callback, trainer, 1, {"accuracy": 0.9})
            assert callback.best_score == 0.9
            assert callback.best_checkpoint_step == step_of(1)

    def test_does_not_update_best_on_worse_metric(self):
        """ModelCheckpoint should not update best_score on worse metric."""
        from artifex.generative_models.training.callbacks import (
            CheckpointConfig,
            ModelCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(CheckpointConfig(dirpath=tmpdir, monitor="loss", mode="min"))
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            end_epoch(callback, trainer, 0, {"loss": 1.0})
            end_epoch(callback, trainer, 1, {"loss": 1.5})

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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            # Simulate 5 epochs with varying losses
            losses = [1.0, 0.8, 0.9, 0.6, 0.7]
            for epoch, loss in enumerate(losses):
                end_epoch(callback, trainer, epoch, {"loss": loss})

            # Should have at most save_top_k checkpoints tracked
            assert len(callback.saved_checkpoint_steps) <= 2
            assert callback.best_checkpoint_step == step_of(3)

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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            # Save 5 checkpoints
            for epoch in range(5):
                end_epoch(callback, trainer, epoch, {"loss": 1.0 - epoch * 0.1})

            # All should be tracked, at their global steps
            assert callback.saved_checkpoint_steps == [step_of(epoch) for epoch in range(5)]
            assert callback.best_checkpoint_step == step_of(4)

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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            end_epoch(callback, trainer, 0, {"loss": 1.0})

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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            # Simulate 5 epochs
            for epoch in range(5):
                end_epoch(callback, trainer, epoch, {"loss": 1.0 - epoch * 0.1})

            # Should only save on epochs 0, 2, 4 (every 2 epochs starting from 0)
            assert callback.saved_checkpoint_steps == [step_of(0), step_of(2), step_of(4)]


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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            # Metric not present
            end_epoch(callback, trainer, 0, {"loss": 1.0})
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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            end_epoch(callback, trainer, 0, {"loss": jnp.array(1.0)})
            end_epoch(callback, trainer, 1, {"loss": jnp.array(0.9)})

            assert callback.best_score == pytest.approx(0.9)


class TestModelCheckpointBestStepTracking:
    """Test best-step tracking on top of Orbax-managed checkpoints."""

    def test_tracks_best_step_for_min_mode(self):
        """The best checkpoint should be tracked by global step for min-mode metrics."""
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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            end_epoch(callback, trainer, 0, {"loss": 1.0})
            end_epoch(callback, trainer, 1, {"loss": 0.8})
            end_epoch(callback, trainer, 2, {"loss": 0.9})

            assert callback.best_checkpoint_step == step_of(1)


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
            trainer = SimpleTrainer(SimpleModel(rngs=nnx.Rngs(0)))

            # First call establishes baseline (may save)
            end_epoch(callback, trainer, 0, {"loss": 0.5})

            # Warmup
            for i in range(100):
                # Worse loss, should not trigger save
                end_epoch(callback, trainer, i + 1, {"loss": 1.0})

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
