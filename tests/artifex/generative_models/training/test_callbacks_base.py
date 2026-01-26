"""Tests for base callback protocol and CallbackList.

Following TDD principles - these tests define the expected behavior
for the callback system.
"""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    TrainingConfig,
)


class TestTrainingCallbackProtocol:
    """Test that TrainingCallback protocol defines expected interface."""

    def test_callback_protocol_exists(self):
        """TrainingCallback protocol should be importable."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        assert TrainingCallback is not None

    def test_callback_has_on_train_begin(self):
        """Protocol should define on_train_begin method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        # Create a mock that satisfies the protocol
        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_train_begin")

    def test_callback_has_on_train_end(self):
        """Protocol should define on_train_end method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_train_end")

    def test_callback_has_on_epoch_begin(self):
        """Protocol should define on_epoch_begin method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_epoch_begin")

    def test_callback_has_on_epoch_end(self):
        """Protocol should define on_epoch_end method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_epoch_end")

    def test_callback_has_on_batch_begin(self):
        """Protocol should define on_batch_begin method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_batch_begin")

    def test_callback_has_on_batch_end(self):
        """Protocol should define on_batch_end method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_batch_end")

    def test_callback_has_on_validation_begin(self):
        """Protocol should define on_validation_begin method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_validation_begin")

    def test_callback_has_on_validation_end(self):
        """Protocol should define on_validation_end method."""
        from artifex.generative_models.training.callbacks import TrainingCallback

        callback = MagicMock(spec=TrainingCallback)
        assert hasattr(callback, "on_validation_end")


class TestBaseCallback:
    """Test BaseCallback provides default no-op implementations."""

    def test_base_callback_exists(self):
        """BaseCallback class should be importable."""
        from artifex.generative_models.training.callbacks import BaseCallback

        assert BaseCallback is not None

    def test_base_callback_implements_protocol(self):
        """BaseCallback should implement TrainingCallback protocol."""
        from artifex.generative_models.training.callbacks import BaseCallback

        callback = BaseCallback()
        # Should have all protocol methods
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_train_end")
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_epoch_end")
        assert hasattr(callback, "on_batch_begin")
        assert hasattr(callback, "on_batch_end")
        assert hasattr(callback, "on_validation_begin")
        assert hasattr(callback, "on_validation_end")

    def test_base_callback_methods_are_noop(self):
        """BaseCallback methods should be no-ops that don't raise."""
        from artifex.generative_models.training.callbacks import BaseCallback

        callback = BaseCallback()
        trainer_mock = MagicMock()

        # All methods should execute without error (positional args for speed)
        callback.on_train_begin(trainer_mock)
        callback.on_train_end(trainer_mock)
        callback.on_epoch_begin(trainer_mock, 0)
        callback.on_epoch_end(trainer_mock, 0, {})
        callback.on_batch_begin(trainer_mock, 0)
        callback.on_batch_end(trainer_mock, 0, {})
        callback.on_validation_begin(trainer_mock)
        callback.on_validation_end(trainer_mock, {})


class TestCallbackList:
    """Test CallbackList container for managing multiple callbacks."""

    def test_callback_list_exists(self):
        """CallbackList should be importable."""
        from artifex.generative_models.training.callbacks import CallbackList

        assert CallbackList is not None

    def test_callback_list_empty_initialization(self):
        """CallbackList should initialize with empty list."""
        from artifex.generative_models.training.callbacks import CallbackList

        callback_list = CallbackList()
        assert len(callback_list.callbacks) == 0

    def test_callback_list_with_callbacks(self):
        """CallbackList should accept callbacks on initialization."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList(callbacks=[cb1, cb2])
        assert len(callback_list.callbacks) == 2

    def test_callback_list_add(self):
        """CallbackList should allow adding callbacks."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        callback_list = CallbackList()
        cb = BaseCallback()
        callback_list.add(cb)
        assert cb in callback_list.callbacks

    def test_callback_list_remove(self):
        """CallbackList should allow removing callbacks."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        cb = BaseCallback()
        callback_list = CallbackList(callbacks=[cb])
        callback_list.remove(cb)
        assert cb not in callback_list.callbacks

    def test_callback_list_dispatches_on_train_begin(self):
        """CallbackList should dispatch on_train_begin to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        callback_list.on_train_begin(trainer_mock)

        cb1.on_train_begin.assert_called_once_with(trainer_mock)
        cb2.on_train_begin.assert_called_once_with(trainer_mock)

    def test_callback_list_dispatches_on_train_end(self):
        """CallbackList should dispatch on_train_end to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        callback_list.on_train_end(trainer_mock)

        cb1.on_train_end.assert_called_once_with(trainer_mock)
        cb2.on_train_end.assert_called_once_with(trainer_mock)

    def test_callback_list_dispatches_on_epoch_begin(self):
        """CallbackList should dispatch on_epoch_begin to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        callback_list.on_epoch_begin(trainer_mock, 5)

        cb1.on_epoch_begin.assert_called_once_with(trainer_mock, 5)
        cb2.on_epoch_begin.assert_called_once_with(trainer_mock, 5)

    def test_callback_list_dispatches_on_epoch_end(self):
        """CallbackList should dispatch on_epoch_end to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        logs = {"loss": 0.5}
        callback_list.on_epoch_end(trainer_mock, 5, logs)

        cb1.on_epoch_end.assert_called_once_with(trainer_mock, 5, logs)
        cb2.on_epoch_end.assert_called_once_with(trainer_mock, 5, logs)

    def test_callback_list_dispatches_on_batch_begin(self):
        """CallbackList should dispatch on_batch_begin to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        callback_list.on_batch_begin(trainer_mock, 10)

        cb1.on_batch_begin.assert_called_once_with(trainer_mock, 10)
        cb2.on_batch_begin.assert_called_once_with(trainer_mock, 10)

    def test_callback_list_dispatches_on_batch_end(self):
        """CallbackList should dispatch on_batch_end to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        logs = {"loss": 0.3}
        callback_list.on_batch_end(trainer_mock, 10, logs)

        cb1.on_batch_end.assert_called_once_with(trainer_mock, 10, logs)
        cb2.on_batch_end.assert_called_once_with(trainer_mock, 10, logs)

    def test_callback_list_dispatches_on_validation_begin(self):
        """CallbackList should dispatch on_validation_begin to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        callback_list.on_validation_begin(trainer_mock)

        cb1.on_validation_begin.assert_called_once_with(trainer_mock)
        cb2.on_validation_begin.assert_called_once_with(trainer_mock)

    def test_callback_list_dispatches_on_validation_end(self):
        """CallbackList should dispatch on_validation_end to all callbacks."""
        from artifex.generative_models.training.callbacks import CallbackList

        cb1 = MagicMock()
        cb2 = MagicMock()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        trainer_mock = MagicMock()
        logs = {"val_loss": 0.4}
        callback_list.on_validation_end(trainer_mock, logs)

        cb1.on_validation_end.assert_called_once_with(trainer_mock, logs)
        cb2.on_validation_end.assert_called_once_with(trainer_mock, logs)

    def test_callback_list_len(self):
        """CallbackList should support len()."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        callback_list = CallbackList(callbacks=[BaseCallback(), BaseCallback()])
        assert len(callback_list) == 2

    def test_callback_list_iter(self):
        """CallbackList should be iterable."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList(callbacks=[cb1, cb2])

        callbacks = list(callback_list)
        assert callbacks == [cb1, cb2]


class TestCallbackIntegrationWithTrainer:
    """Test that callbacks integrate properly with Trainer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple NNX model for testing."""

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.linear = nnx.Linear(4, 4, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                return self.linear(x)

        return SimpleModel(rngs=nnx.Rngs(42))

    @pytest.fixture
    def training_config(self):
        """Create a simple training configuration."""
        return TrainingConfig(
            name="test_training",
            optimizer=OptimizerConfig(
                name="test_optimizer",
                optimizer_type="adam",
                learning_rate=1e-3,
            ),
            batch_size=4,
            num_epochs=2,
        )

    def test_trainer_accepts_callbacks_parameter(self, simple_model, training_config):
        """Trainer should accept a callbacks parameter."""
        from artifex.generative_models.training import Trainer
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        def loss_fn(model, batch, _rng):
            x = batch["x"]
            y = model(x)
            loss = jnp.mean(y**2)
            return loss, {"loss": loss}

        callbacks = CallbackList(callbacks=[BaseCallback()])

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=loss_fn,
            callbacks=callbacks,
        )

        assert trainer.callbacks is not None
        assert len(trainer.callbacks) == 1

    def test_trainer_calls_on_batch_end_during_train_step(self, simple_model, training_config):
        """Trainer should call on_batch_end after each training step."""
        from artifex.generative_models.training import Trainer
        from artifex.generative_models.training.callbacks import CallbackList

        def loss_fn(model, batch, _rng):
            x = batch["x"]
            y = model(x)
            loss = jnp.mean(y**2)
            return loss, {"loss": loss}

        mock_callback = MagicMock()
        callbacks = CallbackList(callbacks=[mock_callback])

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            loss_fn=loss_fn,
            callbacks=callbacks,
        )

        batch = {"x": jax.random.normal(jax.random.PRNGKey(0), (4, 4))}
        trainer.train_step(batch)

        # on_batch_end should have been called
        assert mock_callback.on_batch_end.called


class TestCallbackOverhead:
    """Test that callback system has minimal overhead."""

    def test_empty_callback_list_overhead(self):
        """Empty CallbackList should have near-zero overhead."""
        import time

        from artifex.generative_models.training.callbacks import CallbackList

        callback_list = CallbackList()
        trainer_mock = None  # Minimal object
        logs = {}

        # Warmup
        for _ in range(100):
            callback_list.on_batch_end(trainer_mock, 0, logs)

        # Benchmark
        iterations = 100_000
        start = time.perf_counter()
        for i in range(iterations):
            callback_list.on_batch_end(trainer_mock, i, logs)
        elapsed = time.perf_counter() - start

        # Should be < 1 microsecond per call (100k calls in < 100ms)
        avg_time_us = (elapsed / iterations) * 1_000_000
        assert avg_time_us < 1.0, f"Empty callback overhead too high: {avg_time_us:.3f}us"

    def test_base_callback_overhead(self):
        """BaseCallback no-op methods should have minimal overhead."""
        import time

        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        callback = BaseCallback()
        callback_list = CallbackList(callbacks=[callback])
        trainer_mock = None
        logs = {}

        # Warmup
        for _ in range(100):
            callback_list.on_batch_end(trainer_mock, 0, logs)

        # Benchmark
        iterations = 100_000
        start = time.perf_counter()
        for i in range(iterations):
            callback_list.on_batch_end(trainer_mock, i, logs)
        elapsed = time.perf_counter() - start

        # Should be < 2 microseconds per call with one no-op callback
        avg_time_us = (elapsed / iterations) * 1_000_000
        assert avg_time_us < 2.0, f"BaseCallback overhead too high: {avg_time_us:.3f}us"

    def test_multiple_callbacks_overhead(self):
        """Multiple callbacks should scale linearly with minimal base overhead."""
        import time

        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        # 10 no-op callbacks
        callbacks = [BaseCallback() for _ in range(10)]
        callback_list = CallbackList(callbacks=callbacks)
        trainer_mock = None
        logs = {}

        # Warmup
        for _ in range(100):
            callback_list.on_batch_end(trainer_mock, 0, logs)

        # Benchmark
        iterations = 10_000
        start = time.perf_counter()
        for i in range(iterations):
            callback_list.on_batch_end(trainer_mock, i, logs)
        elapsed = time.perf_counter() - start

        # Should be < 5 microseconds per call with 10 no-op callbacks
        avg_time_us = (elapsed / iterations) * 1_000_000
        assert avg_time_us < 5.0, f"10 callbacks overhead too high: {avg_time_us:.3f}us"

    def test_slots_memory_efficiency(self):
        """CallbackList and BaseCallback should use __slots__ for memory efficiency."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            CallbackList,
        )

        # Verify __slots__ is defined
        assert hasattr(CallbackList, "__slots__"), "CallbackList should use __slots__"
        assert hasattr(BaseCallback, "__slots__"), "BaseCallback should use __slots__"

        # Verify no __dict__ on instances (slots working)
        cb = BaseCallback()
        cl = CallbackList()
        assert not hasattr(cb, "__dict__"), "BaseCallback instance should not have __dict__"
        assert not hasattr(cl, "__dict__"), "CallbackList instance should not have __dict__"
