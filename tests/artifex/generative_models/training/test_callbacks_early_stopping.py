"""Tests for EarlyStopping callback.

Following TDD principles - these tests define the expected behavior
for the EarlyStopping callback.
"""

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest


class TestEarlyStoppingConfig:
    """Test EarlyStoppingConfig dataclass."""

    def test_config_exists(self):
        """EarlyStoppingConfig should be importable."""
        from artifex.generative_models.training.callbacks import EarlyStoppingConfig

        assert EarlyStoppingConfig is not None

    def test_config_default_values(self):
        """EarlyStoppingConfig should have sensible defaults."""
        from artifex.generative_models.training.callbacks import EarlyStoppingConfig

        config = EarlyStoppingConfig()
        assert config.monitor == "val_loss"
        assert config.min_delta == 0.0
        assert config.patience == 10
        assert config.mode == "min"
        assert config.check_finite is True
        assert config.stopping_threshold is None
        assert config.divergence_threshold is None

    def test_config_custom_values(self):
        """EarlyStoppingConfig should accept custom values."""
        from artifex.generative_models.training.callbacks import EarlyStoppingConfig

        config = EarlyStoppingConfig(
            monitor="accuracy",
            min_delta=0.01,
            patience=5,
            mode="max",
            check_finite=False,
            stopping_threshold=0.99,
            divergence_threshold=10.0,
        )
        assert config.monitor == "accuracy"
        assert config.min_delta == 0.01
        assert config.patience == 5
        assert config.mode == "max"
        assert config.check_finite is False
        assert config.stopping_threshold == 0.99
        assert config.divergence_threshold == 10.0


class TestEarlyStoppingBasic:
    """Test basic EarlyStopping functionality."""

    def test_early_stopping_exists(self):
        """EarlyStopping should be importable."""
        from artifex.generative_models.training.callbacks import EarlyStopping

        assert EarlyStopping is not None

    def test_early_stopping_inherits_base_callback(self):
        """EarlyStopping should inherit from BaseCallback."""
        from artifex.generative_models.training.callbacks import (
            BaseCallback,
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig())
        assert isinstance(callback, BaseCallback)

    def test_early_stopping_initializes_state(self):
        """EarlyStopping should initialize tracking state."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig())
        assert callback.wait_count == 0
        assert callback.best_score is None
        assert callback.stopped_epoch is None

    def test_early_stopping_has_should_stop_property(self):
        """EarlyStopping should have should_stop property."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig())
        assert hasattr(callback, "should_stop")
        assert callback.should_stop is False


class TestEarlyStoppingMinMode:
    """Test EarlyStopping in 'min' mode (lower is better)."""

    def test_improvement_resets_wait_count(self):
        """Improvement should reset wait count."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer_mock = MagicMock()

        # First epoch - establishes baseline
        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score == 1.0
        assert callback.wait_count == 0

        # No improvement
        callback.on_epoch_end(trainer_mock, 1, {"loss": 1.0})
        assert callback.wait_count == 1

        # Improvement - should reset
        callback.on_epoch_end(trainer_mock, 2, {"loss": 0.9})
        assert callback.best_score == 0.9
        assert callback.wait_count == 0

    def test_no_improvement_increments_wait_count(self):
        """No improvement should increment wait count."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": 1.1})
        assert callback.wait_count == 1

        callback.on_epoch_end(trainer_mock, 2, {"loss": 1.2})
        assert callback.wait_count == 2

    def test_patience_exceeded_sets_should_stop(self):
        """Exceeding patience should set should_stop."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss", patience=2))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"loss": 1.1})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 2, {"loss": 1.2})
        assert callback.should_stop is True
        assert callback.stopped_epoch == 2


class TestEarlyStoppingMaxMode:
    """Test EarlyStopping in 'max' mode (higher is better)."""

    def test_improvement_in_max_mode(self):
        """Higher values should be improvements in max mode."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="accuracy", mode="max", patience=3))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"accuracy": 0.8})
        assert callback.best_score == 0.8

        # Higher is improvement
        callback.on_epoch_end(trainer_mock, 1, {"accuracy": 0.85})
        assert callback.best_score == 0.85
        assert callback.wait_count == 0

        # Lower is not improvement
        callback.on_epoch_end(trainer_mock, 2, {"accuracy": 0.84})
        assert callback.wait_count == 1


class TestEarlyStoppingMinDelta:
    """Test min_delta threshold for improvements."""

    def test_min_delta_ignores_small_improvements(self):
        """Small improvements below min_delta should not count."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss", min_delta=0.1, patience=3))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score == 1.0

        # Small improvement (0.05 < min_delta=0.1) - should not count
        callback.on_epoch_end(trainer_mock, 1, {"loss": 0.95})
        assert callback.wait_count == 1

        # Large improvement (0.15 > min_delta=0.1) - should count
        callback.on_epoch_end(trainer_mock, 2, {"loss": 0.8})
        assert callback.best_score == 0.8
        assert callback.wait_count == 0


class TestEarlyStoppingCheckFinite:
    """Test check_finite for NaN/Inf detection."""

    def test_nan_triggers_stop(self):
        """NaN values should trigger immediate stop when check_finite=True."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(
            EarlyStoppingConfig(monitor="loss", check_finite=True, patience=10)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": float("nan")})
        assert callback.should_stop is True

    def test_inf_triggers_stop(self):
        """Inf values should trigger immediate stop when check_finite=True."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(
            EarlyStoppingConfig(monitor="loss", check_finite=True, patience=10)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": float("inf")})
        assert callback.should_stop is True

    def test_check_finite_disabled(self):
        """NaN should not trigger stop when check_finite=False."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(
            EarlyStoppingConfig(monitor="loss", check_finite=False, patience=10)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": float("nan")})
        # Should not stop due to NaN, but wait_count increases
        assert callback.should_stop is False


class TestEarlyStoppingThresholds:
    """Test stopping and divergence thresholds."""

    def test_stopping_threshold_min_mode(self):
        """Reaching stopping_threshold should trigger stop in min mode."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(
            EarlyStoppingConfig(monitor="loss", mode="min", stopping_threshold=0.1)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 0.5})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"loss": 0.09})
        assert callback.should_stop is True

    def test_stopping_threshold_max_mode(self):
        """Reaching stopping_threshold should trigger stop in max mode."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(
            EarlyStoppingConfig(monitor="accuracy", mode="max", stopping_threshold=0.99)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"accuracy": 0.9})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"accuracy": 0.995})
        assert callback.should_stop is True

    def test_divergence_threshold(self):
        """Exceeding divergence_threshold should trigger stop."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss", divergence_threshold=10.0))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"loss": 15.0})
        assert callback.should_stop is True


class TestEarlyStoppingMissingMetric:
    """Test behavior when monitored metric is missing."""

    def test_missing_metric_does_not_crash(self):
        """Missing metric should not crash, just skip."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="val_loss"))
        trainer_mock = MagicMock()

        # Metric not present
        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score is None
        assert callback.wait_count == 0

    def test_metric_appears_later(self):
        """Callback should work when metric appears after first epoch."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="val_loss", patience=3))
        trainer_mock = MagicMock()

        # First epoch - metric missing
        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score is None

        # Second epoch - metric appears
        callback.on_epoch_end(trainer_mock, 1, {"val_loss": 0.5})
        assert callback.best_score == 0.5


class TestEarlyStoppingJaxArrays:
    """Test EarlyStopping with JAX arrays."""

    def test_works_with_jax_arrays(self):
        """Should work with JAX array values."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": jnp.array(1.0)})
        callback.on_epoch_end(trainer_mock, 1, {"loss": jnp.array(0.9)})
        assert callback.best_score == pytest.approx(0.9)
        assert callback.wait_count == 0


class TestEarlyStoppingOverhead:
    """Test that EarlyStopping has minimal overhead."""

    def test_overhead_is_minimal(self):
        """EarlyStopping overhead should be minimal."""
        import time

        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss"))
        trainer_mock = None
        logs = {"loss": 0.5}

        # Warmup
        for i in range(100):
            callback.on_epoch_end(trainer_mock, i, logs)

        # Reset state
        callback = EarlyStopping(EarlyStoppingConfig(monitor="loss"))

        # Benchmark
        iterations = 10_000
        start = time.perf_counter()
        for i in range(iterations):
            callback.on_epoch_end(trainer_mock, i, {"loss": 0.5 - i * 0.00001})
        elapsed = time.perf_counter() - start

        # Should be < 10 microseconds per call
        avg_time_us = (elapsed / iterations) * 1_000_000
        assert avg_time_us < 10.0, f"EarlyStopping overhead too high: {avg_time_us:.3f}us"
