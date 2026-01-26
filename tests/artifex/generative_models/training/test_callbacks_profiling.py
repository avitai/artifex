"""Tests for profiling callbacks.

Following TDD principles - these tests define the expected behavior
for the JAXProfiler and MemoryProfiler callbacks.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestProfilingConfig:
    """Test ProfilingConfig dataclass."""

    def test_config_exists(self):
        """ProfilingConfig should be importable."""
        from artifex.generative_models.training.callbacks.profiling import (
            ProfilingConfig,
        )

        assert ProfilingConfig is not None

    def test_config_default_values(self):
        """ProfilingConfig should have sensible defaults."""
        from artifex.generative_models.training.callbacks.profiling import (
            ProfilingConfig,
        )

        config = ProfilingConfig()
        assert config.log_dir == "logs/profiles"
        assert config.start_step == 10
        assert config.end_step == 20
        assert config.trace_memory is True
        assert config.trace_python is False

    def test_config_custom_values(self):
        """ProfilingConfig should accept custom values."""
        from artifex.generative_models.training.callbacks.profiling import (
            ProfilingConfig,
        )

        config = ProfilingConfig(
            log_dir="custom/profiles",
            start_step=5,
            end_step=15,
            trace_memory=False,
            trace_python=True,
        )
        assert config.log_dir == "custom/profiles"
        assert config.start_step == 5
        assert config.end_step == 15
        assert config.trace_memory is False
        assert config.trace_python is True


class TestJAXProfilerBasic:
    """Test basic JAXProfiler functionality."""

    def test_jax_profiler_exists(self):
        """JAXProfiler should be importable."""
        from artifex.generative_models.training.callbacks.profiling import JAXProfiler

        assert JAXProfiler is not None

    def test_jax_profiler_inherits_base_callback(self):
        """JAXProfiler should inherit from BaseCallback."""
        from artifex.generative_models.training.callbacks import BaseCallback
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        callback = JAXProfiler(ProfilingConfig())
        assert isinstance(callback, BaseCallback)

    def test_jax_profiler_initializes_state(self):
        """JAXProfiler should initialize profiling state."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        callback = JAXProfiler(ProfilingConfig())
        assert callback._profiling is False
        assert callback.config is not None

    def test_jax_profiler_stores_config(self):
        """JAXProfiler should store the config."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        config = ProfilingConfig(start_step=5, end_step=10)
        callback = JAXProfiler(config)
        assert callback.config.start_step == 5
        assert callback.config.end_step == 10


class TestJAXProfilerLifecycle:
    """Test JAXProfiler lifecycle hooks."""

    def test_on_train_begin_creates_log_dir(self):
        """on_train_begin should create log directory."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "profiles"
            config = ProfilingConfig(log_dir=str(log_dir))
            callback = JAXProfiler(config)

            trainer = MagicMock()
            callback.on_train_begin(trainer)

            assert log_dir.exists()

    @patch("jax.profiler.start_trace")
    def test_on_batch_begin_starts_profiling_at_start_step(self, mock_start_trace):
        """on_batch_begin should start profiling at start_step."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilingConfig(log_dir=tmpdir, start_step=5, end_step=10)
            callback = JAXProfiler(config)
            trainer = MagicMock()

            # Before start_step - should not start
            callback.on_batch_begin(trainer, batch=4)
            mock_start_trace.assert_not_called()
            assert callback._profiling is False

            # At start_step - should start profiling
            callback.on_batch_begin(trainer, batch=5)
            mock_start_trace.assert_called_once()
            assert callback._profiling is True

    @patch("jax.profiler.stop_trace")
    @patch("jax.profiler.start_trace")
    def test_on_batch_end_stops_profiling_at_end_step(self, _mock_start_trace, mock_stop_trace):
        """on_batch_end should stop profiling at end_step."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilingConfig(log_dir=tmpdir, start_step=5, end_step=10)
            callback = JAXProfiler(config)
            trainer = MagicMock()

            # Start profiling
            callback.on_batch_begin(trainer, batch=5)
            assert callback._profiling is True

            # Before end_step - should not stop
            callback.on_batch_end(trainer, 9, {})
            mock_stop_trace.assert_not_called()

            # At end_step - should stop profiling
            callback.on_batch_end(trainer, 10, {})
            mock_stop_trace.assert_called_once()
            assert callback._profiling is False

    @patch("jax.profiler.stop_trace")
    @patch("jax.profiler.start_trace")
    def test_on_train_end_stops_profiling_if_active(self, _mock_start_trace, mock_stop_trace):
        """on_train_end should stop profiling if still active."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilingConfig(log_dir=tmpdir, start_step=5, end_step=100)
            callback = JAXProfiler(config)
            trainer = MagicMock()

            # Start profiling
            callback.on_batch_begin(trainer, batch=5)
            assert callback._profiling is True

            # End training before end_step
            callback.on_train_end(trainer)
            mock_stop_trace.assert_called_once()
            assert callback._profiling is False

    @patch("jax.profiler.stop_trace")
    def test_on_train_end_does_nothing_if_not_profiling(self, mock_stop_trace):
        """on_train_end should do nothing if not profiling."""
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilingConfig(log_dir=tmpdir)
            callback = JAXProfiler(config)
            trainer = MagicMock()

            # End training without ever starting profiling
            callback.on_train_end(trainer)
            mock_stop_trace.assert_not_called()


class TestMemoryProfileConfig:
    """Test MemoryProfileConfig dataclass."""

    def test_config_exists(self):
        """MemoryProfileConfig should be importable."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
        )

        assert MemoryProfileConfig is not None

    def test_config_default_values(self):
        """MemoryProfileConfig should have sensible defaults."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
        )

        config = MemoryProfileConfig()
        assert config.log_dir == "logs/memory"
        assert config.profile_every_n_steps == 100
        assert config.log_device_memory is True

    def test_config_custom_values(self):
        """MemoryProfileConfig should accept custom values."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
        )

        config = MemoryProfileConfig(
            log_dir="custom/memory",
            profile_every_n_steps=50,
            log_device_memory=False,
        )
        assert config.log_dir == "custom/memory"
        assert config.profile_every_n_steps == 50
        assert config.log_device_memory is False


class TestMemoryProfilerBasic:
    """Test basic MemoryProfiler functionality."""

    def test_memory_profiler_exists(self):
        """MemoryProfiler should be importable."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfiler,
        )

        assert MemoryProfiler is not None

    def test_memory_profiler_inherits_base_callback(self):
        """MemoryProfiler should inherit from BaseCallback."""
        from artifex.generative_models.training.callbacks import BaseCallback
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        callback = MemoryProfiler(MemoryProfileConfig())
        assert isinstance(callback, BaseCallback)

    def test_memory_profiler_initializes_history(self):
        """MemoryProfiler should initialize empty history."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        callback = MemoryProfiler(MemoryProfileConfig())
        assert callback._memory_history == []

    def test_memory_profiler_stores_config(self):
        """MemoryProfiler should store the config."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        config = MemoryProfileConfig(profile_every_n_steps=25)
        callback = MemoryProfiler(config)
        assert callback.config.profile_every_n_steps == 25


class TestMemoryProfilerCollection:
    """Test MemoryProfiler memory collection."""

    @patch("jax.devices")
    def test_on_batch_end_collects_memory_at_interval(self, mock_devices):
        """on_batch_end should collect memory at profile_every_n_steps."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        # Mock device with memory stats
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="gpu:0")
        mock_device.memory_stats.return_value = {
            "bytes_in_use": 1024 * 1024,
            "peak_bytes_in_use": 2048 * 1024,
        }
        mock_devices.return_value = [mock_device]

        config = MemoryProfileConfig(profile_every_n_steps=10)
        callback = MemoryProfiler(config)
        trainer = MagicMock()

        # Before interval - should not collect
        callback.on_batch_end(trainer, batch=5, logs={})
        assert len(callback._memory_history) == 0

        # At interval - should collect
        callback.on_batch_end(trainer, batch=10, logs={})
        assert len(callback._memory_history) == 1
        assert callback._memory_history[0]["step"] == 10
        assert "gpu:0" in callback._memory_history[0]["memory"]

        # At next interval - should collect again
        callback.on_batch_end(trainer, batch=20, logs={})
        assert len(callback._memory_history) == 2

    @patch("jax.devices")
    def test_on_batch_end_handles_no_memory_stats(self, mock_devices):
        """on_batch_end should handle devices without memory_stats."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        # Mock device without memory stats
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="cpu:0")
        mock_device.memory_stats.return_value = None
        mock_devices.return_value = [mock_device]

        config = MemoryProfileConfig(profile_every_n_steps=10)
        callback = MemoryProfiler(config)
        trainer = MagicMock()

        # Should not raise, just skip device with no stats
        callback.on_batch_end(trainer, batch=10, logs={})
        assert len(callback._memory_history) == 1
        assert callback._memory_history[0]["memory"] == {}


class TestMemoryProfilerOutput:
    """Test MemoryProfiler output functionality."""

    @patch("jax.devices")
    def test_on_train_end_saves_memory_profile(self, mock_devices):
        """on_train_end should save memory profile to JSON."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        # Mock device
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="gpu:0")
        mock_device.memory_stats.return_value = {
            "bytes_in_use": 1024,
            "peak_bytes_in_use": 2048,
        }
        mock_devices.return_value = [mock_device]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "memory"
            config = MemoryProfileConfig(
                log_dir=str(log_dir),
                profile_every_n_steps=10,
            )
            callback = MemoryProfiler(config)
            trainer = MagicMock()

            # Collect some data
            callback.on_batch_end(trainer, batch=10, logs={})
            callback.on_batch_end(trainer, batch=20, logs={})

            # Save on train end
            callback.on_train_end(trainer)

            # Check file was created
            profile_file = log_dir / "memory_profile.json"
            assert profile_file.exists()

            # Check content
            with open(profile_file) as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["step"] == 10
            assert data[1]["step"] == 20

    def test_on_train_end_creates_log_dir_if_missing(self):
        """on_train_end should create log directory if it doesn't exist."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "memory"
            config = MemoryProfileConfig(log_dir=str(log_dir))
            callback = MemoryProfiler(config)
            trainer = MagicMock()

            callback.on_train_end(trainer)

            assert log_dir.exists()
            assert (log_dir / "memory_profile.json").exists()

    def test_on_train_end_saves_empty_history(self):
        """on_train_end should save empty history if no data collected."""
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "memory"
            config = MemoryProfileConfig(log_dir=str(log_dir))
            callback = MemoryProfiler(config)
            trainer = MagicMock()

            callback.on_train_end(trainer)

            with open(log_dir / "memory_profile.json") as f:
                data = json.load(f)
            assert data == []


class TestProfilingCallbackExports:
    """Test that profiling callbacks are properly exported."""

    def test_exports_from_callbacks_init(self):
        """Profiling classes should be exported from callbacks __init__."""
        from artifex.generative_models.training.callbacks import (
            JAXProfiler,
            MemoryProfileConfig,
            MemoryProfiler,
            ProfilingConfig,
        )

        assert ProfilingConfig is not None
        assert JAXProfiler is not None
        assert MemoryProfileConfig is not None
        assert MemoryProfiler is not None


class TestProfilingNoInterferenceWithJIT:
    """Test that profiling callbacks don't interfere with JIT compilation."""

    def test_jax_profiler_does_not_interfere_with_jit(self):
        """JAXProfiler should not interfere with jit-compiled functions."""
        import tempfile

        import jax
        import jax.numpy as jnp

        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilingConfig(log_dir=tmpdir, start_step=0, end_step=5)
            callback = JAXProfiler(config)

            # Create a simple jitted function
            @jax.jit
            def simple_computation(x):
                return jnp.sum(x**2)

            trainer = MagicMock()
            callback.on_train_begin(trainer)

            # Run jitted function before, during, and after profiling window
            x = jnp.ones((100, 100))

            # Before profiling starts
            result1 = simple_computation(x)

            # Start profiling (step 0)
            callback.on_batch_begin(trainer, 0)

            # During profiling - JIT should still work
            result2 = simple_computation(x)

            # Stop profiling (step 5)
            callback.on_batch_end(trainer, 5, {})

            # After profiling
            result3 = simple_computation(x)

            callback.on_train_end(trainer)

            # All results should be identical (JIT consistency)
            assert float(result1) == float(result2) == float(result3)

    def test_memory_profiler_does_not_interfere_with_jit(self):
        """MemoryProfiler should not interfere with jit-compiled functions."""
        import tempfile

        import jax
        import jax.numpy as jnp

        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryProfileConfig(log_dir=tmpdir, profile_every_n_steps=1)
            callback = MemoryProfiler(config)

            # Create a simple jitted function
            @jax.jit
            def simple_computation(x):
                return jnp.sum(x**2)

            trainer = MagicMock()
            x = jnp.ones((100, 100))

            # Run jitted function with memory profiling active
            result1 = simple_computation(x)
            callback.on_batch_end(trainer, 1, {})
            result2 = simple_computation(x)
            callback.on_batch_end(trainer, 2, {})
            result3 = simple_computation(x)

            callback.on_train_end(trainer)

            # All results should be identical (JIT consistency)
            assert float(result1) == float(result2) == float(result3)


class TestProfilingMinimalOverhead:
    """Test that profiling callbacks have minimal overhead."""

    def test_jax_profiler_overhead_is_minimal_outside_profiling_window(self):
        """JAXProfiler on_batch_begin/end should have minimal overhead when not profiling."""
        import tempfile
        import time

        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set profiling window to never activate (start_step > iterations)
            config = ProfilingConfig(log_dir=tmpdir, start_step=1000, end_step=2000)
            callback = JAXProfiler(config)
            trainer = MagicMock()
            callback.on_train_begin(trainer)

            # Measure overhead of 1000 callback invocations
            start = time.perf_counter()
            for i in range(1000):
                callback.on_batch_begin(trainer, i)
                callback.on_batch_end(trainer, i, {"loss": 0.5})
            elapsed = time.perf_counter() - start

            callback.on_train_end(trainer)

            # Should complete in under 100ms for 1000 iterations (< 0.1ms per call)
            assert elapsed < 0.1, f"Overhead too high: {elapsed:.4f}s for 1000 iterations"

    def test_memory_profiler_overhead_is_minimal_between_intervals(self):
        """MemoryProfiler should have minimal overhead between collection intervals."""
        import tempfile
        import time

        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set large interval so collection rarely happens
            config = MemoryProfileConfig(log_dir=tmpdir, profile_every_n_steps=10000)
            callback = MemoryProfiler(config)
            trainer = MagicMock()

            # Measure overhead of 1000 callback invocations (none should trigger collection)
            start = time.perf_counter()
            for i in range(1, 1000):  # Start at 1 to avoid step 0 which would trigger
                callback.on_batch_end(trainer, i, {"loss": 0.5})
            elapsed = time.perf_counter() - start

            callback.on_train_end(trainer)

            # Should complete in under 50ms for 1000 iterations (< 0.05ms per call)
            assert elapsed < 0.05, f"Overhead too high: {elapsed:.4f}s for 1000 iterations"
            # No memory was collected since we didn't hit the interval
            assert len(callback._memory_history) == 0


class TestProfilingCallbacksWithCallbackList:
    """Test profiling callbacks work with CallbackList."""

    def test_jax_profiler_in_callback_list(self):
        """JAXProfiler should work in CallbackList."""
        from artifex.generative_models.training.callbacks import CallbackList
        from artifex.generative_models.training.callbacks.profiling import (
            JAXProfiler,
            ProfilingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = JAXProfiler(ProfilingConfig(log_dir=tmpdir))
            callback_list = CallbackList([callback])

            trainer = MagicMock()
            callback_list.on_train_begin(trainer)
            callback_list.on_train_end(trainer)

    def test_memory_profiler_in_callback_list(self):
        """MemoryProfiler should work in CallbackList."""
        from artifex.generative_models.training.callbacks import CallbackList
        from artifex.generative_models.training.callbacks.profiling import (
            MemoryProfileConfig,
            MemoryProfiler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MemoryProfiler(MemoryProfileConfig(log_dir=tmpdir))
            callback_list = CallbackList([callback])

            trainer = MagicMock()
            callback_list.on_train_begin(trainer)
            callback_list.on_batch_end(trainer, batch=100, logs={})
            callback_list.on_train_end(trainer)
