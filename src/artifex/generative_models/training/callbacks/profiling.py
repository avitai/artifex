"""Profiling callbacks for training performance analysis.

Provides JAX-native profiling capabilities including trace-based profiling
for TensorBoard visualization and memory usage tracking.

Note:
    JAX profiler tracing is disabled on macOS due to TensorFlow ARM64
    compatibility issues. The profiler uses TensorFlow/TensorBoard backend
    which hangs on macOS ARM64. See: https://github.com/tensorflow/tensorflow/issues/52138
"""

from __future__ import annotations

import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseCallback, TrainerLike


# Detect platform for conditional profiler behavior
IS_MACOS = platform.system() == "Darwin"


@dataclass(slots=True)
class ProfilingConfig:
    """Configuration for JAX trace profiling.

    Attributes:
        log_dir: Directory to save profiling traces.
        start_step: Step at which to start profiling (skip warmup).
        end_step: Step at which to stop profiling.
        trace_memory: Whether to include memory usage in traces.
        trace_python: Whether to trace Python execution (slower but more detail).
    """

    log_dir: str = "logs/profiles"
    start_step: int = 10
    end_step: int = 20
    trace_memory: bool = True
    trace_python: bool = False


class JAXProfiler(BaseCallback):
    """JAX profiler callback for performance analysis.

    Integrates with JAX's built-in profiler to capture traces that can be
    viewed in TensorBoard or Perfetto. Automatically skips warmup steps
    to get more representative profiling data.

    Features:
        - Integration with JAX's built-in profiler
        - TensorBoard trace visualization
        - Configurable profiling window (start/end steps)
        - Automatic cleanup on training end

    Example:
        ```python
        from artifex.generative_models.training.callbacks import (
            JAXProfiler,
            ProfilingConfig,
        )

        profiler = JAXProfiler(ProfilingConfig(
            log_dir="logs/profiles",
            start_step=10,
            end_step=20,
        ))
        trainer.fit(callbacks=[profiler])

        # View in TensorBoard:
        # tensorboard --logdir logs/profiles
        ```

    Note:
        The profiler captures JAX operations including XLA compilation,
        memory allocation, and device execution. For best results:
        - Set start_step after warmup (JIT compilation)
        - Keep profiling window small (10-20 steps)
        - Use trace_python=True only when debugging Python bottlenecks
    """

    def __init__(self, config: ProfilingConfig) -> None:
        """Initialize JAXProfiler.

        Args:
            config: Profiling configuration.
        """
        self.config = config
        self._profiling: bool = False

    def on_train_begin(self, trainer: TrainerLike) -> None:
        """Create log directory at training start.

        Args:
            trainer: Trainer instance (unused).
        """
        del trainer  # Unused
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    def on_batch_begin(self, trainer: TrainerLike, batch: int) -> None:
        """Start profiling at the configured start step.

        Args:
            trainer: Trainer instance (unused).
            batch: Current batch number.

        Note:
            JAX profiler tracing is disabled on macOS due to TensorFlow
            ARM64 compatibility issues. A warning is emitted and training
            continues without profiling.
        """
        del trainer  # Unused
        if batch == self.config.start_step and not self._profiling:
            # Skip profiling on macOS - TensorFlow/TensorBoard backend causes hangs
            if IS_MACOS:
                warnings.warn(
                    "JAX profiler tracing is disabled on macOS due to TensorFlow "
                    "ARM64 compatibility issues. Training will continue without "
                    "trace profiling. Memory profiling via MemoryProfiler is still "
                    "available.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return

            import jax

            jax.profiler.start_trace(
                self.config.log_dir,
                create_perfetto_link=False,
            )
            self._profiling = True

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None:
        """Stop profiling at the configured end step.

        Args:
            trainer: Trainer instance (unused).
            batch: Current batch number.
            logs: Batch metrics (unused).
        """
        del trainer, logs  # Unused
        if batch == self.config.end_step and self._profiling:
            import jax

            jax.profiler.stop_trace()
            self._profiling = False

    def on_train_end(self, trainer: TrainerLike) -> None:
        """Ensure profiling is stopped if training ends early.

        Args:
            trainer: Trainer instance (unused).
        """
        del trainer  # Unused
        if self._profiling:
            import jax

            jax.profiler.stop_trace()
            self._profiling = False


@dataclass(slots=True)
class MemoryProfileConfig:
    """Configuration for memory profiling.

    Attributes:
        log_dir: Directory to save memory profile data.
        profile_every_n_steps: Collect memory info every N steps.
        log_device_memory: Whether to log device (GPU/TPU) memory stats.
    """

    log_dir: str = "logs/memory"
    profile_every_n_steps: int = 100
    log_device_memory: bool = True


class MemoryProfiler(BaseCallback):
    """Memory usage profiling callback.

    Tracks memory usage during training and saves a timeline to JSON.
    Useful for identifying memory leaks and understanding memory patterns.

    Features:
        - Track JAX device memory usage (GPU/TPU)
        - Log peak memory per step
        - Export memory timeline to JSON
        - Configurable profiling interval

    Example:
        ```python
        from artifex.generative_models.training.callbacks import (
            MemoryProfiler,
            MemoryProfileConfig,
        )

        profiler = MemoryProfiler(MemoryProfileConfig(
            log_dir="logs/memory",
            profile_every_n_steps=50,
        ))
        trainer.fit(callbacks=[profiler])

        # Memory profile saved to logs/memory/memory_profile.json
        ```

    Note:
        Not all devices support memory_stats(). CPU devices typically
        return None, in which case those devices are skipped.
    """

    def __init__(self, config: MemoryProfileConfig) -> None:
        """Initialize MemoryProfiler.

        Args:
            config: Memory profiling configuration.
        """
        self.config = config
        self._memory_history: list[dict[str, Any]] = []

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None:
        """Collect memory stats at configured intervals.

        Args:
            trainer: Trainer instance (unused).
            batch: Current batch number.
            logs: Batch metrics (unused).
        """
        del trainer, logs  # Unused
        if batch % self.config.profile_every_n_steps != 0:
            return

        import jax

        memory_info: dict[str, dict[str, int]] = {}

        if self.config.log_device_memory:
            for device in jax.devices():
                stats = device.memory_stats()
                if stats:
                    memory_info[str(device)] = {
                        "bytes_in_use": stats.get("bytes_in_use", 0),
                        "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                    }

        self._memory_history.append(
            {
                "step": batch,
                "memory": memory_info,
            }
        )

    def on_train_end(self, trainer: TrainerLike) -> None:
        """Save memory profile to JSON file.

        Args:
            trainer: Trainer instance (unused).
        """
        del trainer  # Unused
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        profile_path = log_dir / "memory_profile.json"
        with open(profile_path, "w") as f:
            json.dump(self._memory_history, f, indent=2)
