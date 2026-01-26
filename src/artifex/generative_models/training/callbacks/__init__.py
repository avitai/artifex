"""Training callbacks for generative models.

Provides a lightweight, Protocol-based callback system for training loops.
"""

from artifex.generative_models.training.callbacks.base import (
    BaseCallback,
    CallbackList,
    TrainerLike,
    TrainingCallback,
)
from artifex.generative_models.training.callbacks.checkpoint import (
    CheckpointConfig,
    ModelCheckpoint,
)
from artifex.generative_models.training.callbacks.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
)
from artifex.generative_models.training.callbacks.logging import (
    LoggerCallback,
    LoggerCallbackConfig,
    ProgressBarCallback,
    ProgressBarConfig,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
    WandbLoggerCallback,
    WandbLoggerConfig,
)
from artifex.generative_models.training.callbacks.profiling import (
    JAXProfiler,
    MemoryProfileConfig,
    MemoryProfiler,
    ProfilingConfig,
)


__all__ = [
    # Base classes
    "TrainerLike",
    "TrainingCallback",
    "BaseCallback",
    "CallbackList",
    # Early stopping
    "EarlyStopping",
    "EarlyStoppingConfig",
    # Checkpointing
    "ModelCheckpoint",
    "CheckpointConfig",
    # Logging
    "LoggerCallback",
    "LoggerCallbackConfig",
    "WandbLoggerCallback",
    "WandbLoggerConfig",
    "TensorBoardLoggerCallback",
    "TensorBoardLoggerConfig",
    "ProgressBarCallback",
    "ProgressBarConfig",
    # Profiling
    "ProfilingConfig",
    "JAXProfiler",
    "MemoryProfileConfig",
    "MemoryProfiler",
]
