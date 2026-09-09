"""Training callbacks for generative models.

The callback protocol, the base class, the callback list and the early-stopping
callback are substrax's; this package re-exports them next to the checkpoint,
logging and profiling callbacks that are specific to artifex trainers.
"""

from substrax.callbacks import (
    BaseCallback,
    CallbackList,
    EarlyStoppingCallback,
    EarlyStoppingConfig,
    TrainerLike,
    TrainingCallback,
)

from artifex.generative_models.training.callbacks.checkpoint import (
    CheckpointConfig,
    ModelCheckpoint,
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
    "EarlyStoppingCallback",
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
