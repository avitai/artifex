"""Logging utilities for generative models.

The loggers are substrax's trackers (console, file, Weights & Biases and MLflow),
re-exported here; ``MetricsLogger`` bridges artifex's evaluation metrics into them.
"""

from substrax.tracking import (
    ConsoleLogger,
    create_logger,
    FileLogger,
    Logger,
    MLFlowLogger,
    WandbLogger,
)

from artifex.generative_models.utils.logging.metrics import (
    log_distribution_metrics,
    MetricsLogger,
)


__all__ = [
    # Loggers (substrax.tracking)
    "Logger",
    "ConsoleLogger",
    "FileLogger",
    "MLFlowLogger",
    "WandbLogger",
    "create_logger",
    # Metrics logging
    "MetricsLogger",
    "log_distribution_metrics",
]
