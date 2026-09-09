"""Contracts for the tracking surface artifex re-exports from substrax."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_NAMES = (
    "Logger",
    "ConsoleLogger",
    "FileLogger",
    "WandbLogger",
    "MLFlowLogger",
    "create_logger",
)


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_logging_package_re_exports_substrax(name: str) -> None:
    """The artifex logging package hands out substrax's trackers, not copies."""
    artifex_logging = importlib.import_module("artifex.generative_models.utils.logging")
    substrax_tracking = importlib.import_module("substrax.tracking")

    assert getattr(artifex_logging, name) is getattr(substrax_tracking, name)


@pytest.mark.parametrize(
    "module",
    [
        "artifex.generative_models.utils.logging.logger",
        "artifex.generative_models.utils.logging.wandb",
        "artifex.generative_models.utils.logging.mlflow",
        "artifex.generative_models.core.logging",
    ],
)
def test_the_duplicated_logging_modules_are_gone(module: str) -> None:
    """The logger base, the two SDK loggers and the core re-export have one home."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_metrics_logger_stays_in_artifex_over_the_shared_logger() -> None:
    """MetricsLogger depends on artifex's MetricBase, so it stays; it takes a substrax Logger."""
    import inspect

    from substrax.tracking import Logger

    from artifex.generative_models.utils.logging import log_distribution_metrics, MetricsLogger

    assert MetricsLogger.__module__ == "artifex.generative_models.utils.logging.metrics"
    assert inspect.signature(MetricsLogger.__init__).parameters["logger"].annotation in (
        "Logger",
        Logger,
    )
    assert callable(log_distribution_metrics)


def test_wandb_callback_is_a_logger_callback_over_the_shared_wandb_logger() -> None:
    """The W&B callback constructs substrax's WandbLogger instead of driving the SDK."""
    from unittest.mock import MagicMock, patch

    from substrax.tracking import WandbLogger

    from artifex.generative_models.training.callbacks import (
        LoggerCallback,
        WandbLoggerCallback,
        WandbLoggerConfig,
    )

    assert issubclass(WandbLoggerCallback, LoggerCallback)
    with patch("wandb.init") as init:
        init.return_value = MagicMock(name="run", id="run-id")
        callback = WandbLoggerCallback(
            WandbLoggerConfig(project="p", name="n", mode="offline", resume="allow")
        )

    assert isinstance(callback.logger, WandbLogger)
    assert init.call_args.kwargs["mode"] == "offline"
    assert init.call_args.kwargs["resume"] == "allow"
