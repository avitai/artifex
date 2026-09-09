"""Tests for the runtime-oriented device utility functions."""

from __future__ import annotations

import pytest
from substrax.devices import BatchSizeRecommendation

from artifex.generative_models.utils.jax import device as device_utils


@pytest.fixture
def hardware(monkeypatch: pytest.MonkeyPatch) -> BatchSizeRecommendation:
    """Pin the hardware table so the tests measure artifex's scaling rule alone."""
    recommendation = BatchSizeRecommendation(
        min_batch_size=8, optimal_batch_size=64, critical_batch_size=32, max_memory_batch_size=96
    )
    monkeypatch.setattr(device_utils, "get_batch_size_recommendation", lambda: recommendation)
    return recommendation


def test_default_base_is_the_hardware_optimum(hardware: BatchSizeRecommendation) -> None:
    """A mid-sized model gets the hardware's optimal batch size unchanged."""
    assert device_utils.get_recommended_batch_size(model_params=10_000_000) == 64


def test_large_models_halve_the_batch(hardware: BatchSizeRecommendation) -> None:
    """Above 100M parameters the batch is halved, from the default or an explicit base."""
    assert device_utils.get_recommended_batch_size(model_params=200_000_000) == 32
    assert device_utils.get_recommended_batch_size(200_000_000, base_batch_size=20) == 10


def test_small_models_double_up_to_the_memory_ceiling(hardware: BatchSizeRecommendation) -> None:
    """Below 1M parameters the batch doubles, capped by the hardware's memory estimate."""
    assert device_utils.get_recommended_batch_size(model_params=100_000) == 96
    assert device_utils.get_recommended_batch_size(100_000, base_batch_size=16) == 32


def test_without_a_memory_ceiling_the_doubling_is_uncapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hardware substrax has no memory estimate for imposes no cap."""
    recommendation = BatchSizeRecommendation(
        min_batch_size=1, optimal_batch_size=32, critical_batch_size=16
    )
    monkeypatch.setattr(device_utils, "get_batch_size_recommendation", lambda: recommendation)

    assert device_utils.get_recommended_batch_size(model_params=100) == 64


def test_never_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Halving a base of one still yields a batch of one."""
    recommendation = BatchSizeRecommendation(
        min_batch_size=1, optimal_batch_size=1, critical_batch_size=1
    )
    monkeypatch.setattr(device_utils, "get_batch_size_recommendation", lambda: recommendation)

    assert device_utils.get_recommended_batch_size(model_params=10**9) == 1
