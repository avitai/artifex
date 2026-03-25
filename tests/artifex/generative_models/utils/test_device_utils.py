"""Tests for runtime-oriented device utility functions."""

from __future__ import annotations

from types import SimpleNamespace

from artifex.generative_models.utils.jax import device as device_utils


class TestVerifyDeviceSetup:
    """Test device verification."""

    def test_verify_returns_bool(self) -> None:
        """Verification should return a boolean health verdict."""
        result = device_utils.verify_device_setup(critical_only=True)
        assert isinstance(result, bool)


class TestGetRecommendedBatchSize:
    """Test batch size recommendation."""

    def test_small_model_on_gpu(self, monkeypatch) -> None:
        """Small models on GPU can increase the base batch size."""
        monkeypatch.setattr(
            device_utils,
            "get_device_manager",
            lambda: SimpleNamespace(has_gpu=True),
        )

        batch_size = device_utils.get_recommended_batch_size(model_params=100_000)

        assert batch_size == 64

    def test_large_model_on_gpu(self, monkeypatch) -> None:
        """Large models on GPU should shrink the base batch size."""
        monkeypatch.setattr(
            device_utils,
            "get_device_manager",
            lambda: SimpleNamespace(has_gpu=True),
        )

        batch_size = device_utils.get_recommended_batch_size(model_params=200_000_000)

        assert batch_size == 16

    def test_cpu_runtime_applies_stronger_downscaling(self, monkeypatch) -> None:
        """CPU-only runs should use smaller batches than GPU runs."""
        monkeypatch.setattr(
            device_utils,
            "get_device_manager",
            lambda: SimpleNamespace(has_gpu=False),
        )

        batch_size = device_utils.get_recommended_batch_size(
            model_params=1_000_000,
            base_batch_size=64,
        )

        assert batch_size == 16
