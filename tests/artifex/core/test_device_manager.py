from __future__ import annotations

import pytest

from artifex.generative_models.core import device_manager as device_manager_module


class _FakeDevice:
    def __init__(self, platform: str, identifier: str, kind: str = "fake-device") -> None:
        self.platform = platform
        self._identifier = identifier
        self.device_kind = kind

    def __str__(self) -> str:
        return self._identifier


def test_device_manager_prefers_gpu_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime device selection should prefer a visible GPU over CPU."""
    fake_gpu = _FakeDevice("gpu", "gpu:0", "Fake GPU")
    fake_cpu = _FakeDevice("cpu", "cpu:0", "CPU")
    capabilities = device_manager_module.DeviceCapabilities(
        device_type=device_manager_module.DeviceType.GPU,
        device_count=2,
        total_memory_mb=24_576,
        compute_capability="8.9",
        cuda_version="12.4",
        driver_version="555.00",
        supports_mixed_precision=True,
        supports_distributed=True,
    )

    monkeypatch.setattr(
        device_manager_module,
        "_collect_runtime_state",
        lambda: ((fake_gpu, fake_cpu), capabilities),
    )

    manager = device_manager_module.DeviceManager()

    assert manager.has_gpu is True
    assert manager.gpu_count == 1
    assert manager.device_count == 2
    assert manager.get_default_device() is fake_gpu


def test_device_manager_raises_without_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime helpers should fail clearly when JAX exposes no visible devices."""
    capabilities = device_manager_module.DeviceCapabilities(
        device_type=device_manager_module.DeviceType.CPU,
        device_count=0,
        supports_mixed_precision=False,
        supports_distributed=False,
    )

    monkeypatch.setattr(
        device_manager_module,
        "_collect_runtime_state",
        lambda: ((), capabilities),
    )

    manager = device_manager_module.DeviceManager()

    with pytest.raises(RuntimeError, match="No JAX devices are visible"):
        manager.get_default_device()
