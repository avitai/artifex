"""Runtime device inspection and placement helpers for Artifex."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported runtime device classes."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceCapabilities:
    """Immutable snapshot of the active JAX runtime."""

    device_type: DeviceType
    device_count: int
    default_backend: str | None = None
    visible_devices: tuple[str, ...] = ()
    total_memory_mb: int | None = None
    compute_capability: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    supports_mixed_precision: bool = False
    supports_distributed: bool = False
    error: str | None = None


@contextlib.contextmanager
def _suppress_process_stderr():
    """Temporarily silence backend-probe noise from plugin initialization."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(null_stream.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


def _get_jax():
    """Import JAX lazily so module import stays side-effect free."""
    with _suppress_process_stderr():
        import jax

    return jax


def _device_kind(device: Any) -> str:
    """Return a readable kind string for a JAX device."""
    return getattr(device, "device_kind", "unknown")


def _visible_device_label(device: Any) -> str:
    """Render a stable human-readable device label."""
    return f"{device.platform}:{_device_kind(device)} ({device})"


def _collect_gpu_metadata() -> dict[str, str | int | None]:
    """Collect optional GPU metadata from nvidia-smi when available."""
    metadata: dict[str, str | int | None] = {
        "total_memory_mb": None,
        "compute_capability": None,
        "cuda_version": None,
        "driver_version": None,
    }

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return metadata

    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    if rows:
        first_row = [part.strip() for part in rows[0].split(",")]
        if first_row and first_row[0].isdigit():
            metadata["total_memory_mb"] = int(first_row[0])
        if len(first_row) > 1:
            metadata["compute_capability"] = first_row[1] or None

    try:
        version_output = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return metadata

    for line in version_output.splitlines():
        if "CUDA Version" in line and metadata["cuda_version"] is None:
            metadata["cuda_version"] = line.split("CUDA Version:")[1].strip().split()[0]
        if "Driver Version" in line and metadata["driver_version"] is None:
            parts = [part.strip() for part in line.split("|")]
            for part in parts:
                if "Driver Version" in part:
                    metadata["driver_version"] = part.split("Driver Version:")[1].strip().split()[0]
                    break

    return metadata


def _runtime_device_type(jax_devices: tuple[Any, ...]) -> DeviceType:
    """Choose the primary runtime device type from visible JAX devices."""
    platforms = {device.platform for device in jax_devices}
    if {"gpu", "cuda"} & platforms:
        return DeviceType.GPU
    if "tpu" in platforms:
        return DeviceType.TPU
    return DeviceType.CPU


def _collect_runtime_state() -> tuple[tuple[Any, ...], DeviceCapabilities]:
    """Collect visible JAX devices and the corresponding runtime capabilities."""
    try:
        jax = _get_jax()
    except (ImportError, OSError, RuntimeError) as exc:
        return (), DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_count=0,
            error=str(exc),
        )

    try:
        with _suppress_process_stderr():
            jax_devices = tuple(jax.devices())
            default_backend = jax.default_backend()
    except RuntimeError as exc:
        return (), DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_count=0,
            error=str(exc),
        )

    device_type = _runtime_device_type(jax_devices)
    metadata = _collect_gpu_metadata() if device_type is DeviceType.GPU else {}

    return jax_devices, DeviceCapabilities(
        device_type=device_type,
        device_count=len(jax_devices),
        default_backend=default_backend,
        visible_devices=tuple(_visible_device_label(device) for device in jax_devices),
        total_memory_mb=metadata.get("total_memory_mb"),
        compute_capability=metadata.get("compute_capability"),
        cuda_version=metadata.get("cuda_version"),
        driver_version=metadata.get("driver_version"),
        supports_mixed_precision=device_type in {DeviceType.GPU, DeviceType.TPU},
        supports_distributed=len(jax_devices) > 1,
        error=None,
    )


class DeviceManager:
    """Pure runtime manager for visible JAX devices."""

    def __init__(self) -> None:
        self._devices, self.capabilities = _collect_runtime_state()

    @property
    def devices(self) -> tuple[Any, ...]:
        """Return all visible JAX devices."""
        return self._devices

    @property
    def gpu_devices(self) -> tuple[Any, ...]:
        """Return visible GPU devices."""
        return tuple(device for device in self._devices if device.platform in {"gpu", "cuda"})

    @property
    def cpu_devices(self) -> tuple[Any, ...]:
        """Return visible CPU devices."""
        return tuple(device for device in self._devices if device.platform == "cpu")

    @property
    def tpu_devices(self) -> tuple[Any, ...]:
        """Return visible TPU devices."""
        return tuple(device for device in self._devices if device.platform == "tpu")

    @property
    def has_gpu(self) -> bool:
        """Return True when JAX exposes at least one GPU device."""
        return len(self.gpu_devices) > 0

    @property
    def device_count(self) -> int:
        """Return the number of visible JAX devices."""
        return len(self._devices)

    @property
    def gpu_count(self) -> int:
        """Return the number of visible GPU devices."""
        return len(self.gpu_devices)

    def get_default_device(self) -> Any:
        """Return the preferred default device from the visible runtime."""
        if self.gpu_devices:
            return self.gpu_devices[0]
        if self._devices:
            return self._devices[0]
        raise RuntimeError("No JAX devices are visible in the active runtime")

    def get_device_info(self) -> dict[str, Any]:
        """Return a structured snapshot of the active runtime."""
        default_device = None
        if self._devices:
            default_device = str(self.get_default_device())

        return {
            "backend": self.capabilities.default_backend,
            "capabilities": self.capabilities,
            "jax_devices": [str(device) for device in self._devices],
            "gpu_devices": [str(device) for device in self.gpu_devices],
            "cpu_devices": [str(device) for device in self.cpu_devices],
            "tpu_devices": [str(device) for device in self.tpu_devices],
            "default_device": default_device,
        }

    def distribute_data(
        self,
        data: Any,
        target_devices: tuple[Any, ...] | list[Any] | None = None,
    ) -> list[Any]:
        """Distribute a batch across the selected visible devices."""
        jax = _get_jax()

        if target_devices is None:
            selected_devices = self.gpu_devices if self.gpu_devices else self._devices
        else:
            selected_devices = tuple(target_devices)

        if not selected_devices:
            raise RuntimeError("No JAX devices are visible in the active runtime")

        if len(selected_devices) == 1:
            return [jax.device_put(data, selected_devices[0])]

        batch_size = data.shape[0]
        per_device = batch_size // len(selected_devices)

        shards: list[Any] = []
        for index, device in enumerate(selected_devices):
            start = index * per_device
            stop = (index + 1) * per_device if index < len(selected_devices) - 1 else batch_size
            shards.append(jax.device_put(data[start:stop], device))
        return shards


def get_device_manager() -> DeviceManager:
    """Construct a fresh runtime device manager."""
    return DeviceManager()


def has_gpu() -> bool:
    """Return True when the active runtime exposes a GPU."""
    return DeviceManager().has_gpu


def get_default_device() -> Any:
    """Return the preferred default device from the active runtime."""
    return DeviceManager().get_default_device()


def print_device_info() -> None:
    """Log a concise device summary for the active runtime."""
    manager = DeviceManager()
    info = manager.get_device_info()

    logger.info("Artifex Device Manager")
    logger.info("Backend: %s", info["backend"])
    logger.info("Device Type: %s", manager.capabilities.device_type.value)
    logger.info("Device Count: %s", manager.device_count)
    logger.info("GPU Count: %s", manager.gpu_count)
    logger.info("Default Device: %s", info["default_device"])
    logger.info("Visible Devices: %s", ", ".join(manager.capabilities.visible_devices) or "none")

    if manager.capabilities.total_memory_mb is not None:
        logger.info("GPU Memory (MB): %s", manager.capabilities.total_memory_mb)
    if manager.capabilities.cuda_version:
        logger.info("CUDA Version: %s", manager.capabilities.cuda_version)
    if manager.capabilities.compute_capability:
        logger.info("Compute Capability: %s", manager.capabilities.compute_capability)
    if manager.capabilities.error:
        logger.info("Runtime Error: %s", manager.capabilities.error)
