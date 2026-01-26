"""
Foundation-first device management system for Artifex.

This module provides a comprehensive, type-safe device management architecture
that prioritizes clean design and robust error handling over backward compatibility.
"""

import os
import subprocess
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import jax
from jax import devices


class DeviceType(Enum):
    """Supported device types."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class MemoryStrategy(Enum):
    """Memory allocation strategies."""

    CONSERVATIVE = "conservative"  # 0.6 memory fraction
    BALANCED = "balanced"  # 0.75 memory fraction
    AGGRESSIVE = "aggressive"  # 0.9 memory fraction
    CUSTOM = "custom"  # User-defined


@dataclass(frozen=True)
class DeviceCapabilities:
    """Immutable device capability information."""

    device_type: DeviceType
    device_count: int
    total_memory_mb: int | None = None
    compute_capability: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    supports_mixed_precision: bool = False
    supports_distributed: bool = False


@dataclass
class DeviceConfiguration:
    """Comprehensive device configuration."""

    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    memory_fraction: float | None = None
    enable_x64: bool = False
    enable_jit: bool = True
    platform_priority: list[str] = field(default_factory=lambda: ["cuda", "cpu"])
    environment_variables: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.memory_fraction is not None:
            if not 0.1 <= self.memory_fraction <= 1.0:
                raise ValueError(
                    f"Memory fraction must be between 0.1 and 1.0, got {self.memory_fraction}"
                )

        # Set default memory fraction based on strategy
        if self.memory_fraction is None:
            strategy_fractions = {
                MemoryStrategy.CONSERVATIVE: 0.6,
                MemoryStrategy.BALANCED: 0.75,
                MemoryStrategy.AGGRESSIVE: 0.9,
                MemoryStrategy.CUSTOM: 0.75,  # Default fallback
            }
            self.memory_fraction = strategy_fractions[self.memory_strategy]


class DeviceDetector(Protocol):
    """Protocol for device detection implementations."""

    def detect_capabilities(self) -> DeviceCapabilities:
        """Detect device capabilities."""
        ...


class CUDADetector:
    """CUDA device detection implementation."""

    def detect_capabilities(self) -> DeviceCapabilities:
        """Detect CUDA device capabilities."""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpu_info = result.stdout.strip().split("\n")
            device_count = len(gpu_info)

            if device_count == 0:
                return DeviceCapabilities(DeviceType.CPU, 1)

            # Parse first GPU info
            first_gpu = gpu_info[0].split(",")
            total_memory = int(first_gpu[1].strip()) if len(first_gpu) > 1 else None
            compute_cap = first_gpu[2].strip() if len(first_gpu) > 2 else None

            # Get CUDA version
            cuda_version = self._get_cuda_version()
            driver_version = self._get_driver_version()

            return DeviceCapabilities(
                device_type=DeviceType.GPU,
                device_count=device_count,
                total_memory_mb=total_memory,
                compute_capability=compute_cap,
                cuda_version=cuda_version,
                driver_version=driver_version,
                supports_mixed_precision=True,
                supports_distributed=device_count > 1,
            )

        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return DeviceCapabilities(DeviceType.CPU, 1)

    def _get_cuda_version(self) -> str | None:
        """Extract CUDA version from nvidia-smi."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    return line.split("CUDA Version:")[1].strip().split()[0]
        except Exception:
            pass
        return None

    def _get_driver_version(self) -> str | None:
        """Extract driver version from nvidia-smi."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            for line in result.stdout.split("\n"):
                if "Driver Version" in line:
                    parts = line.split("|")
                    for part in parts:
                        if "Driver Version" in part:
                            return part.split("Driver Version:")[1].strip().split()[0]
        except Exception:
            pass
        return None


class JAXDeviceManager:
    """JAX-specific device management implementation."""

    def __init__(self, config: DeviceConfiguration):
        """Initialize JAX device manager."""
        self.config = config
        self._configure_environment()
        self._devices_cache: list[jax.Device] | None = None

    def _configure_environment(self) -> None:
        """Configure JAX environment variables."""
        base_env = {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(self.config.memory_fraction),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_ENABLE_X64": "1" if self.config.enable_x64 else "0",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "JAX_PLATFORMS": ",".join(self.config.platform_priority),
            "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",
            "JAX_CUDA_PLUGIN_VERIFY": "false",
            # "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",  # Disabled to catch CUDA version mismatches
        }

        # Merge with user-provided environment variables
        env_vars = {**base_env, **self.config.environment_variables}

        for key, value in env_vars.items():
            os.environ[key] = value

    @property
    def devices(self) -> list[jax.Device]:
        """Get all available JAX devices."""
        if self._devices_cache is None:
            self._devices_cache = devices()
        return self._devices_cache

    @property
    def gpu_devices(self) -> list[jax.Device]:
        """Get GPU devices."""
        return [d for d in self.devices if d.platform == "gpu"]

    @property
    def cpu_devices(self) -> list[jax.Device]:
        """Get CPU devices."""
        return [d for d in self.devices if d.platform == "cpu"]

    def get_default_device(self) -> jax.Device:
        """Get the default device for computations."""
        if self.gpu_devices:
            return self.gpu_devices[0]
        return self.cpu_devices[0]

    def distribute_data(
        self, data: jax.Array, target_devices: list[jax.Device] | None = None
    ) -> list[jax.Array]:
        """Distribute data across devices."""
        if target_devices is None:
            target_devices = self.gpu_devices if self.gpu_devices else self.cpu_devices

        if len(target_devices) == 1:
            return [jax.device_put(data, target_devices[0])]

        # Split data evenly across devices
        batch_size = data.shape[0]
        per_device = batch_size // len(target_devices)

        distributed = []
        for i, device in enumerate(target_devices):
            start_idx = i * per_device
            end_idx = (i + 1) * per_device if i < len(target_devices) - 1 else batch_size
            device_data = jax.device_put(data[start_idx:end_idx], device)
            distributed.append(device_data)

        return distributed


class DeviceManager:
    """Unified device management system."""

    def __init__(
        self,
        config: DeviceConfiguration | None = None,
        detector: DeviceDetector | None = None,
    ):
        """Initialize device manager."""
        self.config = config or DeviceConfiguration()
        self.detector = detector or CUDADetector()
        self.capabilities = self.detector.detect_capabilities()
        self.jax_manager = JAXDeviceManager(self.config)

        # Validate configuration against capabilities
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate configuration against detected capabilities."""
        if (
            self.capabilities.device_type == DeviceType.CPU
            and "cuda" in self.config.platform_priority
        ):
            warnings.warn("CUDA requested but no GPU detected. Falling back to CPU.", UserWarning)

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.capabilities.device_type == DeviceType.GPU

    @property
    def device_count(self) -> int:
        """Get total device count."""
        return len(self.jax_manager.devices)

    @property
    def gpu_count(self) -> int:
        """Get GPU device count."""
        return len(self.jax_manager.gpu_devices)

    def get_device_info(self) -> dict[str, Any]:
        """Get comprehensive device information."""
        return {
            "backend": jax.default_backend(),
            "capabilities": self.capabilities,
            "configuration": self.config,
            "jax_devices": [str(d) for d in self.jax_manager.devices],
            "gpu_devices": [str(d) for d in self.jax_manager.gpu_devices],
            "cpu_devices": [str(d) for d in self.jax_manager.cpu_devices],
            "default_device": str(self.jax_manager.get_default_device()),
        }

    def optimize_for_model_size(self, parameter_count: int) -> DeviceConfiguration:
        """Get optimized configuration for model size."""
        if parameter_count < 1e6:  # < 1M parameters
            strategy = MemoryStrategy.CONSERVATIVE
        elif parameter_count < 1e8:  # < 100M parameters
            strategy = MemoryStrategy.BALANCED
        else:  # >= 100M parameters
            strategy = MemoryStrategy.AGGRESSIVE

        return DeviceConfiguration(
            memory_strategy=strategy,
            enable_x64=False,  # Keep float32 for performance
            platform_priority=self.config.platform_priority,
        )


# Global device manager instance
_global_device_manager: DeviceManager | None = None


def get_device_manager(
    config: DeviceConfiguration | None = None, force_reinit: bool = False
) -> DeviceManager:
    """Get global device manager instance."""
    global _global_device_manager

    if _global_device_manager is None or force_reinit:
        _global_device_manager = DeviceManager(config)

    return _global_device_manager


def configure_for_generative_models(
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED, enable_mixed_precision: bool = True
) -> DeviceManager:
    """Configure device manager specifically for generative models."""
    config = DeviceConfiguration(
        memory_strategy=memory_strategy,
        enable_x64=False,  # Float32 for performance
        environment_variables={
            "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",
        },
    )

    return get_device_manager(config, force_reinit=True)


# Convenience functions for backward compatibility (temporary)
def has_gpu() -> bool:
    """Check if GPU is available."""
    return get_device_manager().has_gpu


def get_default_device() -> jax.Device:
    """Get default device."""
    return get_device_manager().jax_manager.get_default_device()


def print_device_info() -> None:
    """Print comprehensive device information."""
    manager = get_device_manager()
    info = manager.get_device_info()

    print("🔍 Artifex Device Manager")
    print("=" * 40)
    print(f"Backend: {info['backend']}")
    print(f"Device Type: {manager.capabilities.device_type.value}")
    print(f"Device Count: {manager.device_count}")
    print(f"GPU Count: {manager.gpu_count}")
    print(f"Memory Strategy: {manager.config.memory_strategy.value}")
    print(f"Memory Fraction: {manager.config.memory_fraction}")
    print(f"Default Device: {info['default_device']}")

    if manager.capabilities.cuda_version:
        print(f"CUDA Version: {manager.capabilities.cuda_version}")
    if manager.capabilities.compute_capability:
        print(f"Compute Capability: {manager.capabilities.compute_capability}")


if __name__ == "__main__":
    # Test the device manager
    print_device_info()

    # Test different configurations
    print("\n🧪 Testing different memory strategies:")
    for strategy in MemoryStrategy:
        config = DeviceConfiguration(memory_strategy=strategy)
        print(f"{strategy.value}: {config.memory_fraction}")
