# Artifex Core Module

This package contains the foundational runtime pieces used by Artifex
generative models: configuration, checkpointing, device inspection, sampling,
evaluation, and the lower-level training/model protocols.

## Device Management

`DeviceManager` is a runtime helper, not a backend bootstrap layer.

It provides:

- inspection of the active JAX runtime
- access to visible devices
- default-device selection
- batch sharding across visible devices
- concise device diagnostics through `print_device_info()`

### Basic Usage

```python
from artifex.generative_models.core import DeviceManager

manager = DeviceManager()
print(manager.has_gpu)
print(manager.device_count)
print(manager.capabilities.default_backend)
print(manager.capabilities.visible_devices)
print(manager.get_default_device())
```

### Device Diagnostics

```python
from artifex.generative_models.core.device_testing import (
    print_test_results,
    run_device_tests,
)

suite = run_device_tests()
print_test_results(suite)
```

### Backend Verification

For explicit backend verification, use the dedicated script instead of the
runtime manager:

```bash
source ./activate.sh
uv run python scripts/verify_gpu_setup.py --json
```

Backend setup itself belongs to contributor bootstrap and package installation,
not to `DeviceManager`.

## Performance Infrastructure

The core package also includes performance analysis tools such as
`HardwareDetector` and `PerformanceEstimator`, plus checkpointing,
gradient-checkpointing, and the shared generative-model protocols.
