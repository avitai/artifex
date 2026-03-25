# Device Management

`DeviceManager` is Artifex's runtime-facing device helper. It inspects the
active JAX runtime, reports the visible devices, selects a default device, and
can shard a batch across the currently visible devices.

It does **not** configure CUDA, mutate `JAX_PLATFORMS`, or manage backend
bootstrap. Backend setup belongs to:

- `./setup.sh` and `source ./activate.sh` for contributors
- `pip install "artifex[cuda12]"` for package users
- `uv run python scripts/verify_gpu_setup.py` for explicit runtime verification

## Supported Surface

```python
from artifex.generative_models.core import DeviceManager
from artifex.generative_models.core.device_manager import (
    get_default_device,
    get_device_manager,
    has_gpu,
    print_device_info,
)
```

## Basic Usage

```python
from artifex.generative_models.core import DeviceManager

device_manager = DeviceManager()

print(device_manager.has_gpu)
print(device_manager.device_count)
print(device_manager.gpu_count)
print(device_manager.capabilities.default_backend)
print(device_manager.capabilities.visible_devices)
print(device_manager.get_default_device())
```

## Runtime Snapshot

`DeviceManager.capabilities` is an immutable snapshot of the currently visible
JAX runtime:

- `device_type`
- `device_count`
- `default_backend`
- `visible_devices`
- optional GPU metadata when `nvidia-smi` is available
- `supports_mixed_precision`
- `supports_distributed`
- `error`

This is intentionally runtime-only state. It reflects what JAX can currently
see, not what Artifex wishes the backend looked like.

## Batch Distribution

```python
import jax.numpy as jnp
from artifex.generative_models.core import DeviceManager

manager = DeviceManager()
batch = jnp.ones((64, 784))
shards = manager.distribute_data(batch)
```

`distribute_data()` uses visible GPU devices when available and otherwise falls
back to the visible runtime devices.

## Verification Workflow

Use the explicit verifier when you want to confirm backend setup:

```bash
source ./activate.sh
uv run python scripts/verify_gpu_setup.py --json
```

That verifier is the authoritative place to answer "am I on CPU or GPU right
now?" DeviceManager should be used after the runtime is already configured.
