# Device Utilities

**Status:** `Supported runtime utility`
**Module:** `artifex.generative_models.utils.jax.device`
**Source:** `src/artifex/generative_models/utils/jax/device.py`

Device identity, explicit placement and the per-hardware batch-size table are
substrax's `substrax.devices` (`detect_devices`, `DeviceInfo`, `DevicePlacement`,
`get_batch_size_recommendation`). This module keeps the one artifex-specific rule
on top of that table.

## Public Helpers

### `get_recommended_batch_size(model_params: int, base_batch_size: int | None = None) -> int`

Returns a batch size for the detected hardware scaled by model size: the starting
point (substrax's optimal batch size for the hardware unless `base_batch_size` is
given) is halved above 100M parameters and doubled below 1M, never below 1 and
never above the memory ceiling substrax estimates for the hardware.

```python
from artifex.generative_models.utils.jax.device import get_recommended_batch_size

batch_size = get_recommended_batch_size(model_params=sum(p.size for p in params))
```

## Diagnostics

Runtime diagnostics (basic arithmetic, an NNX forward and gradient pass, a matrix
multiplication, attention-style ops and a memory allocation) are developer tooling:

```bash
uv run python scripts/gpu_utils.py --test            # every check
uv run python scripts/gpu_utils.py --test-critical   # the two critical checks
uv run python scripts/gpu_utils.py --detailed        # device inventory from substrax
```
