# Artifex Core Module

This package contains the foundational runtime pieces used by Artifex
generative models: configuration, checkpointing, sampling, evaluation, and the
lower-level training/model protocols.

## Devices

Device identity, explicit placement and hardware batch-size recommendations are
substrax's (`substrax.devices`); nothing in `core` wraps them.

```python
import jax
from substrax.devices import detect_devices

info = detect_devices()
print(info.platform, info.kind, info.count)
print(jax.devices()[0])
```

Runtime diagnostics and backend verification are developer tooling under
`scripts/`:

```bash
source ./activate.sh
uv run python scripts/gpu_utils.py --test
uv run python scripts/verify_gpu_setup.py --json
```

## Performance Infrastructure

Hardware specs, FLOP counting and roofline analysis are calibrax's
(`calibrax.profiling`); `core` keeps checkpointing, gradient checkpointing, and
the shared generative-model protocols.
