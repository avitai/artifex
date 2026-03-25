# Device Utilities

**Status:** `Supported runtime utility`
**Module:** `artifex.generative_models.utils.jax.device`
**Source:** `src/artifex/generative_models/utils/jax/device.py`

This page documents the runtime-oriented device helpers layered on top of
`artifex.generative_models.core.device_manager` and
`artifex.generative_models.core.device_testing`.

## Public Helpers

### `verify_device_setup(critical_only: bool = False) -> bool`

Runs the device diagnostics suite and returns its health verdict.

### `get_recommended_batch_size(model_params: int, base_batch_size: int = 32) -> int`

Returns a simple runtime-aware batch-size heuristic based on GPU availability and
model size.
