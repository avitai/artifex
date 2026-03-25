# Training Utils

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.utils`

**Source:** `src/artifex/generative_models/training/utils.py`

`artifex.generative_models.training.utils` is an importable shared helper module used by multiple training runtimes. It is narrower than the generated docs previously implied.

## Current Helpers

- `extract_batch_data(batch, keys=("image", "data"))`
- `expand_dims_to_match(arr, target_ndim)`
- `reshape_for_broadcast(arr, batch_size, target_ndim)`
- `sample_logit_normal(key, shape, loc=0.0, scale=1.0)`
- `sample_u_shaped(key, shape)`
- `extract_model_prediction(output, keys=(...))`

These helpers are intended for shared trainer implementations and custom loops that need the same batch-extraction, broadcasting, and time-sampling behavior as the built-in training modules.
