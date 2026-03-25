# W&B

**Status:** `Supported runtime utility`
**Module:** `artifex.generative_models.utils.logging.wandb`
**Source:** `src/artifex/generative_models/utils/logging/wandb.py`

This page documents the retained Weights & Biases logger integration used by
generative-model training and evaluation workflows.

## Key Symbols

- `WandbLogger`
- `log_scalars(...)`
- `log_image(...)`
- `log_histogram(...)`

## Current Scope

The runtime ships this module as a package-local logging integration. It does not
create a broader top-level `artifex.utils.logging` namespace.
