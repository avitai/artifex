# MLflow

**Status:** `Supported runtime utility`
**Module:** `artifex.generative_models.utils.logging.mlflow`
**Source:** `src/artifex/generative_models/utils/logging/mlflow.py`

This page documents the retained MLflow logger integration used by generative-model
training and evaluation workflows.

## Key Symbols

- `MLFlowLogger`
- `log_scalars(...)`
- `log_artifact(...)`
- `log_model(...)`

## Current Scope

The runtime ships this module as a package-local logging integration. It does not
create a broader top-level `artifex.utils.logging` namespace.
