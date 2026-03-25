# Extensions

**Status:** `Supported runtime extension owner`

**Module:** `artifex.generative_models.extensions.base.extensions`

**Source:** `src/artifex/generative_models/extensions/base/extensions.py`

Shared base classes for attaching domain-specific behavior to otherwise
modality-agnostic generative models.

## Top-Level Module Exports

- `Extension`
- `ModelExtension`
- `ConstraintExtension`
- `AugmentationExtension`
- `SamplingExtension`
- `LossExtension`
- `EvaluationExtension`
- `CallbackExtension`
- `ModalityExtension`
- `ExtensionDict`

## Imported Config Types

Imported config types remain owned by `artifex.generative_models.core.configuration`.

## Class APIs

### `Extension`

- `is_enabled()`

### `ModelExtension`

- `loss_fn()`

### `ConstraintExtension`

- `validate()`
- `project()`

### `AugmentationExtension`

- `augment()`

### `SamplingExtension`

- `modify_score()`
- `filter_samples()`

### `LossExtension`

- `compute_loss()`
- `get_weight_at_step()`

### `EvaluationExtension`

- `compute_metrics()`

### `CallbackExtension`

- `on_train_begin()`
- `on_epoch_end()`
- `on_batch_end()`

### `ModalityExtension`

- `preprocess()`
- `postprocess()`
- `get_input_spec()`

### `ExtensionDict`

- `__contains__()`
