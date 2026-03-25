# Configuration

Artifex uses one typed configuration foundation across the generative stack:

- frozen dataclasses
- `slots=True`
- `kw_only=True`
- validation in `__post_init__`
- typed loading through `from_dict(...)` and YAML helpers

The supported public surface is `artifex.generative_models.core.configuration`, not `core.protocols`.

## Core Types

### `BaseConfig`

`BaseConfig` is the named runtime base for the main model, training, evaluation, and extension config families.

### `ModalityConfig`

`ModalityConfig` is the richer typed document used for modality metadata-driven integrations such as text and molecular modality setup.

### `BaseModalityConfig`

`BaseModalityConfig` is the lightweight runtime base for modality-processing configs used by image, audio, timeseries, tabular, and multi-modal components.

It provides:

- immutable runtime settings
- shared normalization / augmentation / batch-size fields
- cooperative validation through `validate()`
- typed `from_dict(...)` loading

## Modality Runtime Configs

The modality runtime layer now follows the same frozen config contract as the rest of Artifex:

- `ImageModalityConfig`
- `AudioModalityConfig`
- `TimeseriesModalityConfig`
- `TabularModalityConfig`
- `MultiModalModalityConfig`

Sequence-style config fields use tuples at runtime rather than mutable lists.

## Templates

Configuration templates are concrete helpers, not protocols. Artifex uses them through the configuration management layer in `core.configuration.management.templates`.

## Related Pages

- [API Configuration Reference](../api/core/configuration.md)
- [Training Configuration Guide](../user-guide/training/configuration.md)
- [Core Architecture](architecture.md)
