# Onnx

**Status:** `Coming soon`

`artifex.generative_models.inference.onnx` is not shipped yet.

The shared inference runtime stays intentionally narrow today:

- `artifex.generative_models.inference` exports no public helpers from `__all__`.
- `artifex.generative_models.inference.optimization.production` is the only
  retained shared inference owner.
- loading and generation entrypoints remain family-owned; see
  [Inference Overview](../user-guide/inference/overview.md) and
  [Sampling Guide](../user-guide/inference/sampling.md).

See [Inference Reference](index.md) for the current shared inference docs.
