# Pipeline

`EvaluationPipeline` is the retained orchestration layer for explicit
evaluation dependencies. It lives in
`artifex.generative_models.core.evaluation.metrics.pipeline` and builds a
small set of supported runtime metrics.

## Current Contract

- metrics must use explicit `modality:metric` specs
- supported metric specs are `image:fid`, `image:is`, and
  `text:perplexity`
- caller-supplied dependencies are required for `feature_extractor`,
  `classifier`, and `model`
- unsupported metric specs raise during initialization instead of being
  skipped
- registry ownership lives in `calibrax.metrics.MetricRegistry`, not in a
  parallel Artifex wrapper

## Supported Payload Shapes

- `image:fid` expects `{"real": ..., "generated": ...}`
- `image:is` expects `{"generated": ...}`
- `text:perplexity` expects `{"inputs": ...}` or `{"log_probs": ...}`
  and may also receive `{"mask": ...}`

## Example

```python
results = pipeline.evaluate(
    {
        "image": {
            "real": real_images,
            "generated": generated_images,
        },
        "text": {
            "inputs": token_ids,
        },
    }
)
```
