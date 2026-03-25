# Inception Score

`InceptionScore` lives in
`artifex.generative_models.core.evaluation.metrics.image.inception_score`.

## Current Contract

- requires a caller-supplied callable `classifier`
- does not ship a built-in Inception-v3 classifier or placeholder default
- reports `<name>_mean` and `<name>_std` from the configured split count

## Example

```python
metric = InceptionScore(
    classifier=classifier,
    batch_size=32,
    splits=10,
    rngs=nnx.Rngs(0),
)
```
