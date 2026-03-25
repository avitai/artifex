# FID

`FrechetInceptionDistance` lives in
`artifex.generative_models.core.evaluation.metrics.image.fid`.

## Current Contract

- requires a caller-supplied callable `feature_extractor`
- does not ship a built-in Inception-v3 checkpoint or placeholder default
- computes FID from the statistics of extracted real and generated features

## Example

```python
metric = FrechetInceptionDistance(
    feature_extractor=feature_extractor,
    batch_size=32,
    rngs=nnx.Rngs(0),
)
```
