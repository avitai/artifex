# Perplexity

`Perplexity` lives in
`artifex.generative_models.core.evaluation.metrics.text.perplexity`.

## Current Contract

- accepts a caller-supplied callable `model` when you want the metric to
  compute log probabilities from token inputs
- also supports direct `log_probs` input at compute time when the caller
  already owns the language-model forward pass
- lower perplexity remains better

## Example

```python
metric = Perplexity(model=language_model, batch_size=32, rngs=nnx.Rngs(0))
result = metric.compute(inputs=token_ids)
```
