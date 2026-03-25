# Registry

Registry ownership for evaluation metrics lives in CalibraX.

## Current Contract

- use `calibrax.metrics.MetricRegistry` for registry-backed metric lookup
- use CalibraX metric entries, collections, or suites when you need grouped
  metric execution
- Artifex does not ship a second local registry in
  `core.evaluation.metrics`

## Example

```python
from calibrax.metrics import MetricEntry, MetricRegistry, MetricTier


registry = MetricRegistry()
registry.register(
    "custom_score",
    MetricEntry(
        name="custom_score",
        fn=lambda predictions, targets: 1.0,
        tier=MetricTier.PURE_FUNCTION,
        domain="testing",
    ),
)
score_fn = registry.get_function("custom_score")
result = score_fn([1], [1])
```
