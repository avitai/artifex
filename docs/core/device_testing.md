# Device Testing

`artifex.generative_models.core.device_testing` provides a small runtime
diagnostic surface for the active JAX backend.

## Supported API

- `run_device_tests(critical_only: bool = False) -> TestSuite`
- `print_test_results(suite: TestSuite) -> None`
- `TestSeverity`
- `TestResult`
- `TestSuite`

## Design

The module is intentionally runtime-only:

- importing it does not import JAX or Flax eagerly
- diagnostics are plain internal checks, not a custom test framework
- critical failures stop the suite early
- results are returned as immutable dataclasses

## Default diagnostics

The built-in suite verifies:

- basic JAX array computation
- a small Flax NNX forward/gradient path
- matrix multiplication on the active runtime
- generative-model-style attention and noise operations
- optional memory allocation stress

## Example

```python
from artifex.generative_models.core import print_test_results, run_device_tests

suite = run_device_tests()
print_test_results(suite)
```
