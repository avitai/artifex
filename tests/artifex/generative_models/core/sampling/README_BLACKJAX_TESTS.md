# BlackJAX Integration Tests

BlackJAX is a first-class dependency in Artifex. These tests are part of the normal
pytest contract and should run without extra environment-variable toggles.

## Running the Tests

```bash
# Run the full suite
uv run pytest tests/

# Run only the BlackJAX-marked tests
uv run pytest tests/ -m blackjax

# Run this module only
uv run pytest tests/artifex/generative_models/core/sampling/test_blackjax_samplers.py -v
```

If you need a focused local run that excludes them, use standard pytest selection:

```bash
uv run pytest tests/ -m "not blackjax"
```

## Notes

- The `blackjax` marker is for selection, not default skipping.
- Some statistical tests are intentionally `xfail` where behavior depends on sampling variance.
- If a test is slow or memory-intensive, that should be handled with normal pytest markers or
  test design, not a hidden global gate.
