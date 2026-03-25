# BlackJAX Integration Tests

BlackJAX is a first-class dependency in Artifex. Its tests are part of the normal
pytest surface and should run under the same contract as the rest of the suite.

## Running BlackJAX Tests

```bash
# Run the full suite, including BlackJAX tests
uv run pytest tests/

# Run only the BlackJAX-marked tests
uv run pytest tests/ -m blackjax

# Run the BlackJAX sampler tests only
uv run pytest tests/artifex/generative_models/core/sampling/test_blackjax_samplers.py -v
```

If you need a focused local run that excludes them, use standard pytest selection rather
than hidden environment gates:

```bash
uv run pytest tests/ -m "not blackjax"
```

## Notes

- The `blackjax` marker is for selection, not for default skipping.
- Statistical tests may still use `xfail` where the expected behavior is probabilistic.
- Performance-sensitive local runs should rely on normal pytest target and marker selection.

## CI Guidance

CI should treat BlackJAX tests as part of the main test contract. If a separate job exists
for scheduling or sharding reasons, it should still run normal pytest commands rather than
special opt-in environment toggles.
