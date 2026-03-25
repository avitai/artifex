# Templates

Configuration templates are exposed through the public `artifex.configs`
surface and implemented in
`generative_models.core.configuration.management.templates`.

**Canonical module:** `generative_models.core.configuration.management.templates`

## Supported built-ins

- `simple_training`: Generates a typed `TrainingConfig` dictionary with nested optimizer and scheduler defaults.
- `distributed_training`: Generates a typed `DistributedConfig` dictionary for multi-process execution.

Additional templates should be introduced only when they map directly to one
typed configuration class and one stable public API.

## Example

```python
from artifex.configs import template_manager

training_config = template_manager.generate_config(
    "simple_training",
    batch_size=64,
    learning_rate=2e-4,
    scheduler_type="linear",
    total_steps=5000,
)
```
