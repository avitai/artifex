# Configuration Validation Helpers

Supported owner: `artifex.generative_models.core.configuration.validation`

## Overview

This module contains low-level validation helpers used by typed config
Dataclasses in their `__post_init__` hooks. It is not a high-level
compatibility or migration layer.

## Retained Helpers

The live module currently provides helpers such as:

- `validate_positive_int`
- `validate_non_negative_int`
- `validate_positive_float`
- `validate_non_negative_float`
- `validate_positive_tuple`
- `validate_positive_int_tuple`
- `validate_dropout_rate`
- `validate_probability`
- `validate_range`
- `validate_learning_rate`
- `validate_activation`

## Typical Imports

```python
from artifex.generative_models.core.configuration.validation import (
    validate_activation,
    validate_non_negative_float,
    validate_positive_int,
)
```

Use these helpers inside concrete config classes instead of reviving deleted
compatibility shims.

## Related Pages

- [Configuration Overview](configuration.md)
- [Unified Configuration Backend](unified.md)
- [API Configuration Reference](../api/core/configuration.md)
