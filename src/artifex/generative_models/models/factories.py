"""Model factories - DEPRECATED.

All factory functions have been moved to the centralized factory system.

Use:
```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.zoo import zoo
```

This module is kept for backward compatibility only.
"""

import warnings


warnings.warn(
    "artifex.generative_models.models.factories is deprecated. "
    "Use artifex.generative_models.factory instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the new factory for minimal backward compatibility
from artifex.generative_models.factory import create_model


__all__ = ["create_model"]
