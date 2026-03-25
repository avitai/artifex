# Utilities

The top-level `artifex.utils` namespace is intentionally narrow. The current
utility docs cover only the small set of live modules that still ship in the repo.

Most other helpers now live with their owning package instead of a shared
top-level utility umbrella.

## Current Utility Pages

- [Device Utilities](device.md) for `artifex.generative_models.utils.jax.device`
- [Logger](logger.md) for `artifex.generative_models.utils.logging.logger`
- [Metrics](metrics.md) for `artifex.generative_models.utils.logging.metrics`
- [MLflow](mlflow.md) for `artifex.generative_models.utils.logging.mlflow`
- [W&B](wandb.md) for `artifex.generative_models.utils.logging.wandb`
- [Protein Visualization Compatibility](protein.md) for the canonical
  `artifex.visualization.protein_viz` owner and its compatibility alias
- [Dependency Analyzer](dependency_analyzer.md) for
  `artifex.generative_models.utils.code_analysis.dependency_analyzer`
- [File Utils](file_utils.md) for `artifex.utils.file_utils`

```python
from artifex.utils.file_utils import get_valid_output_dir

output_dir = get_valid_output_dir("code_analysis", "reports")
```

## Coming Soon

These pages cover still-relevant utility modules that are planned but not shipped yet.
They are not supported API docs yet.

See [Planned Modules Roadmap](../roadmap/planned-modules.md#utilities) for the
current status of the coming-soon utility families.
