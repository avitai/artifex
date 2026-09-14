# Utilities

The top-level `artifex.utils` namespace is intentionally narrow. The current
utility docs cover only the small set of live modules that still ship in the repo.

Most other helpers now live with their owning package instead of a shared
top-level utility umbrella.

## Current Utility Pages

- [Device Utilities](device.md) for `artifex.generative_models.utils.jax.device`
- [Metrics](metrics.md) for `artifex.generative_models.utils.logging.metrics`; the loggers it
  writes through (`Logger`, `ConsoleLogger`, `FileLogger`, `WandbLogger`, `MLFlowLogger`,
  `create_logger`) are substrax's `substrax.tracking`, re-exported by
  `artifex.generative_models.utils.logging`
- [Protein Visualization Compatibility](protein.md) for the canonical
  `artifex.visualization.protein_viz` owner and its compatibility alias
- [Dependency Analyzer](dependency_analyzer.md) for
  `artifex.generative_models.utils.code_analysis.dependency_analyzer`

Output directories come from `substrax.artifacts.resolve_output_dir`, which honours
`AVITAI_OUTPUT_DIR`; artifex keeps no file-output helper of its own.

## Coming Soon

These pages cover still-relevant utility modules that are planned but not shipped yet.
They are not supported API docs yet.

See [Planned Modules Roadmap](../roadmap/planned-modules.md#utilities) for the
current status of the coming-soon utility families.
