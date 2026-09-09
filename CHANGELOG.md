# Changelog

All notable changes to artifex are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation pages for the two backbone families no artifex model uses yet, the
  Kolmogorov-Arnold Network layers and the Clifford-algebra layers, listing every export
  with its input layout; the layers package docstring states the backbone rule that keeps
  them here.
- `scripts/derive_status.py` measures the model-family surface (each family package, its
  documented exports, and whether it ships a trainer, default configs, tests and examples)
  and fails the quality gate when `docs/models/index.md` names a package or export the
  tree does not have, or omits a family package. Its first run found the audio family
  missing from the table.
- A weekly `upstream-compat.yml` workflow runs the unit test subset against the newest
  jax, flax, optax and orbax-checkpoint releases ahead of the lock. It gates nothing; it
  is where a change like flax 0.12.9 forwarding `is_causal` surfaces first.

### Fixed

- Every value is defined on every path in the source tree: the spectral-norm power
  iteration requires at least one round and says so (`ValueError`) instead of failing on
  an unbound name, an unknown timeseries aggregation or decomposition method raises
  `ValueError` instead of leaving a component undefined, and the ODE and SDE samplers,
  the DiT sampler, the neural spline flow, the graph model, the EGNN layer, the KAN
  initialiser, the benchmark trainer and the file logger no longer read names that
  exist on only one branch.

### Changed

- artifex depends on `substrax` (device identity, meshes, SPMD helpers, the checkpoint
  store, callbacks and trackers shared across the Avitai libraries) and raises its floors
  to `calibrax>=0.1.3`, `jax>=0.11.1` and `orbax-checkpoint>=0.11.33`, the versions every
  sibling tests and the lock already resolved; the substrax floor is 0.1.3, the release whose
  W&B logger forwards init options and whose store keeps every checkpoint on request.
- The datarax floor is 0.1.6, the release whose distributed and checkpoint code is substrax's.
- The calibrax floor is 0.1.4, the release carrying the Fréchet, inception-score, correlation,
  autocorrelation, skewness and RMSD functions the evaluations now call.
- `typer`, `trimesh` and `graphviz` leave the runtime dependencies for the `cli`,
  `geometric` and `analysis` extras (`benchmarks` includes the first two, `dev` the
  last, and `test` all three). Importing `artifex.cli`, `artifex.benchmarks` or
  `artifex.generative_models.utils.code_analysis` without the package raises an
  `ImportError` that names the extra; `artifex` and `artifex.generative_models` never
  needed them.
- Pyright runs in `standard` mode and blocks through the pre-commit hook CI runs;
  the policy table, the contracts and the README say so. The hook set gains
  import-linter (a second contract: `core.layers` never imports `models`), pydoclint
  against a checked-in baseline that can only shrink, and validate-pyproject; the
  ruff hook runs the locked ruff. `hypothesis` joins the test extra.
- Ruff selects the rule set shared with datarax (`ANN`, `ARG`, `B`, `C90`, `PLR`, `PTH`,
  `RET`, `TRY`, with the same five documented exemptions and mccabe at 10). The 850
  file-rule pairs `src/` carried at adoption are recorded in `quality/ruff_baseline.json`
  and rendered into the per-file-ignores table; `scripts/check_ruff_baseline.py` fails when
  a pair grows, when a cleared pair is still listed, or when the table drifts from the
  baseline. Tests, examples, scripts, docs, deployment probes and notebooks keep the
  correctness rules only.
- CI runs every pre-commit hook and checks that `uv.lock` is current in the quality
  gate, runs the unit tests and the package build on Python 3.13 as well as 3.12, and
  checks distributions with `twine check --strict`.
- Publishing uses PyPI trusted publishing (OIDC) instead of an API token, with a
  `github-release` dispatch target that creates the GitHub Release for an existing
  tag and then uploads; `RELEASING.md` records the checklist.

### Removed

- The evaluation helper modules `core.evaluation.metrics.{metric_ops,statistical,distance,
  quality,information}` with their pages. Their computations are calibrax's:
  `kolmogorov_smirnov_distance`, `correlation_preservation`, `distance_to_closest_record`,
  `memorization_rate`, `autocorrelation`, `skewness`, `frechet_distance`,
  `frechet_feature_distance`, `pairwise_rmsd`; the tabular, timeseries, image, text and
  molecular evaluations call them directly (the tabular chi-square statistic stays with the
  tabular suite, and the edge-weighted MSE that stood in for LPIPS is now the demo-mode
  stand-in inside `benchmarks.metrics.image`, named as such). The tabular guards changed:
  correlation preservation is 1.0 below two shared numerical features (was a constant
  0.85 with none), and the memorization rate is 0.0 with no discrete feature (was 1.0,
  every record matching vacuously). Requires calibrax 0.1.4.
- `core.checkpointing` (`setup_checkpoint_manager`, `save_checkpoint`, `load_checkpoint`,
  `save_checkpoint_with_optimizer`, `load_checkpoint_with_optimizer`, `validate_checkpoint`,
  `recover_from_corruption`) with its tests, its page and the `core` lazy exports.
  Checkpoints go through substrax's `OrbaxCheckpointStore`: `ModelCheckpoint` saves into a
  store whose newest `save_top_k` checkpoints are the best because it saves only on
  improvement, and `Trainer.save_checkpoint(step=None)` / `load_checkpoint(step=None)`
  persist the model, optimizer, RNG and extension state as one step-addressed Orbax
  payload instead of a pickle file (`load_checkpoint` raises `FileNotFoundError` when the
  step is absent). The guides teach the store; validation is a restore-and-compare, and
  recovery a loop from the newest readable step.
- `core.performance` (`HardwareSpecs`, `RooflineMetrics`, `HardwareDetector`,
  `PerformanceEstimator`) with its tests and page. Hardware specs, FLOP counting and
  roofline analysis are calibrax's `calibrax.profiling`; `ProductionOptimizer`,
  `ProductionPipeline` and `create_production_optimizer` take a calibrax specification
  dict (`peak_flops`, `memory_bandwidth`, `critical_intensity`) and detect one through
  `detect_hardware_specs()` by default. `PerformanceEstimator` had no consumer.
- `core.device_manager` and `core.device_testing` (`DeviceManager`, `DeviceCapabilities`,
  `DeviceType`, `get_device_manager`, `get_default_device`, `has_gpu`, `print_device_info`,
  `run_device_tests`, `print_test_results`, `TestSuite`, `TestResult`, `TestSeverity`), with
  their three docs pages and the `core` lazy exports. Device identity and placement are
  substrax's `substrax.devices`; `utils.jax.device` keeps one helper,
  `get_recommended_batch_size(model_params, base_batch_size=None)`, which scales substrax's
  hardware batch size by model size (`verify_device_setup` is gone). The runtime diagnostics
  are developer tooling: `scripts/gpu_utils.py --test` and `--test-critical`.
- `utils.logging.logger`, `utils.logging.wandb`, `utils.logging.mlflow` and `core.logging`,
  with their four docs pages. `Logger`, `ConsoleLogger`, `FileLogger`, `WandbLogger`,
  `MLFlowLogger` and `create_logger` are substrax's `substrax.tracking`, re-exported by
  `artifex.generative_models.utils.logging`; `MetricsLogger` stays and takes one of them.
  The substrax loggers take no `**kwargs`, `create_logger` has no `log_to_console`,
  `WandbLogger` has no `anonymous` or `console_log`, `MLFlowLogger` has no `console_log` or
  `log_model`, and `log_dir` attributes are `Path`s. `WandbLoggerCallback` is a
  `LoggerCallback` over `WandbLogger` (its config extends `LoggerCallbackConfig`, so it
  gains `prefix`; the run name defaults to the project name).
- `training.callbacks.base` and `training.callbacks.early_stopping`, with their two docs
  pages. `BaseCallback`, `CallbackList`, `TrainerLike`, `TrainingCallback`,
  `EarlyStoppingConfig` and the early-stopping callback are substrax's and are re-exported
  by `artifex.generative_models.training.callbacks`; the callback is now named
  `EarlyStoppingCallback` (substrax's `EarlyStopping` is the best-metric tracker it is
  built on).
- `artifex.generative_models.scaling` (`mesh_utils`, `sharding`: `ShardingConfig`,
  `ParallelismConfig`, `ShardingStrategy`, `DataParallelStrategy`, `FSDPStrategy`,
  `TensorParallelStrategy`, `PipelineParallelStrategy`, `MultiDimensionalStrategy`) and
  `artifex.generative_models.training.distributed` (`DeviceMeshManager`, `DataParallel`,
  `DevicePlacement`, `HardwareType`, `BatchSizeRecommendation`, `place_on_device`,
  `distribute_batch`, `get_batch_size_recommendation`, `DistributedMetrics`), with their
  tests and nine docs pages. The same code, moved with its tests, is `substrax.mesh`,
  `substrax.spmd` and `substrax.devices`; `ProductionOptimizer` reads `ParallelismConfig`
  from `substrax.mesh`, and the distributed training guide teaches the substrax surface.
- `GradientAccumulator`, `GradientAccumulatorConfig`, `DynamicLossScaler` and
  `DynamicLossScalerConfig`, with the `training.gradient_accumulation` module and its
  two docs pages. Both duplicated upstream-owned tools: wrap the optimizer in
  `optax.MultiSteps(tx, every_k_schedule=k)` for accumulation and differentiate through
  `flax.training.dynamic_scale.DynamicScale` for loss scaling. The Advanced Features
  guide shows both, and tests pin that `k` microbatches through `MultiSteps` equal one
  step on the mean gradient and that `DynamicScale` returns unscaled gradients.

## [0.1.4] - 2026-08-29

Releases up to 0.1.4 predate this file; their notes are the
[GitHub Releases](https://github.com/avitai/artifex/releases).
