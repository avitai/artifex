# Changelog

All notable changes to artifex are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `Trainer.evaluate`, and `Trainer.train` with `val_data`, raised `KeyError('valid_mask')` on a
  fresh install of 0.1.14: its `datarax>=0.1.14` floor resolves datarax 0.1.16, which pads no
  batch and so carries no mask. Evaluation now serves every record once, the last batch holding
  the records left, and weights each batch's metrics by its rows (`datarax.core.spec.batch_length`),
  so the result is still the per-record average.
- `ProteinDataset` implements datarax's `get_records(indices)` in place of `get_batch_at`, which
  datarax 0.1.16 no longer calls; its `supports_indexed_access` override goes, since datarax now
  infers indexed access from `get_records`.
- A training curve recorded with array scalars (a trainer's `jnp.float32` loss) reads back as
  numbers: `TrainingCurvePoint.from_record` reads its values with calibrax's `read_metadata`.
  The plots read the curve through `TrainingCurvePoint` too, so they accept what the benchmark
  wrote and refuse the same malformed records it would.

### Changed

- Requires `datarax>=0.1.16` and `calibrax>=0.1.14`. The relock moves datarax from 0.1.14 and
  calibrax from 0.1.9 (datarax 0.1.16 requires calibrax 0.1.11 or later), and adds lazy-loader;
  every artifex module imports under them.
- A training split smaller than one batch is refused by datarax when the pipeline is built
  (`drop_last needs batch_size <= len(source)`); `Trainer.train`'s own check, which could no
  longer fire, is gone.
- `ProductionOptimizer`, `ProductionPipeline` and `create_production_optimizer` take a calibrax
  `HardwareSpec` (calibrax 0.1.10 made specs records, not dictionaries). Without one, the
  optimizer takes `calibrax.profiling.resolve_hardware_spec(dtype=float32)`: the published spec
  of a chip calibrax lists, else the device's measured ceilings. It used
  `detect_hardware_specs()`, which calibrax 0.1.10 made return `None` for an unlisted device such
  as a CPU, so its roofline analysis was refused there. The spec table's names are calibrax's
  (`a100_sxm4_80gb`, `h100_sxm`, ...); `examples/verify_examples.py` follows.
- `TrainingCurvePoint` is a frozen, keyword-only record that owns its metadata format:
  `to_record()` writes a point into `BenchmarkResult.metadata["training_curve"]` and
  `from_record()` reads one back. `training_curve_from_metadata(metadata)` reads a whole curve.
- `plot_training_curve`, `plot_optimizer_comparison` and `plot_convergence_speed` read their
  metadata through that layer and calibrax's `read_metadata`, and save through the figure they
  drew (`fig.savefig`) rather than matplotlib's current figure. A malformed optimizer result or
  config is refused with its path (`optimizer_configs[1].name`). Saving no longer wraps
  matplotlib's error in an `OSError`, and the `isinstance` checks on the annotated `result`
  argument are gone.
- `adapt_model` returns an `NNXGenerativeModelAdapter`: the registry is calibrax 0.1.14's
  `AdapterRegistry[NNXGenerativeModelAdapter]`, and `register_adapter` takes an adapter class of
  that family (`AdapterClass[TargetT, NNXGenerativeModelAdapter]`). `can_adapt` is a
  `TypeGuard`; `ProteinPointCloudAdapter.can_adapt` returns `TypeGuard[nnx.Module]`.

## [0.1.14] - 2026-09-22

### Added

- `artifex.generative_models.core.sampling.effective_sample_size(samples, *, chain_axis=None,
  sample_axis=0)`: the draws the chains are worth per parameter, from
  `blackjax.diagnostics.effective_sample_size`, with the parameter shape kept where BlackJAX
  squeezes a length-one axis. An antithetic chain reads as worth more than its length, bounded
  by the `draws * log10(draws)` guard on the estimator that Vehtari et al. (2021, Bayesian
  Analysis 16(2), 667, section 3.2) describe. A component whose draws never move reads 0.0 and
  one holding a NaN reads NaN, where BlackJAX 1.6.2 reports the guard value for both: 6602 for
  2000 constant draws. BlackJAX repairs that on main; this answers the two components until it
  is released. Consumers read chain diagnostics here instead of importing BlackJAX.

### Changed

- Requires `blackjax>=1.6.2`; the lock moves it from 1.3 and drops the twelve dependencies
  1.3 pulled in and 1.6.2 does not need (`jaxopt`, `python-fasthtml`, `fastlite`, `apsw` and
  the rest). The sampler wrappers pass unchanged against it.

## [0.1.13] - 2026-09-21

### Changed

- Requires `substrax>=0.1.16`, for what artifex writes rather than what it calls: its checkpoint
  imports are unchanged, but an older substrax stamps a format number the ecosystem no longer
  reads. The lock moves substrax from 0.1.11 and nothing else.

- Requires `datarax>=0.1.14` and `calibrax>=0.1.9`, the latest releases; the lock moves
  datarax from 0.1.13 and calibrax from 0.1.8 and nothing else. Both releases raise their
  substrax floor to 0.1.11, which artifex already requires, and change no API artifex uses.

### Removed

- `TRAINER_FORMAT2`, and reading a checkpoint root written by artifex 0.1.10 or earlier.
  substrax reads one checkpoint format, so the layout that split the older trainer payload into
  items has nothing to describe and `Trainer.load_checkpoint` reads the checkpoint as written.
  With it go the fixture generator, the pinned environment that installed an old substrax to
  write one, the CI step that ran that before three jobs, and the tests and contracts over them.

### Security

- The lock moves anyio from 4.12.1 to 4.14.2 for CVE-2026-63374 and CVE-2026-64847; nothing else moves. 4.14.2 is the first fixed release; 4.15.1 needs typing-extensions 4.16.0, which a single-package upgrade does not allow to move.

## [0.1.12] - 2026-09-18

### Changed

- Requires `substrax>=0.1.11`; the lock moves it from 0.1.10. 0.1.11 caps jax below 0.11.2,
  whose renamed `jax.experimental.hijax.HiPrimitive` flax 0.12.9 imports at module load; a
  resolver given `substrax>=0.1.10` keeps jax 0.11.2 and picks 0.1.10 instead, so a fresh
  install of artifex resolved the failing pair until the floor moved.
- `Trainer.checkpoint_state` returns substrax's format-3 items `model`, `optimizer`, `rng`
  and `extensions` (the optimizer state was under `opt_state`); `save_checkpoint` writes them
  through the store with artifex as the record's producer and returns the checkpoint's
  directory as a `Path`; `load_checkpoint` restores onto the items as templates, reads a root
  written by artifex 0.1.10 or earlier through `TRAINER_FORMAT2` (which
  `substrax.checkpoint.upgrade_checkpoints` also takes to rewrite such a root), and raises
  `substrax.checkpoint.CheckpointNotFoundError`, a `FileNotFoundError`, for a step the
  directory does not hold. A checkpoint written before 0.1.9 restores into no current
  trainer: 0.1.9 moved the optimizer onto substrax's transformation and the optimizer state
  tree changed with it, which that release did not state.
- The configuration owns checkpointing. `TrainingConfig.checkpoint_dir` is `Path | None`:
  `None` puts the checkpoints under `workdir/checkpoints` when the trainer has a workdir,
  through `substrax.checkpoint.resolve_checkpoint_dir`, and nothing creates the directory
  before the first save; a trainer with neither checkpoints nowhere (`train` saves nothing;
  `save_checkpoint` and `load_checkpoint` raise `ValueError`), so it never writes into the
  working directory on its own, where a second run of the same script used to collide with
  the first one's steps. `save_frequency` is the cadence of both `train` and `train_epoch`;
  `max_checkpoints` is how many steps the store keeps. The trainer used to ignore
  `checkpoint_dir` and `max_checkpoints`, keep every checkpoint, and save on a
  `save_interval` of its own in `train`.
- `ModelCheckpoint` saves the model as the `model` item at the trainer's global step, with
  the monitored metric in the record's `metrics` and the epoch in its `epoch`, and picks
  `best_step(monitor, mode=...)`; the trainer it drives exposes `step`
  (`CheckpointingTrainer`), the callback creates no directory before its first save, and
  `CheckpointConfig.dirpath` is required (it defaulted to `checkpoints` under the working
  directory).
- Requires `substrax>=0.1.10` and `datarax>=0.1.13`; the lock moves substrax from 0.1.9 and
  datarax from 0.1.12.

### Removed

- `Trainer(checkpoint_dir=..., save_interval=...)`; `TrainingConfig.checkpoint_dir` and
  `save_frequency` are the one owner of both.

### Security

- The lock moves cryptography from 49.0.0 to 50.0.1 for GHSA-g6cj-pr64-35w5 and PYSEC-2026-3552.
  mlflow 3.15.2 required cryptography below 50, so mlflow (the `logging` extra, with its skinny
  and tracing wheels) moves to 3.16.1, which also leaves the affected range of PYSEC-2026-3865
  (affected through 3.15.2). Both reviewed ignores come out of the security policy; nothing
  under `src/` imports either package.
- The lock moves jupyter-server (the `dev` extra's `jupyter`) from 2.20.0 to 2.21.1 for
  CVE-2026-86049. mlflow 3.15.2's PYSEC-2026-3865, a flaw in the tracking server's gateway
  handler that has no fixed release, is a reviewed ignore in the security policy: nothing
  under `src/` imports mlflow, which the `logging` extra carries for a tracking client.

## [0.1.11] - 2026-09-17

### Changed

- `Trainer.train` drops the epoch's ragged final batch (`drop_last=True`, PyTorch's rule),
  so no padded row reaches a gradient step; `trainer.steps_per_epoch` is the batches the
  data holds (`None` before training) rather than a guessed 100, and a schedule whose
  horizon is not configured (`total_steps` for linear, polynomial and one-cycle schedules,
  `cycle_length` for cosine) spans the run, `num_epochs` times that count; such a trainer
  builds its optimizer when `train` runs, and a `train_step` or `checkpoint_state` before
  that raises naming the missing field. `trainer.schedule` and `trainer.schedule_horizon`
  expose the schedule in use. Training data with fewer records than one batch is refused.
- `Trainer.evaluate` cuts the padded rows of the last batch before the objective sees it
  and weighs each batch's metrics by its records, so the result is the average over the
  data alone.
- `Trainer.train_epoch(steps=None)` runs until `train_data_loader`'s iterator is exhausted,
  or for `steps` batches; it used to take a fixed 100 batches.
- `create_data_pipeline(..., drop_last=False)` exposes datarax's final-batch policy; every
  batch a pipeline serves carries a `valid_mask` leaf.
- Requires `datarax>=0.1.12`; the lock moves it from 0.1.11.

### Removed

- `DataConfig.drop_remainder`, which nothing read; the final-batch policy is the
  pipeline's `drop_last`.

## [0.1.10] - 2026-09-17

### Changed

- Every loss reduces through calibrax's `calibrax.metrics.reduce_values`, so `weights`,
  `reduction` and `axis` mean the same thing in every artifex and calibrax loss. A weighted
  mean is now the weighted mean `sum(w * x) / sum(w)`; it used to be `mean(w * x)`, so a
  weighted loss whose weights do not average to one changes value by that factor.
- `mse_loss`, `mae_loss` and `huber_loss` are calibrax's `mse`, `mae` and `huber_loss` with
  artifex's positional `reduction`, on every path rather than only the default one, and
  `psnr_loss` takes its per-axis MSE from calibrax.
- `measure_inference_latency` times its runs through `calibrax.profiling.time_calls`, so each
  timed call waits for its result and the latency is the compute rather than the dispatch.
- Requires `calibrax>=0.1.8` and `substrax>=0.1.9`; the lock moves both.

### Removed

- `artifex.generative_models.core.losses.base` and its `reduce_loss`; reduce a loss of your
  own with `calibrax.metrics.reduce_values`.
- `charbonnier_loss`; it is `calibrax.metrics.functional.charbonnier_loss`, with the same
  `epsilon` and `alpha` and the shared `mask`, `weights`, `reduction` and `axis` keywords.

## [0.1.9] - 2026-09-17

### Changed

- The optimizer factory builds through `substrax.optim`: `create_optimizer(model, config,
  schedule=None)` maps `OptimizerConfig` onto substrax's specification and returns the
  transformation `substrax.optim.create_transformation` builds for the model, with the
  schedule as the optimizer's learning rate. The model is a new first argument. `nadam` is
  optax's Adam with Nesterov momentum (the previous chain applied plain Adam); `rmsprop` uses
  optax's decay and initial scale (the previous factory passed `beta2` as the decay); a
  `weight_decay` on an optimizer without decoupled decay (`adam`, `sgd`, ...) is refused where
  it was silently ignored; `momentum` reaches `sgd` and `rmsprop` only, and zero means none.
- Every key drawn at a sampling entry point comes through `substrax.rng.key_from`. The
  autoregressive models' `_get_rng_key` takes no default seed: without an owner it raises,
  and a stream the `nnx.Rngs` lacks falls through to its `sample` and `params` streams.
- Requires `substrax>=0.1.8`, the release that carries `substrax.rng` and `substrax.optim`,
  and `datarax>=0.1.11`, the release that takes its own streams through `substrax.rng`; the
  lock moves both.

### Removed

- `artifex.generative_models.core.rng` and `extract_rng_key`; use `substrax.rng.key_from`,
  which takes the same streams and context and raises `MissingRngStreamError`.
- `OptimizerConfig.nesterov` and `OptimizerConfig.initial_accumulator_value`, which substrax's
  specification does not carry; Nesterov momentum is the `nadam` optimizer.

## [0.1.8] - 2026-09-16

### Changed

- Requires `datarax>=0.1.10`; the lock moves datarax from 0.1.8 to 0.1.10 and, through it,
  substrax from 0.1.6 to 0.1.7. artifex reads datarax's sources, pipeline and data-source
  contract, none of which the release changes; its operator contract (the record's key in
  `apply`) reaches no artifex code.
- numpy 2.5 is admitted: the ceiling moves from 2.4 to 2.6, the bound substrax 0.1.7 declares,
  and the lock moves numpy from 2.3.5 to 2.5.3.
- The test suite chooses its JAX backend itself, before JAX is imported: tests run on the CPU
  with eight emulated devices unless `ARTIFEX_TEST_JAX_PLATFORMS` names an accelerator, and
  `ARTIFEX_TEST_DEVICE_COUNT` sets the device count (`0` turns emulation off). An exported
  `JAX_PLATFORMS` no longer moves tests onto a GPU; run
  `ARTIFEX_TEST_JAX_PLATFORMS=cuda uv run pytest` to test on one. See `TESTING.md`.
- GPU-only tests use the substrax pytest plugin's markers:
  `@pytest.mark.accelerator(kind="gpu")` skips unless the run uses a GPU backend, and
  `@pytest.mark.devices(count)` skips below `count` visible devices. The `gpu`, `requires_gpu`,
  `cuda` and `cpu` markers, the `gpu_test_fixture` fixture and `tests/utils/gpu_test_utils.py`
  are removed. Run the GPU tests with `ARTIFEX_TEST_JAX_PLATFORMS=cuda uv run pytest -m accelerator`.
- `ProteinVisualizer.export_to_pdb`, `plot_ramachandran` and `visualize_protein_structure`,
  `plot_optimizer_comparison`, `plot_convergence_speed` and
  `ProteinBenchmarkSuite.visualize_results` write to the path they are given. A relative path is
  no longer moved under `benchmark_results/` or `test_results/`.
- Examples write their figures and files under `$AVITAI_OUTPUT_DIR/<example name>`, or in a
  fresh temporary directory whose path they log, instead of `examples_output/` in the working
  directory. They configure logging only when run as a script or notebook, so importing an
  example leaves logging alone.
- Requires `substrax>=0.1.6`, and the `test` extra installs `substrax[testing]`.
- The generated `.artifex.env`, the pytest environment and the Modal runner set
  `XLA_CLIENT_MEM_FRACTION`, the name jaxlib reads, instead of the deprecated
  `XLA_PYTHON_CLIENT_MEM_FRACTION`. jax refuses a process that sets both; re-running
  `source ./activate.sh` drops the old name from a shell activated before. The pytest
  environment now leaves an exported `XLA_PYTHON_CLIENT_PREALLOCATE` as it is.

### Removed

- `artifex.generative_models.core.jax_config` (`configure_jax`, `auto_configure` and
  `MatmulPrecision`), which was also exported as `artifex.generative_models.jax_config`. Configure
  a JAX process with `substrax.runtime`: build a `JaxRuntime` and pass it to `apply_runtime`, or
  give a process `runtime_environment(...)` before it imports JAX.
- The `ARTIFEX_AUTO_CONFIGURE`, `ARTIFEX_MATMUL_PRECISION` and `ARTIFEX_DETERMINISTIC` environment
  variables. For deterministic GPU runs, set `XLA_FLAGS=--xla_gpu_deterministic_ops=true`; the
  `TF_CUDNN_DETERMINISTIC` and `TF_DETERMINISTIC_OPS` variables the test configuration also set
  are not read by JAX.
- `artifex.utils.file_utils` (`ensure_valid_output_path` and `get_valid_output_dir`), which chose
  an output directory by inspecting the calling function's name. Use
  `substrax.artifacts.resolve_output_dir(name)`, which returns an explicit path,
  `$AVITAI_OUTPUT_DIR/<name>`, or a fresh temporary directory.

### Fixed

- The parallelism guide entered its device meshes with `with mesh:`, which jax deprecates; its
  examples now use `with jax.set_mesh(mesh):`.

## [0.1.7] - 2026-09-11

### Added

- `Trainer.checkpoint_state()` returns the checkpoint pytree (model, optimizer, RNG and
  extension state) that `save_checkpoint` writes, and
  `Trainer.apply_checkpoint_state(payload, step=...)` applies a restored one, so an
  application can store trainer state nested beside its own state in one
  `OrbaxCheckpointStore` checkpoint and validate that checkpoint's metadata before the live
  trainer changes. `load_checkpoint` is now built from the two; the on-disk layout is the
  same. The model and extension entries are live `nnx.State` views, not copies.

### Fixed

- The Trainer API reference documented a pickle checkpoint taking a `path`; it now describes
  the Orbax store keyed by `step` that `save_checkpoint` and `load_checkpoint` use.

## [0.1.6] - 2026-09-09

### Fixed

- `Trainer.train` builds one datarax pipeline per call and starts every later epoch
  with `Pipeline.reset()` (datarax 0.1.8), so each epoch is a permutation of the data in
  a fresh order. It used to build a new `MemorySource` and pipeline every epoch, which
  retraced the pipeline's compiled session each time (about 55 ms per batch on a small
  dataset) and, through the datarax defect fixed in 0.1.8, served overlapping batches.
- `Trainer.train_step` runs its gradient step (base loss, enabled extension losses,
  `nnx.value_and_grad`, the optax update) through one `nnx.jit`-compiled function, traced
  once per batch shape; callbacks, logging and the metric history stay in Python. The
  eager step cost 27 ms for a 2x32x32x1 MLP on CPU against 0.7 ms compiled. For objective
  authors: `loss_fn(model, batch, rng, step)` now runs under `nnx.jit`, so `step` is a
  traced int32 scalar (`jnp` arithmetic, not `int()`), and Python side effects inside the
  objective run at trace time only. `AutoregressiveTrainer.get_teacher_forcing_prob` is
  traceable in `step` and returns a float32 scalar.

## [0.1.5] - 2026-09-09

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
  sibling tests and the lock already resolved; the substrax floor is 0.1.4, the release whose
  W&B logger forwards init options, whose store keeps every checkpoint on request, and whose
  device meshes default to `Auto` axis types (jax 0.11 made `jax.make_mesh` explicit, which
  broke the backward pass of any data-parallel step).
- The datarax floor is 0.1.6, the release whose distributed and checkpoint code is substrax's.
- The calibrax floor is 0.1.5, the release carrying the Fréchet, inception-score, correlation,
  autocorrelation, skewness, RMSD and masked-perplexity functions the evaluations now call.
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

- artifex's own `BenchmarkResult` and the bridge module `benchmarks.core.result_model`
  (`to_calibrax_result`, `from_calibrax_result`, `sanitize_jax_value`, `config_to_dict`).
  `artifex.benchmarks.core.BenchmarkResult` is `calibrax.core.BenchmarkResult`: the
  benchmark name is `name`, the model name is `tags["model_name"]`, `metrics` maps to
  calibrax `Metric` objects, and `save`/`load` use calibrax's JSON layout. Benchmarks build
  results with `Benchmark.result(model_name, metrics, metadata=...)`, which names the result
  after the config and records the config on it, or `benchmark_result(name, model_name,
  metrics, ...)`; `metric_values(result)` reads the `name -> float` view. `timed_run`
  returns a new result carrying `runtime` (results are frozen).
- `artifex.generative_models.core.evaluation` (`FrechetInceptionDistance`, `InceptionScore`,
  `PrecisionRecall`, `DensityPrecisionRecall`, `Perplexity`, `EvaluationPipeline`, the
  `FeatureBasedMetric`/`DistributionMetric`/`SequenceMetric` helpers) with its five docs
  pages and the `core` lazy export. `artifex.benchmarks.metrics` is the one metric class
  layer: `FIDMetric`, `ISMetric`, `PrecisionRecallMetric` (new, with `density_weighted`)
  and `PerplexityMetric` compute through calibrax and register `fid_from_backbone`,
  `inception_score_from_backbone`, `precision_from_backbone`, `recall_from_backbone` and
  `perplexity_from_model` as `frozen_backbone` entries of `calibrax.metrics.MetricRegistry`.
  `ISMetric` reports `inception_score_std` and raises when `splits` exceeds the sample
  count instead of shrinking it; `PerplexityMetric` takes a caller-supplied `model` returning
  token log-probabilities (and a `mask`) in supported mode, and its idle `model_name` is
  gone. The k-means precision-recall heuristic (`KMeansModule`, `compute_precision_recall`,
  the cluster-separation and distance-based scorers) is replaced by the k-NN manifold
  estimator of Kynkäänniemi et al. through calibrax; `PrecisionRecallBenchmark` takes `k`
  instead of `num_clusters`, raises on empty or too-small sample sets, and no longer
  returns fixed scores for a model named `mock_model`. `FIDMetric.compute_statistics`,
  `compute_fid` and the unused `_resize_images` are gone (`compute` is the API); the demo
  Inception mock draws one key so the same images map to the same features. Requires
  calibrax 0.1.5 (masked perplexity).
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
