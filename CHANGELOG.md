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
