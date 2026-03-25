# Artifex Examples

This directory is the contributor-facing root contract for the checked-in example
surface. The reader-facing tutorial catalog lives in
[`docs/examples/index.md`](../docs/examples/index.md), and still-relevant topics
that are not shipped as runnable pairs belong in
[`docs/roadmap/planned-examples.md`](../docs/roadmap/planned-examples.md).

## Scope

Use this root surface when you are working from a source checkout. It documents:

- how contributors run retained examples from the repo root
- which helper workflows are maintained at the root level
- where to find the canonical tutorial catalog and authoring guide

The commands below assume you are in the repository root.

## Setup

```bash
./setup.sh --backend cpu
source ./activate.sh
```

Let JAX choose the available backend by default. If you want to inspect the live
backend before running a GPU-optional example, use:

```bash
uv run python scripts/verify_gpu_setup.py --json
```

## Quickstart Examples

These are the maintained root-level starter commands for the reviewed example
surface:

```bash
uv run python examples/generative_models/framework_features_demo.py
uv run python examples/generative_models/image/vae/vae_mnist.py
uv run python examples/generative_models/image/diffusion/diffusion_mnist.py
uv run python examples/generative_models/geometric/simple_point_cloud_example.py
```

For the broader tutorial catalog, start from
[`docs/examples/index.md`](../docs/examples/index.md).

## Maintained Root Helpers

### Curated Smoke Subset

```bash
./examples/run_all_examples.sh
```

This helper runs a curated CPU-safe smoke subset of reviewed examples. It is not
an auto-discovery runner for every file under `examples/`.

### Root Contract Smoke Checks

```bash
uv run python examples/verify_examples.py
```

This verifier reads the live `examples/README.md`, checks that documented root
example commands resolve to real files, and then runs a small set of maintained
API smoke checks. It does not claim to execute the entire example catalog.

## Authoring And Maintenance

- Reader-facing tutorial examples are maintained as `.py` / `.ipynb` pairs.
- Contributor-facing design rules live in
  [`docs/development/example-documentation-design.md`](../docs/development/example-documentation-design.md).
- Use the Jupytext sync workflow from that guide when you update a tutorial pair.

Useful maintenance commands:

```bash
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py
uv run python scripts/jupytext_converter.py validate examples/path/to/
```

## Catalog Boundaries

Some example families are intentionally outside the retained root quickstart and
smoke-helper surface because they are still being repaired or are specialized
workflows with their own docs entry points. Use the published docs catalog for
supported reader-facing navigation, and use the roadmap page for unshipped or
coming-soon example themes.
