# Artifex Root Examples Guide

This guide covers the maintained root-level workflows under `examples/`. It is a
contributor guide for running, verifying, and updating the checked-in examples
from a source checkout.

For the reader-facing tutorial catalog, use
[`docs/examples/index.md`](../docs/examples/index.md). For relevant but unshipped
or deferred tutorial themes, use
[`docs/roadmap/planned-examples.md`](../docs/roadmap/planned-examples.md).

## Repository Setup

```bash
./setup.sh --backend cpu
source ./activate.sh
```

Examples follow JAX backend defaults. If backend visibility matters for a
GPU-optional workflow, inspect it directly instead of forcing platforms:

```bash
uv run python scripts/verify_gpu_setup.py --json
```

## Reviewed Quickstart Commands

The root guide only publishes commands that point at retained checked-in example
files:

```bash
uv run python examples/generative_models/framework_features_demo.py
uv run python examples/generative_models/image/vae/vae_mnist.py
uv run python examples/generative_models/image/diffusion/diffusion_mnist.py
uv run python examples/generative_models/geometric/simple_point_cloud_example.py
uv run python examples/generative_models/energy/simple_ebm_example.py
```

## Root Maintenance Helpers

### `examples/run_all_examples.sh`

This script runs a small curated smoke subset:

- `examples/generative_models/framework_features_demo.py`
- `examples/generative_models/image/vae/vae_mnist.py`
- `examples/generative_models/geometric/simple_point_cloud_example.py`
- `examples/generative_models/energy/simple_ebm_example.py`

Run it from the repository root after activation:

```bash
./examples/run_all_examples.sh
```

The helper is intentionally narrow. It does not auto-discover every Python file
under `examples/`, and it does not claim coverage for currently open or
specialized example tiers. The diffusion MNIST demo stays in the quickstart
list, but it is excluded from the smoke subset because its visualization path is
longer-running on CPU.

### `examples/verify_examples.py`

The verifier reads the live root README, confirms that every documented
`uv run python examples/...` command resolves to a real file, and then runs a
small set of maintained API smoke checks that back the root examples contract.

Run it with:

```bash
uv run python examples/verify_examples.py
```

## Updating Tutorial Pairs

Reader-facing tutorial examples remain dual-format `.py` / `.ipynb` pairs.
Update them through the documented Jupytext workflow:

```bash
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py
uv run python scripts/jupytext_converter.py validate examples/path/to/
```

See the full authoring rules in
[`docs/development/example-documentation-design.md`](../docs/development/example-documentation-design.md).

## Troubleshooting

- If a documented command fails before model code runs, confirm the referenced
  file still exists and that you are running from the repo root.
- If an example needs GPU visibility, use `scripts/verify_gpu_setup.py` instead
  of forcing backend-selection environment variables in shell setup.
- If you are looking for a tutorial that is not in the root quickstart list,
  check the docs catalog first and the roadmap page second.
