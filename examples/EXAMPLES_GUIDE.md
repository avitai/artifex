# Artifex Root Examples Guide

This guide covers the maintained root-level workflows under `examples/`. It is a
contributor guide for running, verifying, and updating the checked-in examples
from a source checkout.

For the reader-facing tutorial catalog, use
[`docs/tutorials/index.md`](../docs/tutorials/index.md). For relevant but unshipped
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

## Updating Tutorial Pairs

Examples follow substrax's
[Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/)
guide. The script is the only file you author; `substrax-notebooks` writes its
notebook and its page:

```bash
uv run substrax-notebooks convert examples/<topic>/<name>.py   # its clean notebook
uv run substrax-notebooks page examples/<topic>/<name>.py      # run it and write its page
uv run substrax-notebooks test                                 # the shared example tests
```

## Troubleshooting

- If a documented command fails before model code runs, confirm the referenced
  file still exists and that you are running from the repo root.
- If an example needs GPU visibility, use `scripts/verify_gpu_setup.py` instead
  of forcing backend-selection environment variables in shell setup.
- If you are looking for a tutorial that is not in the root quickstart list,
  check the docs catalog first and the roadmap page second.
