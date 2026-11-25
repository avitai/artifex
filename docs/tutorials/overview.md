# Artifex Example Overview

Artifex publishes `docs/examples/` pages only when a real source pair exists
under `examples/generative_models/`: one runnable `.py` file and one paired
`.ipynb` notebook. Not every published pair is a canonical runtime-backed
tutorial, though. Still-relevant topics
without runnable pairs live in [Planned Example Topics](../roadmap/planned-examples.md).

## Published Example Buckets

- Runtime-backed tutorials: examples that instantiate shipped Artifex owners end
  to end, such as [VAE on MNIST](basic/vae-mnist.md),
  [Simple GAN](basic/simple-gan.md), [Diffusion on MNIST](basic/diffusion-mnist.md),
  [DiT Demo](diffusion/dit-demo.md), and the retained protein and geometric
  examples in the main catalog.

## Running Examples From A Source Checkout

```bash
source ./activate.sh
uv run python examples/generative_models/image/vae/vae_mnist.py
uv run python examples/generative_models/protein/protein_extensions_example.py
```

## Working With Example Pairs

- The Python source is the review surface. Examples follow substrax's
  [Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/)
  guide; `uv run substrax-notebooks convert examples/<topic>/<name>.py` writes the
  notebook from the script.
- Reader-facing example docs belong under `docs/examples/` only when the pair
  exists.

## Choosing A Starting Point

- New to Artifex: start with [VAE on MNIST](basic/vae-mnist.md) or
  [Simple GAN](basic/simple-gan.md).
- Interested in current training workflows: start with
  [Advanced Training](../examples/generative_models/advanced_training_example.md) and the
  [Training Guide](../user-guide/training/training-guide.md).
