# Artifex Example Overview

Artifex publishes `docs/examples/` pages only when a real source pair exists
under `examples/generative_models/`: one runnable `.py` file and one paired
`.ipynb` notebook. Not every published pair is a canonical runtime-backed
tutorial, though. The examples catalog now separates runtime-backed tutorials,
standalone pedagogy, exploratory workflows, and validation utilities so the
public teaching surface matches the current code honestly. Still-relevant topics
without runnable pairs live in [Planned Example Topics](../roadmap/planned-examples.md).

## Published Example Buckets

- Runtime-backed tutorials: examples that instantiate shipped Artifex owners end
  to end, such as [VAE on MNIST](basic/vae-mnist.md),
  [Simple GAN](basic/simple-gan.md), [Diffusion on MNIST](basic/diffusion-mnist.md),
  [DiT Demo](diffusion/dit-demo.md), and the retained protein and geometric
  examples in the main catalog.
- Standalone pedagogy: raw JAX/Flax NNX concept walkthroughs that are clearly
  labeled as not instantiating shipped Artifex runtime owners, such as
  [Simple Diffusion](diffusion/simple-diffusion.md),
  [Simple Audio Generation](audio/simple-audio-generation.md),
  [Simple Text Generation](text/simple-text-generation.md), and
  [Simple Image-Text](multimodal/simple-image-text.md).
- Exploratory workflows: useful but non-canonical example tiers that currently
  rely on lower-level components or custom orchestration, such as
  [Advanced GAN](advanced/advanced-gan.md) and
  [Protein Diffusion](protein/protein-diffusion-example.md).
- Validation utilities: quick environment or technology-stack checks, such as
  [Protein Diffusion Technical Validation](protein/protein-diffusion-tech-validation.md).

## Running Examples From A Source Checkout

```bash
source ./activate.sh
uv run python examples/generative_models/image/vae/vae_mnist.py
uv run python examples/generative_models/protein/protein_extensions_example.py
uv run python examples/generative_models/audio/simple_audio_generation.py
```

## Working With Example Pairs

- The Python source is the review surface. Regenerate the notebook from it with
  `uv run python scripts/jupytext_converter.py sync examples/path/to/example.py`.
- Reader-facing example docs belong under `docs/examples/` only when the pair
  exists. The catalog taxonomy then decides whether that pair is runtime-backed,
  standalone, exploratory, or validation-oriented.
- Verification or maintenance scripts such as `examples/verify_examples.py` are
  utilities, not canonical tutorial examples.

## Choosing A Starting Point

- New to Artifex: start with [VAE on MNIST](basic/vae-mnist.md) or
  [Simple GAN](basic/simple-gan.md).
- Interested in current training workflows: start with
  [Advanced Training](advanced/advanced-training.md) and the
  [Training Guide](../user-guide/training/training-guide.md).
- Interested in lightweight concept demos: use the standalone pages for text,
  audio, multimodal, or simple diffusion ideas.
- Interested in lower-level or partially retained surfaces: use the exploratory
  and validation sections in [All Examples](index.md) with the status labels in
  mind.
