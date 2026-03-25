# Advanced GAN

**Status:** `Exploratory workflow`
**Device:** `GPU-optional`

This walkthrough uses lower-level Artifex GAN building blocks and a custom
training loop to compare several GAN families on MNIST-like image data. It does
not instantiate the top-level `ConditionalGAN`, `WGAN`, `DCGAN`, or `LSGAN`
owners end to end, so it is published as exploratory material rather than a
canonical runtime-backed tutorial.

## Files

- Python script: [`examples/generative_models/image/gan/advanced_gan.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/image/gan/advanced_gan.py)
- Jupyter notebook: [`examples/generative_models/image/gan/advanced_gan.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/image/gan/advanced_gan.ipynb)

## Run It

```bash
python examples/generative_models/image/gan/advanced_gan.py
jupyter lab examples/generative_models/image/gan/advanced_gan.ipynb
```

## What This Workflow Actually Uses

- lower-level Artifex GAN building blocks such as
  `ConditionalGenerator`, `ConditionalDiscriminator`, `WGANGenerator`,
  `WGANDiscriminator`, `DCGANGenerator`, `DCGANDiscriminator`,
  `LSGANGenerator`, and `LSGANDiscriminator`
- a custom training loop defined in the example itself
- local MNIST bootstrap through Hugging Face plus Grain rather than a retained
  Artifex example data facade
- Artifex adversarial loss helpers and gradient-penalty utilities

## Why It Is Exploratory

- the example compares lower-level component stacks instead of teaching one
  canonical top-level owner story
- the data-loading and orchestration logic is owned locally by the example
- environment-specific dataset bootstrap issues can still affect execution
  before training begins

## Use This When

Use this pair if you want to inspect how the lower-level GAN components fit
together, adapt the local training loop, or compare family-specific generator
and discriminator stacks in one place.

If you want a retained runtime-backed GAN tutorial instead, start with
[Simple GAN](../basic/simple-gan.md) and the broader GAN API and user-guide
docs.
