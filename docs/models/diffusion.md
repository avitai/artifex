# Audio Diffusion

**Module:** `generative_models.models.audio.diffusion`

**Source:** `generative_models/models/audio/diffusion.py`

## Overview

`AudioDiffusionModel` is the audio-specific diffusion wrapper built on the
shared `DiffusionModel` base. The shipped public surface is intentionally
small:

- `AudioDiffusionConfig` for the typed audio diffusion configuration
- `create_audio_diffusion_config(...)` for constructing the nested config tree
- `AudioDiffusionModel` for waveform diffusion inference and generation

The runtime uses `UNet1DBackboneConfig` through the shared diffusion backbone
factory plus `AudioModalityConfig` for waveform length and normalization
behavior.

## Public Surface

### `AudioDiffusionConfig`

Typed configuration for audio diffusion models.

Retained fields include:

- `backbone`: a `UNet1DBackboneConfig`
- `noise_schedule`: a `NoiseScheduleConfig`
- `input_shape`: the waveform shape `(sequence_length,)`
- `modality_config`: the required `AudioModalityConfig`

### `create_audio_diffusion_config(...)`

Factory helper that builds the full audio diffusion config tree from an
`AudioModalityConfig`, diffusion schedule parameters, and the requested
`unet_channels` width.

### `AudioDiffusionModel`

Audio diffusion model for parallel waveform generation.

Retained behavior:

- `__call__(x, timesteps, *, conditioning=None, **kwargs)` returns the shared
  diffusion output dictionary with `predicted_noise`
- `generate(n_samples=1, duration=None, *, clip_denoised=True)` returns audio
  waveforms shaped to the requested duration
- `preprocess_audio(audio)` normalizes raw audio into the expected input range
- `postprocess_audio(audio)` clips generated audio to `[-1.0, 1.0]`

## Example

```python
from flax import nnx

from artifex.generative_models.modalities.audio import AudioModalityConfig
from artifex.generative_models.models.audio import (
    AudioDiffusionModel,
    create_audio_diffusion_config,
)

modality_config = AudioModalityConfig(sample_rate=16000, duration=1.0)
config = create_audio_diffusion_config(
    modality_config=modality_config,
    num_timesteps=100,
    unet_channels=32,
)
model = AudioDiffusionModel(config, rngs=nnx.Rngs(0))
```
