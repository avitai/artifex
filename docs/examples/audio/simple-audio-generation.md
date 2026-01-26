# Simple Audio Generation

**Level:** Beginner | **Runtime:** ~10 seconds (CPU) | **Format:** Python + Jupyter

**Prerequisites:** Basic neural networks and JAX | **Target Audience:** Users learning audio generation with neural networks

## Overview

This example demonstrates how to generate audio waveforms using neural networks with JAX and Flax NNX. Learn how to build a simple audio generator, create waveform variations, visualize audio in time and frequency domains, and save outputs for playback.

## What You'll Learn

<div class="grid cards" markdown>

- :material-waveform: **Audio Generation**

    ---

    Generate audio waveforms from random latent codes using neural networks

- :material-tune-variant: **Sound Variations**

    ---

    Create variations of a base sound by perturbing latent space

- :material-chart-line: **Waveform Visualization**

    ---

    Plot audio signals in the time domain with proper scaling

- :material-music-note: **Spectrogram Analysis**

    ---

    Convert waveforms to spectrograms using STFT for frequency analysis

</div>

## Files

This example is available in two formats:

- **Python Script**: [`simple_audio_generation.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/audio/simple_audio_generation.py)
- **Jupyter Notebook**: [`simple_audio_generation.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/audio/simple_audio_generation.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/audio/simple_audio_generation.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/audio/simple_audio_generation.ipynb
```

## Key Concepts

### 1. Audio Waveform Representation

Audio signals are represented as 1D arrays of amplitude values over time:

```python
# Audio parameters
sample_rate = 16000  # 16 kHz (16,000 samples per second)
duration = 0.5       # 0.5 seconds
num_samples = int(sample_rate * duration)  # 8,000 samples

# Waveform: array of shape (num_samples,) with values in [-1, 1]
waveform = jnp.array([...])  # Shape: (8000,)
```

**Key Parameters:**

- **Sample Rate**: Number of samples per second (Hz)
  - CD quality: 44.1 kHz
  - Speech: 16 kHz
  - Phone: 8 kHz
- **Duration**: Length of audio in seconds
- **Amplitude**: Waveform values typically in range [-1, 1]

### 2. Neural Audio Generator

The `SimpleAudioGenerator` uses a feedforward network to generate audio from latent codes:

$$\text{waveform} = \text{Generator}(z), \quad z \sim \mathcal{N}(0, I)$$

```python
from flax import nnx

class SimpleAudioGenerator(nnx.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 1.0,
        latent_dim: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.latent_dim = latent_dim
        self.num_samples = int(sample_rate * duration)

        # Generator network: latent → waveform
        self.generator = nnx.Sequential(
            nnx.Linear(latent_dim, 128, rngs=rngs),
            nnx.relu,
            nnx.Linear(128, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, self.num_samples, rngs=rngs),
            nnx.tanh,  # Outputs in [-1, 1]
        )
```

**Architecture Details:**

- Input: Latent vector of shape `(latent_dim,)`
- Hidden layers: Progressive expansion (128 → 256 → 512)
- Output: Waveform of shape `(num_samples,)`
- Activation: `tanh` ensures amplitude in [-1, 1]

### 3. Generating Audio

Generate audio waveforms from random latent codes:

```python
# Create generator
generator = SimpleAudioGenerator(
    sample_rate=16000,
    duration=0.5,
    latent_dim=32,
    rngs=rngs
)

# Generate batch of waveforms
waveforms = generator.generate(batch_size=3, rngs=rngs)
# Shape: (3, 8000) - 3 waveforms, each with 8000 samples

# Save to WAV file (requires scipy or soundfile)
import scipy.io.wavfile as wav
wav.write('generated_audio.wav', sample_rate, waveforms[0])
```

### 4. Creating Variations

Generate variations of a sound by adding noise to the latent code:

$$z_{\text{varied}} = z_{\text{base}} + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

```python
# Base latent vector
base_latent = jax.random.normal(rngs.sample(), (32,))

# Generate variations
variations = generator.generate_with_variation(
    base_latent=base_latent,
    variation_scale=0.2,  # Amount of variation
    num_variations=4,
    rngs=rngs
)
# Shape: (4, 8000) - 4 variations of the base sound
```

**Variation Scale:**

- Small (0.05-0.1): Subtle variations
- Medium (0.1-0.3): Noticeable differences
- Large (0.3+): Very different sounds

### 5. Waveform Visualization

Plot audio signals in the time domain:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_waveforms(waveforms, sample_rate):
    batch_size = waveforms.shape[0]
    num_samples = waveforms.shape[1]
    time = np.linspace(0, num_samples / sample_rate, num_samples)

    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3 * batch_size))

    for i, ax in enumerate(axes):
        ax.plot(time, waveforms[i], linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Waveform {i + 1}")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

### 6. Spectrogram Generation

Convert time-domain waveforms to frequency-domain spectrograms using STFT:

$$S(t, f) = \left| \sum_{n} w[n] \cdot x[n] \cdot e^{-j2\pi fn} \right|$$

```python
def generate_spectrogram(waveform, sample_rate):
    window_size = 512
    hop_size = 128

    # Compute STFT frames
    frames = []
    for i in range(0, len(waveform) - window_size, hop_size):
        frame = waveform[i : i + window_size]
        window = jnp.hanning(window_size)
        windowed_frame = frame * window

        # Compute FFT
        fft = jnp.fft.rfft(windowed_frame)
        frames.append(jnp.abs(fft))

    spectrogram = jnp.stack(frames).T

    # Plot spectrogram in dB scale
    fig, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.linspace(0, len(waveform) / sample_rate, spectrogram.shape[1])
    freq_axis = np.linspace(0, sample_rate / 2, spectrogram.shape[0])

    im = ax.imshow(
        20 * jnp.log10(spectrogram + 1e-10),  # Convert to dB
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        cmap='viridis',
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    plt.show()
```

**STFT Parameters:**

- **Window Size**: Number of samples in each FFT window (trade-off: time vs frequency resolution)
- **Hop Size**: Step size between windows (smaller = more temporal detail)
- **Window Function**: Hanning window reduces spectral leakage

## Code Structure

The example consists of four main components:

1. **SimpleAudioGenerator** - Neural network that generates waveforms from latent codes
2. **visualize_waveforms** - Plot waveforms in time domain
3. **generate_spectrogram** - Convert waveforms to frequency-domain spectrograms
4. **main** - Demo workflow: generate, visualize, and analyze audio

## Features Demonstrated

- ✅ Neural network-based audio generation from latent codes
- ✅ Batch generation of multiple waveforms
- ✅ Variation generation by perturbing latent space
- ✅ Time-domain waveform visualization
- ✅ STFT-based spectrogram computation
- ✅ Frequency-domain analysis with dB scaling
- ✅ Proper audio parameter handling (sample rate, duration)
- ✅ Output saving for playback and analysis

## Experiments to Try

1. **Adjust Audio Parameters**

   ```python
   # Generate longer audio
   generator = SimpleAudioGenerator(
       sample_rate=16000,
       duration=2.0,  # 2 seconds
       latent_dim=64, # More expressive latent space
       rngs=rngs
   )
   ```

2. **Explore Latent Space**

   ```python
   # Interpolate between two sounds
   z1 = jax.random.normal(key1, (32,))
   z2 = jax.random.normal(key2, (32,))

   for alpha in jnp.linspace(0, 1, 10):
       z_interp = (1 - alpha) * z1 + alpha * z2
       waveform = generator.generator(z_interp[None, :])[0]
       # Play or save waveform
   ```

3. **Modify Network Architecture**

   ```python
   # Deeper network for more complex audio
   self.generator = nnx.Sequential(
       nnx.Linear(latent_dim, 256, rngs=rngs),
       nnx.relu,
       nnx.Linear(256, 512, rngs=rngs),
       nnx.relu,
       nnx.Linear(512, 1024, rngs=rngs),
       nnx.relu,
       nnx.Linear(1024, self.num_samples, rngs=rngs),
       nnx.tanh,
   )
   ```

4. **Adjust Spectrogram Parameters**

   ```python
   # Finer frequency resolution
   window_size = 1024  # Larger window
   hop_size = 256      # Smaller hop for more temporal detail
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-arrow-right: **Advanced Audio**

    ---

    Learn WaveNet and autoregressive models

    [:octicons-arrow-right-24: WaveNet Tutorial](#)

- :material-arrow-right: **Conditional Generation**

    ---

    Generate audio conditioned on text or labels

    [:octicons-arrow-right-24: Conditional Audio](#)

- :material-arrow-right: **Audio VAE**

    ---

    Build variational autoencoders for audio

    [:octicons-arrow-right-24: Audio VAE Tutorial](#)

- :material-arrow-right: **Framework Features**

    ---

    Understand Artifex's modality system

    [:octicons-arrow-right-24: Framework Demo](../framework/framework-features-demo.md)

</div>

## Troubleshooting

### Generated Audio Sounds Like Noise

**Symptom:** Generated waveforms are random noise

**Cause:** Untrained generator network

**Solution:** This example shows the generator architecture. For quality audio, you need to train the generator on real audio data:

```python
# Training required for quality results
# See audio VAE or GAN tutorials for training examples
```

### Clipping in Audio Output

**Symptom:** Distorted audio with clipping artifacts

**Cause:** Amplitude exceeds [-1, 1] range

**Solution:** Normalize waveforms

```python
# Ensure amplitudes are in valid range
waveform = jnp.clip(waveform, -1.0, 1.0)

# Or normalize to [-1, 1]
waveform = waveform / jnp.max(jnp.abs(waveform))
```

### Spectrogram Looks Wrong

**Symptom:** Spectrogram is all one color or has artifacts

**Cause:** Incorrect STFT parameters or dB scaling

**Solution:** Adjust window size and add epsilon before log

```python
# Add small epsilon to avoid log(0)
spectrogram_db = 20 * jnp.log10(spectrogram + 1e-10)

# Clip extreme values
spectrogram_db = jnp.clip(spectrogram_db, -80, 0)
```

### Out of Memory Error

**Symptom:** OOM error when generating long audio

**Cause:** Large network output dimension

**Solution:** Generate audio in chunks

```python
# Generate shorter segments
generator = SimpleAudioGenerator(
    sample_rate=16000,
    duration=0.5,  # Shorter duration
    latent_dim=32,
    rngs=rngs
)
```

## Additional Resources

### Documentation

- [Audio Modality Guide](../../user-guide/modalities/audio.md) - Audio-specific features in Artifex

### Related Examples

- [Framework Features Demo](../framework/framework-features-demo.md) - Modality system overview
- [Loss Examples](../losses/loss-examples.md) - Loss functions for audio models

### Papers and Resources

- **WaveNet**: [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)
- **WaveGAN**: [Adversarial Audio Synthesis (Donahue et al., 2018)](https://arxiv.org/abs/1802.04208)
- **Jukebox**: [Jukebox: A Generative Model for Music (Dhariwal et al., 2020)](https://arxiv.org/abs/2005.00341)
- **Audio Signal Processing**: [Digital Signal Processing (Oppenheim & Schafer)](https://www.pearson.com/en-us/subject-catalog/p/discrete-time-signal-processing/P200000003285)
