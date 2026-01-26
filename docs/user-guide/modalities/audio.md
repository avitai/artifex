# Audio Modality Guide

This guide covers working with audio data in Artifex, including audio representations, waveform processing, spectrograms, and best practices for audio-based generative models.

## Overview

Artifex's audio modality provides a unified interface for working with audio data in different representations: raw waveforms, mel-spectrograms, and STFTs (Short-Time Fourier Transforms).

<div class="grid cards" markdown>

- :material-waveform:{ .lg .middle } **Multiple Representations**

    ---

    Support for raw waveforms, mel-spectrograms, and STFT representations

- :material-music-note:{ .lg .middle } **Sample Rate Control**

    ---

    Work with any sample rate from 8kHz to 48kHz and beyond

- :material-chart-timeline-variant:{ .lg .middle } **Spectrogram Processing**

    ---

    Built-in mel-spectrogram and STFT computation with configurable parameters

- :material-database-outline:{ .lg .middle } **Synthetic Datasets**

    ---

    Ready-to-use synthetic audio datasets (sine waves, noise, chirps)

- :material-auto-fix:{ .lg .middle } **Augmentation**

    ---

    Time-domain and frequency-domain audio augmentation techniques

- :material-speedometer:{ .lg .middle } **JAX-Native**

    ---

    Full JAX compatibility with JIT compilation and GPU acceleration

</div>

## Audio Representations

### Supported Formats

Artifex supports three audio representations:

```python
from artifex.generative_models.modalities.audio.base import AudioRepresentation

# Raw waveform (time-domain signal)
AudioRepresentation.RAW_WAVEFORM

# Mel-spectrogram (perceptually-scaled frequency representation)
AudioRepresentation.MEL_SPECTROGRAM

# STFT (Short-Time Fourier Transform)
AudioRepresentation.STFT
```

### Configuring Audio Modality

```python
from artifex.generative_models.modalities.audio import (
    AudioModality,
    AudioModalityConfig,
    AudioRepresentation
)
from flax import nnx

# Initialize RNG
rngs = nnx.Rngs(0)

# Raw waveform configuration
waveform_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=16000,  # 16 kHz
    duration=2.0,       # 2 seconds
    normalize=True
)

waveform_modality = AudioModality(config=waveform_config, rngs=rngs)

# Mel-spectrogram configuration
mel_config = AudioModalityConfig(
    representation=AudioRepresentation.MEL_SPECTROGRAM,
    sample_rate=16000,
    n_mel_channels=80,   # Number of mel bins
    hop_length=256,      # Hop size for STFT
    n_fft=1024,          # FFT size
    duration=2.0,
    normalize=True
)

mel_modality = AudioModality(config=mel_config, rngs=rngs)

# STFT configuration
stft_config = AudioModalityConfig(
    representation=AudioRepresentation.STFT,
    sample_rate=22050,   # CD-quality / 2
    n_fft=2048,
    hop_length=512,
    duration=3.0,
    normalize=True
)

stft_modality = AudioModality(config=stft_config, rngs=rngs)
```

### Audio Shape Properties

```python
# Raw waveform
print(f"Time steps: {waveform_modality.n_time_steps}")  # 32000 (16000 * 2.0)
print(f"Output shape: {waveform_modality.output_shape}")  # (32000,)

# Mel-spectrogram
print(f"Time frames: {mel_modality.n_time_frames}")  # 125 (32000 // 256)
print(f"Output shape: {mel_modality.output_shape}")  # (80, 125)

# STFT
print(f"Frequency bins: {stft_config.n_fft // 2 + 1}")  # 1025
print(f"Output shape: {stft_modality.output_shape}")  # (1025, time_frames)
```

## Audio Datasets

### Synthetic Audio Datasets

Artifex provides synthetic audio datasets for testing and development:

```python
from artifex.generative_models.modalities.audio.datasets import (
    SyntheticAudioDataset,
    create_audio_dataset
)

# Configure audio
audio_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=16000,
    duration=1.0,
    normalize=True
)

# Create synthetic dataset
audio_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=1000,
    audio_types=["sine", "noise", "chirp"],
    name="SyntheticAudio"
)

# Access sample
sample = audio_dataset[0]
print(sample["audio"].shape)  # (16000,) - 1 second at 16kHz
print(sample["audio_type"])   # "sine" or "noise" or "chirp"
print(sample["sample_rate"])  # 16000
print(sample["duration"])     # 1.0
```

#### Audio Types

**Sine Waves** - Pure tones at random frequencies:

```python
audio_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=1000,
    audio_types=["sine"]
)

# Sine waves have:
# - Random frequencies (200-800 Hz)
# - Fixed amplitude (0.5)
# - Clean sinusoidal waveform
```

**White Noise** - Random Gaussian noise:

```python
audio_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=1000,
    audio_types=["noise"]
)

# White noise has:
# - Gaussian distribution
# - Fixed amplitude (0.3)
# - Flat frequency spectrum
```

**Chirp Signals** - Linear frequency sweeps:

```python
audio_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=1000,
    audio_types=["chirp"]
)

# Chirps have:
# - Linear frequency sweep (200-800 Hz)
# - Smooth transitions
# - Useful for testing time-frequency representations
```

#### Mixed Audio Types

```python
# Mix multiple audio types
mixed_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=3000,
    audio_types=["sine", "noise", "chirp"]
)

# Dataset will cycle through types
# Sample 0: sine
# Sample 1: noise
# Sample 2: chirp
# Sample 3: sine (cycles back)
```

### Batching Audio Data

```python
# Get batch using collate function
batch = audio_dataset.collate_fn([
    audio_dataset[0],
    audio_dataset[1],
    audio_dataset[2],
    audio_dataset[3]
])

print(batch["audio"].shape)  # (4, 16000)
print(len(batch["audio_type"]))  # 4
print(batch["sample_rate"])  # [16000, 16000, 16000, 16000]

# Iterate through dataset
for i, sample in enumerate(audio_dataset):
    if i >= 5:
        break
    print(f"Sample {i}: {sample['audio_type']}, shape: {sample['audio'].shape}")
```

### Factory Function

```python
# Create dataset using factory
dataset = create_audio_dataset(
    dataset_type="synthetic",
    config=audio_config,
    n_samples=5000,
    audio_types=["sine", "noise"]
)

# With custom parameters
custom_dataset = create_audio_dataset(
    dataset_type="synthetic",
    config=None,  # Will use defaults
    n_samples=1000,
    audio_types=["chirp"]
)
```

## Audio Preprocessing

### Normalization

```python
import jax.numpy as jnp

def normalize_audio(audio: jax.Array) -> jax.Array:
    """Normalize audio to [-1, 1] range.

    Args:
        audio: Input audio waveform

    Returns:
        Normalized audio
    """
    max_val = jnp.max(jnp.abs(audio))
    return jnp.where(max_val > 0, audio / max_val, audio)

def rms_normalize(audio: jax.Array, target_rms: float = 0.1) -> jax.Array:
    """Normalize audio by RMS (root mean square) energy.

    Args:
        audio: Input audio waveform
        target_rms: Target RMS value

    Returns:
        RMS-normalized audio
    """
    rms = jnp.sqrt(jnp.mean(audio ** 2))
    return audio * (target_rms / (rms + 1e-8))

# Usage
audio = jnp.array([...])  # Raw audio
normalized = normalize_audio(audio)
rms_normalized = rms_normalize(audio, target_rms=0.1)
```

### Resampling

```python
import jax
import jax.numpy as jnp

def resample_audio(
    audio: jax.Array,
    orig_sr: int,
    target_sr: int
) -> jax.Array:
    """Resample audio to different sample rate.

    Args:
        audio: Input audio waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    # Simple linear interpolation resampling
    orig_length = len(audio)
    target_length = int(orig_length * target_sr / orig_sr)

    # Create interpolation indices
    orig_indices = jnp.linspace(0, orig_length - 1, target_length)

    # Interpolate
    resampled = jnp.interp(orig_indices, jnp.arange(orig_length), audio)

    return resampled

# Usage
audio = jnp.array([...])  # Audio at 16kHz
audio_22k = resample_audio(audio, orig_sr=16000, target_sr=22050)
audio_8k = resample_audio(audio, orig_sr=16000, target_sr=8000)
```

### Duration Adjustment

```python
def trim_or_pad(
    audio: jax.Array,
    target_length: int,
    pad_value: float = 0.0
) -> jax.Array:
    """Trim or pad audio to target length.

    Args:
        audio: Input audio waveform
        target_length: Target number of samples
        pad_value: Value to use for padding

    Returns:
        Trimmed or padded audio
    """
    current_length = len(audio)

    if current_length >= target_length:
        # Trim
        return audio[:target_length]
    else:
        # Pad
        padding = jnp.full((target_length - current_length,), pad_value)
        return jnp.concatenate([audio, padding])

# Usage
audio = jnp.array([...])  # Variable length audio
fixed_length = trim_or_pad(audio, target_length=16000)
print(fixed_length.shape)  # (16000,)
```

## Spectrogram Processing

### JIT-Compatible STFT (Recommended)

For production use, the `SpectralAnalysis` extension provides a fully JIT-compatible STFT implementation using `jax.scipy.signal.stft`:

```python
from artifex.generative_models.extensions.audio_processing.spectral import SpectralAnalysis
from artifex.generative_models.core.configuration import ExtensionConfig
from flax import nnx
import jax.numpy as jnp

rngs = nnx.Rngs(0)

# Configure spectral analysis
config = ExtensionConfig(
    weight=1.0,
    enabled=True,
    extensions={
        "spectral": {
            "sample_rate": 22050,
            "n_fft": 2048,
            "hop_length": 512,
            "window_type": "hann",  # "hann", "hamming", "blackman"
            "n_mels": 128
        }
    }
)

# Create spectral analysis module
spectral = SpectralAnalysis(config=config, rngs=rngs)

# Generate test audio
audio = jnp.sin(2 * jnp.pi * 440 * jnp.linspace(0, 1, 22050))

# Compute STFT (JIT-compatible, GPU-accelerated)
stft_magnitude = spectral.compute_stft(audio)
print(f"STFT shape: {stft_magnitude.shape}")

# Compute various spectrograms
power_spec = spectral.compute_spectrogram(audio)
mel_spec = spectral.compute_mel_spectrogram(audio)
log_mel_spec = spectral.compute_log_mel_spectrogram(audio)
mfcc = spectral.compute_mfcc(audio, n_mfcc=13)

# Spectral features
centroid = spectral.compute_spectral_centroid(audio)
bandwidth = spectral.compute_spectral_bandwidth(audio)
rolloff = spectral.compute_spectral_rolloff(audio)

# Extract all features at once
features = spectral.extract_spectral_features(audio)
```

**Key benefits:**

- Fully JIT-compatible using `jax.scipy.signal.stft`
- GPU-accelerated computation
- Supports batch processing via `jax.vmap`
- Comprehensive spectral feature extraction

## Audio Augmentation

### Time-Domain Augmentations

```python
import jax
import jax.numpy as jnp

def time_shift(audio: jax.Array, key, max_shift: int = 1600):
    """Randomly shift audio in time.

    Args:
        audio: Input audio waveform
        key: Random key
        max_shift: Maximum shift in samples

    Returns:
        Time-shifted audio
    """
    shift = jax.random.randint(key, (), -max_shift, max_shift)
    return jnp.roll(audio, int(shift))

def time_stretch(audio: jax.Array, key, rate_range=(0.8, 1.2)):
    """Randomly stretch or compress audio in time.

    Args:
        audio: Input audio waveform
        key: Random key
        rate_range: (min_rate, max_rate) for stretching

    Returns:
        Time-stretched audio
    """
    rate = jax.random.uniform(key, minval=rate_range[0], maxval=rate_range[1])

    # Simple resampling for time stretching
    orig_length = len(audio)
    new_length = int(orig_length / rate)

    indices = jnp.linspace(0, orig_length - 1, new_length)
    stretched = jnp.interp(indices, jnp.arange(orig_length), audio)

    # Pad or trim to original length
    if len(stretched) < orig_length:
        padding = jnp.zeros(orig_length - len(stretched))
        stretched = jnp.concatenate([stretched, padding])
    else:
        stretched = stretched[:orig_length]

    return stretched

def add_gaussian_noise(audio: jax.Array, key, noise_level: float = 0.005):
    """Add Gaussian noise to audio.

    Args:
        audio: Input audio waveform
        key: Random key
        noise_level: Standard deviation of noise

    Returns:
        Noisy audio
    """
    noise = noise_level * jax.random.normal(key, audio.shape)
    return audio + noise

def random_gain(audio: jax.Array, key, gain_range=(0.7, 1.3)):
    """Apply random gain (amplitude scaling).

    Args:
        audio: Input audio waveform
        key: Random key
        gain_range: (min_gain, max_gain)

    Returns:
        Gain-adjusted audio
    """
    gain = jax.random.uniform(key, minval=gain_range[0], maxval=gain_range[1])
    return audio * gain

# Usage
audio = jnp.array([...])  # Audio waveform
key = jax.random.key(0)
keys = jax.random.split(key, 4)

shifted = time_shift(audio, keys[0], max_shift=1600)
stretched = time_stretch(audio, keys[1], rate_range=(0.9, 1.1))
noisy = add_gaussian_noise(audio, keys[2], noise_level=0.005)
gained = random_gain(audio, keys[3], gain_range=(0.8, 1.2))
```

### Frequency-Domain Augmentations

```python
def frequency_mask(
    spectrogram: jax.Array,
    key,
    num_masks: int = 1,
    mask_param: int = 10
):
    """Apply frequency masking to spectrogram (SpecAugment).

    Args:
        spectrogram: Input spectrogram (freq_bins, time_frames)
        key: Random key
        num_masks: Number of masks to apply
        mask_param: Maximum width of mask

    Returns:
        Masked spectrogram
    """
    masked_spec = spectrogram.copy()
    n_freq_bins = spectrogram.shape[0]

    keys = jax.random.split(key, num_masks * 2)

    for i in range(num_masks):
        # Random mask width
        mask_width = jax.random.randint(keys[2*i], (), 0, mask_param)

        # Random starting frequency
        start_freq = jax.random.randint(
            keys[2*i + 1], (), 0, n_freq_bins - mask_width
        )

        # Apply mask
        masked_spec = masked_spec.at[start_freq:start_freq+mask_width, :].set(0)

    return masked_spec

def time_mask(
    spectrogram: jax.Array,
    key,
    num_masks: int = 1,
    mask_param: int = 10
):
    """Apply time masking to spectrogram (SpecAugment).

    Args:
        spectrogram: Input spectrogram (freq_bins, time_frames)
        key: Random key
        num_masks: Number of masks to apply
        mask_param: Maximum width of mask

    Returns:
        Masked spectrogram
    """
    masked_spec = spectrogram.copy()
    n_time_frames = spectrogram.shape[1]

    keys = jax.random.split(key, num_masks * 2)

    for i in range(num_masks):
        # Random mask width
        mask_width = jax.random.randint(keys[2*i], (), 0, mask_param)

        # Random starting time
        start_time = jax.random.randint(
            keys[2*i + 1], (), 0, n_time_frames - mask_width
        )

        # Apply mask
        masked_spec = masked_spec.at[:, start_time:start_time+mask_width].set(0)

    return masked_spec

def spec_augment(
    spectrogram: jax.Array,
    key,
    freq_masks: int = 2,
    time_masks: int = 2,
    freq_mask_param: int = 15,
    time_mask_param: int = 20
):
    """Apply SpecAugment (frequency + time masking).

    Args:
        spectrogram: Input spectrogram
        key: Random key
        freq_masks: Number of frequency masks
        time_masks: Number of time masks
        freq_mask_param: Max frequency mask width
        time_mask_param: Max time mask width

    Returns:
        Augmented spectrogram
    """
    keys = jax.random.split(key, 2)

    # Apply frequency masking
    spec = frequency_mask(spectrogram, keys[0], freq_masks, freq_mask_param)

    # Apply time masking
    spec = time_mask(spec, keys[1], time_masks, time_mask_param)

    return spec

# Usage (using SpectralAnalysis from above)
mel_spec = spectral.compute_mel_spectrogram(audio)
key = jax.random.key(0)

augmented_spec = spec_augment(
    mel_spec,
    key,
    freq_masks=2,
    time_masks=2,
    freq_mask_param=15,
    time_mask_param=20
)
```

### Complete Augmentation Pipeline

```python
@jax.jit
def augment_audio(audio: jax.Array, key):
    """Apply comprehensive audio augmentation pipeline.

    Args:
        audio: Input audio waveform
        key: Random key

    Returns:
        Augmented audio
    """
    keys = jax.random.split(key, 5)

    # Time-domain augmentations
    audio = time_shift(audio, keys[0], max_shift=800)
    audio = add_gaussian_noise(audio, keys[1], noise_level=0.005)
    audio = random_gain(audio, keys[2], gain_range=(0.8, 1.2))
    audio = time_stretch(audio, keys[3], rate_range=(0.95, 1.05))

    # Normalize after augmentation
    audio = normalize_audio(audio)

    return audio

def augment_audio_batch(audio_batch: jax.Array, key):
    """Augment batch of audio waveforms.

    Args:
        audio_batch: Batch of audio (N, n_samples)
        key: Random key

    Returns:
        Augmented batch
    """
    batch_size = audio_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    # Vectorize over batch
    augmented = jax.vmap(augment_audio)(audio_batch, keys)

    return augmented

# Usage in training
key = jax.random.key(0)
for batch in data_loader:
    key, subkey = jax.random.split(key)
    augmented_audio = augment_audio_batch(batch["audio"], subkey)
    # Use augmented_audio for training
```

## Working with Different Sample Rates

### Common Sample Rates

```python
# Telephone quality (8 kHz)
telephone_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=8000,
    duration=2.0
)

# Standard speech (16 kHz)
speech_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=16000,
    duration=2.0
)

# High-quality speech (22.05 kHz)
hq_speech_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=22050,
    duration=2.0
)

# CD quality (44.1 kHz)
cd_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=44100,
    duration=2.0
)

# Studio quality (48 kHz)
studio_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=48000,
    duration=2.0
)
```

### Sample Rate Conversion

```python
def convert_sample_rate(
    audio: jax.Array,
    orig_sr: int,
    target_sr: int
) -> jax.Array:
    """Convert audio to different sample rate.

    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    return resample_audio(audio, orig_sr, target_sr)

# Usage
audio_16k = jnp.array([...])  # Audio at 16 kHz

# Upsample to 22.05 kHz
audio_22k = convert_sample_rate(audio_16k, 16000, 22050)

# Downsample to 8 kHz
audio_8k = convert_sample_rate(audio_16k, 16000, 8000)

print(f"16 kHz: {len(audio_16k)} samples")
print(f"22.05 kHz: {len(audio_22k)} samples")
print(f"8 kHz: {len(audio_8k)} samples")
```

## Complete Examples

### Example 1: Audio Generation Dataset

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.audio import (
    AudioModality,
    AudioModalityConfig,
    AudioRepresentation
)
from artifex.generative_models.modalities.audio.datasets import (
    SyntheticAudioDataset
)

# Setup
rngs = nnx.Rngs(0)

# Configure
audio_config = AudioModalityConfig(
    representation=AudioRepresentation.RAW_WAVEFORM,
    sample_rate=16000,
    duration=1.0,
    normalize=True
)

modality = AudioModality(config=audio_config, rngs=rngs)

# Create datasets
train_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=10000,
    audio_types=["sine", "noise", "chirp"]
)

val_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=1000,
    audio_types=["sine", "noise", "chirp"]
)

# Training loop
batch_size = 32
num_epochs = 10
key = jax.random.key(42)

for epoch in range(num_epochs):
    num_batches = len(train_dataset) // batch_size

    for i in range(num_batches):
        # Get batch
        batch_samples = [train_dataset[j] for j in range(i*batch_size, (i+1)*batch_size)]
        batch = train_dataset.collate_fn(batch_samples)
        audio_batch = batch["audio"]

        # Apply augmentation
        key, subkey = jax.random.split(key)
        augmented = augment_audio_batch(audio_batch, subkey)

        # Training step
        # loss = train_step(model, augmented)

    # Validation (no augmentation)
    val_batches = len(val_dataset) // batch_size
    for i in range(val_batches):
        val_samples = [val_dataset[j] for j in range(i*batch_size, (i+1)*batch_size)]
        val_batch = val_dataset.collate_fn(val_samples)
        # val_loss = validate_step(model, val_batch["audio"])

    print(f"Epoch {epoch + 1}/{num_epochs} complete")
```

### Example 2: Mel-Spectrogram Training

```python
# Configure for mel-spectrograms
mel_config = AudioModalityConfig(
    representation=AudioRepresentation.MEL_SPECTROGRAM,
    sample_rate=16000,
    n_mel_channels=80,
    hop_length=256,
    n_fft=1024,
    duration=2.0,
    normalize=True
)

mel_modality = AudioModality(config=mel_config, rngs=rngs)

# Create dataset
mel_dataset = SyntheticAudioDataset(
    config=mel_config,
    n_samples=5000,
    audio_types=["sine", "chirp"]
)

# Create SpectralAnalysis for mel-spectrogram computation
spectral_config = ExtensionConfig(
    weight=1.0,
    enabled=True,
    extensions={
        "spectral": {
            "sample_rate": 16000,
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 80
        }
    }
)
spectral = SpectralAnalysis(config=spectral_config, rngs=rngs)

# Training with spectrograms
for epoch in range(num_epochs):
    for i in range(len(mel_dataset) // batch_size):
        # Get audio batch
        batch_samples = [mel_dataset[j] for j in range(i*batch_size, (i+1)*batch_size)]
        batch = mel_dataset.collate_fn(batch_samples)
        audio_batch = batch["audio"]

        # Compute mel-spectrograms (batch processing via vmap)
        mel_batch = jax.vmap(spectral.compute_mel_spectrogram)(audio_batch)

        # Apply SpecAugment
        key, subkey = jax.random.split(key)
        augmented_specs = jax.vmap(lambda spec, k: spec_augment(spec, k))(
            mel_batch,
            jax.random.split(subkey, batch_size)
        )

        # Training step
        # loss = train_step(model, augmented_specs)
```

### Example 3: Custom Audio Dataset

```python
from typing import Iterator
from artifex.generative_models.modalities.base import BaseDataset

class CustomAudioDataset(BaseDataset):
    """Custom audio dataset loading from files."""

    def __init__(
        self,
        config: AudioModalityConfig,
        audio_paths: list[str],
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, split, rngs=rngs)
        self.audio_paths = audio_paths
        self.audio_samples = self._load_audio_files()

    def _load_audio_files(self):
        """Load audio files."""
        samples = []
        for path in self.audio_paths:
            # In practice: use librosa, soundfile, etc.
            # For demo, generate synthetic
            audio = jax.random.uniform(
                jax.random.key(hash(path)),
                (16000,)
            )
            samples.append(audio)
        return samples

    def __len__(self) -> int:
        return len(self.audio_samples)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for i, audio in enumerate(self.audio_samples):
            yield {
                "audio": audio,
                "index": jnp.array(i),
                "path": self.audio_paths[i]
            }

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        batch_audio = [self.audio_samples[int(idx)] for idx in indices]
        batch_paths = [self.audio_paths[int(idx)] for idx in indices]

        return {
            "audio": jnp.stack(batch_audio),
            "indices": indices,
            "paths": batch_paths
        }

# Usage
audio_paths = ["/path/to/audio1.wav", "/path/to/audio2.wav", ...]

custom_dataset = CustomAudioDataset(
    config=audio_config,
    audio_paths=audio_paths,
    rngs=rngs
)
```

## Best Practices

### DO

!!! tip "Audio Loading"
    - Use appropriate sample rate for your task
    - Normalize audio to consistent amplitude
    - Trim or pad to consistent duration for batching
    - Validate audio shapes before training
    - Cache processed audio when possible
    - Use synthetic datasets for testing pipelines

!!! tip "Spectrograms"
    - Choose appropriate n_fft and hop_length for task
    - Use mel-spectrograms for perceptual tasks
    - Apply log scaling to spectrograms
    - Consider phase information for reconstruction tasks
    - Use appropriate number of mel bins (typically 40-128)

!!! tip "Augmentation"
    - Apply augmentation only during training
    - Balance augmentation strength with audio quality
    - Use SpecAugment for spectrogram-based models
    - Test augmentations by listening to samples
    - Use JIT compilation for performance

### DON'T

!!! danger "Common Mistakes"
    - Mix different sample rates in same batch
    - Forget to normalize audio amplitudes
    - Apply augmentation during validation
    - Use non-JAX operations in data pipeline
    - Load full audio files if working with clips
    - Ignore clipping (values > 1.0 or < -1.0)

!!! danger "Performance Issues"
    - Load audio from disk during training
    - Compute spectrograms on-the-fly every epoch
    - Use Python loops for audio processing
    - Keep multiple copies of audio in memory
    - Use very long audio clips on limited GPU memory

!!! danger "Quality Issues"
    - Over-augment audio (too much distortion)
    - Use inappropriate sample rates
    - Mix time-domain and frequency-domain representations
    - Ignore audio phase for waveform reconstruction
    - Apply excessive noise

## Summary

This guide covered:

- **Audio representations** - Waveforms, mel-spectrograms, STFT
- **Audio datasets** - Synthetic datasets with various audio types
- **Preprocessing** - Normalization, resampling, duration adjustment
- **Spectrograms** - JIT-compatible STFT, magnitude, power, and mel-spectrograms
- **SpectralAnalysis extension** - Production-ready spectral processing with `jax.scipy.signal.stft`
- **Augmentation** - Time-domain and frequency-domain techniques
- **Sample rates** - Working with different sample rates
- **Complete examples** - Training pipelines and custom datasets
- **Best practices** - DOs and DON'Ts for audio data

## Next Steps

<div class="grid cards" markdown>

- :material-layers-triple:{ .lg .middle } **[Multi-modal Guide](multimodal.md)**

    ---

    Working with multiple modalities and aligned multi-modal datasets

- :material-image:{ .lg .middle } **[Image Modality Guide](image.md)**

    ---

    Deep dive into image datasets, preprocessing, and augmentation

- :material-text:{ .lg .middle } **[Text Modality Guide](text.md)**

    ---

    Learn about text tokenization, vocabulary management, and sequences

- :material-api:{ .lg .middle } **[Data API Reference](../../api/data/loaders.md)**

    ---

    Complete API documentation for all dataset classes and functions

</div>
