#!/usr/bin/env python
"""Simple audio generation example using the Artifex framework.

This example demonstrates basic audio waveform generation using
neural networks with JAX/Flax.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx


class SimpleAudioGenerator(nnx.Module):
    """Simple audio waveform generator using neural networks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 1.0,
        latent_dim: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the audio generator.

        Args:
            sample_rate: Sample rate in Hz
            duration: Duration of generated audio in seconds
            latent_dim: Dimension of latent space
            rngs: Random number generators
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.latent_dim = latent_dim
        self.num_samples = int(sample_rate * duration)

        # Simple generator network
        self.generator = nnx.Sequential(
            nnx.Linear(latent_dim, 128, rngs=rngs),
            nnx.relu,
            nnx.Linear(128, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, self.num_samples, rngs=rngs),
            nnx.tanh,  # Audio samples in [-1, 1]
        )

    def generate(self, batch_size: int = 1, *, rngs: nnx.Rngs):
        """Generate audio waveforms.

        Args:
            batch_size: Number of waveforms to generate
            rngs: Random number generators

        Returns:
            Generated audio waveforms
        """
        # Sample from latent space
        z = jax.random.normal(rngs.sample(), (batch_size, self.latent_dim))

        # Generate waveforms
        waveforms = self.generator(z)

        return waveforms

    def generate_with_variation(
        self, base_latent, variation_scale=0.1, num_variations=4, *, rngs: nnx.Rngs
    ):
        """Generate variations of a base sound.

        Args:
            base_latent: Base latent vector
            variation_scale: Scale of variations
            num_variations: Number of variations to generate
            rngs: Random number generators

        Returns:
            Array of waveform variations
        """
        variations = []
        key = rngs.sample()

        for i in range(num_variations):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, base_latent.shape) * variation_scale
            varied_latent = base_latent + noise
            waveform = self.generator(varied_latent[None, :])[0]
            variations.append(waveform)

        return jnp.stack(variations)


def visualize_waveforms(waveforms, sample_rate, title="Generated Audio Waveforms"):
    """Visualize audio waveforms.

    Args:
        waveforms: Array of waveforms [batch, samples]
        sample_rate: Sample rate in Hz
        title: Plot title
    """
    batch_size = waveforms.shape[0]
    num_samples = waveforms.shape[1]
    time = np.linspace(0, num_samples / sample_rate, num_samples)

    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, waveforms[i], linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Waveform {i + 1}")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def generate_spectrogram(waveform, sample_rate):
    """Generate a simple spectrogram visualization.

    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz

    Returns:
        Spectrogram plot
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Simple FFT-based spectrogram
    window_size = 512
    hop_size = 128

    # Pad waveform if needed
    padding = window_size - (len(waveform) % window_size)
    if padding != window_size:
        waveform = jnp.pad(waveform, (0, padding), mode="constant")

    # Compute STFT frames
    frames = []
    for i in range(0, len(waveform) - window_size, hop_size):
        frame = waveform[i : i + window_size]
        # Apply Hann window
        window = jnp.hanning(window_size)
        windowed_frame = frame * window
        # Compute FFT
        fft = jnp.fft.rfft(windowed_frame)
        frames.append(jnp.abs(fft))

    spectrogram = jnp.stack(frames).T

    # Plot spectrogram
    time_axis = np.linspace(0, len(waveform) / sample_rate, spectrogram.shape[1])
    freq_axis = np.linspace(0, sample_rate / 2, spectrogram.shape[0])

    im = ax.imshow(
        20 * jnp.log10(spectrogram + 1e-10),  # Convert to dB
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        cmap="viridis",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")

    return fig


def main():
    """Run the simple audio generation example."""
    print("=" * 60)
    print("Simple Audio Generation Example")
    print("=" * 60)

    # Set random seed
    seed = 42
    key = jax.random.key(seed)
    rngs = nnx.Rngs(params=key, sample=key)

    # Create audio generator
    print("\nCreating audio generator...")
    sample_rate = 16000  # 16 kHz
    duration = 0.5  # 0.5 seconds

    generator = SimpleAudioGenerator(
        sample_rate=sample_rate, duration=duration, latent_dim=32, rngs=rngs
    )

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print(f"Samples per waveform: {generator.num_samples}")

    # Generate random audio
    print("\nGenerating random audio waveforms...")
    waveforms = generator.generate(batch_size=3, rngs=rngs)
    print(f"Generated waveforms shape: {waveforms.shape}")

    # Visualize waveforms
    print("\nVisualizing waveforms...")
    fig1 = visualize_waveforms(waveforms, sample_rate, "Random Generated Audio")

    # Generate variations of a sound
    print("\nGenerating variations of a base sound...")
    base_key = jax.random.key(123)
    base_rngs = nnx.Rngs(sample=base_key)
    base_latent = jax.random.normal(base_rngs.sample(), (32,))

    variations = generator.generate_with_variation(
        base_latent, variation_scale=0.2, num_variations=4, rngs=base_rngs
    )
    print(f"Variations shape: {variations.shape}")

    # Visualize variations
    fig2 = visualize_waveforms(variations, sample_rate, "Audio Variations")

    # Generate spectrogram for one waveform
    print("\nGenerating spectrogram...")
    fig3 = generate_spectrogram(waveforms[0], sample_rate)

    # Save figures
    import os

    output_dir = "examples_output"
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, "audio_waveforms.png"))
    fig2.savefig(os.path.join(output_dir, "audio_variations.png"))
    fig3.savefig(os.path.join(output_dir, "audio_spectrogram.png"))

    print(f"\nResults saved to {output_dir}/")
    print("Simple audio generation example completed successfully!")


if __name__ == "__main__":
    main()
