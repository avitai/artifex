"""Audio modality for generative models.

This module provides complete audio generation capabilities including:
- Multiple representation support (waveforms, spectrograms)
- WaveNet and diffusion-based generation models
- Complete audio evaluation metrics
- Integration with benchmark framework

Example:
    >>> from artifex.generative_models.modalities.audio import AudioModality, AudioRepresentation
    >>> modality = AudioModality(representation=AudioRepresentation.RAW_WAVEFORM)
    >>> audio = modality.generate(n_samples=1, duration=2.0)
"""

from .base import (
    AudioGenerationProtocol,
    AudioModality,
    AudioModalityConfig,
    AudioRepresentation,
    create_audio_modality,
)
from .datasets import (
    create_audio_dataset,
    generate_synthetic_audio,
)
from .evaluation import (
    AudioEvaluationSuite,
    AudioMetrics,
    compute_audio_metrics,
)
from .representations import (
    AudioProcessor,
    SpectrogramProcessor,
    WaveformProcessor,
)


__all__ = [
    # Core modality
    "AudioModality",
    "AudioGenerationProtocol",
    "AudioModalityConfig",
    "AudioRepresentation",
    "create_audio_modality",
    # Representation processing
    "AudioProcessor",
    "WaveformProcessor",
    "SpectrogramProcessor",
    # Dataset handling
    "generate_synthetic_audio",
    "create_audio_dataset",
    # Evaluation
    "AudioEvaluationSuite",
    "AudioMetrics",
    "compute_audio_metrics",
]
