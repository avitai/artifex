"""Audio processing extensions for generative models.

This package provides audio analysis and processing utilities
for generative models working with audio data.
"""

from .spectral import SpectralAnalysis
from .temporal import TemporalAnalysis


__all__ = [
    "SpectralAnalysis",
    "TemporalAnalysis",
]
