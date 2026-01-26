"""Image-specific evaluation metrics."""

from .fid import FrechetInceptionDistance
from .inception_score import InceptionScore


__all__ = [
    "FrechetInceptionDistance",
    "InceptionScore",
]
