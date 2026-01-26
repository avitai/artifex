"""Probability distributions for generative models.

This package provides implementations of various probability distributions
used in generative models, including continuous and discrete distributions.
"""

from artifex.generative_models.core.distributions.continuous import (
    Beta,
    Normal,
)
from artifex.generative_models.core.distributions.discrete import (
    Bernoulli,
    Categorical,
    OneHotCategorical,
)
from artifex.generative_models.core.distributions.mixture import (
    Mixture,
    MixtureOfGaussians,
)
from artifex.generative_models.core.distributions.transformations import (
    AffineTransform,
    TransformedDistribution,
)


__all__ = [
    # Continuous distributions
    "Normal",
    "Beta",
    # Discrete distributions
    "Bernoulli",
    "Categorical",
    "OneHotCategorical",
    # Mixture models
    "Mixture",
    "MixtureOfGaussians",
    # Transformations
    "AffineTransform",
    "TransformedDistribution",
]
