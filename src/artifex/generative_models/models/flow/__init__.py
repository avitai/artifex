"""Flow-based generative models package."""

# Base classes
from artifex.generative_models.models.flow.base import (
    FlowLayer,
    NormalizingFlow,
)

# Conditional flows
from artifex.generative_models.models.flow.conditional import (
    ConditionalCouplingLayer,
    ConditionalNormalizingFlow,
    ConditionalRealNVP,
)

# Factory functions have been moved to the centralized factory
# Use: from artifex.generative_models.factory import create_model
# Glow implementation
from artifex.generative_models.models.flow.glow import (
    ActNormLayer,
    AffineCouplingLayer,
    Glow,
    GlowBlock,
    InvertibleConv1x1,
)

# IAF implementation
from artifex.generative_models.models.flow.iaf import (
    IAF,
    IAFLayer,
)

# MAF implementation
from artifex.generative_models.models.flow.maf import (
    MADE,
    MAF,
    MAFLayer,
)

# Neural Spline Flow implementation
from artifex.generative_models.models.flow.neural_spline import (
    NeuralSplineFlow,
    RationalQuadraticSplineTransform,
    SplineCouplingLayer,
)

# RealNVP implementation
from artifex.generative_models.models.flow.real_nvp import (
    CouplingLayer,
    RealNVP,
)


# Aliases for compatibility
GlowFlow = Glow
Flow = NormalizingFlow

__all__ = [
    # Base classes
    "FlowLayer",
    "NormalizingFlow",
    "Flow",  # Alias
    # IAF
    "IAF",
    "IAFLayer",
    # MAF
    "MAF",
    "MAFLayer",
    "MADE",
    # Neural Spline Flow
    "NeuralSplineFlow",
    "SplineCouplingLayer",
    "RationalQuadraticSplineTransform",
    # RealNVP
    "RealNVP",
    "CouplingLayer",
    # Glow
    "Glow",
    "GlowFlow",  # Alias
    "GlowBlock",
    "ActNormLayer",
    "InvertibleConv1x1",
    "AffineCouplingLayer",
    # Conditional flows
    "ConditionalNormalizingFlow",
    "ConditionalRealNVP",
    "ConditionalCouplingLayer",
]
