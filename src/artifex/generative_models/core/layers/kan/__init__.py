"""Kolmogorov-Arnold Network (KAN) layers.

Provides spline-based and polynomial-basis KAN layer variants for use
as drop-in replacements for standard dense layers in neural networks.

Adapted from jaxKAN (MIT license) with Artifex conventions:
- Flax NNX module base (``nnx.Module``)
- ``*, rngs: nnx.Rngs`` keyword-only parameter
- ``deterministic: bool`` in ``__call__``
- Shared DRY initialization via ``_initialize_kan_params``

Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks" (arXiv:2404.19756)
"""

from artifex.generative_models.core.layers.kan.basis_layers import (
    ChebyshevKANLayer,
    FourierKANLayer,
    LegendreKANLayer,
    RBFKANLayer,
    SineKANLayer,
)
from artifex.generative_models.core.layers.kan.config import KANConfig
from artifex.generative_models.core.layers.kan.conv import ConvKANLayer
from artifex.generative_models.core.layers.kan.grids import (
    BSplineBasis,
    DenseKANGrid,
    EfficientKANGrid,
    RBFKANGrid,
)
from artifex.generative_models.core.layers.kan.spline import (
    DenseKANLayer,
    EfficientKANLayer,
)


# Layer type registry for factory function
_KAN_LAYER_REGISTRY: dict[str, type] = {
    "dense": DenseKANLayer,
    "efficient": EfficientKANLayer,
    "chebyshev": ChebyshevKANLayer,
    "fourier": FourierKANLayer,
    "legendre": LegendreKANLayer,
    "rbf": RBFKANLayer,
    "sine": SineKANLayer,
    "conv": ConvKANLayer,
}


def create_kan_layer(layer_type: str, **kwargs: object) -> object:
    """Factory function to create a KAN layer by type name.

    Args:
        layer_type: One of ``"dense"``, ``"efficient"``,
            ``"chebyshev"``, ``"fourier"``, ``"legendre"``,
            ``"rbf"``, ``"sine"``, ``"conv"``.
        **kwargs: Keyword arguments forwarded to the layer constructor.

    Returns:
        An instantiated KAN layer module.

    Raises:
        ValueError: If ``layer_type`` is not recognised.
    """
    cls = _KAN_LAYER_REGISTRY.get(layer_type)
    if cls is None:
        valid = sorted(_KAN_LAYER_REGISTRY)
        raise ValueError(f"Unknown KAN layer type '{layer_type}'. Valid types: {valid}")
    return cls(**kwargs)


__all__ = [
    # Config
    "KANConfig",
    # Grids
    "BSplineBasis",
    "DenseKANGrid",
    "EfficientKANGrid",
    "RBFKANGrid",
    # Spline layers
    "DenseKANLayer",
    "EfficientKANLayer",
    # Basis layers
    "ChebyshevKANLayer",
    "FourierKANLayer",
    "LegendreKANLayer",
    "RBFKANLayer",
    "SineKANLayer",
    # Convolutional
    "ConvKANLayer",
    # Factory
    "create_kan_layer",
]
